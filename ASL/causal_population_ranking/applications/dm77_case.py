"""Synthetic DM77-aligned population-health application benchmark.

This module is deliberately self-contained and reproducible.  It compares the
proposed contrastive causal ranker against open prognostic stratification
baselines (ACG-inspired, not ACG implementations), public causal meta-learners,
and non-personalized allocation policies under identical constraints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contrastive_ranker import (
    fit_contrastive_causal_ranker,
    sample_stable_ranking_pairs,
)


FEATURE_COLUMNS = (
    "age", "female", "chronic_count", "frailty", "prior_admissions",
    "prior_ed_visits", "medication_count", "social_vulnerability",
    "digital_access", "rural", "need_score",
)
PROFILE_LABELS = {
    1: "remote_monitoring",
    2: "home_care",
    3: "multidisciplinary_case_management",
}
PROFILE_COST = {1: 1.0, 2: 2.0, 3: 3.5}


@dataclass(frozen=True)
class DM77Cohort:
    learner: pd.DataFrame
    oracle: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class DM77Supervision:
    opportunities: pd.DataFrame
    signal_columns: tuple[str, ...]
    audit: dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def generate_dm77_cohort(
    n: int,
    *,
    seed: int,
    response_scenario: str = "partially_aligned",
) -> DM77Cohort:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(72, 11, n), 35, 98)
    female = rng.binomial(1, 0.54, n)
    chronic_count = np.clip(rng.poisson(2.6, n), 0, 9)
    frailty = np.clip(
        _sigmoid(-3.8 + 0.045 * age + 0.28 * chronic_count + rng.normal(0, 0.7, n)),
        0, 1,
    )
    prior_admissions = np.clip(
        rng.poisson(0.25 + 0.16 * chronic_count + 0.7 * frailty), 0, 8
    )
    prior_ed = np.clip(
        rng.poisson(0.45 + 0.20 * chronic_count + 0.8 * frailty), 0, 12
    )
    medication_count = np.clip(
        np.rint(1.5 + 1.25 * chronic_count + rng.normal(0, 2.0, n)), 0, 25
    ).astype(int)
    social = np.clip(rng.beta(2.0, 3.2, n) + 0.12 * (age > 82), 0, 1)
    digital = np.clip(
        1.0 - 0.008 * (age - 50) - 0.28 * social + rng.normal(0, 0.16, n),
        0, 1,
    )
    rural = rng.binomial(1, _sigmoid(-0.6 + 1.2 * social))
    latent_response = rng.normal(0, 1, n)
    risk_linear = (
        -3.1 + 0.035 * (age - 60) + 0.34 * chronic_count + 1.35 * frailty
        + 0.22 * prior_admissions + 0.13 * prior_ed + 0.55 * social
    )
    baseline_risk = _sigmoid(risk_linear)
    need_score = np.clip(
        0.22 * chronic_count + 1.6 * frailty + 0.17 * prior_admissions
        + 0.08 * prior_ed + 0.45 * social,
        0, None,
    )
    need_level = np.digitize(
        need_score,
        np.quantile(need_score, [0.15, 0.35, 0.58, 0.78, 0.93]),
    )

    y0 = np.clip(
        358.0 - 31.0 * baseline_risk - 2.2 * prior_admissions
        + rng.normal(0, 3.0, n),
        250, 365,
    )
    window = np.exp(-0.5 * ((baseline_risk - 0.48) / 0.22) ** 2)
    monitoring = (
        1.0 + 6.2 * digital * baseline_risk + 1.4 * (chronic_count >= 2)
        - 2.5 * frailty + 0.8 * latent_response
    )
    home = (
        0.5 + 7.5 * frailty + 2.0 * rural + 1.5 * prior_admissions
        - 1.0 * digital + 0.8 * latent_response
    )
    case = (
        0.8 + 2.0 * chronic_count + 4.8 * social + 2.6 * window
        - 5.5 * np.maximum(baseline_risk - 0.82, 0) ** 2
        + 1.1 * latent_response
    )
    if response_scenario == "risk_benefit_aligned":
        monitoring += 7.0 * baseline_risk
        home += 5.0 * baseline_risk
        case += 8.0 * baseline_risk
    elif response_scenario == "risk_benefit_misaligned":
        monitoring += 7.0 * window - 7.0 * baseline_risk
        home += 5.5 * window - 5.0 * baseline_risk
        case += 9.0 * window - 9.0 * baseline_risk
    elif response_scenario == "mixed_response":
        non_response = rng.random((n, 3)) < np.array([0.22, 0.18, 0.25])
        negative = rng.random((n, 3)) < np.array([0.05, 0.04, 0.08])
    elif response_scenario != "partially_aligned":
        raise ValueError(f"Unknown DM77 response scenario: {response_scenario}")

    effects = np.column_stack([monitoring, home, case])
    if response_scenario == "mixed_response":
        effects[non_response] = rng.normal(0, 0.15, int(non_response.sum()))
        effects[negative] -= rng.uniform(1.0, 3.5, int(negative.sum()))
    effects = np.clip(effects, -4.0, 18.0)
    potential = np.column_stack([y0, *(np.clip(y0 + effects[:, j], 0, 365) for j in range(3))])

    eligible = np.column_stack([
        chronic_count >= 1,
        (frailty >= 0.22) | (prior_admissions >= 1),
        (chronic_count >= 3) | (social >= 0.42) | (prior_ed >= 3),
    ])
    logits = np.column_stack([
        np.zeros(n),
        -1.2 + 0.7 * baseline_risk + 0.9 * digital + 0.15 * chronic_count,
        -1.4 + 1.2 * frailty + 0.5 * rural + 0.2 * prior_admissions,
        -1.7 + 0.32 * chronic_count + 0.9 * social + 0.12 * prior_ed,
    ])
    logits[:, 1:][~eligible] = -20.0
    probability = _softmax(logits)
    treatment = np.array([rng.choice(4, p=p) for p in probability], dtype=int)
    observed = potential[np.arange(n), treatment] + rng.normal(0, 1.5, n)
    observed = np.clip(observed, 0, 365)
    future_utilization = np.clip(
        (365.0 - observed) / 6.0 + 0.6 * prior_ed + rng.normal(0, 1.0, n),
        0,
        None,
    )

    split_labels = np.array(
        ["nuisance_train", "rank_train", "validation", "calibration", "test"]
    )
    split_probability = np.array([0.30, 0.30, 0.15, 0.10, 0.15])
    split = rng.choice(split_labels, size=n, p=split_probability)
    learner = pd.DataFrame({
        "patient_id": [f"dm77_{i:06d}" for i in range(n)],
        "split": split,
        "age": age,
        "female": female,
        "chronic_count": chronic_count,
        "frailty": frailty,
        "prior_admissions": prior_admissions,
        "prior_ed_visits": prior_ed,
        "medication_count": medication_count,
        "social_vulnerability": social,
        "digital_access": digital,
        "rural": rural,
        "need_score": need_score,
        "need_level": need_level,
        "assigned_profile": treatment,
        "observed_outcome": observed,
        "future_utilization": future_utilization,
    })
    for p in range(1, 4):
        learner[f"eligible_{p}"] = eligible[:, p - 1]
    oracle = pd.DataFrame({
        "patient_id": learner.patient_id,
        "true_baseline_risk": baseline_risk,
        "y0": potential[:, 0],
        "y1": potential[:, 1],
        "y2": potential[:, 2],
        "y3": potential[:, 3],
        "tau_1": potential[:, 1] - potential[:, 0],
        "tau_2": potential[:, 2] - potential[:, 0],
        "tau_3": potential[:, 3] - potential[:, 0],
    })
    correlation = {
        f"risk_tau_{p}_spearman": float(spearmanr(baseline_risk, oracle[f"tau_{p}"]).statistic)
        for p in range(1, 4)
    }
    return DM77Cohort(
        learner=learner,
        oracle=oracle,
        audit={
            "data_level": "fully_synthetic_dm77_aligned",
            "patients": n,
            "response_scenario": response_scenario,
            "oracle_used_for_training": False,
            **correlation,
        },
    )


def _processor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = ["female", "rural"]
    numeric = [c for c in FEATURE_COLUMNS if c not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("categorical", encoder, categorical),
        ("numeric", StandardScaler(), numeric),
    ])


def _predict_propensity(model: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 4), dtype=float)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return out


def build_dm77_supervision(learner: pd.DataFrame, config: Mapping[str, Any]) -> DM77Supervision:
    nuisance = learner.loc[learner.split.eq("nuisance_train")].reset_index(drop=True)
    target = learner.loc[learner.split.ne("nuisance_train")].reset_index(drop=True)
    settings = config["causal_supervision"]
    processor = _processor(nuisance)
    nx = np.asarray(processor.fit_transform(nuisance[list(FEATURE_COLUMNS)]), dtype=np.float32)
    tx = np.asarray(processor.transform(target[list(FEATURE_COLUMNS)]), dtype=np.float32)
    a = nuisance.assigned_profile.to_numpy(int)
    y = nuisance.observed_outcome.to_numpy(float)
    ta = target.assigned_profile.to_numpy(int)
    ty = target.observed_outcome.to_numpy(float)
    repeats = int(settings.get("nuisance_repeats", 3))
    folds = int(settings.get("nuisance_folds", 3))
    clip = float(settings.get("propensity_clip", 0.04))
    seed = int(settings.get("nuisance_seed", 1001))
    repeated: list[np.ndarray] = []
    for repeat in range(repeats):
        splitter = KFold(folds, shuffle=True, random_state=seed + repeat * 101)
        predictions = []
        for fold, (fit_index, _) in enumerate(splitter.split(nx)):
            model_seed = seed + repeat * 101 + fold
            prop = HistGradientBoostingClassifier(
                max_iter=int(settings.get("propensity_model_max_iter", 50)),
                max_leaf_nodes=15,
                learning_rate=0.06,
                random_state=model_seed,
            ).fit(nx[fit_index], a[fit_index])
            e = np.clip(_predict_propensity(prop, tx), clip, 1 - clip)
            e /= e.sum(axis=1, keepdims=True)
            mu = np.zeros((len(target), 4), dtype=float)
            for p in range(4):
                idx = fit_index[a[fit_index] == p]
                if len(idx) < 12:
                    raise ValueError(f"Too few observations for DM77 profile {p}")
                model = HistGradientBoostingRegressor(
                    max_iter=int(settings.get("outcome_model_max_iter", 60)),
                    max_leaf_nodes=15,
                    learning_rate=0.06,
                    random_state=model_seed + p + 1,
                ).fit(nx[idx], y[idx])
                mu[:, p] = model.predict(tx)
            dr = mu.copy()
            rows = np.arange(len(target))
            dr[rows, ta] += (ty - mu[rows, ta]) / e[rows, ta]
            predictions.append(dr)
        mean_dr = np.mean(predictions, axis=0)
        repeated.append(np.column_stack([mean_dr[:, p] - mean_dr[:, 0] for p in range(1, 4)]))

    blocks = []
    for p in range(1, 4):
        mask = target[f"eligible_{p}"].to_numpy(bool)
        block = target.loc[mask, ["patient_id", "split", *FEATURE_COLUMNS]].copy()
        block["profile_id"] = p
        block["profile_name"] = PROFILE_LABELS[p]
        for r, contrast in enumerate(repeated):
            block[f"_dr_repeat_{r}"] = contrast[mask, p - 1]
        blocks.append(block)
    opportunities = pd.concat(blocks, ignore_index=True)
    columns = tuple(f"_dr_repeat_{r}" for r in range(repeats))
    return DM77Supervision(
        opportunities=opportunities,
        signal_columns=columns,
        audit={
            "supervision": "repeated_doubly_robust_multi_profile_contrasts",
            "oracle_inputs_used": False,
            "nuisance_rows": len(nuisance),
            "opportunities": len(opportunities),
            "repeats": repeats,
            "folds": folds,
        },
    )


def sample_pairs(
    opportunities: pd.DataFrame,
    signal_columns: tuple[str, ...],
    *,
    split: str,
    maximum_pairs: int,
    minimum_gap: float,
    minimum_repeat_agreement: float,
    pair_type_fractions: Mapping[str, float],
    top_region_fraction: float,
    top_pair_multiplier: float,
    gap_clip_quantile: float,
    seed: int,
) -> pd.DataFrame:
    """DM77 wrapper around the shared reliable pair sampler."""
    return sample_stable_ranking_pairs(
        opportunities,
        signal_columns,
        split=split,
        unit_id_column="patient_id",
        treatment_column="profile_name",
        maximum_pairs=maximum_pairs,
        minimum_signal_difference=minimum_gap,
        minimum_repeat_agreement=minimum_repeat_agreement,
        pair_type_fractions=pair_type_fractions,
        top_region_fraction=top_region_fraction,
        top_pair_multiplier=top_pair_multiplier,
        gap_clip_quantile=gap_clip_quantile,
        seed=seed,
    )


def _fit_x_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    processor: ColumnTransformer,
    profile: int,
    seed: int,
) -> np.ndarray:
    subset = train.loc[train.assigned_profile.isin([0, profile])].reset_index(drop=True)
    x = np.asarray(processor.transform(subset[list(FEATURE_COLUMNS)]), dtype=float)
    xt = np.asarray(processor.transform(test[list(FEATURE_COLUMNS)]), dtype=float)
    d = subset.assigned_profile.eq(profile).to_numpy(int)
    y = subset.observed_outcome.to_numpy(float)
    mu0 = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15, random_state=seed).fit(x[d == 0], y[d == 0])
    mu1 = HistGradientBoostingRegressor(max_iter=60, max_leaf_nodes=15, random_state=seed + 1).fit(x[d == 1], y[d == 1])
    d1 = y[d == 1] - mu0.predict(x[d == 1])
    d0 = mu1.predict(x[d == 0]) - y[d == 0]
    tau1 = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, random_state=seed + 2).fit(x[d == 1], d1)
    tau0 = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, random_state=seed + 3).fit(x[d == 0], d0)
    prop = HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=15, random_state=seed + 4).fit(x, d)
    e = np.clip(prop.predict_proba(xt)[:, 1], 0.05, 0.95)
    return (1.0 - e) * tau1.predict(xt) + e * tau0.predict(xt)


def _milp_allocate(opportunities: pd.DataFrame, value: np.ndarray, *, budget: float, capacities: Mapping[int, int]) -> np.ndarray:
    frame = opportunities.reset_index(drop=True)
    n = len(frame)
    patients = sorted(frame.patient_id.unique())
    patient_index = {p: i for i, p in enumerate(patients)}
    profile_rows = {p: np.flatnonzero(frame.profile_id.to_numpy(int) == p) for p in capacities}
    rows, cols, data, lower, upper = [], [], [], [], []
    r = 0
    for patient in patients:
        idx = np.flatnonzero(frame.patient_id.to_numpy() == patient)
        rows.extend([r] * len(idx)); cols.extend(idx); data.extend([1.0] * len(idx))
        lower.append(-np.inf); upper.append(1.0); r += 1
    for p, cap in capacities.items():
        idx = profile_rows[p]
        rows.extend([r] * len(idx)); cols.extend(idx); data.extend([1.0] * len(idx))
        lower.append(-np.inf); upper.append(float(cap)); r += 1
    cost = frame.profile_id.map(PROFILE_COST).to_numpy(float)
    rows.extend([r] * n); cols.extend(range(n)); data.extend(cost.tolist())
    lower.append(-np.inf); upper.append(float(budget))
    matrix = coo_matrix((data, (rows, cols)), shape=(r + 1, n)).tocsr()
    result = milp(
        c=-np.asarray(value, dtype=float),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 30.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"DM77 MILP failed: {result.message}")
    return result.x >= 0.5


def _concordance(score: np.ndarray, truth: np.ndarray, seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    m = min(20000, len(score) * 8)
    left = rng.integers(0, len(score), m)
    right = rng.integers(0, len(score), m)
    valid = (left != right) & (truth[left] != truth[right])
    return float(np.mean((score[left[valid]] - score[right[valid]]) * (truth[left[valid]] - truth[right[valid]]) > 0))


def _cross_profile_concordance(
    score: np.ndarray,
    truth: np.ndarray,
    profile: np.ndarray,
    patient: np.ndarray,
    seed: int = 2,
) -> float:
    rng = np.random.default_rng(seed)
    m = min(30000, len(score) * 12)
    left = rng.integers(0, len(score), m)
    right = rng.integers(0, len(score), m)
    valid = (
        (left != right)
        & (patient[left] != patient[right])
        & (profile[left] != profile[right])
        & (truth[left] != truth[right])
    )
    if not valid.any():
        return float("nan")
    product = (score[left[valid]] - score[right[valid]]) * (
        truth[left[valid]] - truth[right[valid]]
    )
    return float(np.mean(product > 0.0))


def _within_patient_recommendation_accuracy(
    score: np.ndarray,
    truth: np.ndarray,
    patient: np.ndarray,
) -> float:
    correct = []
    frame = pd.DataFrame({
        "patient": patient,
        "score": score,
        "truth": truth,
        "row": np.arange(len(score)),
    })
    for _, group in frame.groupby("patient", sort=False):
        if len(group) < 2:
            continue
        predicted = int(group.iloc[int(np.argmax(group.score.to_numpy()))].row)
        actual = int(group.iloc[int(np.argmax(group.truth.to_numpy()))].row)
        correct.append(predicted == actual)
    return float(np.mean(correct)) if correct else float("nan")


def _ndcg_fraction(score: np.ndarray, truth: np.ndarray, fraction: float = 0.25) -> float:
    k = max(1, min(len(score), int(round(len(score) * fraction))))
    relevance = pd.Series(truth).rank(method="average", pct=True).to_numpy(float)
    predicted = np.argsort(-score)[:k]
    ideal = np.argsort(-relevance)[:k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[predicted] * discount))
    idcg = float(np.sum(relevance[ideal] * discount))
    return dcg / idcg if idcg > 0 else float("nan")


def build_dm77_baselines(
    cohort: DM77Cohort,
    supervision: DM77Supervision,
    ranker: Any,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    learner = cohort.learner
    nuisance = learner.loc[learner.split.eq("nuisance_train")].reset_index(drop=True)
    calibration = supervision.opportunities.loc[supervision.opportunities.split.eq("calibration")].reset_index(drop=True)
    test = supervision.opportunities.loc[supervision.opportunities.split.eq("test")].reset_index(drop=True)
    test_patient = learner.set_index("patient_id").loc[test.patient_id].reset_index()
    processor = _processor(nuisance)
    nx = np.asarray(processor.fit_transform(nuisance[list(FEATURE_COLUMNS)]), dtype=float)
    tx = np.asarray(processor.transform(test_patient[list(FEATURE_COLUMNS)]), dtype=float)
    cal_signal = calibration[list(supervision.signal_columns)].mean(axis=1).to_numpy(float)
    cal_score = ranker.score(calibration)
    test_score = ranker.score(test)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(cal_score, cal_signal)
    prometheus = calibrator.predict(test_score)

    util_ridge = Ridge(alpha=1.0).fit(nx, nuisance.future_utilization)
    util_gbdt = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, random_state=311).fit(nx, nuisance.future_utilization)
    util_lr_score = util_ridge.predict(tx)
    util_gbdt_score = util_gbdt.predict(tx)
    train_util_gbdt = util_gbdt.predict(nx)
    thresholds = np.quantile(train_util_gbdt, [0.20, 0.40, 0.60, 0.80, 0.95])
    utilization_band = np.digitize(util_gbdt_score, thresholds)
    morbidity_band = np.clip(test_patient.chronic_count.to_numpy(int), 0, 5)
    need = test_patient.need_score.to_numpy(float)

    profile_mean = {}
    rank_train = supervision.opportunities.loc[supervision.opportunities.split.eq("rank_train")]
    for p in PROFILE_LABELS:
        profile_mean[p] = float(rank_train.loc[rank_train.profile_id.eq(p), list(supervision.signal_columns)].to_numpy(float).mean())
    mean_vector = test.profile_id.map(profile_mean).to_numpy(float)

    t_learner: dict[int, np.ndarray] = {}
    x_learner: dict[int, np.ndarray] = {}
    dr_gbdt: dict[int, np.ndarray] = {}
    rank_processor = _processor(rank_train)
    rx = np.asarray(rank_processor.fit_transform(rank_train[list(FEATURE_COLUMNS)]), dtype=float)
    test_op_x = np.asarray(rank_processor.transform(test[list(FEATURE_COLUMNS)]), dtype=float)
    for p in PROFILE_LABELS:
        mu = {}
        for arm in (0, p):
            idx = nuisance.assigned_profile.eq(arm).to_numpy()
            mu[arm] = HistGradientBoostingRegressor(max_iter=70, max_leaf_nodes=15, random_state=500 + 10 * p + arm).fit(nx[idx], nuisance.loc[idx, "observed_outcome"])
        t_learner[p] = mu[p].predict(tx) - mu[0].predict(tx)
        x_learner[p] = _fit_x_learner(nuisance, test_patient, processor, p, 600 + p * 10)
        mask = rank_train.profile_id.eq(p).to_numpy()
        target = rank_train.loc[mask, list(supervision.signal_columns)].mean(axis=1)
        dr_model = HistGradientBoostingRegressor(max_iter=70, max_leaf_nodes=15, random_state=700 + p).fit(rx[mask], target)
        dr_gbdt[p] = dr_model.predict(test_op_x)

    values: dict[str, np.ndarray] = {
        "PROMETHEUS-Contrastive": prometheus,
        "Open morbidity bands": morbidity_band * np.maximum(mean_vector, 0.05),
        "Open utilization bands": utilization_band * np.maximum(mean_vector, 0.05),
        "Prospective utilization Ridge": util_lr_score * np.maximum(mean_vector, 0.05),
        "Prospective utilization GBDT": util_gbdt_score * np.maximum(mean_vector, 0.05),
        "DM77 need-first": need * np.maximum(mean_vector, 0.05),
        "Profile-mean DR": mean_vector,
    }
    profile_array = test.profile_id.to_numpy(int)
    values["T-learner GBDT"] = np.asarray([t_learner[p][i] for i, p in enumerate(profile_array)])
    values["X-learner GBDT"] = np.asarray([x_learner[p][i] for i, p in enumerate(profile_array)])
    values["DR-learner GBDT"] = np.asarray([dr_gbdt[p][i] for i, p in enumerate(profile_array)])
    values["DR-Random Forest"] = np.zeros(len(test))
    for p in PROFILE_LABELS:
        mask_train = rank_train.profile_id.eq(p).to_numpy()
        mask_test = profile_array == p
        target = rank_train.loc[mask_train, list(supervision.signal_columns)].mean(axis=1)
        rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=12, random_state=800+p, n_jobs=-1).fit(rx[mask_train], target)
        values["DR-Random Forest"][mask_test] = rf.predict(test_op_x[mask_test])
    rng = np.random.default_rng(901)
    values["Random"] = rng.normal(0, 1, len(test))

    oracle = cohort.oracle.set_index("patient_id")
    truth = np.asarray([
        oracle.loc[patient, f"tau_{profile}"]
        for patient, profile in zip(test.patient_id, profile_array)
    ], dtype=float)
    values["Oracle"] = truth.copy()

    test_count = test.patient_id.nunique()
    capacities = {
        1: max(1, int(round(test_count * float(config["allocation"]["capacity_fraction_monitoring"])) )),
        2: max(1, int(round(test_count * float(config["allocation"]["capacity_fraction_home_care"])) )),
        3: max(1, int(round(test_count * float(config["allocation"]["capacity_fraction_case_management"])) )),
    }
    budget = float(config["allocation"]["budget_multiplier"]) * sum(
        capacities[p] * PROFILE_COST[p] for p in capacities
    )
    result_rows = []
    score_rows = []
    patient_array = test.patient_id.astype(str).to_numpy()
    oracle_selected = _milp_allocate(test, truth, budget=budget, capacities=capacities)
    oracle_value = float(truth[oracle_selected].sum())
    for method, score in values.items():
        selected = _milp_allocate(test, score, budget=budget, capacities=capacities)
        true_value = float(truth[selected].sum())
        result_rows.append({
            "application": "DM77",
            "method": method,
            "optimizer": "MILP",
            "true_value": true_value,
            "oracle_value": oracle_value,
            "normalized_value": true_value / oracle_value if oracle_value else np.nan,
            "regret": oracle_value - true_value,
            "served": int(selected.sum()),
            "ranking_concordance": _concordance(score, truth),
            "cross_profile_concordance": _cross_profile_concordance(
                score, truth, profile_array, patient_array
            ),
            "within_patient_recommendation_accuracy": _within_patient_recommendation_accuracy(
                score, truth, patient_array
            ),
            "ndcg_at_25pct": _ndcg_fraction(score, truth, 0.25),
            "oracle_access_for_policy_construction": method == "Oracle",
            "oracle_access_for_evaluation": True,
        })
        score_rows.extend({
            "patient_id": pid,
            "profile_id": int(p),
            "method": method,
            "allocator_value": float(v),
            "true_effect_evaluation_only": float(t),
            "selected": bool(s),
        } for pid, p, v, t, s in zip(test.patient_id, profile_array, score, truth, selected))
    return pd.DataFrame(result_rows), pd.DataFrame(score_rows)


def run_dm77_case(config: Mapping[str, Any], output_dir: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    n = int(config["dm77"]["patients_smoke" if smoke else "patients"])
    cohort = generate_dm77_cohort(
        n,
        seed=int(config["dm77"]["seed"]),
        response_scenario=str(config["dm77"]["response_scenario"]),
    )
    supervision = build_dm77_supervision(cohort.learner, config["dm77"])
    ranking = dict(config["dm77"]["ranking"])
    if smoke:
        ranking.update({
            "epochs": 4, "patience": 2,
            "maximum_contrastive_pairs": 600,
        })
    pairs = config["dm77"]["pairs"]
    pair_fractions = pairs.get("pair_type_fractions", {
        "within_unit": 0.30,
        "within_treatment": 0.35,
        "global_cross_treatment": 0.35,
    })
    pair_common = {
        "minimum_repeat_agreement": float(pairs.get("minimum_repeat_agreement", 0.80)),
        "pair_type_fractions": pair_fractions,
        "top_region_fraction": float(pairs.get("top_region_fraction", 0.25)),
        "top_pair_multiplier": float(pairs.get("top_pair_multiplier", 2.0)),
        "gap_clip_quantile": float(pairs.get("gap_clip_quantile", 0.95)),
    }
    train_pairs = sample_pairs(
        supervision.opportunities, supervision.signal_columns,
        split="rank_train", maximum_pairs=int(800 if smoke else pairs["maximum_train_pairs"]),
        minimum_gap=float(pairs["minimum_gap"]), seed=int(pairs["train_seed"]),
        **pair_common,
    )
    validation_pairs = sample_pairs(
        supervision.opportunities, supervision.signal_columns,
        split="validation", maximum_pairs=int(300 if smoke else pairs["maximum_validation_pairs"]),
        minimum_gap=float(pairs["minimum_gap"]), seed=int(pairs["validation_seed"]),
        **pair_common,
    )
    ranker = fit_contrastive_causal_ranker(
        supervision.opportunities,
        train_pairs,
        validation_pairs,
        feature_columns=FEATURE_COLUMNS,
        treatment_column="profile_name",
        signal_columns=supervision.signal_columns,
        unit_id_column="patient_id",
        config=ranking,
    )
    results, scores = build_dm77_baselines(cohort, supervision, ranker, config["dm77"])
    results.to_csv(output / "dm77_policy_results.csv", index=False)
    scores.to_csv(output / "dm77_opportunity_scores.csv", index=False)
    pd.DataFrame([
        {"method": "PROMETHEUS-Contrastive", "family": "proposed causal ranking", "profile_specific": True, "official_acg": False},
        {"method": "Open morbidity bands", "family": "open ACG-inspired prognostic", "profile_specific": False, "official_acg": False},
        {"method": "Open utilization bands", "family": "open ACG-inspired prognostic", "profile_specific": False, "official_acg": False},
        {"method": "Prospective utilization Ridge", "family": "open prognostic", "profile_specific": False, "official_acg": False},
        {"method": "Prospective utilization GBDT", "family": "open prognostic", "profile_specific": False, "official_acg": False},
        {"method": "DM77 need-first", "family": "descriptive need", "profile_specific": False, "official_acg": False},
        {"method": "T-learner GBDT", "family": "public causal meta-learner", "profile_specific": True, "official_acg": False},
        {"method": "X-learner GBDT", "family": "public causal meta-learner", "profile_specific": True, "official_acg": False},
        {"method": "DR-learner GBDT", "family": "public doubly robust learner", "profile_specific": True, "official_acg": False},
        {"method": "DR-Random Forest", "family": "public doubly robust learner", "profile_specific": True, "official_acg": False},
    ]).to_csv(output / "dm77_methodology.csv", index=False)
    (output / "dm77_dgp_audit.json").write_text(json.dumps(cohort.audit, indent=2), encoding="utf-8")
    (output / "dm77_supervision_audit.json").write_text(json.dumps(supervision.audit, indent=2), encoding="utf-8")
    (output / "dm77_ranking_audit.json").write_text(json.dumps(ranker.audit, indent=2), encoding="utf-8")
    return {
        "results": results,
        "cohort_audit": cohort.audit,
        "supervision_audit": supervision.audit,
        "ranking_audit": ranker.audit,
    }


def load_and_run_dm77(config_path: str | Path, output_dir: str | Path, *, smoke: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return run_dm77_case(config, output_dir, smoke=smoke)
