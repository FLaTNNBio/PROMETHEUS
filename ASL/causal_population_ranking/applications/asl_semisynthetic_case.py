"""ASL real-covariate semi-synthetic PROMETHEUS benchmark.

The patient population and baseline covariates are read from the pseudonymized
ASL ETL output. Treatment assignment, potential outcomes and follow-up outcomes
are simulated because those longitudinal causal fields are not present in the
current extract. Oracle quantities are evaluation-only.
"""
from __future__ import annotations

import copy
import gzip
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contrastive_ranker import fit_contrastive_causal_ranker
from .dm77_case import (
    DM77Supervision,
    PROFILE_COST,
    PROFILE_LABELS,
    _concordance,
    _cross_profile_concordance,
    _milp_allocate,
    _ndcg_fraction,
    _predict_propensity,
    _sigmoid,
    _softmax,
    _within_patient_recommendation_accuracy,
    sample_pairs,
)


ASL_FEATURE_COLUMNS = (
    "age",
    "female",
    "morbidity_proxy",
    "frailty_proxy",
    "prior_admissions",
    "log_los_days",
    "log_hospital_reimbursement",
    "log_exemption_records",
    "log_distinct_diagnoses",
    "income_exemption",
    "district",
)


@dataclass(frozen=True)
class ASLSemiSyntheticCohort:
    learner: pd.DataFrame
    oracle: pd.DataFrame
    baseline_summary: pd.DataFrame
    audit: dict[str, Any]


def _read_patient_features(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"ASL input not found: {path}")

    if path.is_dir():
        candidates = list(path.rglob("patient_features_2025.csv.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"patient_features_2025.csv.gz not found under {path}"
            )
        return pd.read_csv(candidates[0], compression="gzip")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist()
                if name.replace("\\", "/").endswith(
                    "curated/patient_features_2025.csv.gz"
                )
            ]
            if not members:
                members = [
                    name for name in archive.namelist()
                    if name.endswith("patient_features_2025.csv.gz")
                ]
            if len(members) != 1:
                raise FileNotFoundError(
                    "Expected exactly one patient_features_2025.csv.gz "
                    f"inside {path}; found {len(members)}"
                )
            with archive.open(members[0]) as raw:
                with gzip.GzipFile(fileobj=raw) as stream:
                    return pd.read_csv(stream)

    if path.name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        "ASL input must be output_acg_2025.zip, a directory containing the "
        "curated files, patient_features_2025.csv.gz, or a CSV file."
    )


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(float)


def _binary_any(frame: pd.DataFrame, *columns: str) -> np.ndarray:
    values = np.zeros(len(frame), dtype=bool)
    for column in columns:
        if column in frame:
            values |= _numeric(frame, column) > 0
    return values.astype(int)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-8:
        return np.zeros_like(values)
    return (values - float(np.nanmean(values))) / scale


def _processor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = ["district"]
    numeric = [c for c in ASL_FEATURE_COLUMNS if c not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("categorical", encoder, categorical),
        ("numeric", StandardScaler(), numeric),
    ])


def generate_asl_semisynthetic_cohort(
    input_path: str | Path,
    n: int | None,
    *,
    seed: int,
    response_scenario: str = "partially_aligned",
    minimum_age: int = 18,
) -> ASLSemiSyntheticCohort:
    source = _read_patient_features(input_path)
    required = {"patient_key", "age_at_index", "deceased_by_index"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required ASL columns: {sorted(missing)}")

    source_rows = len(source)
    active = source.loc[
        (_numeric(source, "deceased_by_index") == 0)
        & (_numeric(source, "age_at_index", -1) >= minimum_age)
    ].copy()
    active = active.drop_duplicates("patient_key", keep="first")
    eligible_rows = len(active)
    if eligible_rows < 100:
        raise ValueError("Too few active ASL patients after eligibility filtering")

    rng = np.random.default_rng(seed)
    if n is not None and n > 0 and n < len(active):
        selected = rng.choice(len(active), size=int(n), replace=False)
        active = active.iloc[np.sort(selected)].reset_index(drop=True)
    else:
        active = active.reset_index(drop=True)

    age = np.clip(_numeric(active, "age_at_index"), minimum_age, 110)
    female = active.get("sex", pd.Series("U", index=active.index)).astype(str).str.upper().eq("F").to_numpy(int)
    pathology = _binary_any(
        active,
        "pathology_exemption_registry_flag",
        "pathology_or_other_exemption_file_flag",
    )
    invalidity = _binary_any(
        active,
        "invalidity_exemption_registry_flag",
        "invalidity_exemption_file_flag",
    )
    income = _binary_any(
        active,
        "income_exemption_registry_flag",
        "income_exemption_file_flag",
    )
    exemption_records = np.clip(_numeric(active, "exemption_record_count"), 0, None)
    exemption_groups = np.clip(_numeric(active, "distinct_exemption_group_count"), 0, None)
    admissions = np.clip(_numeric(active, "hospitalization_count"), 0, None)
    los = np.clip(_numeric(active, "total_reported_los_days"), 0, None)
    reimbursement = np.clip(_numeric(active, "total_hospital_reimbursement"), 0, None)
    diagnoses = np.clip(_numeric(active, "distinct_diagnosis_count"), 0, None)
    district = (
        active.get("enrollment_district", pd.Series("missing", index=active.index))
        .fillna("missing").astype(str)
    )

    log_exemption = np.log1p(exemption_records)
    log_admissions = np.log1p(admissions)
    log_los = np.log1p(los)
    log_reimbursement = np.log1p(reimbursement)
    log_diagnoses = np.log1p(diagnoses)

    morbidity_proxy = np.clip(
        0.70 * exemption_groups
        + 0.90 * pathology
        + 0.35 * log_diagnoses
        + 0.20 * log_exemption,
        0,
        12,
    )
    frailty_proxy = _sigmoid(
        -4.35
        + 0.047 * age
        + 1.05 * invalidity
        + 0.42 * log_admissions
        + 0.24 * log_los
        + 0.16 * log_diagnoses
    )
    need_score = np.clip(
        0.018 * np.maximum(age - 45, 0)
        + 0.75 * morbidity_proxy
        + 1.65 * frailty_proxy
        + 0.65 * invalidity
        + 0.32 * income
        + 0.55 * log_admissions
        + 0.18 * log_los,
        0,
        None,
    )
    need_level = np.digitize(
        need_score,
        np.quantile(need_score, [0.15, 0.35, 0.58, 0.78, 0.93]),
    )

    baseline_risk = _sigmoid(
        -4.65
        + 0.036 * np.maximum(age - 40, 0)
        + 0.62 * morbidity_proxy
        + 1.20 * frailty_proxy
        + 0.58 * log_admissions
        + 0.20 * log_los
        + 0.16 * income
    )
    latent_response = rng.normal(0, 1, len(active))
    y0 = np.clip(
        362.0
        - 36.0 * baseline_risk
        - 2.8 * admissions
        - 0.18 * los
        + rng.normal(0, 2.5, len(active)),
        220,
        365,
    )
    response_window = np.exp(-0.5 * ((baseline_risk - 0.48) / 0.20) ** 2)
    remote = (
        0.6
        + 2.0 * pathology
        + 1.0 * log_exemption
        + 2.8 * response_window
        - 2.1 * frailty_proxy
        + 0.65 * latent_response
    )
    home = (
        0.4
        + 7.0 * frailty_proxy
        + 2.2 * invalidity
        + 1.2 * log_admissions
        + 0.35 * log_los
        - 1.0 * response_window
        + 0.70 * latent_response
    )
    case = (
        0.5
        + 1.5 * morbidity_proxy
        + 1.0 * income
        + 0.45 * log_reimbursement
        + 2.4 * response_window
        - 4.5 * np.maximum(baseline_risk - 0.84, 0) ** 2
        + 0.90 * latent_response
    )

    if response_scenario == "risk_benefit_aligned":
        remote += 6.5 * baseline_risk
        home += 5.0 * baseline_risk
        case += 7.5 * baseline_risk
    elif response_scenario == "risk_benefit_misaligned":
        remote += 6.5 * response_window - 7.5 * baseline_risk
        home += 5.0 * response_window - 5.5 * baseline_risk
        case += 8.0 * response_window - 8.5 * baseline_risk
    elif response_scenario == "mixed_response":
        pass
    elif response_scenario != "partially_aligned":
        raise ValueError(f"Unknown ASL response scenario: {response_scenario}")

    effects = np.column_stack([remote, home, case])
    if response_scenario == "mixed_response":
        non_response = rng.random(effects.shape) < np.array([0.24, 0.18, 0.23])
        negative = rng.random(effects.shape) < np.array([0.05, 0.04, 0.07])
        effects[non_response] = rng.normal(0, 0.18, int(non_response.sum()))
        effects[negative] -= rng.uniform(1.0, 4.0, int(negative.sum()))
    effects = np.clip(effects, -5.0, 18.0)
    potential = np.column_stack([
        y0,
        *(np.clip(y0 + effects[:, j], 0, 365) for j in range(3)),
    ])

    positive_morbidity = morbidity_proxy[morbidity_proxy > 0]
    morbidity_cutoff = (
        float(np.quantile(positive_morbidity, 0.55))
        if len(positive_morbidity) else 0.5
    )
    positive_cost = reimbursement[reimbursement > 0]
    cost_cutoff = (
        float(np.quantile(positive_cost, 0.75))
        if len(positive_cost) else 1.0
    )
    high_morbidity = morbidity_proxy >= max(morbidity_cutoff, 0.5)
    high_cost = reimbursement >= max(cost_cutoff, 1.0)
    eligible = np.column_stack([
        (pathology == 1) | (exemption_groups >= 1) | (diagnoses >= 2),
        (invalidity == 1) | (admissions >= 1) | (frailty_proxy >= 0.30) | (age >= 78),
        high_morbidity | high_cost | ((income == 1) & (pathology == 1)),
    ])

    logits = np.column_stack([
        np.zeros(len(active)),
        -1.05 + 0.55 * pathology + 0.30 * log_exemption + 0.25 * baseline_risk,
        -0.90 + 1.00 * frailty_proxy + 0.55 * invalidity + 0.30 * log_admissions,
        -1.15 + 0.35 * morbidity_proxy + 0.25 * income + 0.12 * log_reimbursement,
    ])
    logits[:, 1:][~eligible] = -20.0
    probability = _softmax(logits)
    treatment = np.array(
        [rng.choice(4, p=row) for row in probability], dtype=int
    )
    observed = potential[np.arange(len(active)), treatment] + rng.normal(
        0, 1.5, len(active)
    )
    observed = np.clip(observed, 0, 365)
    future_utilization = np.clip(
        (365.0 - observed) / 6.0
        + 0.45 * log_admissions
        + rng.normal(0, 0.9, len(active)),
        0,
        None,
    )

    split_labels = np.array([
        "nuisance_train", "rank_train", "validation", "calibration", "test"
    ])
    split_probability = np.array([0.30, 0.30, 0.15, 0.10, 0.15])
    split = rng.choice(split_labels, size=len(active), p=split_probability)

    learner = pd.DataFrame({
        "patient_id": active.patient_key.astype(str).to_numpy(),
        "split": split,
        "age": age,
        "female": female,
        "morbidity_proxy": morbidity_proxy,
        "frailty_proxy": frailty_proxy,
        "prior_admissions": admissions,
        "log_los_days": log_los,
        "log_hospital_reimbursement": log_reimbursement,
        "log_exemption_records": log_exemption,
        "log_distinct_diagnoses": log_diagnoses,
        "income_exemption": income,
        "district": district.to_numpy(),
        "need_score": need_score,
        "need_level": need_level,
        "assigned_profile": treatment,
        "observed_outcome": observed,
        "future_utilization": future_utilization,
    })
    for profile in range(1, 4):
        learner[f"eligible_{profile}"] = eligible[:, profile - 1]

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

    baseline_summary = pd.DataFrame([
        {"variable": "age", "mean": float(np.mean(age)), "sd": float(np.std(age)), "minimum": float(np.min(age)), "maximum": float(np.max(age))},
        {"variable": "female", "mean": float(np.mean(female)), "sd": float(np.std(female)), "minimum": float(np.min(female)), "maximum": float(np.max(female))},
        {"variable": "pathology_exemption", "mean": float(np.mean(pathology)), "sd": float(np.std(pathology)), "minimum": 0.0, "maximum": 1.0},
        {"variable": "invalidity_exemption", "mean": float(np.mean(invalidity)), "sd": float(np.std(invalidity)), "minimum": 0.0, "maximum": 1.0},
        {"variable": "income_exemption", "mean": float(np.mean(income)), "sd": float(np.std(income)), "minimum": 0.0, "maximum": 1.0},
        {"variable": "hospitalization_count", "mean": float(np.mean(admissions)), "sd": float(np.std(admissions)), "minimum": float(np.min(admissions)), "maximum": float(np.max(admissions))},
        {"variable": "morbidity_proxy", "mean": float(np.mean(morbidity_proxy)), "sd": float(np.std(morbidity_proxy)), "minimum": float(np.min(morbidity_proxy)), "maximum": float(np.max(morbidity_proxy))},
        {"variable": "frailty_proxy", "mean": float(np.mean(frailty_proxy)), "sd": float(np.std(frailty_proxy)), "minimum": float(np.min(frailty_proxy)), "maximum": float(np.max(frailty_proxy))},
    ])
    correlations = {
        f"risk_tau_{p}_spearman": float(
            spearmanr(baseline_risk, oracle[f"tau_{p}"]).statistic
        )
        for p in range(1, 4)
    }
    treatment_counts = learner.assigned_profile.value_counts().sort_index().to_dict()
    return ASLSemiSyntheticCohort(
        learner=learner,
        oracle=oracle,
        baseline_summary=baseline_summary,
        audit={
            "data_level": "real_asl_covariates_semisynthetic_treatment_and_outcomes",
            "input_path": str(Path(input_path).expanduser().resolve()),
            "source_rows": int(source_rows),
            "active_adult_rows_before_sampling": int(eligible_rows),
            "patients_used": int(len(active)),
            "minimum_age": int(minimum_age),
            "response_scenario": response_scenario,
            "real_components": [
                "patient population",
                "age and sex",
                "exemption-derived morbidity",
                "hospitalization utilization and reimbursement",
                "diagnosis burden",
                "district",
            ],
            "simulated_components": [
                "care-profile assignment",
                "potential outcomes",
                "observed follow-up outcome",
                "individual treatment effects",
            ],
            "oracle_used_for_training": False,
            "treatment_counts": {str(k): int(v) for k, v in treatment_counts.items()},
            "eligibility_rate_profile_1": float(eligible[:, 0].mean()),
            "eligibility_rate_profile_2": float(eligible[:, 1].mean()),
            "eligibility_rate_profile_3": float(eligible[:, 2].mean()),
            **correlations,
        },
    )


def _stabilize_multinomial_propensity(
    probability: np.ndarray,
    minimum_probability: float,
) -> np.ndarray:
    """Shrink multinomial propensities toward uniform support.

    The affine transformation preserves row sums and guarantees every arm a
    probability of at least ``minimum_probability``.  Clipping followed by
    renormalization does not preserve the requested lower bound.
    """
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2:
        raise ValueError("Multinomial propensity must be a two-dimensional array")
    arm_count = probability.shape[1]
    floor = float(minimum_probability)
    if not 0.0 <= floor < 1.0 / arm_count:
        raise ValueError(
            "minimum_probability must lie in [0, 1 / number_of_arms)"
        )
    probability = np.clip(probability, 0.0, None)
    row_sum = probability.sum(axis=1, keepdims=True)
    probability = np.divide(
        probability,
        row_sum,
        out=np.full_like(probability, 1.0 / arm_count),
        where=row_sum > 0,
    )
    return (1.0 - arm_count * floor) * probability + floor


def _stratified_bootstrap_indices(
    treatment: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = []
    for arm in np.unique(treatment):
        indices = np.flatnonzero(treatment == arm)
        if not len(indices):
            continue
        blocks.append(rng.choice(indices, size=len(indices), replace=True))
    if not blocks:
        raise ValueError("Cannot bootstrap an empty nuisance sample")
    combined = np.concatenate(blocks)
    rng.shuffle(combined)
    return combined


def build_asl_supervision(
    learner: pd.DataFrame,
    config: Mapping[str, Any],
) -> DM77Supervision:
    nuisance = learner.loc[learner.split.eq("nuisance_train")].reset_index(drop=True)
    target = learner.loc[learner.split.ne("nuisance_train")].reset_index(drop=True)
    settings = config["causal_supervision"]
    processor = _processor(nuisance)
    nx = np.asarray(
        processor.fit_transform(nuisance[list(ASL_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    tx = np.asarray(
        processor.transform(target[list(ASL_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    a = nuisance.assigned_profile.to_numpy(int)
    y = nuisance.observed_outcome.to_numpy(float)
    ta = target.assigned_profile.to_numpy(int)
    ty = target.observed_outcome.to_numpy(float)
    repeats = int(settings.get("nuisance_repeats", 3))
    clip = float(settings.get("propensity_clip", 0.04))
    seed = int(settings.get("nuisance_seed", 8102))
    fit_mode = str(settings.get("nuisance_fit_mode", "full_bootstrap"))
    if fit_mode not in {"full_bootstrap", "full_sample"}:
        raise ValueError(
            "nuisance_fit_mode must be 'full_bootstrap' or 'full_sample'"
        )

    arm_counts = {int(arm): int(np.sum(a == arm)) for arm in range(4)}
    minimum_arm_count = int(settings.get("minimum_nuisance_arm_count", 20))
    too_small = {arm: count for arm, count in arm_counts.items() if count < minimum_arm_count}
    if too_small:
        raise ValueError(
            "Too few nuisance observations for one or more treatment arms: "
            f"{too_small}; required at least {minimum_arm_count}"
        )

    repeated: list[np.ndarray] = []
    propensity_minima: list[float] = []
    propensity_ess: list[list[float]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 1000)
        if fit_mode == "full_bootstrap":
            idx = _stratified_bootstrap_indices(a, rng)
        else:
            idx = np.arange(len(nuisance))
        model_seed = seed + repeat * 100
        prop = HistGradientBoostingClassifier(
            max_iter=int(settings.get("propensity_model_max_iter", 60)),
            max_leaf_nodes=15,
            learning_rate=0.06,
            random_state=model_seed,
        ).fit(nx[idx], a[idx])
        e = _stabilize_multinomial_propensity(
            _predict_propensity(prop, tx),
            clip,
        )
        propensity_minima.append(float(e.min()))
        propensity_ess.append([
            float((e[:, arm].sum() ** 2) / np.square(e[:, arm]).sum())
            for arm in range(e.shape[1])
        ])

        mu = np.zeros((len(target), 4), dtype=float)
        for profile in range(4):
            arm_idx = idx[a[idx] == profile]
            if len(arm_idx) < minimum_arm_count:
                raise ValueError(
                    f"Too few nuisance observations for treatment arm {profile}: "
                    f"{len(arm_idx)}"
                )
            model = HistGradientBoostingRegressor(
                max_iter=int(settings.get("outcome_model_max_iter", 60)),
                max_leaf_nodes=15,
                learning_rate=0.06,
                random_state=model_seed + profile + 1,
            ).fit(nx[arm_idx], y[arm_idx])
            mu[:, profile] = model.predict(tx)

        dr = mu.copy()
        rows = np.arange(len(target))
        dr[rows, ta] += (ty - mu[rows, ta]) / e[rows, ta]
        repeated.append(np.column_stack([
            dr[:, profile] - dr[:, 0]
            for profile in range(1, 4)
        ]))

    blocks = []
    for profile in range(1, 4):
        mask = target[f"eligible_{profile}"].to_numpy(bool)
        block = target.loc[
            mask,
            ["patient_id", "split", *ASL_FEATURE_COLUMNS],
        ].copy()
        block["profile_id"] = profile
        block["profile_name"] = PROFILE_LABELS[profile]
        for repeat, contrast in enumerate(repeated):
            block[f"_dr_repeat_{repeat}"] = contrast[mask, profile - 1]
        blocks.append(block)
    opportunities = pd.concat(blocks, ignore_index=True)
    columns = tuple(f"_dr_repeat_{repeat}" for repeat in range(repeats))
    return DM77Supervision(
        opportunities=opportunities,
        signal_columns=columns,
        audit={
            "supervision": "repeated_doubly_robust_real_covariate_semisynthetic_contrasts",
            "oracle_inputs_used": False,
            "nuisance_fit_mode": fit_mode,
            "nuisance_rows": int(len(nuisance)),
            "nuisance_arm_counts": {str(k): int(v) for k, v in arm_counts.items()},
            "minimum_nuisance_arm_count": minimum_arm_count,
            "opportunities": int(len(opportunities)),
            "repeats": repeats,
            "propensity_floor_requested": clip,
            "minimum_stabilized_propensity": float(min(propensity_minima)),
            "mean_propensity_effective_sample_size_by_arm": {
                str(arm): float(np.mean([row[arm] for row in propensity_ess]))
                for arm in range(4)
            },
        },
    )


def _asl_supervision_quality(
    cohort: ASLSemiSyntheticCohort,
    supervision: DM77Supervision,
) -> pd.DataFrame:
    """Evaluation-only diagnostics available in the semi-synthetic benchmark."""
    opportunities = supervision.opportunities
    signal = opportunities[list(supervision.signal_columns)].mean(axis=1).to_numpy(float)
    oracle = cohort.oracle.set_index("patient_id")
    truth = np.asarray([
        oracle.loc[patient, f"tau_{profile}"]
        for patient, profile in zip(
            opportunities.patient_id,
            opportunities.profile_id.to_numpy(int),
        )
    ], dtype=float)
    rows: list[dict[str, Any]] = []
    scopes = [("all", np.ones(len(opportunities), dtype=bool))]
    scopes.extend(
        (f"split:{split}", opportunities.split.eq(split).to_numpy())
        for split in opportunities.split.unique()
    )
    scopes.extend(
        (f"profile:{profile}", opportunities.profile_id.eq(profile).to_numpy())
        for profile in sorted(opportunities.profile_id.unique())
    )
    for scope, mask in scopes:
        if int(mask.sum()) < 3:
            continue
        rho = spearmanr(signal[mask], truth[mask]).statistic
        rows.append({
            "scope": scope,
            "opportunities": int(mask.sum()),
            "dr_true_spearman": float(rho) if np.isfinite(rho) else np.nan,
            "dr_mean": float(np.mean(signal[mask])),
            "true_effect_mean": float(np.mean(truth[mask])),
            "mean_bias": float(np.mean(signal[mask] - truth[mask])),
            "rmse": float(np.sqrt(np.mean(np.square(signal[mask] - truth[mask])))),
        })
    return pd.DataFrame(rows)

def _fit_x_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    processor: ColumnTransformer,
    profile: int,
    seed: int,
) -> np.ndarray:
    subset = train.loc[train.assigned_profile.isin([0, profile])].reset_index(drop=True)
    x = np.asarray(processor.transform(subset[list(ASL_FEATURE_COLUMNS)]), dtype=float)
    xt = np.asarray(processor.transform(test[list(ASL_FEATURE_COLUMNS)]), dtype=float)
    d = subset.assigned_profile.eq(profile).to_numpy(int)
    y = subset.observed_outcome.to_numpy(float)
    if int((d == 0).sum()) < 10 or int((d == 1).sum()) < 10:
        return np.full(len(test), np.nan)
    mu0 = HistGradientBoostingRegressor(
        max_iter=60, max_leaf_nodes=15, random_state=seed
    ).fit(x[d == 0], y[d == 0])
    mu1 = HistGradientBoostingRegressor(
        max_iter=60, max_leaf_nodes=15, random_state=seed + 1
    ).fit(x[d == 1], y[d == 1])
    d1 = y[d == 1] - mu0.predict(x[d == 1])
    d0 = mu1.predict(x[d == 0]) - y[d == 0]
    tau1 = HistGradientBoostingRegressor(
        max_iter=50, max_leaf_nodes=15, random_state=seed + 2
    ).fit(x[d == 1], d1)
    tau0 = HistGradientBoostingRegressor(
        max_iter=50, max_leaf_nodes=15, random_state=seed + 3
    ).fit(x[d == 0], d0)
    prop = HistGradientBoostingClassifier(
        max_iter=50, max_leaf_nodes=15, random_state=seed + 4
    ).fit(x, d)
    e = np.clip(prop.predict_proba(xt)[:, 1], 0.05, 0.95)
    return (1.0 - e) * tau1.predict(xt) + e * tau0.predict(xt)



def _treatment_one_hot(treatment: np.ndarray, arm_count: int) -> np.ndarray:
    """Return a deterministic dense treatment indicator matrix."""
    treatment = np.asarray(treatment, dtype=int)
    if treatment.ndim != 1:
        raise ValueError("Treatment labels must be one-dimensional")
    if np.any((treatment < 0) | (treatment >= arm_count)):
        raise ValueError("Treatment label outside the configured arm range")
    result = np.zeros((len(treatment), arm_count), dtype=float)
    result[np.arange(len(treatment)), treatment] = 1.0
    return result


def _fit_s_learner(
    train_x: np.ndarray,
    train_treatment: np.ndarray,
    train_outcome: np.ndarray,
    test_x: np.ndarray,
    test_profile: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    """Multi-treatment S-learner using one shared outcome model.

    The model receives both patient covariates and a one-hot treatment label.
    Candidate-profile effects are obtained by contrasting the predicted outcome
    under the candidate profile with the predicted outcome under reference care.
    """
    arm_count = int(max(max(train_treatment), max(test_profile), 0)) + 1
    train_augmented = np.column_stack([
        np.asarray(train_x, dtype=float),
        _treatment_one_hot(train_treatment, arm_count),
    ])
    model = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        max_leaf_nodes=15,
        learning_rate=0.06,
        random_state=int(seed),
    ).fit(train_augmented, np.asarray(train_outcome, dtype=float))
    candidate = np.column_stack([
        np.asarray(test_x, dtype=float),
        _treatment_one_hot(test_profile, arm_count),
    ])
    reference = np.column_stack([
        np.asarray(test_x, dtype=float),
        _treatment_one_hot(np.zeros(len(test_profile), dtype=int), arm_count),
    ])
    return model.predict(candidate) - model.predict(reference)


def _fit_r_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    processor: ColumnTransformer,
    profile: int,
    *,
    max_iter: int,
    n_splits: int,
    propensity_clip: float,
    seed: int,
) -> np.ndarray:
    """Pairwise profile-vs-reference R-learner with cross-fitted residuals."""
    subset = train.loc[train.assigned_profile.isin([0, profile])].reset_index(drop=True)
    x = np.asarray(processor.transform(subset[list(ASL_FEATURE_COLUMNS)]), dtype=float)
    xt = np.asarray(processor.transform(test[list(ASL_FEATURE_COLUMNS)]), dtype=float)
    d = subset.assigned_profile.eq(profile).to_numpy(int)
    y = subset.observed_outcome.to_numpy(float)
    arm_counts = np.bincount(d, minlength=2)
    folds = min(int(n_splits), int(arm_counts.min()))
    if folds < 2:
        return np.full(len(test), np.nan)

    outcome_hat = np.zeros(len(subset), dtype=float)
    propensity_hat = np.zeros(len(subset), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(x, d)):
        outcome_model = HistGradientBoostingRegressor(
            max_iter=int(max_iter),
            max_leaf_nodes=15,
            learning_rate=0.06,
            random_state=int(seed) + 10 * fold + 1,
        ).fit(x[fit_idx], y[fit_idx])
        propensity_model = HistGradientBoostingClassifier(
            max_iter=int(max_iter),
            max_leaf_nodes=15,
            learning_rate=0.06,
            random_state=int(seed) + 10 * fold + 2,
        ).fit(x[fit_idx], d[fit_idx])
        outcome_hat[holdout_idx] = outcome_model.predict(x[holdout_idx])
        propensity_hat[holdout_idx] = propensity_model.predict_proba(x[holdout_idx])[:, 1]

    propensity_hat = np.clip(
        propensity_hat,
        float(propensity_clip),
        1.0 - float(propensity_clip),
    )
    treatment_residual = d - propensity_hat
    outcome_residual = y - outcome_hat
    transformed_outcome = outcome_residual / treatment_residual
    r_weight = np.square(treatment_residual)
    effect_model = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        max_leaf_nodes=15,
        learning_rate=0.06,
        random_state=int(seed) + 999,
    ).fit(x, transformed_outcome, sample_weight=r_weight)
    return effect_model.predict(xt)


def _fit_pooled_dr_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    signal_columns: tuple[str, ...],
    *,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    """Shared profile-conditioned regression on cross-fitted DR signals.

    This is the closest non-ranking comparator to PROMETHEUS: it uses the same
    learner-safe DR supervision and a single model across all profiles, but it
    has no pairwise ranking objective and no contrastive representation loss.
    """
    processor = _processor(train)
    train_x = np.asarray(
        processor.fit_transform(train[list(ASL_FEATURE_COLUMNS)]), dtype=float
    )
    test_x = np.asarray(
        processor.transform(test[list(ASL_FEATURE_COLUMNS)]), dtype=float
    )
    profile_count = len(PROFILE_LABELS)
    train_profile = train.profile_id.to_numpy(int) - 1
    test_profile = test.profile_id.to_numpy(int) - 1
    train_augmented = np.column_stack([
        train_x,
        _treatment_one_hot(train_profile, profile_count),
    ])
    test_augmented = np.column_stack([
        test_x,
        _treatment_one_hot(test_profile, profile_count),
    ])
    target = train[list(signal_columns)].mean(axis=1).to_numpy(float)
    model = HistGradientBoostingRegressor(
        max_iter=int(max_iter),
        max_leaf_nodes=15,
        learning_rate=0.06,
        random_state=int(seed),
    ).fit(train_augmented, target)
    return model.predict(test_augmented)


def build_asl_baselines(
    cohort: ASLSemiSyntheticCohort,
    supervision: DM77Supervision,
    ranker: Any,
    config: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    learner = cohort.learner
    nuisance = learner.loc[learner.split.eq("nuisance_train")].reset_index(drop=True)
    calibration = supervision.opportunities.loc[
        supervision.opportunities.split.eq("calibration")
    ].reset_index(drop=True)
    test = supervision.opportunities.loc[
        supervision.opportunities.split.eq("test")
    ].reset_index(drop=True)
    test_patient = learner.set_index("patient_id").loc[test.patient_id].reset_index()
    baseline_settings = dict(config.get("baselines", {}))
    baseline_seed = int(baseline_settings.get("seed", 8201))
    baseline_max_iter = int(
        baseline_settings.get("max_iter_smoke" if smoke else "max_iter", 35 if smoke else 80)
    )

    processor = _processor(nuisance)
    nx = np.asarray(
        processor.fit_transform(nuisance[list(ASL_FEATURE_COLUMNS)]), dtype=float
    )
    tx = np.asarray(
        processor.transform(test_patient[list(ASL_FEATURE_COLUMNS)]), dtype=float
    )
    cal_signal = calibration[list(supervision.signal_columns)].mean(axis=1).to_numpy(float)
    cal_score = ranker.score(calibration)
    test_score = ranker.score(test)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(cal_score, cal_signal)
    prometheus = calibrator.predict(test_score)

    util_ridge = Ridge(alpha=1.0).fit(nx, nuisance.future_utilization)
    util_gbdt = HistGradientBoostingRegressor(
        max_iter=35 if smoke else 80, max_leaf_nodes=15, random_state=8311
    ).fit(nx, nuisance.future_utilization)
    util_ridge_score = util_ridge.predict(tx)
    util_gbdt_score = util_gbdt.predict(tx)
    train_util = util_gbdt.predict(nx)
    thresholds = np.quantile(train_util, [0.20, 0.40, 0.60, 0.80, 0.95])
    utilization_band = np.digitize(util_gbdt_score, thresholds)
    morbidity_train = nuisance.morbidity_proxy.to_numpy(float)
    morbidity_thresholds = np.quantile(morbidity_train, [0.20, 0.40, 0.60, 0.80, 0.95])
    morbidity_band = np.digitize(test_patient.morbidity_proxy.to_numpy(float), morbidity_thresholds)
    need = test_patient.need_score.to_numpy(float)

    profile_mean: dict[int, float] = {}
    rank_train = supervision.opportunities.loc[
        supervision.opportunities.split.eq("rank_train")
    ].reset_index(drop=True)
    for profile in PROFILE_LABELS:
        profile_mean[profile] = float(
            rank_train.loc[
                rank_train.profile_id.eq(profile),
                list(supervision.signal_columns),
            ].to_numpy(float).mean()
        )
    mean_vector = test.profile_id.map(profile_mean).to_numpy(float)

    profile_array = test.profile_id.to_numpy(int)
    s_learner = _fit_s_learner(
        nx,
        nuisance.assigned_profile.to_numpy(int),
        nuisance.observed_outcome.to_numpy(float),
        tx,
        profile_array,
        max_iter=baseline_max_iter,
        seed=baseline_seed + 100,
    )
    pooled_dr = _fit_pooled_dr_learner(
        rank_train,
        test,
        supervision.signal_columns,
        max_iter=baseline_max_iter,
        seed=baseline_seed + 200,
    )

    t_learner: dict[int, np.ndarray] = {}
    x_learner: dict[int, np.ndarray] = {}
    r_learner: dict[int, np.ndarray] = {}
    dr_gbdt: dict[int, np.ndarray] = {}
    rank_processor = _processor(rank_train)
    rx = np.asarray(
        rank_processor.fit_transform(rank_train[list(ASL_FEATURE_COLUMNS)]), dtype=float
    )
    test_op_x = np.asarray(
        rank_processor.transform(test[list(ASL_FEATURE_COLUMNS)]), dtype=float
    )
    for profile in PROFILE_LABELS:
        mu: dict[int, HistGradientBoostingRegressor] = {}
        for arm in (0, profile):
            idx = nuisance.assigned_profile.eq(arm).to_numpy()
            mu[arm] = HistGradientBoostingRegressor(
                max_iter=35 if smoke else 70,
                max_leaf_nodes=15,
                random_state=8500 + 10 * profile + arm,
            ).fit(nx[idx], nuisance.loc[idx, "observed_outcome"])
        t_learner[profile] = mu[profile].predict(tx) - mu[0].predict(tx)
        x_learner[profile] = _fit_x_learner(
            nuisance, test_patient, processor, profile, baseline_seed + 300 + profile * 10
        )
        r_learner[profile] = _fit_r_learner(
            nuisance,
            test_patient,
            processor,
            profile,
            max_iter=baseline_max_iter,
            n_splits=int(baseline_settings.get("r_learner_folds", 3)),
            propensity_clip=float(baseline_settings.get("r_learner_propensity_clip", 0.05)),
            seed=baseline_seed + 400 + profile * 20,
        )
        mask = rank_train.profile_id.eq(profile).to_numpy()
        target = rank_train.loc[
            mask, list(supervision.signal_columns)
        ].mean(axis=1)
        dr_model = HistGradientBoostingRegressor(
            max_iter=35 if smoke else 70, max_leaf_nodes=15, random_state=8700 + profile
        ).fit(rx[mask], target)
        dr_gbdt[profile] = dr_model.predict(test_op_x)

    values: dict[str, np.ndarray] = {
        "PROMETHEUS-Contrastive": prometheus,
        "ASL morbidity-first": morbidity_band * np.maximum(mean_vector, 0.05),
        "ASL utilization-first": utilization_band * np.maximum(mean_vector, 0.05),
        "ASL need-first": need * np.maximum(mean_vector, 0.05),
        "Prospective utilization Ridge": util_ridge_score * np.maximum(mean_vector, 0.05),
        "Prospective utilization GBDT": util_gbdt_score * np.maximum(mean_vector, 0.05),
        "Profile-mean DR": mean_vector,
    }
    values["S-learner GBDT"] = s_learner
    values["Pooled DR-learner GBDT"] = pooled_dr
    values["T-learner GBDT"] = np.asarray([
        t_learner[p][i] for i, p in enumerate(profile_array)
    ])
    values["X-learner GBDT"] = np.asarray([
        x_learner[p][i] for i, p in enumerate(profile_array)
    ])
    values["R-learner GBDT"] = np.asarray([
        r_learner[p][i] for i, p in enumerate(profile_array)
    ])
    values["DR-learner GBDT"] = np.asarray([
        dr_gbdt[p][i] for i, p in enumerate(profile_array)
    ])
    values["DR-Random Forest"] = np.zeros(len(test))
    for profile in PROFILE_LABELS:
        mask_train = rank_train.profile_id.eq(profile).to_numpy()
        mask_test = profile_array == profile
        target = rank_train.loc[
            mask_train, list(supervision.signal_columns)
        ].mean(axis=1)
        rf = RandomForestRegressor(
            n_estimators=40 if smoke else 120,
            min_samples_leaf=12,
            random_state=8800 + profile,
            n_jobs=-1,
        ).fit(rx[mask_train], target)
        values["DR-Random Forest"][mask_test] = rf.predict(test_op_x[mask_test])
    rng = np.random.default_rng(8901)
    values["Random"] = rng.normal(0, 1, len(test))

    oracle = cohort.oracle.set_index("patient_id")
    truth = np.asarray([
        oracle.loc[patient, f"tau_{profile}"]
        for patient, profile in zip(test.patient_id, profile_array)
    ], dtype=float)
    values["Oracle"] = truth.copy()

    test_count = test.patient_id.nunique()
    capacities = {
        1: max(1, int(round(test_count * float(config["allocation"]["capacity_fraction_monitoring"])))),
        2: max(1, int(round(test_count * float(config["allocation"]["capacity_fraction_home_care"])))),
        3: max(1, int(round(test_count * float(config["allocation"]["capacity_fraction_case_management"])))),
    }
    budget = float(config["allocation"]["budget_multiplier"]) * sum(
        capacities[p] * PROFILE_COST[p] for p in capacities
    )
    result_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    patient_array = test.patient_id.astype(str).to_numpy()
    oracle_selected = _milp_allocate(test, truth, budget=budget, capacities=capacities)
    oracle_value = float(truth[oracle_selected].sum())
    for method, score in values.items():
        score = np.asarray(score, dtype=float)
        if not np.isfinite(score).all():
            score = np.nan_to_num(score, nan=float(np.nanmedian(score)))
        selected = _milp_allocate(test, score, budget=budget, capacities=capacities)
        true_value = float(truth[selected].sum())
        result_rows.append({
            "application": "ASL-real-X-semisynthetic-TY",
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
            "harm_at_25pct": float(np.mean(
                truth[np.argsort(-score)[:max(1, int(round(0.25 * len(score))))]] < 0.0
            )),
            "selected_harm_rate": float(np.mean(truth[selected] < 0.0))
            if selected.any() else 0.0,
            "oracle_access_for_policy_construction": method == "Oracle",
            "oracle_access_for_evaluation": True,
        })
        score_rows.extend({
            "patient_id": patient,
            "profile_id": int(profile),
            "method": method,
            "allocator_value": float(value),
            "true_effect_evaluation_only": float(effect),
            "selected": bool(is_selected),
        } for patient, profile, value, effect, is_selected in zip(
            test.patient_id, profile_array, score, truth, selected
        ))
    return pd.DataFrame(result_rows), pd.DataFrame(score_rows)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    value = spearmanr(left.to_numpy(float), right.to_numpy(float)).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _asl_profile_diagnostics(score_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize score scale, ranking quality and selection by care profile."""
    oracle_selection = score_rows.loc[
        score_rows.method.eq("Oracle"),
        ["patient_id", "profile_id", "selected"],
    ].rename(columns={"selected": "oracle_selected"})
    frame = score_rows.merge(
        oracle_selection,
        on=["patient_id", "profile_id"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    for (method, profile_id), group in frame.groupby(
        ["method", "profile_id"],
        sort=True,
    ):
        selected_count = int(group.selected.sum())
        total_selected = int(frame.loc[frame.method.eq(method), "selected"].sum())
        oracle_selected_count = int(group.oracle_selected.fillna(False).sum())
        records.append({
            "method": str(method),
            "profile_id": int(profile_id),
            "opportunities": int(len(group)),
            "mean_allocator_value": float(group.allocator_value.mean()),
            "std_allocator_value": float(group.allocator_value.std(ddof=0)),
            "mean_true_effect_evaluation_only": float(
                group.true_effect_evaluation_only.mean()
            ),
            "within_profile_spearman": _safe_spearman(
                group.allocator_value,
                group.true_effect_evaluation_only,
            ),
            "selected_count": selected_count,
            "selected_rate": float(group.selected.mean()),
            "selected_share_within_method": (
                selected_count / total_selected if total_selected else 0.0
            ),
            "oracle_selected_count": oracle_selected_count,
            "oracle_selected_rate": float(
                group.oracle_selected.fillna(False).mean()
            ),
            "selected_count_minus_oracle": selected_count - oracle_selected_count,
        })
    return pd.DataFrame(records)


def _asl_patient_profile_confusion(score_rows: pd.DataFrame) -> pd.DataFrame:
    """Compare each method's preferred profile with the oracle profile."""
    records: list[dict[str, Any]] = []
    for method, group in score_rows.groupby("method", sort=True):
        score = group.pivot(
            index="patient_id",
            columns="profile_id",
            values="allocator_value",
        )
        truth = group.pivot(
            index="patient_id",
            columns="profile_id",
            values="true_effect_evaluation_only",
        )
        common = score.index.intersection(truth.index)
        score = score.loc[common]
        truth = truth.loc[common]
        predicted = score.idxmax(axis=1)
        oracle = truth.idxmax(axis=1)
        table = pd.crosstab(oracle, predicted, dropna=False)
        for oracle_profile in sorted(truth.columns):
            denominator = int((oracle == oracle_profile).sum())
            for predicted_profile in sorted(score.columns):
                count = int(
                    table.loc[oracle_profile, predicted_profile]
                    if oracle_profile in table.index
                    and predicted_profile in table.columns
                    else 0
                )
                records.append({
                    "method": str(method),
                    "oracle_profile_id": int(oracle_profile),
                    "predicted_profile_id": int(predicted_profile),
                    "patients": count,
                    "row_fraction": count / denominator if denominator else 0.0,
                })
    return pd.DataFrame(records)


def run_asl_semisynthetic_case(
    config: Mapping[str, Any],
    input_path: str | Path,
    output_dir: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    settings = copy.deepcopy(config["asl_semisynthetic"])
    if smoke:
        settings["causal_supervision"].update({
            "nuisance_repeats": 3,
            "propensity_model_max_iter": 45,
            "outcome_model_max_iter": 50,
        })
    n = int(settings["patients_smoke" if smoke else "patients"])
    cohort = generate_asl_semisynthetic_cohort(
        input_path,
        n,
        seed=int(settings["seed"]),
        response_scenario=str(settings["response_scenario"]),
        minimum_age=int(settings.get("minimum_age", 18)),
    )
    supervision = build_asl_supervision(cohort.learner, settings)
    supervision_quality = _asl_supervision_quality(cohort, supervision)
    ranking = dict(settings["ranking"])
    if smoke:
        ranking.update({
            "epochs": 20,
            "patience": 5,
            "minimum_training_epochs": 12,
            "maximum_contrastive_pairs": 2000,
        })
        legacy = ranking.setdefault("legacy_triplet", ranking.setdefault("triplet", {}))
        legacy["maximum_triplets"] = min(int(legacy.get("maximum_triplets", 2000)), 2000)
        causal = ranking.setdefault("causal_contrastive", {})
        causal["maximum_anchors"] = min(int(causal.get("maximum_anchors", 3000)), 3000)
        causal["warmup_epochs"] = min(
            int(causal.get("warmup_epochs", 2)), 2
        )
        causal["ramp_epochs"] = min(
            int(causal.get("ramp_epochs", 5)), 5
        )
        causal["minimum_post_ramp_epochs"] = min(
            int(causal.get("minimum_post_ramp_epochs", 3)), 3
        )
        pretraining = ranking.setdefault("contrastive_pretraining", {})
        pretraining["epochs"] = min(int(pretraining.get("epochs", 6)), 6)
        pretraining["patience"] = min(int(pretraining.get("patience", 3)), 3)
        pretraining["maximum_anchors"] = min(
            int(pretraining.get("maximum_anchors", 1500)), 1500
        )
        pretraining["maximum_validation_anchors"] = min(
            int(pretraining.get("maximum_validation_anchors", 800)), 800
        )
        pretraining["maximum_validation_triplets"] = min(
            int(pretraining.get("maximum_validation_triplets", 1000)), 1000
        )
        finetuning = ranking.setdefault("ranking_finetuning", {})
        finetuning["freeze_encoder_epochs"] = min(
            int(finetuning.get("freeze_encoder_epochs", 2)), 2
        )
    pairs = settings["pairs"]
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
        supervision.opportunities,
        supervision.signal_columns,
        split="rank_train",
        maximum_pairs=int(6000 if smoke else pairs["maximum_train_pairs"]),
        minimum_gap=float(pairs["minimum_gap"]),
        seed=int(pairs["train_seed"]),
        **pair_common,
    )
    validation_pairs = sample_pairs(
        supervision.opportunities,
        supervision.signal_columns,
        split="validation",
        maximum_pairs=int(1800 if smoke else pairs["maximum_validation_pairs"]),
        minimum_gap=float(pairs["minimum_gap"]),
        seed=int(pairs["validation_seed"]),
        **pair_common,
    )
    ranker = fit_contrastive_causal_ranker(
        supervision.opportunities,
        train_pairs,
        validation_pairs,
        feature_columns=ASL_FEATURE_COLUMNS,
        treatment_column="profile_name",
        signal_columns=supervision.signal_columns,
        unit_id_column="patient_id",
        config=ranking,
    )
    results, scores = build_asl_baselines(
        cohort, supervision, ranker, settings, smoke=smoke
    )
    profile_diagnostics = _asl_profile_diagnostics(scores)
    profile_confusion = _asl_patient_profile_confusion(scores)
    results.to_csv(output / "asl_policy_results.csv", index=False)
    scores.to_csv(output / "asl_opportunity_scores.csv", index=False)
    profile_diagnostics.to_csv(
        output / "asl_profile_diagnostics.csv",
        index=False,
    )
    profile_confusion.to_csv(
        output / "asl_patient_profile_confusion.csv",
        index=False,
    )
    cohort.baseline_summary.to_csv(output / "asl_baseline_summary.csv", index=False)
    supervision_quality.to_csv(output / "asl_supervision_quality.csv", index=False)
    methodology_rows = [
        {"method": "PROMETHEUS-Contrastive", "family": "proposed shared causal ranker", "profile_specific": True, "shared_across_profiles": True, "causal_supervision": True},
        {"method": "Pooled DR-learner GBDT", "family": "shared treatment-conditioned DR regression", "profile_specific": True, "shared_across_profiles": True, "causal_supervision": True},
        {"method": "S-learner GBDT", "family": "shared observed-outcome meta-learner", "profile_specific": True, "shared_across_profiles": True, "causal_supervision": True},
        {"method": "T-learner GBDT", "family": "outcome-regression meta-learner", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": True},
        {"method": "X-learner GBDT", "family": "imputed-effect meta-learner", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": True},
        {"method": "R-learner GBDT", "family": "orthogonalized residual meta-learner", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": True},
        {"method": "DR-learner GBDT", "family": "profile-specific DR regression", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": True},
        {"method": "DR-Random Forest", "family": "profile-specific DR regression", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": True},
        {"method": "Profile-mean DR", "family": "profile-level causal mean", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": True},
        {"method": "ASL morbidity-first", "family": "descriptive morbidity heuristic", "profile_specific": False, "shared_across_profiles": False, "causal_supervision": False},
        {"method": "ASL utilization-first", "family": "descriptive utilization heuristic", "profile_specific": False, "shared_across_profiles": False, "causal_supervision": False},
        {"method": "ASL need-first", "family": "descriptive need heuristic", "profile_specific": False, "shared_across_profiles": False, "causal_supervision": False},
        {"method": "Prospective utilization Ridge", "family": "prognostic risk model", "profile_specific": False, "shared_across_profiles": False, "causal_supervision": False},
        {"method": "Prospective utilization GBDT", "family": "prognostic risk model", "profile_specific": False, "shared_across_profiles": False, "causal_supervision": False},
        {"method": "Random", "family": "random-score negative control", "profile_specific": False, "shared_across_profiles": False, "causal_supervision": False},
        {"method": "Oracle", "family": "evaluation-only oracle", "profile_specific": True, "shared_across_profiles": False, "causal_supervision": False},
    ]
    pd.DataFrame(methodology_rows).to_csv(
        output / "asl_methodology.csv", index=False
    )
    (output / "asl_cohort_audit.json").write_text(
        json.dumps(cohort.audit, indent=2), encoding="utf-8"
    )
    supervision_audit = dict(supervision.audit)
    overall_quality = supervision_quality.loc[
        supervision_quality.scope.eq("all")
    ].iloc[0].to_dict()
    supervision_audit["evaluation_only_quality"] = overall_quality
    (output / "asl_supervision_audit.json").write_text(
        json.dumps(supervision_audit, indent=2), encoding="utf-8"
    )
    (output / "asl_ranking_audit.json").write_text(
        json.dumps(ranker.audit, indent=2), encoding="utf-8"
    )
    best = results.loc[
        ~results.oracle_access_for_policy_construction
    ].sort_values("normalized_value", ascending=False).iloc[0]
    lines = [
        "# ASL real-covariate semi-synthetic benchmark",
        "",
        "This run uses the real pseudonymized ASL baseline population and covariates.",
        "Care-profile assignment, potential outcomes and follow-up outcomes are simulated",
        "because the current ASL extract does not contain treatment exposure and longitudinal outcome fields.",
        "",
        f"- Patients used: **{cohort.audit['patients_used']:,}**",
        f"- Response scenario: **{cohort.audit['response_scenario']}**",
        f"- Smoke/diagnostic run: **{bool(smoke)}**",
        f"- DR-vs-true Spearman (evaluation only): **{float(overall_quality['dr_true_spearman']):.4f}**",
        f"- Best non-oracle method: **{best.method}**",
        f"- Best normalized value: **{float(best.normalized_value):.4f}**",
        "",
        "Smoke runs are engineering diagnostics and are not suitable for scientific comparison.",
        "These results validate the method on a real ASL covariate distribution; they are not",
        "an estimate of the clinical effectiveness of services delivered by the ASL.",
    ]
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "results": results,
        "cohort_audit": cohort.audit,
        "supervision_audit": supervision.audit,
        "ranking_audit": ranker.audit,
    }
