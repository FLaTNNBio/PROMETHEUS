"""Learner-safe causal supervision for the semi-synthetic EMS case study.

The doubly robust contrasts in this module are transient supervision signals.
They are used to decide which member of a pair should rank first; they are not
reported as individual treatment-effect estimates and are never sorted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .synthetic import LEARNER_FEATURE_COLUMNS


OPPORTUNITY_LABELS = {
    1: "nurse_supported_vs_basic",
    2: "medicalized_vs_nurse_supported",
}


@dataclass(frozen=True)
class EMSSupervision:
    """Transient repeated DR signals plus a non-oracle audit trail."""

    opportunities: pd.DataFrame
    repeat_signal_columns: tuple[str, ...]
    audit: dict[str, Any]


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        column
        for column in LEARNER_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric = [
        column for column in LEARNER_FEATURE_COLUMNS
        if column not in categorical
    ]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("categorical", encoder, categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )


def _predict_propensity(
    model: HistGradientBoostingClassifier,
    matrix: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(matrix)
    result = np.zeros((len(matrix), 3), dtype=float)
    for position, label in enumerate(model.classes_):
        result[:, int(label)] = raw[:, position]
    return result


def build_ems_causal_supervision(
    learner_data: pd.DataFrame,
    config: Mapping[str, Any],
) -> EMSSupervision:
    """Create repeated DR pair targets without opening evaluation-only truth."""

    forbidden_prefixes = (
        "true_",
        "oracle_",
        "potential_outcome",
        "latent_",
        "simulated_post_response",
    )
    leaked = [
        column
        for column in learner_data.columns
        if column.startswith(forbidden_prefixes)
    ]
    if leaked:
        raise ValueError(f"Evaluation-only fields entered EMS supervision: {leaked}")

    nuisance = learner_data.loc[
        learner_data.split.eq("nuisance_train")
    ].reset_index(drop=True)
    target = learner_data.loc[
        learner_data.split.ne("nuisance_train")
    ].reset_index(drop=True)
    if nuisance.empty or target.empty:
        raise ValueError("EMS supervision requires nuisance and target splits")

    settings = config["causal_supervision"]
    seed = int(settings["nuisance_seed"])
    folds = int(settings["nuisance_folds"])
    repeats = int(settings["nuisance_repeats"])
    clip = float(settings["propensity_clip"])
    if folds < 2 or repeats < 1 or not 0.0 < clip < 1.0 / 3.0:
        raise ValueError("Invalid EMS nuisance settings")

    processor = _preprocessor(nuisance)
    nuisance_x = np.asarray(
        processor.fit_transform(nuisance[list(LEARNER_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    target_x = np.asarray(
        processor.transform(target[list(LEARNER_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    treatment = nuisance.assigned_tier.to_numpy(int)
    outcome = nuisance.observed_outcome.to_numpy(float)
    target_treatment = target.assigned_tier.to_numpy(int)
    target_outcome = target.observed_outcome.to_numpy(float)
    repeat_contrasts: list[np.ndarray] = []

    for repeat in range(repeats):
        fold_seed = seed + repeat * 1009
        splitter = KFold(n_splits=folds, shuffle=True, random_state=fold_seed)
        dr_predictions = []
        for fold, (fit_index, _) in enumerate(splitter.split(nuisance_x)):
            model_seed = fold_seed + fold * 37
            fit_treatment = treatment[fit_index]
            if set(np.unique(fit_treatment)) != {0, 1, 2}:
                raise ValueError(
                    "Every EMS nuisance fold must contain all response tiers"
                )
            propensity_model = HistGradientBoostingClassifier(
                max_iter=int(settings["propensity_model_max_iter"]),
                max_leaf_nodes=15,
                learning_rate=0.06,
                random_state=model_seed,
            )
            propensity_model.fit(nuisance_x[fit_index], fit_treatment)
            propensity = np.clip(
                _predict_propensity(propensity_model, target_x),
                clip,
                1.0 - clip,
            )
            propensity /= propensity.sum(axis=1, keepdims=True)

            mu = np.zeros((len(target), 3), dtype=float)
            for tier in range(3):
                arm_index = fit_index[fit_treatment == tier]
                if len(arm_index) < 8:
                    raise ValueError(
                        f"Too few tier-{tier} observations in EMS nuisance fold"
                    )
                outcome_model = HistGradientBoostingRegressor(
                    max_iter=int(settings["outcome_model_max_iter"]),
                    max_leaf_nodes=15,
                    learning_rate=0.06,
                    random_state=model_seed + tier + 1,
                )
                outcome_model.fit(nuisance_x[arm_index], outcome[arm_index])
                mu[:, tier] = outcome_model.predict(target_x)
            dr = mu.copy()
            rows = np.arange(len(target))
            dr[rows, target_treatment] += (
                target_outcome - mu[rows, target_treatment]
            ) / propensity[rows, target_treatment]
            dr_predictions.append(dr)
        mean_dr = np.mean(dr_predictions, axis=0)
        repeat_contrasts.append(np.column_stack([
            mean_dr[:, 1] - mean_dr[:, 0],
            mean_dr[:, 2] - mean_dr[:, 1],
        ]))

    records = []
    for opportunity_position, tier in enumerate((1, 2)):
        block = target[
            ["mission_id", "split", *LEARNER_FEATURE_COLUMNS]
        ].copy()
        block["opportunity_tier"] = tier
        block["opportunity_type"] = OPPORTUNITY_LABELS[tier]
        for repeat, contrasts in enumerate(repeat_contrasts):
            block[f"_dr_repeat_{repeat}"] = contrasts[:, opportunity_position]
        records.append(block)
    opportunities = pd.concat(records, ignore_index=True)
    repeat_columns = tuple(
        f"_dr_repeat_{repeat}" for repeat in range(repeats)
    )

    return EMSSupervision(
        opportunities=opportunities,
        repeat_signal_columns=repeat_columns,
        audit={
            "supervision_target": "transient_repeated_doubly_robust_contrasts",
            "reported_as_individual_cate": False,
            "individual_cate_estimated_then_sorted": False,
            "oracle_inputs_used": False,
            "nuisance_train_rows": int(len(nuisance)),
            "supervised_missions": int(len(target)),
            "opportunities": int(len(opportunities)),
            "feature_count_after_encoding": int(nuisance_x.shape[1]),
            "nuisance_seed": seed,
            "nuisance_folds": folds,
            "nuisance_repeats": repeats,
            "propensity_clip": clip,
        },
    )


def sample_direct_ranking_pairs(
    opportunities: pd.DataFrame,
    repeat_signal_columns: tuple[str, ...],
    *,
    split: str,
    maximum_pairs: int,
    minimum_signal_difference: float,
    minimum_repeat_agreement: float,
    seed: int,
) -> pd.DataFrame:
    """Sample mission-to-mission order constraints from transient DR signals."""

    subset = opportunities.loc[
        opportunities.split.eq(split)
    ].reset_index(drop=True)
    if len(subset) < 2 or maximum_pairs < 1:
        raise ValueError(f"Not enough {split!r} EMS opportunities for pairs")
    signals = subset[list(repeat_signal_columns)].to_numpy(float)
    mean_signal = signals.mean(axis=1)
    rng = np.random.default_rng(int(seed))
    accepted: dict[tuple[int, int], tuple[int, int, float, float]] = {}
    batch = max(4096, min(maximum_pairs * 3, 100_000))
    attempts = 0
    while len(accepted) < maximum_pairs and attempts < 30:
        left = rng.integers(0, len(subset), size=batch)
        right = rng.integers(0, len(subset), size=batch)
        gap = mean_signal[left] - mean_signal[right]
        repeat_gap = signals[left] - signals[right]
        agreement = np.mean(np.sign(repeat_gap) == np.sign(gap)[:, None], axis=1)
        valid = (
            subset.mission_id.to_numpy()[left]
            != subset.mission_id.to_numpy()[right]
        ) & (np.abs(gap) >= float(minimum_signal_difference)) & (
            agreement >= float(minimum_repeat_agreement)
        ) & (gap != 0.0)
        for a, b, value, reliability in zip(
            left[valid], right[valid], gap[valid], agreement[valid]
        ):
            high, low = (int(a), int(b)) if value > 0 else (int(b), int(a))
            accepted.setdefault(
                (high, low),
                (high, low, float(abs(value)), float(reliability)),
            )
            if len(accepted) >= maximum_pairs:
                break
        attempts += 1
    if not accepted:
        raise ValueError(f"No stable direct-ranking pairs found for {split!r}")
    pairs = pd.DataFrame(
        list(accepted.values()),
        columns=["high_index", "low_index", "signal_gap", "repeat_agreement"],
    )
    pairs["weight"] = (
        pairs.signal_gap.clip(upper=pairs.signal_gap.quantile(0.95))
        * pairs.repeat_agreement
    )
    pairs["weight"] /= pairs.weight.mean()
    return pairs
