"""Cross-fitted causal supervision for exact care-profile comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .signals import (
    repeated_doubly_robust_signals,
    robustify_repeated_signals,
)

from .cross_fitting import fit_repeated_partitioned_nuisance


NO_NEW_PROFILE = "no_new_profile"
REQUIRED_SPLITS = ("nuisance_train", "rank_train", "validation", "test")
ORACLE_PREFIXES = ("true_", "oracle_", "potential_outcome", "latent_")
FORBIDDEN_FEATURES = {
    "clinician_assigned_need_level",
    "observed_treatment_profile",
    "observed_outcome",
    "recommended_profile_id",
    "allocated_profile_id",
    "raw_priority_score",
    "calibrated_incremental_benefit",
    "dr_pseudo_outcome",
}


@dataclass(frozen=True)
class ProfileCausalSupervisionResult:
    patient_splits: pd.DataFrame
    supervision: pd.DataFrame
    nuisance_predictions: pd.DataFrame
    support_diagnostics: pd.DataFrame
    supported_opportunities: pd.DataFrame
    audit: dict[str, Any]


def _oracle_columns(columns: Sequence[str]) -> list[str]:
    return sorted(
        column for column in map(str, columns)
        if column.startswith(ORACLE_PREFIXES)
    )


def assign_patient_splits(
    learner: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    stratify_column: str = "observed_treatment_profile",
) -> pd.DataFrame:
    """Assign deterministic patient-disjoint splits stratified by observed profile."""

    required = {"patient_id", stratify_column}
    missing = sorted(required.difference(learner.columns))
    if missing:
        raise ValueError(f"Missing split-assignment columns: {missing}")
    if learner.patient_id.astype(str).duplicated().any():
        raise ValueError("Profile supervision requires one learner row per patient")
    if tuple(fractions) != REQUIRED_SPLITS:
        raise ValueError(f"Split fractions must preserve order {REQUIRED_SPLITS}")
    probabilities = np.asarray([float(fractions[name]) for name in REQUIRED_SPLITS])
    if np.any(probabilities <= 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Split fractions must be positive and sum to one")

    rng = np.random.default_rng(int(seed))
    assignments: dict[str, str] = {}
    strata = learner[["patient_id", stratify_column]].copy()
    strata["patient_id"] = strata.patient_id.astype(str)
    strata[stratify_column] = strata[stratify_column].astype(str)
    for _, group in strata.groupby(stratify_column, sort=True):
        patients = rng.permutation(group.patient_id.to_numpy(str))
        expected = probabilities * len(patients)
        counts = np.floor(expected).astype(int)
        remainder = len(patients) - int(counts.sum())
        priorities = np.argsort(-(expected - counts), kind="stable")
        counts[priorities[:remainder]] += 1
        cursor = 0
        for split, count in zip(REQUIRED_SPLITS, counts):
            for patient_id in patients[cursor:cursor + count]:
                assignments[str(patient_id)] = split
            cursor += int(count)

    result = strata.copy()
    result["split"] = result.patient_id.map(assignments)
    if result.split.isna().any() or result.patient_id.duplicated().any():
        raise AssertionError("Every patient must receive exactly one protocol split")
    return result.rename(columns={stratify_column: "split_stratum"})


def _add_features(
    learner: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    selected = [*map(str, numeric_features), *map(str, categorical_features)]
    missing = sorted(set(selected).difference(learner.columns))
    if missing:
        raise ValueError(f"Missing causal-supervision features: {missing}")
    forbidden = sorted(
        column for column in selected
        if column in FORBIDDEN_FEATURES or column.startswith(ORACLE_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"Forbidden causal-supervision features: {forbidden}")

    enriched = learner.copy()
    output_features = list(map(str, numeric_features))
    levels: dict[str, list[str]] = {}
    numeric = enriched[output_features].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Numeric causal-supervision features must be finite")
    enriched[output_features] = numeric
    for column in map(str, categorical_features):
        categories = sorted(enriched[column].dropna().astype(str).unique().tolist())
        if not categories:
            raise ValueError(f"Categorical feature {column!r} has no observed levels")
        levels[column] = categories
        for category in categories:
            feature = f"{column}__{category}"
            enriched[feature] = enriched[column].astype(str).eq(category).astype(float)
            output_features.append(feature)
    return enriched, output_features, levels


def _effective_sample_size(treatment: np.ndarray, propensity: np.ndarray) -> float:
    weights = treatment / propensity + (1 - treatment) / (1.0 - propensity)
    denominator = float(np.square(weights).sum())
    return 0.0 if denominator <= 0.0 else float(weights.sum() ** 2 / denominator)


def build_profile_causal_supervision(
    learner: pd.DataFrame,
    opportunities: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    split_seed: int,
    nuisance_seed: int,
    profile_candidates: Sequence[tuple[str, int]] | None = None,
) -> ProfileCausalSupervisionResult:
    """Build non-oracle repeated DR supervision for every admissible profile.

    Each nuisance task contains only patients eligible for the target profile whose
    exact observed treatment is that profile or ``no_new_profile``. Patients assigned
    to another profile are never folded into the comparator arm.
    """

    learner_required = {
        "patient_id", "observed_treatment_profile", "observed_outcome",
    }
    opportunity_required = {"patient_id", "care_profile_id", "care_profile_index"}
    missing_learner = sorted(learner_required.difference(learner.columns))
    missing_opportunities = sorted(opportunity_required.difference(opportunities.columns))
    if missing_learner or missing_opportunities:
        raise ValueError(
            f"Missing profile-supervision columns: learner={missing_learner}, "
            f"opportunities={missing_opportunities}"
        )
    leaked = {
        "learner": _oracle_columns(learner.columns),
        "opportunities": _oracle_columns(opportunities.columns),
    }
    if leaked["learner"] or leaked["opportunities"]:
        raise ValueError(f"Oracle columns cannot enter causal supervision: {leaked}")
    if int(split_seed) == int(nuisance_seed):
        raise ValueError("Split and nuisance model seeds must be distinct")

    numeric_features = tuple(map(str, settings["numeric_features"]))
    categorical_features = tuple(map(str, settings.get("categorical_features", ())))
    enriched, feature_columns, categorical_levels = _add_features(
        learner, numeric_features, categorical_features
    )
    splits = assign_patient_splits(
        enriched,
        settings["split_fractions"],
        seed=int(split_seed),
    )
    split_map = splits.set_index("patient_id").split

    folds = int(settings["folds"])
    repeats = int(settings["repeats"])
    repeat_stride = int(settings["repeat_seed_stride"])
    minimum_arm_count = int(settings["minimum_arm_count_per_split"])
    minimum_ess = float(settings["minimum_effective_sample_size_per_split"])
    minimum_profile_ess = float(settings["minimum_profile_effective_sample_size"])
    minimum_overlap = float(settings["minimum_overlap_fraction_per_split"])
    propensity_clip = tuple(map(float, settings["propensity_clip"]))
    overlap_bounds = tuple(map(float, settings["overlap_support_bounds"]))
    winsor_quantiles = tuple(map(float, settings["winsorize_quantiles"]))
    if folds < 2 or repeats < 1 or repeat_stride < 1:
        raise ValueError("Folds, repeats and repeat seed stride must be positive")
    if minimum_arm_count < 1 or minimum_ess <= 0.0 or minimum_profile_ess <= 0.0:
        raise ValueError("Arm-count and ESS support thresholds must be positive")
    if not 0.0 <= minimum_overlap <= 1.0:
        raise ValueError("Minimum overlap fraction must be in [0, 1]")
    if not 0.0 < propensity_clip[0] < propensity_clip[1] < 1.0:
        raise ValueError("Propensity clipping bounds must lie strictly inside (0, 1)")
    if not 0.0 < overlap_bounds[0] < overlap_bounds[1] < 1.0:
        raise ValueError("Overlap support bounds must lie strictly inside (0, 1)")

    if profile_candidates is None:
        profile_table = opportunities[[
            "care_profile_id", "care_profile_index"
        ]].drop_duplicates()
    else:
        profile_table = pd.DataFrame(
            profile_candidates,
            columns=("care_profile_id", "care_profile_index"),
        )
    profile_table = profile_table.sort_values(
        ["care_profile_index", "care_profile_id"]
    ).reset_index(drop=True)
    if profile_table.care_profile_id.astype(str).duplicated().any():
        raise ValueError("Profile candidate identifiers must be unique")
    supervision_frames: list[pd.DataFrame] = []
    nuisance_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    nuisance_audits: dict[str, Any] = {}
    profile_support: dict[str, bool] = {}

    for profile_id in profile_table.care_profile_id.astype(str):
        eligible_ids = set(
            opportunities.loc[
                opportunities.care_profile_id.astype(str).eq(profile_id), "patient_id"
            ].astype(str)
        )
        observed = enriched.observed_treatment_profile.astype(str)
        analysis = enriched.loc[
            enriched.patient_id.astype(str).isin(eligible_ids)
            & observed.isin((profile_id, NO_NEW_PROFILE))
        ].copy().reset_index(drop=True)
        analysis["patient_id"] = analysis.patient_id.astype(str)
        analysis["care_profile_id"] = profile_id
        analysis["profile_treatment"] = analysis.observed_treatment_profile.astype(str).eq(
            profile_id
        ).astype(int)
        analysis["split"] = analysis.patient_id.map(split_map)
        if analysis.split.isna().any():
            raise AssertionError("Profile analysis rows must inherit the global patient split")

        counts: dict[str, dict[int, int]] = {}
        for split in REQUIRED_SPLITS:
            split_treatment = analysis.loc[
                analysis.split.eq(split), "profile_treatment"
            ]
            counts[split] = {
                arm: int(split_treatment.eq(arm).sum()) for arm in (0, 1)
            }
        count_supported = all(
            counts[split][arm] >= minimum_arm_count
            for split in REQUIRED_SPLITS for arm in (0, 1)
        ) and all(
            counts["nuisance_train"][arm] >= folds for arm in (0, 1)
        )
        if not count_supported:
            profile_support[profile_id] = False
            nuisance_audits[profile_id] = {
                "status": "not_fitted_insufficient_split_arm_counts",
                "split_arm_counts": counts,
            }
            for split in REQUIRED_SPLITS:
                diagnostic_rows.append({
                    "care_profile_id": profile_id,
                    "split": split,
                    "n": int(sum(counts[split].values())),
                    "control_n": counts[split][0],
                    "treated_n": counts[split][1],
                    "effective_sample_size": np.nan,
                    "profile_effective_sample_size": np.nan,
                    "overlap_fraction": np.nan,
                    "propensity_min": np.nan,
                    "propensity_max": np.nan,
                    "arm_count_supported": False,
                    "ess_supported": False,
                    "profile_ess_supported": False,
                    "overlap_supported": False,
                    "profile_empirical_support": False,
                    "support_status": "insufficient_split_arm_counts",
                })
            continue

        aggregate, repetition_predictions, nuisance_diagnostic = (
            fit_repeated_partitioned_nuisance(
                learner=analysis,
                features=feature_columns,
                folds=folds,
                model=str(settings["model"]),
                seed=int(nuisance_seed),
                repeats=repeats,
                repeat_seed_stride=repeat_stride,
                training_split="nuisance_train",
                clip=propensity_clip,
            )
        )
        nuisance_audits[profile_id] = {
            "status": "fitted",
            "split_arm_counts": counts,
            **nuisance_diagnostic,
        }
        repeated_array = np.stack([
            prediction[["e_hat", "mu0_hat", "mu1_hat"]].to_numpy(float)
            for prediction in repetition_predictions
        ], axis=1)
        repeated_dr = repeated_doubly_robust_signals(
            analysis.profile_treatment.to_numpy(float),
            analysis.observed_outcome.to_numpy(float),
            repeated_array,
            propensity_clip_epsilon=float(propensity_clip[0]),
        )
        robust = robustify_repeated_signals(
            repeated_dr,
            analysis.split.to_numpy(str),
            aggregation=str(settings["aggregation"]),
            winsorize=bool(settings["winsorize"]),
            winsorize_quantiles=winsor_quantiles,
        )

        profile_rows: list[dict[str, Any]] = []
        split_support: dict[str, bool] = {}
        for split in REQUIRED_SPLITS:
            mask = analysis.split.eq(split).to_numpy()
            treatment = analysis.loc[mask, "profile_treatment"].to_numpy(int)
            propensity = aggregate.loc[mask, "e_hat"].to_numpy(float)
            ess = _effective_sample_size(treatment, propensity)
            overlap = float(np.mean(
                (propensity >= overlap_bounds[0]) & (propensity <= overlap_bounds[1])
            ))
            ess_supported = ess >= minimum_ess
            overlap_supported = overlap >= minimum_overlap
            split_support[split] = bool(ess_supported and overlap_supported)
            profile_rows.append({
                "care_profile_id": profile_id,
                "split": split,
                "n": int(mask.sum()),
                "control_n": counts[split][0],
                "treated_n": counts[split][1],
                "effective_sample_size": ess,
                "overlap_fraction": overlap,
                "propensity_min": float(propensity.min()),
                "propensity_max": float(propensity.max()),
                "arm_count_supported": True,
                "ess_supported": bool(ess_supported),
                "overlap_supported": bool(overlap_supported),
            })
        all_treatment = analysis.profile_treatment.to_numpy(int)
        all_propensity = aggregate.e_hat.to_numpy(float)
        profile_ess = _effective_sample_size(all_treatment, all_propensity)
        profile_ess_supported = profile_ess >= minimum_profile_ess
        supported = all(split_support.values()) and profile_ess_supported
        profile_support[profile_id] = supported
        for row in profile_rows:
            row["profile_empirical_support"] = supported
            row["profile_effective_sample_size"] = profile_ess
            row["profile_ess_supported"] = bool(profile_ess_supported)
            row["support_status"] = (
                "supported" if supported else "insufficient_ess_or_overlap"
            )
            diagnostic_rows.append(row)

        supervision = analysis[[
            "patient_id", "care_profile_id", "observed_treatment_profile",
            "profile_treatment", "split", "observed_outcome",
        ]].copy()
        supervision["e_hat"] = aggregate.e_hat.to_numpy(float)
        supervision["mu0_hat"] = aggregate.mu0_hat.to_numpy(float)
        supervision["mu1_hat"] = aggregate.mu1_hat.to_numpy(float)
        supervision["dr_pseudo_outcome_raw"] = robust.raw_aggregate
        supervision["dr_pseudo_outcome"] = robust.robust_aggregate
        supervision["dr_reliability_weight"] = robust.reliability_weight
        supervision["empirical_overlap"] = (
            supervision.e_hat.between(*overlap_bounds, inclusive="both")
        )
        supervision["profile_empirical_support"] = supported
        supervision["causal_supervision_status"] = np.where(
            supervision.empirical_overlap & supported,
            "supported",
            "excluded_from_ranking_supervision",
        )
        for repeat in range(repeats):
            supervision[f"dr_pseudo_outcome_repeat_{repeat}"] = repeated_dr[:, repeat]
        supervision_frames.append(supervision)

        for repeat, prediction in enumerate(repetition_predictions):
            nuisance = analysis[[
                "patient_id", "care_profile_id", "profile_treatment", "split"
            ]].copy()
            nuisance["nuisance_repeat"] = repeat
            for column in ("e_hat", "mu0_hat", "mu1_hat", "nuisance_fold"):
                nuisance[column] = prediction[column].to_numpy()
            nuisance_frames.append(nuisance)

    supervision_columns = [
        "patient_id", "care_profile_id", "observed_treatment_profile",
        "profile_treatment", "split", "observed_outcome", "e_hat", "mu0_hat",
        "mu1_hat", "dr_pseudo_outcome_raw", "dr_pseudo_outcome",
        "dr_reliability_weight", "empirical_overlap", "profile_empirical_support",
        "causal_supervision_status",
        *[f"dr_pseudo_outcome_repeat_{repeat}" for repeat in range(repeats)],
    ]
    supervision = (
        pd.concat(supervision_frames, ignore_index=True)
        if supervision_frames else pd.DataFrame(columns=supervision_columns)
    )
    nuisance_predictions = (
        pd.concat(nuisance_frames, ignore_index=True)
        if nuisance_frames else pd.DataFrame(columns=[
            "patient_id", "care_profile_id", "profile_treatment", "split",
            "nuisance_repeat", "e_hat", "mu0_hat", "mu1_hat", "nuisance_fold",
        ])
    )
    support_diagnostics = pd.DataFrame(diagnostic_rows)
    supported_opportunities = opportunities.copy()
    supported_opportunities["empirical_support"] = (
        supported_opportunities.care_profile_id.astype(str).map(profile_support).fillna(False)
    ).astype(bool)
    supported_opportunities["empirical_support_status"] = np.where(
        supported_opportunities.empirical_support,
        "profile_comparison_supported",
        "profile_comparison_not_supported",
    )

    repeat_seeds = [
        int(nuisance_seed) + repeat * repeat_stride for repeat in range(repeats)
    ]
    audit = {
        "protocol": "profile_vs_no_new_repeated_dr_v1",
        "treatment": "exact_observed_treatment_profile",
        "comparator": NO_NEW_PROFILE,
        "other_profiles_used_as_comparator": False,
        "feature_columns": feature_columns,
        "categorical_levels": categorical_levels,
        "oracle_columns_detected": leaked,
        "oracle_used": False,
        "split_seed": int(split_seed),
        "nuisance_seed": int(nuisance_seed),
        "nuisance_repeat_seeds": repeat_seeds,
        "split_fractions": {
            name: float(settings["split_fractions"][name]) for name in REQUIRED_SPLITS
        },
        "split_counts": {
            name: int(splits.split.eq(name).sum()) for name in REQUIRED_SPLITS
        },
        "nuisance_outcomes_opened_from_splits": ["nuisance_train"],
        "downstream_outcomes_used_for_nuisance_fitting": False,
        "dr_uses_each_rows_observed_treatment_and_outcome": True,
        "split_local_robustification": True,
        "support_thresholds": {
            "minimum_arm_count_per_split": minimum_arm_count,
            "minimum_effective_sample_size_per_split": minimum_ess,
            "minimum_profile_effective_sample_size": minimum_profile_ess,
            "minimum_overlap_fraction_per_split": minimum_overlap,
            "overlap_support_bounds": list(overlap_bounds),
        },
        "profiles_considered": int(len(profile_table)),
        "profiles_fitted": int(sum(
            details["status"] == "fitted" for details in nuisance_audits.values()
        )),
        "profiles_supported": int(sum(profile_support.values())),
        "unsupported_profiles": sorted(
            profile for profile, supported in profile_support.items() if not supported
        ),
        "profile_support": profile_support,
        "nuisance_diagnostics": nuisance_audits,
    }
    return ProfileCausalSupervisionResult(
        patient_splits=splits,
        supervision=supervision,
        nuisance_predictions=nuisance_predictions,
        support_diagnostics=support_diagnostics,
        supported_opportunities=supported_opportunities,
        audit=audit,
    )
