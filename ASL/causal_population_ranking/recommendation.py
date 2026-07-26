"""Validation-only calibration and pre-allocation profile recommendation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


CALIBRATION_METHODS = (
    "pooled_isotonic",
)
ORACLE_PREFIXES = ("true_", "oracle_", "potential_outcome", "latent_")
VALIDATION_SPLIT = "validation"


@dataclass(frozen=True)
class MonotoneCalibrator:
    method: str
    score_min: float
    score_max: float
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray

    def predict(self, score: Sequence[float]) -> np.ndarray:
        values = np.asarray(score, dtype=float)
        result = np.full(values.shape, np.nan, dtype=float)
        supported = (
            np.isfinite(values)
            & (values >= self.score_min)
            & (values <= self.score_max)
        )
        if supported.any():
            result[supported] = np.interp(
                values[supported], self.x_thresholds, self.y_thresholds
            )
        return result

    def as_contract(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "score_min": float(self.score_min),
            "score_max": float(self.score_max),
            "x_thresholds": [float(value) for value in self.x_thresholds],
            "y_thresholds": [float(value) for value in self.y_thresholds],
            "outside_range_policy": "abstain_no_clipped_extrapolation",
            "monotone_in_raw_priority_score": True,
        }


@dataclass(frozen=True)
class ProfileRecommendationResult:
    calibration_diagnostics: pd.DataFrame
    calibrated_opportunities: pd.DataFrame
    recommendations: pd.DataFrame
    decision_contract: dict[str, Any]
    audit: dict[str, Any]


def _oracle_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in map(str, frame.columns)
        if column != "oracle_used" and column.startswith(ORACLE_PREFIXES)
    )


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _safe_numeric(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{name} requires finite numeric values: {list(columns)}")


def _fit_isotonic(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    model = IsotonicRegression(increasing=True, out_of_bounds="nan")
    model.fit(x, y, sample_weight=sample_weight)
    return (
        np.asarray(model.X_thresholds_, dtype=float),
        np.asarray(model.y_thresholds_, dtype=float),
    )


def _fit_candidate(
    method: str,
    frame: pd.DataFrame,
    *,
    monotonic_bins: int,
    profile_shrinkage_strength: float,
) -> MonotoneCalibrator:
    x = frame.raw_priority_score.to_numpy(float)
    y = frame.dr_pseudo_outcome.to_numpy(float)
    weight = frame.dr_reliability_weight.to_numpy(float)
    score_min, score_max = float(x.min()), float(x.max())
    if method == "pooled_isotonic":
        thresholds = _fit_isotonic(x, y, None)
    elif method == "reliability_weighted_pooled_isotonic":
        thresholds = _fit_isotonic(x, y, weight)
    elif method == "monotonic_binned":
        unique_scores = np.unique(x)
        bin_count = min(int(monotonic_bins), len(unique_scores))
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bin_count + 1)))
        if len(edges) < 2:
            raise ValueError("Monotonic binned calibration needs varying scores")
        bin_index = np.searchsorted(edges[1:-1], x, side="right")
        binned = pd.DataFrame({"x": x, "y": y, "weight": weight, "bin": bin_index})
        rows = []
        for _, group in binned.groupby("bin", sort=True):
            rows.append({
                "x": float(np.average(group.x, weights=group.weight)),
                "y": float(np.average(group.y, weights=group.weight)),
                "weight": float(group.weight.sum()),
            })
        aggregate = pd.DataFrame(rows)
        thresholds = _fit_isotonic(
            aggregate.x.to_numpy(float),
            aggregate.y.to_numpy(float),
            aggregate.weight.to_numpy(float),
        )
    elif method == "profile_mean_shrinkage":
        global_mean = float(np.average(y, weights=weight))
        target_by_profile: dict[str, float] = {}
        for profile, group in frame.groupby("care_profile_id", sort=True):
            group_weight = group.dr_reliability_weight.to_numpy(float)
            profile_mean = float(
                np.average(group.dr_pseudo_outcome.to_numpy(float), weights=group_weight)
            )
            effective_n = float(group_weight.sum())
            shrinkage = effective_n / (effective_n + float(profile_shrinkage_strength))
            target_by_profile[str(profile)] = (
                shrinkage * profile_mean + (1.0 - shrinkage) * global_mean
            )
        shrunk_target = frame.care_profile_id.astype(str).map(target_by_profile).to_numpy(float)
        thresholds = _fit_isotonic(x, shrunk_target, weight)
    else:
        raise ValueError(f"Unknown calibration method {method!r}")
    return MonotoneCalibrator(
        method=method,
        score_min=score_min,
        score_max=score_max,
        x_thresholds=thresholds[0],
        y_thresholds=thresholds[1],
    )


def _rank_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(np.unique(prediction)) < 2 or len(np.unique(target)) < 2:
        return 0.0
    value = pd.Series(prediction).corr(pd.Series(target), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def _empty_calibration_diagnostics(
    methods: Sequence[str], validation_rows: int, status: str
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "method": method,
            "status": status,
            "fit_partition": VALIDATION_SPLIT,
            "validation_rows": int(validation_rows),
            "validation_dr_mae": np.nan,
            "validation_reliability_weighted_dr_mae": np.nan,
            "validation_rank_correlation": np.nan,
            "rank_order_violations": np.nan,
            "finite": False,
            "monotone": False,
            "selected": False,
            "oracle_used": False,
        }
        for method in methods
    ])


def _fit_validation_calibrator(
    priority_scores: pd.DataFrame,
    supervision: pd.DataFrame,
    *,
    primary_variant: str,
    methods: Sequence[str],
    fit_splits: Sequence[str],
    minimum_validation_rows: int,
    monotonic_bins: int,
    profile_shrinkage_strength: float,
    residual_quantile: float,
) -> tuple[MonotoneCalibrator | None, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _require(
        priority_scores,
        {"patient_id", "care_profile_id", "split", "method", "raw_priority_score"},
        "priority_scores",
    )
    _require(
        supervision,
        {
            "patient_id", "care_profile_id", "split", "dr_pseudo_outcome",
            "dr_reliability_weight", "causal_supervision_status",
        },
        "supervision",
    )
    primary = priority_scores.loc[
        priority_scores.method.astype(str).eq(str(primary_variant))
        & priority_scores.raw_priority_score.notna(),
        ["patient_id", "care_profile_id", "split", "raw_priority_score"],
    ].copy()
    primary["patient_id"] = primary.patient_id.astype(str)
    primary["care_profile_id"] = primary.care_profile_id.astype(str)
    if primary[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Primary priority scores must be unique by patient and profile")
    fit_splits = tuple(map(str, fit_splits))
    if not fit_splits or "test" in fit_splits or "rank_train" in fit_splits:
        raise ValueError(
            "Calibration may use only non-test splits independent of rank training"
        )
    target = supervision.loc[
        supervision.split.astype(str).isin(fit_splits)
        & supervision.causal_supervision_status.astype(str).eq("supported"),
        [
            "patient_id", "care_profile_id", "dr_pseudo_outcome",
            "dr_reliability_weight",
        ],
    ].copy()
    target["patient_id"] = target.patient_id.astype(str)
    target["care_profile_id"] = target.care_profile_id.astype(str)
    if target[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Validation DR targets must be unique by patient and profile")
    validation = target.merge(
        primary.loc[primary.split.astype(str).isin(fit_splits)].drop(columns="split"),
        on=["patient_id", "care_profile_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(validation):
        _safe_numeric(
            validation,
            ("raw_priority_score", "dr_pseudo_outcome", "dr_reliability_weight"),
            "validation calibration",
        )
        if (validation.dr_reliability_weight <= 0.0).any():
            raise ValueError("Validation reliability weights must be positive")
    methods = tuple(map(str, methods))
    if methods != CALIBRATION_METHODS:
        raise ValueError(
            f"Calibration methods must preserve the frozen order {CALIBRATION_METHODS}"
        )
    if (
        len(validation) < int(minimum_validation_rows)
        or validation.raw_priority_score.nunique() < 2
    ):
        diagnostics = _empty_calibration_diagnostics(
            methods, len(validation), "not_fitted_insufficient_validation_support"
        )
        return None, diagnostics, validation, {
            "status": "not_fitted_insufficient_validation_support",
            "validation_rows": int(len(validation)),
            "selected_method": None,
            "residual_quantile_probability": float(residual_quantile),
            "residual_quantile_days": None,
        }

    fitted: dict[str, MonotoneCalibrator] = {}
    rows = []
    x = validation.raw_priority_score.to_numpy(float)
    y = validation.dr_pseudo_outcome.to_numpy(float)
    weight = validation.dr_reliability_weight.to_numpy(float)
    for method in methods:
        try:
            calibrator = _fit_candidate(
                method,
                validation,
                monotonic_bins=int(monotonic_bins),
                profile_shrinkage_strength=float(profile_shrinkage_strength),
            )
            prediction = calibrator.predict(x)
            finite = bool(np.isfinite(prediction).all())
            order = np.argsort(x, kind="stable")
            violations = int(np.sum(np.diff(prediction[order]) < -1e-10)) if finite else -1
            monotone = bool(finite and violations == 0)
            absolute_error = np.abs(prediction - y)
            mae = float(np.mean(absolute_error)) if finite else np.nan
            weighted_mae = (
                float(np.average(absolute_error, weights=weight)) if finite else np.nan
            )
            correlation = _rank_correlation(prediction, y) if finite else np.nan
            status = "candidate_valid" if finite and monotone else "candidate_invalid"
            if finite and monotone:
                fitted[method] = calibrator
        except (TypeError, ValueError, FloatingPointError) as error:
            finite = False
            monotone = False
            violations = -1
            mae = np.nan
            weighted_mae = np.nan
            correlation = np.nan
            status = f"candidate_invalid:{type(error).__name__}"
        rows.append({
            "method": method,
            "status": status,
            "fit_partition": ",".join(fit_splits),
            "validation_rows": int(len(validation)),
            "validation_dr_mae": mae,
            "validation_reliability_weighted_dr_mae": weighted_mae,
            "validation_rank_correlation": correlation,
            "rank_order_violations": violations,
            "finite": finite,
            "monotone": monotone,
            "selected": False,
            "oracle_used": False,
        })
    diagnostics = pd.DataFrame(rows)
    valid = diagnostics.loc[diagnostics.method.isin(fitted)].copy()
    if valid.empty:
        return None, diagnostics, validation, {
            "status": "not_fitted_no_finite_monotone_candidate",
            "validation_rows": int(len(validation)),
            "selected_method": None,
            "residual_quantile_probability": float(residual_quantile),
            "residual_quantile_days": None,
        }
    valid["selection_rank_correlation"] = valid.validation_rank_correlation.fillna(
        -np.inf
    )
    selected_method = str(
        valid.sort_values(
            ["validation_dr_mae", "selection_rank_correlation", "method"],
            ascending=[True, False, True],
            kind="stable",
        ).iloc[0].method
    )
    diagnostics.loc[diagnostics.method.eq(selected_method), "selected"] = True
    diagnostics.loc[diagnostics.method.eq(selected_method), "status"] = "selected"
    selected = fitted[selected_method]
    residual = np.abs(selected.predict(x) - y)
    residual_value = float(np.quantile(residual, float(residual_quantile)))
    contract = {
        "status": "fitted",
        "fit_partitions": list(fit_splits),
        "target": "ranker_independent_cross_fitted_and_heldout_robust_dr_signal_days",
        "validation_rows": int(len(validation)),
        "selected_method": selected_method,
        "selection_rule": (
            "minimum_validation_dr_mae_then_maximum_rank_correlation_then_method_name"
        ),
        "residual_quantile_probability": float(residual_quantile),
        "residual_quantile_days": residual_value,
        "residual_diagnostic_semantics": (
            "validation_calibration_dispersion_not_individual_effect_interval"
        ),
        **selected.as_contract(),
    }
    return selected, diagnostics, validation, contract


def _profile_support_table(support_diagnostics: pd.DataFrame) -> pd.DataFrame:
    _require(
        support_diagnostics,
        {
            "care_profile_id", "profile_empirical_support",
            "profile_effective_sample_size",
        },
        "support_diagnostics",
    )
    if support_diagnostics.empty:
        return pd.DataFrame(columns=[
            "care_profile_id", "profile_empirical_support",
            "profile_effective_sample_size",
        ])
    frame = support_diagnostics[[
        "care_profile_id", "profile_empirical_support",
        "profile_effective_sample_size",
    ]].copy()
    frame["care_profile_id"] = frame.care_profile_id.astype(str)
    frame["profile_effective_sample_size"] = pd.to_numeric(
        frame.profile_effective_sample_size, errors="coerce"
    )
    return frame.groupby("care_profile_id", as_index=False).agg(
        profile_empirical_support=("profile_empirical_support", "all"),
        profile_effective_sample_size=("profile_effective_sample_size", "max"),
    )


def _profile_validation_evidence(
    validation: pd.DataFrame,
    *,
    minimum_rows: int,
    one_sided_z: float,
    null_days: float,
    evidence_mode: str,
) -> pd.DataFrame:
    """Compute a profile-level validation guardrail, never an individual interval."""

    if int(minimum_rows) < 2 or float(one_sided_z) <= 0.0:
        raise ValueError("Profile evidence requires at least two rows and positive z")
    if evidence_mode != "population_pooled_nonnull_guardrail":
        raise ValueError("Unsupported profile-evidence mode")
    pooled_outcome = validation.dr_pseudo_outcome.to_numpy(float)
    pooled_weight = validation.dr_reliability_weight.to_numpy(float)
    pooled_count = int(len(validation))
    pooled_weight_sum = float(pooled_weight.sum())
    pooled_effective_n = float(
        pooled_weight_sum**2 / np.square(pooled_weight).sum()
    )
    pooled_mean = float(np.average(pooled_outcome, weights=pooled_weight))
    pooled_variance = float(np.average(
        np.square(pooled_outcome - pooled_mean), weights=pooled_weight
    ))
    pooled_standard_error = float(np.sqrt(
        pooled_variance / max(pooled_effective_n, 1.0)
    ))
    pooled_lower_bound = pooled_mean - float(one_sided_z) * pooled_standard_error
    pooled_positive = bool(
        pooled_count >= int(minimum_rows)
        and pooled_lower_bound > float(null_days)
    )
    rows = []
    for profile, group in validation.groupby("care_profile_id", sort=True):
        outcome = group.dr_pseudo_outcome.to_numpy(float)
        weight = group.dr_reliability_weight.to_numpy(float)
        count = int(len(group))
        weight_sum = float(weight.sum())
        effective_n = float(weight_sum**2 / np.square(weight).sum())
        mean = float(np.average(outcome, weights=weight))
        variance = float(np.average(np.square(outcome - mean), weights=weight))
        standard_error = float(np.sqrt(variance / max(effective_n, 1.0)))
        lower_bound = mean - float(one_sided_z) * standard_error
        rows.append({
            "care_profile_id": str(profile),
            "profile_validation_rows": count,
            "profile_validation_effective_n": effective_n,
            "profile_validation_dr_mean_days": mean,
            "profile_validation_dr_standard_error_days": standard_error,
            "profile_validation_evidence_lower_bound_days": lower_bound,
            "profile_positive_validation_evidence": pooled_positive,
            "profile_evidence_scope": evidence_mode,
            "population_evidence_rows": pooled_count,
            "population_evidence_effective_n": pooled_effective_n,
            "population_evidence_dr_mean_days": pooled_mean,
            "population_evidence_standard_error_days": pooled_standard_error,
            "population_evidence_lower_bound_days": pooled_lower_bound,
            "profile_evidence_oracle_used": False,
        })
    return pd.DataFrame(rows, columns=[
        "care_profile_id", "profile_validation_rows",
        "profile_validation_effective_n", "profile_validation_dr_mean_days",
        "profile_validation_dr_standard_error_days",
        "profile_validation_evidence_lower_bound_days",
        "profile_positive_validation_evidence", "profile_evidence_scope",
        "population_evidence_rows", "population_evidence_effective_n",
        "population_evidence_dr_mean_days",
        "population_evidence_standard_error_days",
        "population_evidence_lower_bound_days", "profile_evidence_oracle_used",
    ])


def _fallback_record(base: pd.Series, reason: str, common: Mapping[str, Any]) -> dict:
    return {
        "patient_id": str(base.patient_id),
        "split": str(base.split),
        "baseline_need_level": base.baseline_need_level,
        "current_care_profile": str(base.current_care_profile),
        "current_care_profile_level": base.current_care_profile_level,
        "recommended_actionable_level": base.baseline_need_level,
        "recommended_profile_id": None,
        "recommended_raw_priority_score": np.nan,
        "calibrated_incremental_benefit": np.nan,
        "recommendation_abstained": True,
        "recommendation_status": "abstained",
        "recommendation_reason": reason,
        "selected_profile_effective_sample_size": np.nan,
        "selected_patient_empirical_support": False,
        "selected_profile_positive_validation_evidence": False,
        "selected_profile_validation_evidence_lower_bound_days": np.nan,
        "benefit_margin_over_threshold_days": np.nan,
        "margin_exceeds_validation_residual_quantile": False,
        **common,
    }


def calibrate_and_recommend_profiles(
    priority_scores: pd.DataFrame,
    supervision: pd.DataFrame,
    supported_opportunities: pd.DataFrame,
    baseline_need: pd.DataFrame,
    current_care: pd.DataFrame,
    patient_splits: pd.DataFrame,
    support_diagnostics: pd.DataFrame,
    *,
    primary_variant: str,
    calibration_contract: Mapping[str, Any],
    recommendation_contract: Mapping[str, Any],
    implementation_settings: Mapping[str, Any],
) -> ProfileRecommendationResult:
    """Calibrate the raw ordinal priority and freeze one pre-allocation decision.

    Only validation DR rows fit or select the calibrator. Capacity and synthetic
    truth columns are never inputs, and the original raw score is never modified.
    """

    frames = {
        "priority_scores": priority_scores,
        "supervision": supervision,
        "supported_opportunities": supported_opportunities,
        "baseline_need": baseline_need,
        "current_care": current_care,
        "patient_splits": patient_splits,
        "support_diagnostics": support_diagnostics,
    }
    leaked = {name: _oracle_columns(frame) for name, frame in frames.items()}
    if any(leaked.values()):
        raise ValueError(f"Oracle columns cannot enter profile recommendation: {leaked}")
    audit_flags_true = [
        name for name, frame in frames.items()
        if "oracle_used" in frame and frame.oracle_used.fillna(False).astype(bool).any()
    ]
    if audit_flags_true:
        raise ValueError(
            f"Inputs marked as oracle-used cannot enter recommendation: {audit_flags_true}"
        )
    if priority_scores.empty:
        priority_scores = pd.DataFrame(columns=[
            "patient_id", "care_profile_id", "split", "method",
            "raw_priority_score",
        ])
    _require(baseline_need, {"patient_id", "baseline_need_level"}, "baseline_need")
    _require(
        current_care,
        {"patient_id", "current_care_profile", "current_care_profile_level"},
        "current_care",
    )
    _require(patient_splits, {"patient_id", "split"}, "patient_splits")
    _require(
        supported_opportunities,
        {
            "patient_id", "care_profile_id", "care_profile_index",
            "care_profile_level", "current_care_profile_level", "eligibility",
            "discretionary_rank_candidate", "empirical_support",
        },
        "supported_opportunities",
    )
    methods = tuple(map(str, calibration_contract["candidate_methods"]))
    calibrator, diagnostics, validation, calibrator_contract = (
        _fit_validation_calibrator(
            priority_scores,
            supervision,
            primary_variant=str(primary_variant),
            methods=methods,
            fit_splits=calibration_contract["fit_partitions"],
            minimum_validation_rows=int(
                implementation_settings["minimum_validation_rows"]
            ),
            monotonic_bins=int(implementation_settings["monotonic_bins"]),
            profile_shrinkage_strength=float(
                implementation_settings["profile_mean_shrinkage_strength"]
            ),
            residual_quantile=float(
                implementation_settings["uncertainty_residual_quantile"]
            ),
        )
    )

    base = baseline_need[["patient_id", "baseline_need_level"]].copy()
    base["patient_id"] = base.patient_id.astype(str)
    current = current_care[[
        "patient_id", "current_care_profile", "current_care_profile_level"
    ]].copy()
    current["patient_id"] = current.patient_id.astype(str)
    splits = patient_splits[["patient_id", "split"]].copy()
    splits["patient_id"] = splits.patient_id.astype(str)
    if any(frame.patient_id.duplicated().any() for frame in (base, current, splits)):
        raise ValueError("Need, current care and split tables require unique patients")
    base = (
        base.merge(current, on="patient_id", how="inner", validate="one_to_one")
        .merge(splits, on="patient_id", how="inner", validate="one_to_one")
    )
    if len(base) != len(baseline_need):
        raise ValueError("Every patient requires need, current-care and split records")

    score = priority_scores.loc[
        priority_scores.method.astype(str).eq(str(primary_variant))
        & priority_scores.raw_priority_score.notna(),
        ["patient_id", "care_profile_id", "raw_priority_score"],
    ].copy()
    score["patient_id"] = score.patient_id.astype(str)
    score["care_profile_id"] = score.care_profile_id.astype(str)
    opportunity = supported_opportunities[[
        "patient_id", "care_profile_id", "care_profile_index", "care_profile_level",
        "current_care_profile_level", "eligibility", "discretionary_rank_candidate",
        "empirical_support",
    ]].copy()
    opportunity["patient_id"] = opportunity.patient_id.astype(str)
    opportunity["care_profile_id"] = opportunity.care_profile_id.astype(str)
    opportunity = opportunity.loc[
        opportunity.eligibility.astype(bool)
        & opportunity.discretionary_rank_candidate.astype(bool)
        & opportunity.empirical_support.astype(bool)
    ]
    candidates = opportunity.merge(
        score,
        on=["patient_id", "care_profile_id"],
        how="inner",
        validate="one_to_one",
    )
    profile_support = _profile_support_table(support_diagnostics)
    candidates = candidates.merge(
        profile_support, on="care_profile_id", how="left", validate="many_to_one"
    )
    evidence_splits = tuple(map(
        str, implementation_settings["profile_evidence_splits"]
    ))
    if not evidence_splits or "test" in evidence_splits or "rank_train" in evidence_splits:
        raise ValueError(
            "Profile evidence may use only non-test splits independent of rank training"
        )
    profile_evidence_frame = supervision.loc[
        supervision.causal_supervision_status.eq("supported")
        & supervision.split.astype(str).isin(evidence_splits),
        [
            "care_profile_id", "dr_pseudo_outcome", "dr_reliability_weight",
        ],
    ].copy()
    profile_evidence = _profile_validation_evidence(
        profile_evidence_frame,
        minimum_rows=int(implementation_settings["minimum_profile_validation_rows"]),
        one_sided_z=float(implementation_settings["profile_evidence_one_sided_z"]),
        null_days=float(implementation_settings["profile_evidence_null_days"]),
        evidence_mode=str(implementation_settings["profile_evidence_mode"]),
    )
    candidates = candidates.merge(
        profile_evidence, on="care_profile_id", how="left", validate="many_to_one"
    )
    candidates["profile_empirical_support"] = candidates[
        "profile_empirical_support"
    ].fillna(False).astype(bool)
    candidates["profile_effective_sample_size"] = pd.to_numeric(
        candidates.profile_effective_sample_size, errors="coerce"
    )
    candidates["profile_positive_validation_evidence"] = candidates[
        "profile_positive_validation_evidence"
    ].fillna(False).astype(bool)
    minimum_ess = float(
        recommendation_contract["support_requirements"][
            "minimum_profile_effective_sample_size"
        ]
    )
    candidates["same_or_higher_level"] = (
        pd.to_numeric(candidates.care_profile_level, errors="coerce")
        >= pd.to_numeric(candidates.current_care_profile_level, errors="coerce")
    )
    candidates["patient_empirical_support"] = (
        candidates.empirical_support.astype(bool)
        & candidates.profile_empirical_support.astype(bool)
        & candidates.profile_effective_sample_size.ge(minimum_ess)
    )
    candidates["calibrated_incremental_benefit"] = (
        calibrator.predict(candidates.raw_priority_score.to_numpy(float))
        if calibrator is not None
        else np.nan
    )
    candidates["score_inside_validation_calibration_range"] = candidates[
        "calibrated_incremental_benefit"
    ].notna()
    candidates = candidates.loc[candidates.same_or_higher_level].copy()
    candidates = candidates.sort_values(
        ["patient_id", "raw_priority_score", "care_profile_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    if len(candidates) and calibrator is not None:
        finite = candidates.calibrated_incremental_benefit.notna()
        ordered = candidates.loc[finite].sort_values("raw_priority_score", kind="stable")
        ordering_violations = int(
            np.sum(np.diff(ordered.calibrated_incremental_benefit.to_numpy(float)) < -1e-10)
        )
    else:
        ordering_violations = 0

    threshold = float(recommendation_contract["primary_minimum_benefit_threshold_days"])
    residual_days = calibrator_contract.get("residual_quantile_days")
    selected_method = calibrator_contract.get("selected_method")
    records = []
    for _, patient in base.sort_values("patient_id", kind="stable").iterrows():
        group = candidates.loc[candidates.patient_id.eq(str(patient.patient_id))].copy()
        supported = group.loc[group.patient_empirical_support]
        evidence_supported = supported.loc[
            supported.profile_positive_validation_evidence
        ]
        in_range = evidence_supported.loc[
            evidence_supported.score_inside_validation_calibration_range
        ]
        common = {
            "eligible_supported_candidate_count": int(len(group)),
            "empirically_supported_candidate_count": int(len(supported)),
            "profile_evidence_supported_candidate_count": int(
                len(evidence_supported)
            ),
            "score_range_supported_candidate_count": int(len(in_range)),
            "minimum_benefit_threshold_days": threshold,
            "selected_calibration_method": selected_method,
            "calibration_score_min": (
                np.nan if calibrator is None else calibrator.score_min
            ),
            "calibration_score_max": (
                np.nan if calibrator is None else calibrator.score_max
            ),
            "validation_residual_quantile_probability": float(
                implementation_settings["uncertainty_residual_quantile"]
            ),
            "validation_residual_quantile_days": (
                np.nan if residual_days is None else float(residual_days)
            ),
            "capacity_inputs_used": False,
            "oracle_used": False,
        }
        if calibrator is None:
            records.append(_fallback_record(patient, "CALIBRATION_UNAVAILABLE", common))
            continue
        if group.empty:
            records.append(
                _fallback_record(patient, "NO_ELIGIBLE_SUPPORTED_PROFILE", common)
            )
            continue
        if supported.empty:
            records.append(
                _fallback_record(patient, "INSUFFICIENT_EMPIRICAL_SUPPORT", common)
            )
            continue
        if evidence_supported.empty:
            records.append(
                _fallback_record(
                    patient, "PROFILE_VALIDATION_EVIDENCE_NOT_POSITIVE", common
                )
            )
            continue
        if in_range.empty:
            records.append(
                _fallback_record(patient, "SCORE_OUTSIDE_VALIDATION_RANGE", common)
            )
            continue
        best = in_range.sort_values(
            ["calibrated_incremental_benefit", "raw_priority_score", "care_profile_id"],
            ascending=[False, False, True],
            kind="stable",
        ).iloc[0]
        unsupported_higher = evidence_supported.loc[
            ~evidence_supported.score_inside_validation_calibration_range
            & evidence_supported.raw_priority_score.gt(float(best.raw_priority_score))
        ]
        if not unsupported_higher.empty:
            records.append(
                _fallback_record(patient, "TOP_SCORE_OUTSIDE_VALIDATION_RANGE", common)
            )
            continue
        benefit = float(best.calibrated_incremental_benefit)
        if not benefit > threshold:
            records.append(
                _fallback_record(patient, "BENEFIT_NOT_STRICTLY_ABOVE_THRESHOLD", common)
            )
            continue
        margin = benefit - threshold
        records.append({
            "patient_id": str(patient.patient_id),
            "split": str(patient.split),
            "baseline_need_level": patient.baseline_need_level,
            "current_care_profile": str(patient.current_care_profile),
            "current_care_profile_level": patient.current_care_profile_level,
            "recommended_actionable_level": int(best.care_profile_level),
            "recommended_profile_id": str(best.care_profile_id),
            "recommended_raw_priority_score": float(best.raw_priority_score),
            "calibrated_incremental_benefit": benefit,
            "recommendation_abstained": False,
            "recommendation_status": "recommended",
            "recommendation_reason": "SUPPORTED_BENEFIT_STRICTLY_ABOVE_THRESHOLD",
            "selected_profile_effective_sample_size": float(
                best.profile_effective_sample_size
            ),
            "selected_patient_empirical_support": True,
            "selected_profile_positive_validation_evidence": True,
            "selected_profile_validation_evidence_lower_bound_days": float(
                best.profile_validation_evidence_lower_bound_days
            ),
            "benefit_margin_over_threshold_days": margin,
            "margin_exceeds_validation_residual_quantile": bool(
                residual_days is not None and margin > float(residual_days)
            ),
            **common,
        })
    recommendations = pd.DataFrame(records)
    reason_counts = {
        str(reason): int(count)
        for reason, count in recommendations.recommendation_reason.value_counts().items()
    }
    recommended = ~recommendations.recommendation_abstained.astype(bool)
    deintensification = recommended & (
        pd.to_numeric(recommendations.recommended_actionable_level, errors="coerce")
        < pd.to_numeric(recommendations.current_care_profile_level, errors="coerce")
    )
    audit = {
        "status": (
            "calibrated_and_recommended"
            if calibrator is not None
            else "calibration_unavailable_all_patients_abstained"
        ),
        "primary_variant": str(primary_variant),
        "calibration_fit_partitions": list(map(
            str, calibration_contract["fit_partitions"]
        )),
        "calibration_validation_rows": int(len(validation)),
        "calibration_candidate_methods": list(methods),
        "selected_calibration_method": selected_method,
        "test_dr_used_for_calibration_or_selection": False,
        "oracle_columns_detected": leaked,
        "oracle_used": False,
        "capacity_inputs_used": False,
        "capacity_fields_present_but_ignored": sorted(
            column for column in supported_opportunities.columns
            if "capacity" in str(column) or "resource_cost" in str(column)
        ),
        "raw_priority_score_modified": False,
        "raw_to_calibrated_ordering_violations": ordering_violations,
        "raw_score_has_causal_zero": False,
        "calibrated_output_is_individual_cate": False,
        "patient_support_semantics": (
            "patient_profile_opportunity_inherits_nonoracle_profile_overlap_and_ess_gate"
        ),
        "profile_validation_evidence": profile_evidence.to_dict(orient="records"),
        "profile_validation_evidence_fit_partitions": list(evidence_splits),
        "profile_validation_evidence_mode": str(
            implementation_settings["profile_evidence_mode"]
        ),
        "profile_validation_evidence_oracle_used": False,
        "patients": int(len(recommendations)),
        "patients_recommended": int(recommended.sum()),
        "patients_abstained": int((~recommended).sum()),
        "deintensification_recommendations": int(deintensification.sum()),
        "recommendation_reason_counts": reason_counts,
        "benefit_threshold_days": threshold,
        "benefit_threshold_operator": "strictly_greater_than",
        "uncertainty_diagnostic": (
            "validation_absolute_calibration_residual_quantile_no_individual_coverage_claim"
        ),
    }
    decision_contract = {
        "stage": "validation_calibration_and_recommendation_before_allocation",
        "calibration": calibrator_contract,
        "recommendation_policy": {
            "candidate_policy": recommendation_contract["candidate_policy"],
            "deintensification_supported": bool(
                recommendation_contract["deintensification_supported_in_v1"]
            ),
            "lateral_profile_change_supported": bool(
                recommendation_contract["lateral_profile_change_supported"]
            ),
            "selector": recommendation_contract["selector"],
            "minimum_benefit_threshold_days": threshold,
            "minimum_benefit_operator": "strictly_greater_than",
            "support_requirements": dict(recommendation_contract["support_requirements"]),
            "profile_validation_evidence_guardrail": {
                "fit_partitions": list(evidence_splits),
                "mode": str(implementation_settings["profile_evidence_mode"]),
                "minimum_rows": int(
                    implementation_settings["minimum_profile_validation_rows"]
                ),
                "one_sided_z": float(
                    implementation_settings["profile_evidence_one_sided_z"]
                ),
                "null_days": float(
                    implementation_settings["profile_evidence_null_days"]
                ),
                "require_lower_bound_strictly_above_null": True,
                "individual_interval_claim": False,
                "oracle_used": False,
            },
            "tie_breakers": list(recommendation_contract["tie_breakers"]),
            "fallback": recommendation_contract[
                "unsupported_or_below_threshold_policy"
            ],
        },
        "raw_priority_score_preserved": True,
        "raw_priority_score_has_causal_zero": False,
        "capacity_inputs_used": False,
        "oracle_used": False,
        "audit": audit,
    }
    return ProfileRecommendationResult(
        calibration_diagnostics=diagnostics,
        calibrated_opportunities=candidates,
        recommendations=recommendations,
        decision_contract=decision_contract,
        audit=audit,
    )


__all__ = [
    "CALIBRATION_METHODS",
    "MonotoneCalibrator",
    "ProfileRecommendationResult",
    "calibrate_and_recommend_profiles",
]
