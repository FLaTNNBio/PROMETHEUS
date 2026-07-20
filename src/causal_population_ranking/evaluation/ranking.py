"""Evaluation of a direct multi-profile causal priority ranking.

Held-out doubly robust signals are the observational evaluation target. Synthetic
ground truth is accepted only by the explicitly named oracle function, after the
model, ranking, and thresholds have been frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from causal_population_ranking.allocation import solve_binary_allocation


DEFAULT_FRACTIONS = (0.05, 0.10, 0.20)


def _numeric_arrays(*values) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if not arrays or any(array.ndim != 1 for array in arrays):
        raise ValueError("Evaluation arrays must be one-dimensional")
    if len({len(array) for array in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("Evaluation arrays must be non-empty and aligned")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Evaluation arrays must be finite")
    return arrays


def _labels(values, expected_length: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != expected_length:
        raise ValueError(f"{name} must be one-dimensional and aligned")
    return result.astype(str)


def _fractions(values) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not 0.0 < value < 1.0 for value in result):
        raise ValueError("Evaluation fractions must be in (0, 1)")
    return result


def _pair_indices(length: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if length < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    if int(max_pairs) < 1:
        raise ValueError("max_pairs must be positive")
    exact_pairs = length * (length - 1) // 2
    if exact_pairs <= int(max_pairs):
        return np.triu_indices(length, k=1)
    rng = np.random.default_rng(int(seed))
    left = rng.integers(0, length, size=int(max_pairs) * 2)
    right = rng.integers(0, length, size=int(max_pairs) * 2)
    distinct = left != right
    return left[distinct][: int(max_pairs)], right[distinct][: int(max_pairs)]


def _concordance(
    score: np.ndarray,
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not mask.any():
        return float("nan")
    product = (score[left[mask]] - score[right[mask]]) * (
        target[left[mask]] - target[right[mask]]
    )
    return float(np.mean((product > 0) + 0.5 * (product == 0)))


def pairwise_concordance(
    score,
    target,
    max_pairs: int = 200_000,
    seed: int = 0,
    minimum_target_gap: float = 0.0,
) -> float:
    """Fraction of comparable pairs ordered consistently with an evaluation target."""

    score, target = _numeric_arrays(score, target)
    if float(minimum_target_gap) < 0:
        raise ValueError("minimum_target_gap cannot be negative")
    left, right = _pair_indices(len(score), max_pairs, seed)
    comparable = np.abs(target[left] - target[right]) > float(minimum_target_gap)
    return _concordance(score, target, left, right, comparable)


def profile_concordance_metrics(
    score,
    target,
    care_profile,
    max_pairs: int = 200_000,
    seed: int = 0,
    minimum_target_gap: float = 0.0,
) -> dict:
    """Separate pooled concordance into within- and cross-profile comparisons."""

    score, target = _numeric_arrays(score, target)
    profile = _labels(care_profile, len(score), "care_profile")
    if float(minimum_target_gap) < 0:
        raise ValueError("minimum_target_gap cannot be negative")
    left, right = _pair_indices(len(score), max_pairs, seed)
    comparable = np.abs(target[left] - target[right]) > float(minimum_target_gap)
    within = comparable & (profile[left] == profile[right])
    cross = comparable & (profile[left] != profile[right])
    return {
        "global_pairwise_concordance": _concordance(
            score, target, left, right, comparable
        ),
        "within_profile_pairwise_concordance": _concordance(
            score, target, left, right, within
        ),
        "cross_profile_pairwise_concordance": _concordance(
            score, target, left, right, cross
        ),
        "evaluated_global_pairs": int(comparable.sum()),
        "evaluated_within_profile_pairs": int(within.sum()),
        "evaluated_cross_profile_pairs": int(cross.sum()),
    }


def _rank_weighted_point(score: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score, kind="mergesort")
    ranked = target[order]
    count = np.arange(1, len(ranked) + 1, dtype=float)
    fraction = count / len(ranked)
    targeting = np.cumsum(ranked) / count - float(np.mean(ranked))
    return (
        float(np.trapezoid(targeting, fraction)),
        float(np.trapezoid(fraction * targeting, fraction)),
    )


def rank_weighted_metrics(
    score,
    target,
    patient_ids=None,
    bootstrap_samples: int = 0,
    seed: int = 0,
    prefix: str = "heldout_dr",
) -> dict:
    """Compute AUTOC/QINI metrics with an optional patient-cluster bootstrap."""

    score, target = _numeric_arrays(score, target)
    if int(bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples cannot be negative")
    if patient_ids is None:
        patients = np.arange(len(score)).astype(str)
    else:
        patients = _labels(patient_ids, len(score), "patient_ids")
    if not str(prefix).strip():
        raise ValueError("prefix cannot be empty")

    metric_names = ("autoc", "qini")
    point = _rank_weighted_point(score, target)
    result = {
        f"{prefix}_{name}": value for name, value in zip(metric_names, point)
    }
    result.update(
        {
            f"{prefix}_rows": int(len(score)),
            f"{prefix}_patients": int(len(np.unique(patients))),
            f"{prefix}_bootstrap_samples": int(bootstrap_samples),
        }
    )
    if int(bootstrap_samples) == 0:
        return result

    unique_patients = np.unique(patients)
    rows_by_patient = {
        patient: np.flatnonzero(patients == patient) for patient in unique_patients
    }
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(bootstrap_samples), len(metric_names)), dtype=float)
    for repeat in range(int(bootstrap_samples)):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rows = np.concatenate([rows_by_patient[patient] for patient in sampled])
        draws[repeat] = _rank_weighted_point(score[rows], target[rows])
    for column, name in enumerate(metric_names):
        result[f"{prefix}_{name}_standard_error"] = (
            float(np.std(draws[:, column], ddof=1)) if len(draws) > 1 else 0.0
        )
        result[f"{prefix}_{name}_ci_lower"] = float(
            np.quantile(draws[:, column], 0.025)
        )
        result[f"{prefix}_{name}_ci_upper"] = float(
            np.quantile(draws[:, column], 0.975)
        )
    return result


def observational_ranking_metrics(
    score,
    heldout_dr_signal,
    propensity,
    care_profile,
    patient_ids=None,
    fractions=DEFAULT_FRACTIONS,
    propensity_clip_epsilon: float = 0.02,
    minimum_signal_gap: float = 0.0,
    max_pairs: int = 200_000,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict:
    """Evaluate a frozen ranking without using synthetic or latent ground truth."""

    score, signal, propensity = _numeric_arrays(score, heldout_dr_signal, propensity)
    profile = _labels(care_profile, len(score), "care_profile")
    fractions = _fractions(fractions)
    if not 0.0 < float(propensity_clip_epsilon) < 0.5:
        raise ValueError("propensity_clip_epsilon must be in (0, 0.5)")
    if (propensity < 0).any() or (propensity > 1).any():
        raise ValueError("propensity values must be in [0, 1]")

    grouped = profile_concordance_metrics(
        score,
        signal,
        profile,
        max_pairs=max_pairs,
        seed=seed,
        minimum_target_gap=minimum_signal_gap,
    )
    result = {
        "evaluation_target": "heldout_doubly_robust_signal",
        "oracle_target": False,
        "overlap_coverage": float(
            np.mean(
                (propensity >= propensity_clip_epsilon)
                & (propensity <= 1.0 - propensity_clip_epsilon)
            )
        ),
        **grouped,
        **rank_weighted_metrics(
            score,
            signal,
            patient_ids=patient_ids,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            prefix="heldout_dr_rate",
        ),
    }
    order = np.argsort(-score, kind="mergesort")
    for fraction in fractions:
        count = max(1, int(np.ceil(fraction * len(score))))
        selected = order[:count]
        label = f"{int(round(100 * fraction))}pct"
        result[f"heldout_dr_benefit_at_{label}"] = float(signal[selected].mean())
        result[f"heldout_dr_value_at_{label}"] = float(
            signal[selected].sum() / len(signal)
        )
    return result


def oracle_ranking_metrics(
    score,
    true_benefit,
    care_profile,
    fractions=DEFAULT_FRACTIONS,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict:
    """Evaluate a frozen synthetic ranking against evaluation-only ground truth."""

    score, benefit = _numeric_arrays(score, true_benefit)
    profile = _labels(care_profile, len(score), "care_profile")
    fractions = _fractions(fractions)
    learned_order = np.argsort(-score, kind="mergesort")
    oracle_order = np.argsort(-benefit, kind="mergesort")
    result = {
        "evaluation_target": "synthetic_true_benefit",
        "oracle_evaluation_only": True,
        "spearman": float(spearmanr(score, benefit).statistic),
        "kendall": float(kendalltau(score, benefit).statistic),
        **profile_concordance_metrics(
            score, benefit, profile, max_pairs=max_pairs, seed=seed
        ),
        **rank_weighted_metrics(score, benefit, seed=seed, prefix="oracle_rate"),
    }

    relevance = benefit - float(benefit.min())
    for fraction in fractions:
        count = max(1, int(np.ceil(fraction * len(score))))
        learned = learned_order[:count]
        oracle = oracle_order[:count]
        label = f"{int(round(100 * fraction))}pct"
        discount = 1.0 / np.log2(np.arange(2, count + 2))
        ideal_dcg = float(np.sum(relevance[oracle] * discount))
        learned_dcg = float(np.sum(relevance[learned] * discount))
        result.update(
            {
                f"benefit_at_{label}": float(benefit[learned].mean()),
                f"value_at_{label}": float(benefit[learned].sum()),
                f"regret_at_{label}": float(
                    benefit[oracle].sum() - benefit[learned].sum()
                ),
                f"oracle_top_{label}_recovery": float(
                    len(np.intersect1d(learned, oracle, assume_unique=True)) / count
                ),
                f"ndcg_at_{label}": (
                    learned_dcg / ideal_dcg if ideal_dcg > 1e-12 else 1.0
                ),
            }
        )
    return result


def evaluate_profile_ranking_scores(
    priority_scores: pd.DataFrame,
    supervision: pd.DataFrame,
    profile_truth: pd.DataFrame,
    *,
    split: str = "test",
    fractions=DEFAULT_FRACTIONS,
    seed: int,
) -> pd.DataFrame:
    """Evaluate frozen method scores on held-out DR signals and synthetic truth."""

    score_required = {
        "patient_id", "care_profile_id", "split", "method", "method_score",
        "score_role", "oracle_used",
    }
    supervision_required = {
        "patient_id", "care_profile_id", "split", "dr_pseudo_outcome", "e_hat",
        "causal_supervision_status",
    }
    truth_required = {
        "patient_id", "care_profile_id", "true_profile_benefit", "evaluation_only",
    }
    for name, frame, required in (
        ("priority_scores", priority_scores, score_required),
        ("supervision", supervision, supervision_required),
        ("profile_truth", profile_truth, truth_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing ranking-evaluation columns: {missing}")
    if priority_scores.oracle_used.fillna(False).astype(bool).any():
        raise ValueError("Operational priority scores cannot contain oracle use")
    if profile_truth.empty or not profile_truth.evaluation_only.astype(bool).all():
        raise ValueError("Profile ranking oracle metrics require evaluation-only truth")

    heldout = supervision.loc[
        supervision.split.astype(str).eq(str(split))
        & supervision.causal_supervision_status.astype(str).eq("supported"),
        ["patient_id", "care_profile_id", "dr_pseudo_outcome", "e_hat"],
    ].copy()
    heldout["patient_id"] = heldout.patient_id.astype(str)
    heldout["care_profile_id"] = heldout.care_profile_id.astype(str)
    truth = profile_truth[[
        "patient_id", "care_profile_id", "true_profile_benefit"
    ]].copy()
    truth["patient_id"] = truth.patient_id.astype(str)
    truth["care_profile_id"] = truth.care_profile_id.astype(str)
    rows: list[dict[str, object]] = []
    for method in sorted(priority_scores.method.astype(str).unique()):
        scored = priority_scores.loc[
            priority_scores.method.astype(str).eq(method)
            & priority_scores.split.astype(str).eq(str(split)),
            ["patient_id", "care_profile_id", "method_score", "score_role"],
        ].copy()
        scored["patient_id"] = scored.patient_id.astype(str)
        scored["care_profile_id"] = scored.care_profile_id.astype(str)
        if scored[["patient_id", "care_profile_id"]].duplicated().any():
            raise ValueError("Ranking evaluation requires unique method opportunities")
        joined = (
            scored.merge(
                heldout,
                on=["patient_id", "care_profile_id"],
                how="inner",
                validate="one_to_one",
            )
            .merge(
                truth,
                on=["patient_id", "care_profile_id"],
                how="inner",
                validate="one_to_one",
            )
        )
        if len(joined) < 2:
            continue
        method_seed = int.from_bytes(
            hashlib.sha256(f"{int(seed)}|{method}".encode("utf-8")).digest()[:4],
            "big",
        )
        observational = observational_ranking_metrics(
            joined.method_score,
            joined.dr_pseudo_outcome,
            joined.e_hat,
            joined.care_profile_id,
            patient_ids=joined.patient_id,
            fractions=fractions,
            seed=method_seed,
        )
        oracle = oracle_ranking_metrics(
            joined.method_score,
            joined.true_profile_benefit,
            joined.care_profile_id,
            fractions=fractions,
            seed=method_seed,
        )
        score_role = str(joined.score_role.iloc[0])
        globally_comparable = score_role != "within_profile_only_ordinal_baseline"
        rows.append({
            "method": method,
            "score_role": score_role,
            "metric": "evaluated_opportunities",
            "value": float(len(joined)),
            "split": str(split),
            "uses_oracle": False,
            "oracle_evaluation_only": True,
            "globally_comparable": globally_comparable,
        })
        for uses_oracle, metrics in ((False, observational), (True, oracle)):
            for metric, value in metrics.items():
                if isinstance(value, (bool, str)):
                    continue
                valid_scope = globally_comparable or (
                    "within_profile" in str(metric)
                    or str(metric) == "overlap_coverage"
                )
                if not valid_scope:
                    continue
                rows.append({
                    "method": method,
                    "score_role": score_role,
                    "metric": str(metric),
                    "value": float(value),
                    "split": str(split),
                    "uses_oracle": bool(uses_oracle),
                    "oracle_evaluation_only": True,
                    "globally_comparable": globally_comparable,
                })
    return pd.DataFrame(rows)


def linear_calibration_metrics(prediction, target, prefix: str) -> dict:
    """Describe a validation-only mapping; never reinterpret raw rank as a CATE."""

    prediction, target = _numeric_arrays(prediction, target)
    if np.std(prediction) <= 1e-12:
        slope, intercept = float("nan"), float(np.mean(target))
    else:
        design = np.column_stack([np.ones(len(prediction)), prediction])
        intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        f"{prefix}_calibration_slope": float(slope),
        f"{prefix}_calibration_intercept": float(intercept),
    }


def evaluate_profile_recommendations(
    recommendations: pd.DataFrame,
    profile_truth: pd.DataFrame,
    supported_opportunities: pd.DataFrame,
    *,
    benefit_threshold_days: float,
    split: str = "test",
) -> pd.DataFrame:
    """Evaluate frozen recommendations against synthetic truth after unblinding."""

    recommendation_required = {
        "patient_id", "split", "baseline_need_level", "current_care_profile",
        "current_care_profile_level", "recommended_actionable_level",
        "recommended_profile_id", "recommendation_abstained",
        "eligible_supported_candidate_count", "score_range_supported_candidate_count",
    }
    truth_required = {
        "patient_id", "care_profile_id", "true_profile_benefit", "evaluation_only",
    }
    opportunity_required = {
        "patient_id", "care_profile_id", "care_profile_level", "eligibility",
        "discretionary_rank_candidate", "empirical_support",
        "current_care_profile_level",
    }
    for name, frame, required in (
        ("recommendations", recommendations, recommendation_required),
        ("profile_truth", profile_truth, truth_required),
        ("supported_opportunities", supported_opportunities, opportunity_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing recommendation-evaluation columns: {missing}")
    if profile_truth.empty or not profile_truth.evaluation_only.astype(bool).all():
        raise ValueError("Recommendation oracle metrics require evaluation-only truth")
    evaluated = recommendations.loc[
        recommendations.split.astype(str).eq(str(split))
    ].copy()
    if evaluated.empty:
        raise ValueError(f"No recommendation rows are available for split {split!r}")
    evaluated["patient_id"] = evaluated.patient_id.astype(str)
    if evaluated.patient_id.duplicated().any():
        raise ValueError("Recommendation evaluation requires one row per patient")

    opportunities = supported_opportunities.loc[
        supported_opportunities.eligibility.astype(bool)
        & supported_opportunities.discretionary_rank_candidate.astype(bool)
        & supported_opportunities.empirical_support.astype(bool)
        & (
            pd.to_numeric(
                supported_opportunities.care_profile_level, errors="coerce"
            )
            >= pd.to_numeric(
                supported_opportunities.current_care_profile_level, errors="coerce"
            )
        ),
        ["patient_id", "care_profile_id", "care_profile_level"],
    ].copy()
    opportunities["patient_id"] = opportunities.patient_id.astype(str)
    opportunities["care_profile_id"] = opportunities.care_profile_id.astype(str)
    truth = profile_truth[[
        "patient_id", "care_profile_id", "true_profile_benefit"
    ]].copy()
    truth["patient_id"] = truth.patient_id.astype(str)
    truth["care_profile_id"] = truth.care_profile_id.astype(str)
    candidates = opportunities.merge(
        truth,
        on=["patient_id", "care_profile_id"],
        how="inner",
        validate="one_to_one",
    )
    candidates["true_profile_benefit"] = pd.to_numeric(
        candidates.true_profile_benefit, errors="coerce"
    )
    if not np.isfinite(candidates.true_profile_benefit.to_numpy(float)).all():
        raise ValueError("Recommendation truth benefits must be finite")

    oracle_rows = []
    for row in evaluated.itertuples(index=False):
        patient_candidates = candidates.loc[
            candidates.patient_id.eq(str(row.patient_id))
        ].sort_values(
            ["true_profile_benefit", "care_profile_id"],
            ascending=[False, True],
            kind="stable",
        )
        best = patient_candidates.iloc[0] if len(patient_candidates) else None
        if best is not None and float(best.true_profile_benefit) > float(
            benefit_threshold_days
        ):
            oracle_profile = str(best.care_profile_id)
            oracle_level = int(best.care_profile_level)
            oracle_benefit = float(best.true_profile_benefit)
        else:
            oracle_profile = None
            oracle_level = int(row.baseline_need_level)
            oracle_benefit = 0.0
        recommended_profile = (
            None if pd.isna(row.recommended_profile_id) else str(row.recommended_profile_id)
        )
        if recommended_profile is None:
            recommended_benefit = 0.0
        else:
            selected_truth = patient_candidates.loc[
                patient_candidates.care_profile_id.eq(recommended_profile),
                "true_profile_benefit",
            ]
            if len(selected_truth) != 1:
                raise ValueError(
                    "Every recommended profile requires one supported oracle benefit"
                )
            recommended_benefit = float(selected_truth.iloc[0])
        oracle_rows.append({
            "patient_id": str(row.patient_id),
            "oracle_profile_id": oracle_profile,
            "oracle_actionable_level": oracle_level,
            "oracle_policy_benefit": oracle_benefit,
            "recommended_true_benefit": recommended_benefit,
            "recommendation_regret": oracle_benefit - recommended_benefit,
            "oracle_profile_exact_agreement": recommended_profile == oracle_profile,
            "oracle_profile_within_one_level": abs(
                int(row.recommended_actionable_level) - oracle_level
            ) <= 1,
        })
    oracle = pd.DataFrame(oracle_rows)
    merged = evaluated.merge(oracle, on="patient_id", how="inner", validate="one_to_one")
    abstained = merged.recommendation_abstained.astype(bool)
    recommended = ~abstained
    current_level = pd.to_numeric(merged.current_care_profile_level, errors="coerce")
    recommended_level = pd.to_numeric(
        merged.recommended_actionable_level, errors="coerce"
    )
    if not np.isfinite(current_level).all() or not np.isfinite(recommended_level).all():
        raise ValueError("Recommendation levels must be finite for oracle evaluation")
    recommended_profile = merged.recommended_profile_id.fillna("").astype(str)
    lateral = (
        recommended
        & recommended_level.eq(current_level)
        & recommended_profile.ne(merged.current_care_profile.astype(str))
    )
    rows: list[dict[str, object]] = []

    def add(metric: str, value: float, uses_oracle: bool = False) -> None:
        rows.append({
            "metric": metric,
            "value": float(value),
            "split": str(split),
            "uses_oracle": bool(uses_oracle),
            "oracle_evaluation_only": True,
        })

    add("patients", len(merged))
    add("recommendation_rate", recommended.mean())
    add("abstention_rate", abstained.mean())
    add("escalation_rate", (recommended & recommended_level.gt(current_level)).mean())
    add("deintensification_rate", (
        recommended & recommended_level.lt(current_level)
    ).mean())
    add("lateral_profile_change_rate", lateral.mean())
    add(
        "mean_eligible_supported_candidate_count",
        merged.eligible_supported_candidate_count.mean(),
    )
    add(
        "mean_score_range_supported_candidate_count",
        merged.score_range_supported_candidate_count.mean(),
    )
    for level, count in recommended_level.astype(int).value_counts().sort_index().items():
        add(f"recommended_actionable_level_{level}_fraction", count / len(merged))
    add(
        "oracle_profile_exact_agreement",
        merged.oracle_profile_exact_agreement.mean(),
        True,
    )
    add(
        "oracle_profile_within_one_level_agreement",
        merged.oracle_profile_within_one_level.mean(),
        True,
    )
    add(
        "expected_recommended_true_benefit_days",
        merged.recommended_true_benefit.mean(),
        True,
    )
    add(
        "oracle_policy_value_days",
        merged.oracle_policy_benefit.mean(),
        True,
    )
    add(
        "mean_recommendation_regret_days",
        merged.recommendation_regret.mean(),
        True,
    )
    return pd.DataFrame(rows)


def evaluate_profile_allocation(
    decisions: pd.DataFrame,
    allocation_candidates: pd.DataFrame,
    diagnostic_selections: pd.DataFrame,
    profile_truth: pd.DataFrame,
    *,
    shared_budget_limit: float,
    pool_capacities: Mapping[str, float],
    profile_capacities: Mapping[str, float],
    tie_seed: int,
    tie_break_epsilon: float,
    solver_settings: Mapping[str, object],
) -> pd.DataFrame:
    """Evaluate a frozen allocation against synthetic truth after oracle opening."""

    decision_required = {
        "patient_id", "recommended_profile_id", "recommendation_abstained",
        "allocated_profile_id", "deferred_recommendation",
        "allocated_calibrated_benefit", "allocated_incremental_resource_cost",
        "allocated_incremental_capacity_requirements", "recommendation_preserved",
        "alternative_profile_substituted",
    }
    candidate_required = {
        "patient_id", "care_profile_id", "incremental_resource_cost",
        "capacity_requirements",
    }
    truth_required = {
        "patient_id", "care_profile_id", "true_profile_benefit", "evaluation_only",
    }
    for name, frame, required in (
        ("decisions", decisions, decision_required),
        ("allocation_candidates", allocation_candidates, candidate_required),
        ("profile_truth", profile_truth, truth_required),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing allocation-evaluation columns: {missing}")
    if profile_truth.empty or not profile_truth.evaluation_only.astype(bool).all():
        raise ValueError("Allocation oracle metrics require evaluation-only truth")
    if decisions.patient_id.astype(str).duplicated().any():
        raise ValueError("Allocation evaluation requires one decision per patient")
    candidates = allocation_candidates.copy()
    candidates["patient_id"] = candidates.patient_id.astype(str)
    candidates["care_profile_id"] = candidates.care_profile_id.astype(str)
    truth = profile_truth[[
        "patient_id", "care_profile_id", "true_profile_benefit"
    ]].copy()
    truth["patient_id"] = truth.patient_id.astype(str)
    truth["care_profile_id"] = truth.care_profile_id.astype(str)
    candidates = candidates.merge(
        truth,
        on=["patient_id", "care_profile_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(candidates) != len(allocation_candidates):
        raise ValueError("Every allocation candidate requires one oracle benefit")
    true_benefit = pd.to_numeric(
        candidates.true_profile_benefit, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(true_benefit).all():
        raise ValueError("Allocation oracle benefits must be finite")
    oracle = solve_binary_allocation(
        candidates,
        true_benefit,
        shared_budget_limit=float(shared_budget_limit),
        pool_capacities=pool_capacities,
        profile_capacities=profile_capacities,
        tie_seed=int(tie_seed),
        tie_break_epsilon=float(tie_break_epsilon),
        solver_settings=solver_settings,
    )
    oracle_curve_solutions = {1.0: oracle}
    actual_keys = {
        (str(row.patient_id), str(row.allocated_profile_id))
        for row in decisions.loc[decisions.allocated_profile_id.notna()].itertuples(
            index=False
        )
    }
    oracle_keys = {
        (str(row.patient_id), str(row.care_profile_id))
        for row in candidates.loc[oracle.selected].itertuples(index=False)
    }
    benefit_by_key = {
        (str(row.patient_id), str(row.care_profile_id)): float(row.true_profile_benefit)
        for row in candidates.itertuples(index=False)
    }
    actual_true_value = float(sum(benefit_by_key[key] for key in actual_keys))
    oracle_true_value = float(sum(benefit_by_key[key] for key in oracle_keys))
    regret = oracle_true_value - actual_true_value
    if regret < -1e-6:
        raise AssertionError("Oracle allocation value cannot be below learned allocation")
    union = actual_keys | oracle_keys
    intersection = actual_keys & oracle_keys
    actual_profiles = pd.Series(
        [profile for _, profile in actual_keys], dtype="object"
    ).value_counts()
    oracle_profiles = pd.Series(
        [profile for _, profile in oracle_keys], dtype="object"
    ).value_counts()
    profile_ids = set(actual_profiles.index) | set(oracle_profiles.index)
    profile_overlap = (
        sum(min(int(actual_profiles.get(p, 0)), int(oracle_profiles.get(p, 0))) for p in profile_ids)
        / max(len(actual_keys), len(oracle_keys), 1)
    )
    recommended = ~decisions.recommendation_abstained.astype(bool)
    deferred = decisions.deferred_recommendation.astype(bool)
    allocated = decisions.allocated_profile_id.notna()
    budget_used = float(decisions.allocated_incremental_resource_cost.sum())
    pool_usage = {str(pool): 0.0 for pool in pool_capacities}
    for value in decisions.loc[
        allocated, "allocated_incremental_capacity_requirements"
    ]:
        requirements = json.loads(str(value) or "{}")
        for pool, amount in requirements.items():
            pool_usage[str(pool)] += float(amount)

    rows: list[dict[str, object]] = []

    def add(metric: str, value: float, uses_oracle: bool = False) -> None:
        rows.append({
            "metric": metric,
            "value": float(value),
            "scope": "full_synthetic_population",
            "uses_oracle": bool(uses_oracle),
            "oracle_evaluation_only": True,
        })

    add("patients", len(decisions))
    add("actionable_recommendations", recommended.sum())
    add("allocated_recommendations", allocated.sum())
    add("deferred_recommendations", deferred.sum())
    add("recommendation_to_allocation_gap_rate", deferred.mean())
    add(
        "conditional_deferral_rate_among_recommended",
        deferred.sum() / max(int(recommended.sum()), 1),
    )
    add("shared_budget_limit", shared_budget_limit)
    add("shared_budget_used", budget_used)
    add(
        "shared_budget_utilization",
        budget_used / shared_budget_limit if shared_budget_limit > 0.0 else 0.0,
    )
    for pool, available in sorted(pool_capacities.items()):
        add(f"capacity_used:{pool}", pool_usage[str(pool)])
        add(
            f"capacity_utilization:{pool}",
            pool_usage[str(pool)] / float(available) if float(available) > 0.0 else 0.0,
        )
    add(
        "recommendation_preservation_rate",
        decisions.recommendation_preserved.astype(bool).mean(),
    )
    add(
        "alternative_profile_substitution_rate",
        decisions.alternative_profile_substituted.astype(bool).mean(),
    )
    add("allocated_true_value_days", actual_true_value, True)
    add("oracle_allocation_true_value_days", oracle_true_value, True)
    add("allocation_regret_days", max(0.0, regret), True)
    add(
        "oracle_selected_opportunity_jaccard",
        len(intersection) / len(union) if union else 1.0,
        True,
    )
    add("oracle_selected_profile_distribution_overlap", profile_overlap, True)

    if not diagnostic_selections.empty:
        selections = diagnostic_selections.copy()
        selections["patient_id"] = selections.patient_id.astype(str)
        selections["care_profile_id"] = selections.care_profile_id.astype(str)
        selections = selections.merge(
            truth,
            on=["patient_id", "care_profile_id"],
            how="left",
            validate="many_to_one",
        )
        if selections.true_profile_benefit.isna().any():
            raise ValueError("Every diagnostic selection requires one oracle benefit")
        curve = selections.loc[
            selections.diagnostic_type.eq("cardinal_budget_curve")
        ]
        for setting, group in curve.groupby("setting", sort=True):
            multiplier = float(setting)
            learned_value = float(group.true_profile_benefit.sum())
            oracle_curve = oracle_curve_solutions.get(multiplier)
            if oracle_curve is None:
                oracle_curve = solve_binary_allocation(
                    candidates,
                    true_benefit,
                    shared_budget_limit=float(shared_budget_limit) * multiplier,
                    pool_capacities=pool_capacities,
                    profile_capacities=profile_capacities,
                    tie_seed=int(tie_seed),
                    tie_break_epsilon=float(tie_break_epsilon),
                    solver_settings=solver_settings,
                )
                oracle_curve_solutions[multiplier] = oracle_curve
            oracle_value = float(true_benefit[oracle_curve.selected].sum())
            label = str(multiplier).replace(".", "p")
            add(f"budget_curve_learned_true_value:{label}", learned_value, True)
            add(f"budget_curve_oracle_true_value:{label}", oracle_value, True)
            add(
                f"budget_curve_allocation_regret:{label}",
                max(0.0, oracle_value - learned_value),
                True,
            )
        ordinal = selections.loc[
            selections.diagnostic_type.eq("fixed_count_ordinal")
        ]
        for setting, group in ordinal.groupby("setting", sort=True):
            label = str(float(setting)).replace(".", "p")
            add(
                f"fixed_count_ordinal_true_value:{label}",
                float(group.true_profile_benefit.sum()),
                True,
            )
    return pd.DataFrame(rows)
