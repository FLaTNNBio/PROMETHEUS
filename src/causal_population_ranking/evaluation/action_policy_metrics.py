"""Evaluation-only metrics for patient-action rankings and policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _aligned(*values) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value) for value in values)
    if not arrays or any(array.ndim != 1 for array in arrays):
        raise ValueError("Evaluation arrays must be one-dimensional")
    if len({len(array) for array in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("Evaluation arrays must be non-empty and aligned")
    return arrays


def _quantiles(values) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not 0.0 < value < 1.0 for value in result):
        raise ValueError("Top-q fractions must be in (0,1)")
    return result


def synthetic_top_q_metrics(
    score,
    true_cate,
    quantiles=(0.05, 0.10, 0.20),
) -> dict:
    """Measure top-q benefit, value, oracle recovery, and regret.

    This function is evaluation-only: ``true_cate`` must be joined after model
    fitting, ranking, calibration, and observational allocation are fixed.
    """

    score, cate = _aligned(score, true_cate)
    score = score.astype(float)
    cate = cate.astype(float)
    if not np.isfinite(score).all() or not np.isfinite(cate).all():
        raise ValueError("Top-q evaluation requires finite score and true CATE")
    result = {"top_q_oracle_evaluation_only": True}
    learned_order = np.argsort(-score, kind="mergesort")
    oracle_order = np.argsort(-cate, kind="mergesort")
    for quantile in _quantiles(quantiles):
        count = max(1, int(np.ceil(quantile * len(score))))
        learned = learned_order[:count]
        oracle = oracle_order[:count]
        label = f"{int(round(100 * quantile))}pct"
        learned_value = float(cate[learned].sum())
        oracle_value = float(cate[oracle].sum())
        relevance = cate - float(cate.min())
        discount = 1.0 / np.log2(np.arange(2, count + 2))
        ideal_dcg = float(np.sum(relevance[oracle] * discount))
        learned_dcg = float(np.sum(relevance[learned] * discount))
        result.update({
            f"benefit_at_{label}": float(cate[learned].mean()),
            f"oracle_top_{label}_recovery": float(
                len(np.intersect1d(learned, oracle, assume_unique=True)) / count
            ),
            f"value_at_{label}": learned_value,
            f"regret_at_{label}": oracle_value - learned_value,
            f"ndcg_at_{label}": (
                learned_dcg / ideal_dcg if ideal_dcg > 1e-12 else 1.0
            ),
        })
    return result


def normalized_oracle_efficiency(value, random_value, oracle_value) -> float:
    """Fraction of evaluation-only oracle gain recovered above random allocation."""

    value = float(value)
    random_value = float(random_value)
    oracle_value = float(oracle_value)
    if not np.isfinite((value, random_value, oracle_value)).all():
        raise ValueError("Oracle-efficiency values must be finite")
    denominator = oracle_value - random_value
    return float((value - random_value) / denominator) if abs(denominator) > 1e-12 else float("nan")


def _rate_point(score: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score, kind="mergesort")
    ranked = signal[order]
    count = np.arange(1, len(ranked) + 1, dtype=float)
    fraction = count / len(ranked)
    toc = np.cumsum(ranked) / count - float(np.mean(ranked))
    # RATE with alpha(q)=1 (AUTOC) and alpha(q)=q (QINI family).
    autoc = float(np.trapezoid(toc, fraction))
    qini = float(np.trapezoid(fraction * toc, fraction))
    return autoc, qini


def rank_weighted_effect_metrics(
    score,
    signal,
    patient_ids=None,
    bootstrap_samples: int = 0,
    seed: int = 0,
    prefix: str = "heldout_dr",
) -> dict:
    """Empirical AUTOC/QINI RATE metrics with optional patient-cluster bootstrap.

    The target may be a held-out DR signal or, after all decisions are fixed, a
    synthetic evaluation oracle. The function itself never participates in model
    fitting, checkpointing, calibration, or allocation.
    """

    score, signal = _aligned(score, signal)
    score = score.astype(float)
    signal = signal.astype(float)
    if not np.isfinite(score).all() or not np.isfinite(signal).all():
        raise ValueError("RATE inputs must be finite")
    if patient_ids is None:
        patients = np.asarray([str(index) for index in range(len(score))])
    else:
        patients = np.asarray(patient_ids).astype(str)
        if patients.shape != (len(score),):
            raise ValueError("RATE patient clusters must align")
    autoc, qini = _rate_point(score, signal)
    result = {
        f"{prefix}_rate_autoc": autoc,
        f"{prefix}_rate_qini": qini,
        f"{prefix}_rate_rows": int(len(score)),
        f"{prefix}_rate_patients": int(len(np.unique(patients))),
        f"{prefix}_rate_bootstrap_samples": int(bootstrap_samples),
    }
    if int(bootstrap_samples) < 1:
        return result
    unique = np.unique(patients)
    by_patient = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(bootstrap_samples), 2), dtype=float)
    for repeat in range(int(bootstrap_samples)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_patient[patient] for patient in sampled])
        draws[repeat] = _rate_point(score[index], signal[index])
    for column, name in enumerate(("autoc", "qini")):
        result[f"{prefix}_rate_{name}_standard_error"] = float(
            np.std(draws[:, column], ddof=1)
        ) if len(draws) > 1 else 0.0
        result[f"{prefix}_rate_{name}_ci_lower"] = float(
            np.quantile(draws[:, column], 0.025)
        )
        result[f"{prefix}_rate_{name}_ci_upper"] = float(
            np.quantile(draws[:, column], 0.975)
        )
    return result


def actionwise_off_policy_metrics(
    treatment,
    outcome,
    propensity,
    mu0,
    mu1,
    dr_signal,
    selected_for_action,
) -> dict:
    """Action-specific incremental policy values from OR, IPW, and DR estimators."""

    treatment, outcome, propensity, mu0, mu1, dr_signal, selected = _aligned(
        treatment, outcome, propensity, mu0, mu1, dr_signal, selected_for_action
    )
    treatment = treatment.astype(float)
    outcome = outcome.astype(float)
    propensity = np.clip(propensity.astype(float), 1e-6, 1.0 - 1e-6)
    mu0 = mu0.astype(float)
    mu1 = mu1.astype(float)
    dr_signal = dr_signal.astype(float)
    selected = selected.astype(bool)
    if not all(np.isfinite(value).all() for value in (
        treatment, outcome, propensity, mu0, mu1, dr_signal
    )):
        raise ValueError("Action-wise OPE inputs must be finite")
    policy = selected.astype(float)
    ipw_signal = (
        treatment * outcome / propensity
        - (1.0 - treatment) * outcome / (1.0 - propensity)
    )
    importance = policy * (
        treatment / propensity + (1.0 - treatment) / (1.0 - propensity)
    )
    positive = importance > 0
    ess = (
        float(importance[positive].sum() ** 2 / np.square(importance[positive]).sum())
        if positive.any() and np.square(importance[positive]).sum() > 0 else 0.0
    )
    return {
        "or_incremental_policy_value": float(np.mean(policy * (mu1 - mu0))),
        "ipw_incremental_policy_value": float(np.mean(policy * ipw_signal)),
        "dr_incremental_policy_value": float(np.mean(policy * dr_signal)),
        "selected_fraction": float(np.mean(selected)),
        "selected_count": int(selected.sum()),
        "effective_sample_size": ess,
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
    }


def budget_curve_area(budget_fraction, value) -> float:
    fraction, value = _aligned(budget_fraction, value)
    fraction = fraction.astype(float)
    value = value.astype(float)
    if not np.isfinite(fraction).all() or not np.isfinite(value).all():
        raise ValueError("Budget curve inputs must be finite")
    order = np.argsort(fraction, kind="mergesort")
    if np.any(np.diff(fraction[order]) <= 0):
        raise ValueError("Budget fractions must be unique")
    return float(np.trapezoid(value[order], fraction[order]))


def allocation_budget_metrics(true_cate, selected, cost, oracle_selected) -> dict:
    cate, selected, cost, oracle_selected = _aligned(
        true_cate, selected, cost, oracle_selected
    )
    cate = cate.astype(float)
    selected = selected.astype(bool)
    oracle_selected = oracle_selected.astype(bool)
    cost = cost.astype(float)
    if not np.isfinite(cate).all() or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Budget evaluation requires finite CATE and non-negative costs")
    value = float(cate[selected].sum())
    oracle_value = float(cate[oracle_selected].sum())
    return {
        "benefit_at_budget": float(cate[selected].mean()) if selected.any() else float("nan"),
        "value_at_budget": value,
        "regret_at_budget": oracle_value - value,
        "budget_used_for_evaluation": float(cost[selected].sum()),
        "selected_at_budget": int(selected.sum()),
        "oracle_evaluation_only": True,
    }


def linear_calibration_metrics(prediction, target, prefix: str) -> dict:
    prediction, target = _aligned(prediction, target)
    prediction = prediction.astype(float)
    target = target.astype(float)
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Linear calibration inputs must be finite")
    if np.std(prediction) <= 1e-12:
        slope, intercept = float("nan"), float(np.mean(target))
    else:
        design = np.column_stack([np.ones(len(prediction)), prediction])
        intercept, slope = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        f"{prefix}_calibration_slope": float(slope),
        f"{prefix}_calibration_intercept": float(intercept),
    }


def score_group_diagnostics(
    raw_score,
    calibrated_benefit,
    target,
    method: str,
    partition: str,
    target_name: str,
    oracle_target: bool,
    groups: int = 10,
) -> pd.DataFrame:
    """Return equal-frequency score-group diagnostics without calibrating raw score."""

    score, calibrated, target = _aligned(raw_score, calibrated_benefit, target)
    score = score.astype(float)
    calibrated = calibrated.astype(float)
    target = target.astype(float)
    if int(groups) < 2 or not all(
        np.isfinite(value).all() for value in (score, calibrated, target)
    ):
        raise ValueError("Score-group diagnostics require finite arrays and >=2 groups")
    order = np.argsort(score, kind="mergesort")
    group_index = np.empty(len(score), dtype=int)
    group_index[order] = np.minimum(
        int(groups) - 1,
        np.floor(np.arange(len(score)) * int(groups) / len(score)).astype(int),
    )
    rows = []
    for group in range(int(groups)):
        mask = group_index == group
        if not mask.any():
            continue
        rows.append({
            "method": str(method),
            "partition": str(partition),
            "target": str(target_name),
            "oracle_target": bool(oracle_target),
            "score_group": group + 1,
            "score_group_count": int(mask.sum()),
            "raw_score_min": float(score[mask].min()),
            "raw_score_max": float(score[mask].max()),
            "raw_score_mean": float(score[mask].mean()),
            "calibrated_benefit_mean": float(calibrated[mask].mean()),
            "target_benefit_mean": float(target[mask].mean()),
            "calibration_bias": float(np.mean(calibrated[mask] - target[mask])),
        })
    return pd.DataFrame(rows)


def ranking_stability_metrics(
    score_left,
    score_right,
    selected_left,
    selected_right,
    true_cate,
    quantiles=(0.05, 0.10, 0.20),
) -> dict:
    """Compare two runs over the exact same patient-action opportunities."""

    left, right, selected_left, selected_right, cate = _aligned(
        score_left, score_right, selected_left, selected_right, true_cate
    )
    left = left.astype(float)
    right = right.astype(float)
    selected_left = selected_left.astype(bool)
    selected_right = selected_right.astype(bool)
    cate = cate.astype(float)
    if not all(np.isfinite(value).all() for value in (left, right, cate)):
        raise ValueError("Stability metrics require finite aligned inputs")
    result = {
        "ranking_spearman": float(spearmanr(left, right).statistic),
        "ranking_kendall": float(kendalltau(left, right).statistic),
        "allocation_identical_fraction": float(np.mean(selected_left == selected_right)),
        "policy_value_absolute_difference": float(abs(
            cate[selected_left].sum() - cate[selected_right].sum()
        )),
    }
    intersection = np.sum(selected_left & selected_right)
    union = np.sum(selected_left | selected_right)
    result["allocation_jaccard"] = float(intersection / union) if union else 1.0
    for quantile in _quantiles(quantiles):
        count = max(1, int(np.ceil(quantile * len(left))))
        left_top = set(np.argsort(-left, kind="mergesort")[:count].tolist())
        right_top = set(np.argsort(-right, kind="mergesort")[:count].tolist())
        label = f"{int(round(100 * quantile))}pct"
        result[f"top_{label}_jaccard"] = float(
            len(left_top & right_top) / len(left_top | right_top)
        )

    if selected_left.any() and selected_right.any():
        boundary_left = float(np.min(left[selected_left]))
        boundary_right = float(np.min(right[selected_right]))
        margin_left = np.abs(left - boundary_left)
        margin_right = np.abs(right - boundary_right)
        threshold_left = float(np.quantile(margin_left, 0.20))
        threshold_right = float(np.quantile(margin_right, 0.20))
        near_boundary = (margin_left <= threshold_left) | (margin_right <= threshold_right)
        result["near_boundary_allocation_instability"] = float(np.mean(
            selected_left[near_boundary] != selected_right[near_boundary]
        )) if near_boundary.any() else float("nan")
    else:
        result["near_boundary_allocation_instability"] = float("nan")
    return result
