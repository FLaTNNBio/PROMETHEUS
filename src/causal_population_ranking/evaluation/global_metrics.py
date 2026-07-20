from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import autoc, pairwise_concordance


def global_pairwise_concordance(
    score,
    true_benefit,
    transition_index,
    max_pairs: int = 200_000,
    seed: int = 0,
) -> dict:
    """Evaluation-only oracle concordance on patient-transition opportunities."""

    score = np.asarray(score, dtype=float)
    benefit = np.asarray(true_benefit, dtype=float)
    transition = np.asarray(transition_index, dtype=np.int64)
    if score.ndim != 1 or score.shape != benefit.shape or score.shape != transition.shape:
        raise ValueError("Global concordance arrays must align")
    if not all(np.isfinite(value).all() for value in (score, benefit)):
        raise ValueError("Global concordance inputs must be finite")
    if len(score) < 2:
        return {key: float("nan") for key in (
            "within_transition_concordance", "cross_transition_concordance", "global_concordance"
        )}
    rng = np.random.default_rng(seed)
    draw = min(max_pairs, max(5_000, len(score) * 50))
    left = rng.integers(0, len(score), draw)
    right = rng.integers(0, len(score), draw)
    comparable = (left != right) & (benefit[left] != benefit[right])
    within = comparable & (transition[left] == transition[right])
    cross = comparable & (transition[left] != transition[right])

    def concordance(mask) -> float:
        if not mask.any():
            return float("nan")
        product = (score[left[mask]] - score[right[mask]]) * (benefit[left[mask]] - benefit[right[mask]])
        return float(np.mean((product > 0) + 0.5 * (product == 0)))

    return {
        "within_transition_concordance": concordance(within),
        "cross_transition_concordance": concordance(cross),
        "within_action_concordance": concordance(within),
        "cross_action_concordance": concordance(cross),
        "global_concordance": concordance(comparable),
        "evaluated_within_pairs": int(within.sum()),
        "evaluated_cross_pairs": int(cross.sum()),
        "oracle_evaluation_only": True,
    }


def hard_cross_action_concordance(
    score,
    true_cate,
    action_index,
    support_flag=None,
    max_pairs: int = 200_000,
    close_gap_quantile: float = 0.35,
    seed: int = 0,
) -> dict:
    """Concordance on supported cross-action pairs with deliberately close CATEs."""

    score = np.asarray(score, dtype=float)
    cate = np.asarray(true_cate, dtype=float)
    action = np.asarray(action_index, dtype=int)
    support = (
        np.ones(len(score), dtype=bool)
        if support_flag is None else np.asarray(support_flag, dtype=bool)
    )
    if any(value.shape != score.shape for value in (cate, action, support)):
        raise ValueError("Hard cross-action metric arrays must align")
    if not 0.0 < float(close_gap_quantile) < 1.0:
        raise ValueError("close_gap_quantile must be in (0,1)")
    rng = np.random.default_rng(seed)
    draw = min(int(max_pairs), max(10_000, len(score) * 100))
    left = rng.integers(0, len(score), draw)
    right = rng.integers(0, len(score), draw)
    cross_supported = (
        (left != right) & (action[left] != action[right])
        & support[left] & support[right] & (cate[left] != cate[right])
    )
    if not cross_supported.any():
        return {"hard_cross_action_concordance": float("nan"), "hard_cross_action_pairs": 0}
    gap = np.abs(cate[left] - cate[right])
    threshold = float(np.quantile(gap[cross_supported], close_gap_quantile))
    hard = cross_supported & (gap <= threshold)
    product = (score[left[hard]] - score[right[hard]]) * (cate[left[hard]] - cate[right[hard]])
    return {
        "hard_cross_action_concordance": float(
            np.mean((product > 0) + 0.5 * (product == 0))
        ) if len(product) else float("nan"),
        "hard_cross_action_pairs": int(len(product)),
        "hard_cross_action_cate_gap_threshold": threshold,
        "hard_cross_action_requires_overlap_support": True,
    }


def transition_ranking_metrics(
    frame: pd.DataFrame,
    score_column: str = "raw_score",
    benefit_column: str = "true_benefit",
) -> dict[str, dict]:
    result = {}
    for transition, group in frame.groupby("transition", sort=True):
        score = group[score_column].to_numpy(float)
        benefit = group[benefit_column].to_numpy(float)
        result[str(transition)] = {
            "n_opportunities": int(len(group)),
            "pairwise_concordance": pairwise_concordance(score, benefit, seed=71),
            "autoc": autoc(score, benefit),
        }
        for capacity in (0.05, 0.10, 0.20):
            count = max(1, int(np.floor(capacity * len(group))))
            selected = np.argsort(-score, kind="mergesort")[:count]
            result[str(transition)][f"benefit_at_{int(capacity * 100)}pct"] = float(
                benefit[selected].mean()
            )
    return result


def allocation_value(frame: pd.DataFrame, selected) -> dict:
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(frame),):
        raise ValueError("Allocation decisions must align with evaluation opportunities")
    benefit = frame.true_benefit.to_numpy(float)
    cost = frame.cost.to_numpy(float)
    value = float(benefit[selected].sum())
    used = float(cost[selected].sum())
    return {
        "global_allocation_value": value,
        "capacity_used": used,
        "benefit_per_capacity_unit": value / used if used > 0 else float("nan"),
        "selected_opportunities": int(selected.sum()),
    }


def compare_allocation_to_oracle(
    evaluation_frame: pd.DataFrame,
    learned_selected,
    oracle_selected,
    baseline_selected,
) -> dict:
    benefit = evaluation_frame.true_benefit.to_numpy(float)
    learned = float(benefit[np.asarray(learned_selected, dtype=bool)].sum())
    oracle = float(benefit[np.asarray(oracle_selected, dtype=bool)].sum())
    baseline = float(benefit[np.asarray(baseline_selected, dtype=bool)].sum())
    denominator = oracle - baseline
    return {
        "learned_global_value": learned,
        "oracle_global_value": oracle,
        "baseline_global_value": baseline,
        "global_regret": oracle - learned,
        "fraction_of_oracle_benefit": (learned - baseline) / denominator if abs(denominator) > 1e-12 else float("nan"),
        "oracle_evaluation_only": True,
        "oracle_and_learned_constraints_identical": True,
    }


def pooled_opportunity_autoc(score, true_benefit) -> float:
    """Oracle pooled AUTOC whose units are opportunities, not unique patients."""

    return autoc(np.asarray(score, dtype=float), np.asarray(true_benefit, dtype=float))
