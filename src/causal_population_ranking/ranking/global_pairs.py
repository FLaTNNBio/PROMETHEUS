from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .opportunities import OpportunityArrays


@dataclass(frozen=True)
class GlobalPairBatch:
    left: np.ndarray
    right: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    is_cross: np.ndarray
    direction_agreement: np.ndarray
    diagnostics: dict

    def __len__(self) -> int:
        return len(self.left)

    def subset(self, mask) -> "GlobalPairBatch":
        mask = np.asarray(mask)
        return GlobalPairBatch(
            self.left[mask], self.right[mask], self.target[mask], self.weights[mask],
            self.is_cross[mask], self.direction_agreement[mask], dict(self.diagnostics),
        )


def _empty() -> GlobalPairBatch:
    return GlobalPairBatch(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32),
        np.empty(0, dtype=bool), np.empty(0, dtype=np.float32), {},
    )


def _budgets(total: int, blocks: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    if total < 0:
        raise ValueError("Pair budgets must be non-negative")
    if not blocks:
        return {}
    quotient, remainder = divmod(total, len(blocks))
    return {block: quotient + int(index < remainder) for index, block in enumerate(blocks)}


def _availability_weighted_budgets(
    total: int,
    blocks: list[tuple[int, int]],
    opportunities: OpportunityArrays,
) -> dict[tuple[int, int], int]:
    """Allocate pair draws in proportion to block opportunity counts."""

    if total < 0:
        raise ValueError("Pair budgets must be non-negative")
    weights = []
    for left, right in blocks:
        left_n = int(np.sum(opportunities.transition_index == left))
        right_n = int(np.sum(opportunities.transition_index == right))
        weights.append(left_n * (left_n - 1) // 2 if left == right else left_n * right_n)
    weight = np.asarray(weights, dtype=float)
    if not len(weight) or weight.sum() <= 0:
        return {block: 0 for block in blocks}
    raw = total * weight / weight.sum()
    budget = np.floor(raw).astype(int)
    remainder = int(total - budget.sum())
    if remainder:
        order = np.argsort(-(raw - budget), kind="mergesort")
        budget[order[:remainder]] += 1
    return {block: int(value) for block, value in zip(blocks, budget)}


def _candidate_indices(
    left_members: np.ndarray,
    right_members: np.ndarray,
    same_block: bool,
    target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if target < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    maximum = (
        len(left_members) * (len(left_members) - 1) // 2
        if same_block else len(left_members) * len(right_members)
    )
    if maximum < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    draw = min(maximum, max(target * 12, target + 64))
    if maximum <= 250_000 and draw >= maximum // 3:
        if same_block:
            local_left, local_right = np.triu_indices(len(left_members), 1)
            left, right = left_members[local_left], left_members[local_right]
        else:
            left = np.repeat(left_members, len(right_members))
            right = np.tile(right_members, len(left_members))
        permutation = rng.permutation(len(left))[:draw]
        return left[permutation], right[permutation]
    encoded: set[tuple[int, int]] = set()
    for _ in range(max(draw * 30, 500)):
        if len(encoded) >= draw:
            break
        left_value = int(left_members[int(rng.integers(len(left_members)))])
        right_value = int(right_members[int(rng.integers(len(right_members)))])
        if same_block and left_value == right_value:
            continue
        if same_block and left_value > right_value:
            left_value, right_value = right_value, left_value
        encoded.add((left_value, right_value))
    if not encoded:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    values = np.asarray(sorted(encoded), dtype=np.int64)
    permutation = rng.permutation(len(values))[:draw]
    return values[permutation, 0], values[permutation, 1]


def _block_pairs(
    opportunities: OpportunityArrays,
    block: tuple[int, int],
    budget: int,
    min_signal_gap_days: float,
    min_direction_agreement: float,
    allow_same_patient_cross_pairs: bool,
    reliability_weighting: bool,
    max_weight: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    left_transition, right_transition = block
    left_members = np.flatnonzero(opportunities.transition_index == left_transition)
    right_members = np.flatnonzero(opportunities.transition_index == right_transition)
    left, right = _candidate_indices(
        left_members, right_members, left_transition == right_transition,
        budget, np.random.default_rng(seed),
    )
    if not len(left):
        return {}, {"candidate_count": 0, "valid_count": 0}
    gap = opportunities.signal[left] - opportunities.signal[right]
    direction = np.sign(gap)
    repeated_direction = np.sign(opportunities.repeated_signal[left] - opportunities.repeated_signal[right])
    agreement = np.mean(repeated_direction == direction[:, None], axis=1)
    gap_valid = np.abs(gap) > float(min_signal_gap_days)
    stable = agreement >= float(min_direction_agreement)
    distinct_patient = (
        opportunities.patient_ids[left] != opportunities.patient_ids[right]
        if left_transition != right_transition and not allow_same_patient_cross_pairs
        else np.ones(len(left), dtype=bool)
    )
    valid = gap_valid & stable & distinct_patient & (direction != 0)
    chosen = np.flatnonzero(valid)[:budget]
    chosen_gap = gap[chosen]
    raw_weight = np.minimum(float(max_weight), np.abs(chosen_gap))
    if not reliability_weighting or not len(raw_weight) or raw_weight.mean() <= 0:
        weights = np.ones(len(chosen), dtype=np.float32)
    else:
        weights = (raw_weight / raw_weight.mean()).astype(np.float32)
    arrays = {
        "left": left[chosen].astype(np.int64), "right": right[chosen].astype(np.int64),
        "target": direction[chosen].astype(np.float32), "weights": weights,
        "agreement": agreement[chosen].astype(np.float32),
    }
    diagnostic = {
        "candidate_count": int(len(left)), "valid_count": int(len(chosen)),
        "discarded_ambiguous_fraction": float(np.mean(~gap_valid)),
        "discarded_unstable_fraction": float(np.mean(gap_valid & ~stable)),
        "discarded_same_patient_fraction": float(np.mean(gap_valid & stable & ~distinct_patient)),
        "mean_direction_agreement": float(agreement[chosen].mean()) if len(chosen) else float("nan"),
    }
    return arrays, diagnostic


def sample_global_ranking_pairs(
    opportunities: OpportunityArrays,
    within_pairs: int,
    cross_pairs: int,
    min_signal_gap_days: float,
    min_direction_agreement: float,
    seed: int,
    balance_transition_pairs: bool = True,
    allow_same_patient_cross_pairs: bool = True,
    reliability_weighting: bool = False,
    max_weight: float = 25.0,
) -> GlobalPairBatch:
    """Sample stable pairs with explicit balanced budgets for every `(k, l)` block."""

    if min_signal_gap_days < 0 or not 0.0 <= min_direction_agreement <= 1.0:
        raise ValueError("Invalid global ranking reliability settings")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    transitions = sorted(np.unique(opportunities.transition_index).tolist())
    within_blocks = [(transition, transition) for transition in transitions]
    cross_blocks = [(left, right) for left in transitions for right in transitions if left < right]
    budget_function = _budgets if balance_transition_pairs else None
    within_budgets = (
        budget_function(int(within_pairs), within_blocks)
        if budget_function else _availability_weighted_budgets(
            int(within_pairs), within_blocks, opportunities
        )
    )
    cross_budgets = (
        budget_function(int(cross_pairs), cross_blocks)
        if budget_function else _availability_weighted_budgets(
            int(cross_pairs), cross_blocks, opportunities
        )
    )
    budgets = {**within_budgets, **cross_budgets}
    parts, block_diagnostics = [], {}
    for block_index, block in enumerate((*within_blocks, *cross_blocks)):
        arrays, diagnostics = _block_pairs(
            opportunities, block, budgets[block], min_signal_gap_days,
            min_direction_agreement, allow_same_patient_cross_pairs,
            reliability_weighting, max_weight, seed + 1009 * block_index,
        )
        block_diagnostics[f"{block[0]}:{block[1]}"] = diagnostics
        if arrays:
            arrays["is_cross"] = np.full(len(arrays["left"]), block[0] != block[1], dtype=bool)
            parts.append(arrays)
    if not parts:
        return _empty()
    left = np.concatenate([part["left"] for part in parts])
    right = np.concatenate([part["right"] for part in parts])
    target = np.concatenate([part["target"] for part in parts])
    weights = np.concatenate([part["weights"] for part in parts])
    agreement = np.concatenate([part["agreement"] for part in parts])
    is_cross = np.concatenate([part["is_cross"] for part in parts])
    candidate_weights = [max(1, value["candidate_count"]) for value in block_diagnostics.values()]
    same_patient_cross = is_cross & (opportunities.patient_ids[left] == opportunities.patient_ids[right])
    diagnostics = {
        "requested_within_pairs": int(within_pairs), "requested_cross_pairs": int(cross_pairs),
        "valid_within_pairs": int((~is_cross).sum()), "valid_cross_pairs": int(is_cross.sum()),
        "pair_counts_per_transition_pair": {
            key: value["valid_count"] for key, value in block_diagnostics.items()
        },
        "per_transition_pair": block_diagnostics,
        "discarded_ambiguous_fraction": float(np.average(
            [value.get("discarded_ambiguous_fraction", 0.0) for value in block_diagnostics.values()],
            weights=candidate_weights,
        )),
        "discarded_unstable_fraction": float(np.average(
            [value.get("discarded_unstable_fraction", 0.0) for value in block_diagnostics.values()],
            weights=candidate_weights,
        )),
        "mean_direction_agreement": float(agreement.mean()),
        "same_patient_cross_pair_fraction": float(same_patient_cross.sum() / is_cross.sum()) if is_cross.any() else 0.0,
        "balanced_transition_pairs": bool(balance_transition_pairs),
    }
    return GlobalPairBatch(left, right, target, weights, is_cross, agreement, diagnostics)


def sample_global_contrastive_pairs(
    opportunities: OpportunityArrays,
    pairs: int,
    seed: int,
    positive_max_gap_days: float,
    negative_min_gap_days: float,
    source: str = "causal",
    scaled_x: np.ndarray | None = None,
    pair_scope: str = "pooled",
    require_stable_direction: bool = False,
    min_direction_agreement: float = 0.66,
    reliability_weighting: bool = False,
) -> GlobalPairBatch:
    """Sample pooled contrastive pairs using global day-gap or covariate rules."""

    if pairs < 1 or not 0 <= positive_max_gap_days < negative_min_gap_days:
        raise ValueError("Contrastive thresholds must satisfy 0 <= positive < negative")
    if source not in {"causal", "random", "covariate"}:
        raise ValueError("Unknown global contrastive source")
    if pair_scope not in {"pooled", "within_action", "balanced_mixed"}:
        raise ValueError(
            "Contrastive pair_scope must be pooled, within_action, or balanced_mixed"
        )
    if not 0.0 <= min_direction_agreement <= 1.0:
        raise ValueError("Invalid contrastive direction-agreement threshold")
    rng = np.random.default_rng(seed)
    signal = rng.permutation(opportunities.signal) if source == "random" else opportunities.signal
    transitions = sorted(np.unique(opportunities.transition_index).tolist())
    within_blocks = [(transition, transition) for transition in transitions]
    pooled_blocks = list(combinations_with_replacement(transitions, 2))
    cross_blocks = [block for block in pooled_blocks if block[0] != block[1]]
    if pair_scope == "within_action":
        blocks = within_blocks
        budgets = _budgets(pairs, blocks)
    elif pair_scope == "balanced_mixed" and cross_blocks:
        requested_within = pairs // 2
        blocks = within_blocks + cross_blocks
        budgets = {
            **_budgets(requested_within, within_blocks),
            **_budgets(pairs - requested_within, cross_blocks),
        }
    else:
        blocks = pooled_blocks
        budgets = _budgets(pairs, blocks)
    requested_within = int(sum(
        budget for block, budget in budgets.items() if block[0] == block[1]
    ))
    requested_cross = int(pairs - requested_within)
    parts, block_counts = [], {}
    positive_block_counts, negative_block_counts = {}, {}
    available_positive_counts, available_negative_counts = {}, {}
    for block_index, block in enumerate(blocks):
        left_members = np.flatnonzero(opportunities.transition_index == block[0])
        right_members = np.flatnonzero(opportunities.transition_index == block[1])
        left, right = _candidate_indices(
            left_members, right_members, block[0] == block[1], budgets[block] * 2,
            np.random.default_rng(seed + 2027 * block_index),
        )
        if not len(left):
            block_counts[f"{block[0]}:{block[1]}"] = 0
            continue
        if source == "covariate":
            x = np.asarray(scaled_x) if scaled_x is not None else None
            if x is None or x.shape[0] != len(opportunities.x):
                raise ValueError("Covariate contrastive sampling requires aligned scaled_x")
            distance = np.linalg.norm(x[left] - x[right], axis=1)
            low, high = np.quantile(distance, (0.20, 0.80))
            similar, valid = distance <= low, (distance <= low) | (distance >= high)
        else:
            gap = np.abs(signal[left] - signal[right])
            similar, valid = gap <= positive_max_gap_days, (gap <= positive_max_gap_days) | (gap >= negative_min_gap_days)
        pair_stability = np.ones(len(left), dtype=float)
        if require_stable_direction or reliability_weighting:
            aggregate_gap = opportunities.signal[left] - opportunities.signal[right]
            repeated_gap = opportunities.repeated_signal[left] - opportunities.repeated_signal[right]
            direction_agreement = np.mean(
                np.sign(repeated_gap) == np.sign(aggregate_gap)[:, None], axis=1
            )
            positive_stability = np.mean(
                np.abs(repeated_gap) <= positive_max_gap_days, axis=1
            )
            pair_stability = np.where(
                similar, positive_stability, direction_agreement
            )
            if require_stable_direction:
                valid &= pair_stability >= float(min_direction_agreement)
        positive_candidates = np.flatnonzero(valid & similar)
        negative_candidates = np.flatnonzero(valid & ~similar)
        positive_target = budgets[block] // 2
        chosen_positive = positive_candidates[:positive_target]
        negative_target = budgets[block] - len(chosen_positive)
        chosen_negative = negative_candidates[:negative_target]
        remaining = budgets[block] - len(chosen_positive) - len(chosen_negative)
        if remaining:
            extra_positive = positive_candidates[len(chosen_positive):][:remaining]
            chosen_positive = np.concatenate([chosen_positive, extra_positive])
            remaining -= len(extra_positive)
        if remaining:
            extra_negative = negative_candidates[len(chosen_negative):][:remaining]
            chosen_negative = np.concatenate([chosen_negative, extra_negative])
        chosen = np.concatenate([chosen_positive, chosen_negative]).astype(np.int64)
        if len(chosen) > 1:
            chosen = chosen[rng.permutation(len(chosen))]
        selected_stability = pair_stability[chosen]
        selected_weights = (
            np.clip(2.0 * selected_stability - 1.0, 0.05, 1.0)
            if reliability_weighting
            else np.ones(len(chosen), dtype=float)
        )
        block_name = f"{block[0]}:{block[1]}"
        block_counts[block_name] = int(len(chosen))
        positive_block_counts[block_name] = int(len(chosen_positive))
        negative_block_counts[block_name] = int(len(chosen_negative))
        available_positive_counts[block_name] = int(len(positive_candidates))
        available_negative_counts[block_name] = int(len(negative_candidates))
        if len(chosen):
            parts.append({
                "left": left[chosen], "right": right[chosen],
                "target": similar[chosen].astype(np.float32),
                "is_cross": np.full(len(chosen), block[0] != block[1], dtype=bool),
                "weights": selected_weights.astype(np.float32),
                "stability": selected_stability.astype(np.float32),
            })
    if not parts:
        return _empty()
    left = np.concatenate([part["left"] for part in parts]).astype(np.int64)
    right = np.concatenate([part["right"] for part in parts]).astype(np.int64)
    target = np.concatenate([part["target"] for part in parts]).astype(np.float32)
    is_cross = np.concatenate([part["is_cross"] for part in parts])
    weights = np.concatenate([part["weights"] for part in parts]).astype(np.float32)
    stability = np.concatenate([part["stability"] for part in parts]).astype(np.float32)
    diagnostics = {
        "source": source, "positive_pairs": int((target > 0.5).sum()),
        "negative_pairs": int((target <= 0.5).sum()), "within_pairs": int((~is_cross).sum()),
        "cross_pairs": int(is_cross.sum()), "pair_counts_per_transition_pair": block_counts,
        "requested_within_pairs": requested_within,
        "requested_cross_pairs": requested_cross,
        "positive_pairs_per_transition_pair": positive_block_counts,
        "negative_pairs_per_transition_pair": negative_block_counts,
        "available_stable_positive_pairs_per_transition_pair": available_positive_counts,
        "available_stable_negative_pairs_per_transition_pair": available_negative_counts,
        "threshold_scale": "pooled_outcome_days" if source != "covariate" else "pooled_covariate_distance",
        "pair_scope": pair_scope,
        "require_stable_direction": bool(require_stable_direction),
        "reliability_weighting": bool(reliability_weighting),
        "mean_reliability_weight": float(weights.mean()),
        "min_reliability_weight": float(weights.min()),
        "max_reliability_weight": float(weights.max()),
        "mean_pair_stability": float(stability.mean()),
    }
    return GlobalPairBatch(
        left, right, target, weights, is_cross, stability, diagnostics,
    )
