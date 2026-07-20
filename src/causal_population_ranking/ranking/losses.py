from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .pair_sampler import PairSampler


@dataclass(frozen=True)
class PairBatch:
    left: np.ndarray
    right: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    candidate_count: int
    discarded_fraction: float
    mean_direction_agreement: float = 1.0
    stability_discarded_fraction: float = 0.0


def _normalized_reliability_weights(
    signal_gap: np.ndarray,
    enabled: bool,
    max_weight: float,
) -> np.ndarray:
    if max_weight <= 0:
        raise ValueError("pair_reliability_weighting.max_weight must be positive")
    if not enabled:
        return np.ones(len(signal_gap), dtype=np.float32)
    raw = np.minimum(float(max_weight), np.abs(signal_gap)).astype(np.float64)
    mean = float(raw.mean()) if len(raw) else 0.0
    if mean <= 0 or not np.isfinite(mean):
        return np.ones(len(raw), dtype=np.float32)
    return (raw / mean).astype(np.float32)


def sample_ranking_pairs(
    signal,
    pairs: int,
    min_signal_gap: float,
    seed: int,
    reliability_weighting: bool = False,
    max_weight: float = 1.0,
    repeated_signal=None,
    min_direction_agreement: float = 0.0,
) -> PairBatch:
    """Sample stable orderable pairs from one transition only.

    When repeated DR signals are supplied, a pair is retained only if the sign
    of its repetition-specific gap agrees sufficiently often with the ordering
    induced by the aggregated signal.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError("Need a finite one-dimensional signal with at least two rows")
    if pairs < 1 or min_signal_gap < 0:
        raise ValueError("pairs must be positive and min_signal_gap non-negative")
    if not 0.0 <= min_direction_agreement <= 1.0:
        raise ValueError("min_direction_agreement must be in [0, 1]")
    if repeated_signal is None:
        repeated = signal[:, None]
    else:
        repeated = np.asarray(repeated_signal, dtype=float)
        if repeated.ndim != 2 or repeated.shape[0] != len(signal):
            raise ValueError("Repeated signal must have shape [rows, repetitions]")
        if repeated.shape[1] < 1 or not np.isfinite(repeated).all():
            raise ValueError("Repeated signal must be finite and non-empty")
    total = len(signal) * (len(signal) - 1)
    candidate_count = min(total, max(pairs, 4 * pairs))
    left, right = PairSampler(len(signal), candidate_count, seed).sample()
    gap = signal[left] - signal[right]
    direction = np.sign(gap)
    repeated_direction = np.sign(repeated[left] - repeated[right])
    agreement = np.mean(repeated_direction == direction[:, None], axis=1)
    gap_valid = np.abs(gap) > float(min_signal_gap)
    stable = agreement >= float(min_direction_agreement)
    valid = gap_valid & stable
    discarded_fraction = float(1.0 - valid.mean())
    stability_discarded_fraction = float(np.mean(gap_valid & ~stable))
    left = left[valid][:pairs]
    right = right[valid][:pairs]
    gap = gap[valid][:pairs]
    agreement = agreement[valid][:pairs]
    target = np.sign(gap).astype(np.float32)
    weights = _normalized_reliability_weights(gap, reliability_weighting, max_weight)
    return PairBatch(
        left, right, target, weights, candidate_count, discarded_fraction,
        float(agreement.mean()) if len(agreement) else float("nan"),
        stability_discarded_fraction,
    )


def pairwise_causal_ranking_loss(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    direction: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard pairwise logistic loss guided by fixed DR signal ordering."""
    if score_i.shape != score_j.shape or score_i.shape != direction.shape:
        raise ValueError("Scores and pair directions must have matching shapes")
    if not torch.all((direction == -1) | (direction == 1)):
        raise ValueError("Pair directions must be -1 or +1")
    raw = F.softplus(-direction * (score_i - score_j))
    if weights is not None:
        if weights.shape != raw.shape:
            raise ValueError("Pair weights must align with pair losses")
        raw = raw * weights
    return raw.mean()


def fit_response_bin_boundaries(signal, num_bins: int) -> np.ndarray:
    """Fit response-stratum boundaries on rank-training signals only."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < num_bins or num_bins < 2:
        raise ValueError("Need at least one finite training signal per response bin")
    if not np.isfinite(signal).all():
        raise ValueError("Response-bin training signal must be finite")
    return np.quantile(signal, np.arange(1, num_bins) / num_bins).astype(float)


def apply_response_bin_boundaries(signal, boundaries) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    if signal.ndim != 1 or boundaries.ndim != 1:
        raise ValueError("Signals and response-bin boundaries must be one-dimensional")
    if not np.isfinite(signal).all() or not np.isfinite(boundaries).all():
        raise ValueError("Signals and response-bin boundaries must be finite")
    return np.searchsorted(boundaries, signal, side="right").astype(np.int64)


def permute_response_labels(labels, seed: int) -> np.ndarray:
    """Preserve stratum counts while breaking the patient-label assignment."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("Response labels must be one-dimensional")
    result = np.random.default_rng(seed).permutation(labels)
    if len(labels) > 1 and len(np.unique(labels)) > 1 and np.array_equal(result, labels):
        result = np.roll(result, 1)
    return result


def sample_contrastive_pairs(
    labels,
    pairs: int,
    seed: int,
    negative_bin_separation: int = 2,
    positive_negative_pair_ratio: float = 1.0,
    ordered_labels: bool = True,
    signal=None,
    reliability_weighting: bool = False,
    max_weight: float = 1.0,
) -> PairBatch:
    """Sample positive and negative pairs from a single transition."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) < 2 or pairs < 1:
        raise ValueError("Need labels for at least two rows and a positive pair budget")
    if negative_bin_separation < 1 or positive_negative_pair_ratio <= 0:
        raise ValueError("Invalid contrastive sampling configuration")
    if signal is not None:
        signal = np.asarray(signal, dtype=float)
        if signal.shape != labels.shape:
            raise ValueError("Reliability signal must align with contrastive labels")

    rng = np.random.default_rng(seed)
    members = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
    positive_labels = [label for label, index in members.items() if len(index) >= 2]
    negative_labels = [
        (left, right)
        for left in members
        for right in members
        if left < right
        and (not ordered_labels or abs(left - right) >= negative_bin_separation)
    ]
    positive_target = int(round(pairs * positive_negative_pair_ratio / (1.0 + positive_negative_pair_ratio)))
    negative_target = pairs - positive_target
    if not positive_labels:
        positive_target = 0
        negative_target = pairs
    if not negative_labels:
        negative_target = 0
        positive_target = pairs

    left, right, target = [], [], []
    for _ in range(positive_target):
        label = positive_labels[int(rng.integers(len(positive_labels)))]
        i, j = rng.choice(members[label], size=2, replace=False)
        left.append(int(i)); right.append(int(j)); target.append(1.0)
    for _ in range(negative_target):
        low, high = negative_labels[int(rng.integers(len(negative_labels)))]
        left.append(int(rng.choice(members[low])))
        right.append(int(rng.choice(members[high])))
        target.append(0.0)
    if not left:
        return PairBatch(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), pairs, 1.0,
        )
    permutation = rng.permutation(len(left))
    left_array = np.asarray(left, dtype=np.int64)[permutation]
    right_array = np.asarray(right, dtype=np.int64)[permutation]
    target_array = np.asarray(target, dtype=np.float32)[permutation]
    gap = (
        signal[left_array] - signal[right_array]
        if signal is not None else np.ones(len(left_array), dtype=float)
    )
    weights = _normalized_reliability_weights(gap, reliability_weighting, max_weight)
    return PairBatch(left_array, right_array, target_array, weights, pairs, float(1 - len(left_array) / pairs))


def causal_contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    similar: torch.Tensor,
    margin: float = 1.0,
    normalize_embeddings: bool = False,
    weights: torch.Tensor | None = None,
    return_diagnostics: bool = False,
):
    """Margin loss on transition-conditioned latent representations."""
    if margin <= 0:
        raise ValueError("contrastive_margin must be positive")
    if z_i.shape != z_j.shape or z_i.ndim != 2 or similar.shape != (len(z_i),):
        raise ValueError("Contrastive representations and labels must align")
    if normalize_embeddings:
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
    distance = torch.linalg.vector_norm(z_i - z_j, dim=1)
    positive = similar > 0.5
    negative = ~positive

    zero = distance.sum() * 0.0
    positive_loss = zero
    negative_loss = zero
    if positive.any():
        values = distance[positive].square()
        if weights is not None:
            local = weights[positive]
            local = local / local.mean().clamp_min(1e-12)
            values = values * local
        positive_loss = values.mean()
    if negative.any():
        values = F.relu(margin - distance[negative]).square()
        if weights is not None:
            local = weights[negative]
            local = local / local.mean().clamp_min(1e-12)
            values = values * local
        negative_loss = values.mean()
    loss = positive_loss + negative_loss
    diagnostics = {
        "positive_pair_count": int(positive.sum().item()),
        "negative_pair_count": int(negative.sum().item()),
        "mean_positive_distance": float(distance[positive].mean().detach().cpu()) if positive.any() else float("nan"),
        "mean_negative_distance": float(distance[negative].mean().detach().cpu()) if negative.any() else float("nan"),
        "active_negative_margin_fraction": float((distance[negative] < margin).float().mean().detach().cpu()) if negative.any() else 0.0,
    }
    return (loss, diagnostics) if return_diagnostics else loss


def prometheus_objective(
    ranking_loss: torch.Tensor,
    contrastive_loss: torch.Tensor,
    l2_penalty: torch.Tensor,
    lambda_con: float,
    lambda_reg: float,
) -> torch.Tensor:
    if lambda_con < 0 or lambda_reg < 0:
        raise ValueError("Objective coefficients must be non-negative")
    return ranking_loss + lambda_con * contrastive_loss + lambda_reg * l2_penalty
