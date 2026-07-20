from __future__ import annotations
import numpy as np


def fit_capacity_threshold(validation_score, capacity: float) -> float:
    """Fit a score threshold on validation for a fixed service capacity."""
    score=np.asarray(validation_score,float)
    if not 0<capacity<1:raise ValueError("capacity must be in (0,1)")
    if not len(score) or not np.isfinite(score).all():raise ValueError("Need finite validation scores")
    return float(np.quantile(score,1-capacity))


def passes_capacity_threshold(score,threshold):return np.asarray(score,float)>=float(threshold)


def select_at_capacity(score, capacity: float | int, eligible=None) -> np.ndarray:
    """Select a deterministic top-B set with no threshold-tie overflow."""
    score = np.asarray(score, dtype=float)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("Need finite one-dimensional scores")
    eligible = np.ones(len(score), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if eligible.shape != score.shape:
        raise ValueError("Eligibility must align with scores")
    eligible_index = np.flatnonzero(eligible)
    if isinstance(capacity, (int, np.integer)):
        budget = int(capacity)
    else:
        if not 0 <= float(capacity) <= 1:
            raise ValueError("Fractional capacity must be in [0,1]")
        budget = int(np.floor(float(capacity) * len(score)))
    budget = min(max(budget, 0), len(eligible_index))
    selected = np.zeros(len(score), dtype=bool)
    if budget:
        # Stable sorting makes index order the deterministic tie breaker.
        order = eligible_index[np.argsort(-score[eligible_index], kind="mergesort")]
        selected[order[:budget]] = True
    if int(selected.sum()) > budget:
        raise AssertionError("Capacity allocation exceeds B_k")
    return selected
