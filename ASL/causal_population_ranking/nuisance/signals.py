"""Robust non-oracle supervision signals for exact care-profile comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustSignalResult:
    raw_aggregate: np.ndarray
    robust_repeated: np.ndarray
    robust_aggregate: np.ndarray
    reliability_weight: np.ndarray
    winsor_bounds: dict[str, list[float]]


def repeated_doubly_robust_signals(
    profile_treatment,
    outcome,
    nuisance_repetitions,
    propensity_clip_epsilon: float = 0.02,
) -> np.ndarray:
    """Return one fixed DR ordering signal per row and nuisance repetition.

    ``nuisance_repetitions`` has shape ``[rows, repetitions, 3]`` containing
    propensity, comparator-outcome prediction, and profile-outcome prediction.
    The result is noisy ranking supervision, not a calibrated individual effect.
    """

    if not 0.0 < float(propensity_clip_epsilon) < 0.5:
        raise ValueError("propensity_clip_epsilon must be in (0, 0.5)")
    treatment = np.asarray(profile_treatment, dtype=float)
    observed = np.asarray(outcome, dtype=float)
    nuisance = np.asarray(nuisance_repetitions, dtype=float)
    if treatment.ndim != 1 or observed.shape != treatment.shape:
        raise ValueError("Treatment and outcome must be aligned one-dimensional arrays")
    if (
        nuisance.ndim != 3
        or nuisance.shape[0] != len(treatment)
        or nuisance.shape[2] != 3
    ):
        raise ValueError(
            "Expected nuisance repetitions with shape [rows, repetitions, 3]"
        )
    if nuisance.shape[1] < 1 or not all(
        np.isfinite(value).all() for value in (treatment, observed, nuisance)
    ):
        raise ValueError(
            "Repeated DR inputs must be finite and contain at least one repetition"
        )

    propensity = np.clip(
        nuisance[:, :, 0],
        propensity_clip_epsilon,
        1.0 - propensity_clip_epsilon,
    )
    comparator_prediction = nuisance[:, :, 1]
    profile_prediction = nuisance[:, :, 2]
    treatment_column = treatment[:, None]
    outcome_column = observed[:, None]
    signal = (
        profile_prediction
        - comparator_prediction
        + treatment_column
        / propensity
        * (outcome_column - profile_prediction)
        - (1.0 - treatment_column)
        / (1.0 - propensity)
        * (outcome_column - comparator_prediction)
    )
    if not np.isfinite(signal).all():
        raise ValueError("Repeated cross-fitted doubly robust signals are not finite")
    return signal


def aggregate_repeated_signals(signals, method: str = "median") -> np.ndarray:
    """Aggregate repeated supervision row-wise without oracle information."""

    values = np.asarray(signals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(
            "Expected finite repeated signals with shape [rows, repetitions]"
        )
    if method == "median":
        result = np.median(values, axis=1)
    elif method == "mean":
        result = np.mean(values, axis=1)
    else:
        raise ValueError("Signal aggregation method must be 'median' or 'mean'")
    return result.astype(float)


def robustify_repeated_signals(
    signals,
    split_labels,
    aggregation: str = "median",
    winsorize: bool = True,
    winsorize_quantiles=(0.01, 0.99),
) -> RobustSignalResult:
    """Apply split-local winsorization and derive reliability weights."""

    values = np.asarray(signals, dtype=float)
    labels = np.asarray(split_labels).astype(str)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("Repeated DR signals and split labels must align")
    if not np.isfinite(values).all():
        raise ValueError("DR signals must be finite")
    quantiles = tuple(float(value) for value in winsorize_quantiles)
    if len(quantiles) != 2 or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
        raise ValueError("winsorize_quantiles must satisfy 0 <= lower < upper <= 1")

    raw_aggregate = aggregate_repeated_signals(values, aggregation)
    robust = values.copy()
    bounds: dict[str, list[float]] = {}
    if winsorize:
        for split in sorted(set(labels)):
            mask = labels == split
            lower, upper = np.quantile(values[mask], quantiles)
            robust[mask] = np.clip(robust[mask], lower, upper)
            bounds[str(split)] = [float(lower), float(upper)]
    robust_aggregate = aggregate_repeated_signals(robust, aggregation)
    row_median = np.median(robust, axis=1, keepdims=True)
    dispersion = np.median(np.abs(robust - row_median), axis=1)
    reliability = 1.0 / (1.0 + dispersion)
    if reliability.mean() > 0:
        reliability /= reliability.mean()
    return RobustSignalResult(
        raw_aggregate=raw_aggregate,
        robust_repeated=robust,
        robust_aggregate=robust_aggregate,
        reliability_weight=reliability.astype(float),
        winsor_bounds=bounds,
    )
