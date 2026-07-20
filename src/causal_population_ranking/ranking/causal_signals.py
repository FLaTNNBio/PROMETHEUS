from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RobustSignalResult:
    raw_aggregate: np.ndarray
    robust_repeated: np.ndarray
    robust_aggregate: np.ndarray
    reliability_weight: np.ndarray
    winsor_bounds: dict[str, list[float]]


def doubly_robust_signal(
    transition_treatment: torch.Tensor,
    outcome: torch.Tensor,
    propensity: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    propensity_clip_epsilon: float = 0.02,
) -> torch.Tensor:
    """Return a transition-specific cross-fitted doubly robust signal.

    ``transition_treatment`` is the binary escalation indicator D^(k).  The
    returned Gamma is noisy supervision for ordering patients; it is not an
    observed or calibrated individual treatment effect.
    """
    if not 0.0 < propensity_clip_epsilon < 0.5:
        raise ValueError("propensity_clip_epsilon must be in (0, 0.5)")
    arrays = (transition_treatment, outcome, propensity, mu0, mu1)
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("D, outcome, propensity, mu0, and mu1 must be one-dimensional")
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("D, outcome, propensity, mu0, and mu1 must align")
    e = propensity.clamp(propensity_clip_epsilon, 1.0 - propensity_clip_epsilon)
    d = transition_treatment
    gamma = (
        mu1
        - mu0
        + d / e * (outcome - mu1)
        - (1.0 - d) / (1.0 - e) * (outcome - mu0)
    )
    if not torch.isfinite(gamma).all():
        raise ValueError("Cross-fitted doubly robust signal is not finite")
    return gamma


def repeated_doubly_robust_signals(
    transition_treatment,
    outcome,
    nuisance_repetitions,
    propensity_clip_epsilon: float = 0.02,
) -> np.ndarray:
    """Return one fixed DR signal per row and nuisance repetition.

    ``nuisance_repetitions`` must have shape ``[rows, repetitions, 3]`` with
    propensity, lower-arm outcome prediction, and upper-arm outcome prediction.
    No oracle quantity is accepted by this function.
    """
    if not 0.0 < propensity_clip_epsilon < 0.5:
        raise ValueError("propensity_clip_epsilon must be in (0, 0.5)")
    treatment = np.asarray(transition_treatment, dtype=float)
    observed = np.asarray(outcome, dtype=float)
    nuisance = np.asarray(nuisance_repetitions, dtype=float)
    if treatment.ndim != 1 or observed.shape != treatment.shape:
        raise ValueError("Treatment and outcome must be aligned one-dimensional arrays")
    if nuisance.ndim != 3 or nuisance.shape[0] != len(treatment) or nuisance.shape[2] != 3:
        raise ValueError("Expected nuisance repetitions with shape [rows, repetitions, 3]")
    if nuisance.shape[1] < 1 or not all(
        np.isfinite(value).all() for value in (treatment, observed, nuisance)
    ):
        raise ValueError("Repeated DR inputs must be finite and contain at least one repetition")
    propensity = np.clip(
        nuisance[:, :, 0], propensity_clip_epsilon, 1.0 - propensity_clip_epsilon
    )
    mu0 = nuisance[:, :, 1]
    mu1 = nuisance[:, :, 2]
    d = treatment[:, None]
    y = observed[:, None]
    signal = mu1 - mu0 + d / propensity * (y - mu1) - (1.0 - d) / (
        1.0 - propensity
    ) * (y - mu0)
    if not np.isfinite(signal).all():
        raise ValueError("Repeated cross-fitted doubly robust signals are not finite")
    return signal


def repeated_causal_signals(
    transition_treatment,
    outcome,
    nuisance_repetitions,
    propensity_clip_epsilon: float = 0.02,
    estimator: str = "dr",
) -> np.ndarray:
    """Build repeated non-oracle causal supervision for signal ablations.

    ``dr`` is the primary estimator. The alternatives deliberately weaken one
    nuisance component and are intended for prespecified ablation studies only.
    None of the estimators accepts synthetic ground truth.
    """

    treatment = np.asarray(transition_treatment, dtype=float)
    observed = np.asarray(outcome, dtype=float)
    nuisance = np.asarray(nuisance_repetitions, dtype=float)
    if treatment.ndim != 1 or observed.shape != treatment.shape:
        raise ValueError("Treatment and outcome must be aligned one-dimensional arrays")
    if nuisance.ndim != 3 or nuisance.shape[0] != len(treatment) or nuisance.shape[2] != 3:
        raise ValueError("Expected nuisance repetitions with shape [rows, repetitions, 3]")
    if not 0.0 < propensity_clip_epsilon < 0.5:
        raise ValueError("propensity_clip_epsilon must be in (0, 0.5)")
    if nuisance.shape[1] < 1 or not all(
        np.isfinite(value).all() for value in (treatment, observed, nuisance)
    ):
        raise ValueError("Repeated causal-signal inputs must be finite")
    method = str(estimator)
    if method == "dr":
        return repeated_doubly_robust_signals(
            treatment, observed, nuisance, propensity_clip_epsilon
        )
    propensity = np.clip(
        nuisance[:, :, 0], propensity_clip_epsilon, 1.0 - propensity_clip_epsilon
    )
    mu0, mu1 = nuisance[:, :, 1], nuisance[:, :, 2]
    d, y = treatment[:, None], observed[:, None]
    if method == "outcome_regression":
        signal = mu1 - mu0
    elif method == "ipw":
        signal = d * y / propensity - (1.0 - d) * y / (1.0 - propensity)
    elif method == "naive_outcome":
        signal = np.repeat((2.0 * d - 1.0) * y, nuisance.shape[1], axis=1)
    else:
        raise ValueError(
            "causal signal estimator must be dr, outcome_regression, ipw, or naive_outcome"
        )
    if not np.isfinite(signal).all():
        raise ValueError("Repeated causal signals are not finite")
    return signal.astype(float)


def aggregate_repeated_signals(signals, method: str = "median") -> np.ndarray:
    """Aggregate repeated DR supervision row-wise without oracle information."""
    values = np.asarray(signals, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite repeated signals with shape [rows, repetitions]")
    if method == "median":
        result = np.median(values, axis=1)
    elif method == "mean":
        result = np.mean(values, axis=1)
    else:
        raise ValueError("DR aggregation method must be 'median' or 'mean'")
    return result.astype(float)


def robustify_repeated_signals(
    signals,
    split_labels,
    aggregation: str = "median",
    winsorize: bool = True,
    winsorize_quantiles=(0.01, 0.99),
) -> RobustSignalResult:
    """Create split-local robust DR supervision without opening another split.

    Winsorization bounds are estimated independently inside each protocol split,
    so test outcomes cannot affect rank training, validation, or calibration.
    Raw signals remain available to the caller for audit export.
    """

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
            if not mask.any():
                continue
            lower, upper = np.quantile(values[mask], quantiles)
            robust[mask] = np.clip(robust[mask], lower, upper)
            bounds[str(split)] = [float(lower), float(upper)]
    robust_aggregate = aggregate_repeated_signals(robust, aggregation)
    median = np.median(robust, axis=1, keepdims=True)
    dispersion = np.median(np.abs(robust - median), axis=1)
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


def dr_signal_diagnostics(
    signal,
    propensity,
    treatment,
    action_id: str,
    split: str,
) -> dict:
    """Summarize robust supervision and inverse-probability weight stability."""

    dr = np.asarray(signal, dtype=float)
    e = np.asarray(propensity, dtype=float)
    d = np.asarray(treatment, dtype=int)
    if dr.ndim != 1 or e.shape != dr.shape or d.shape != dr.shape or not len(dr):
        raise ValueError("DR diagnostic arrays must be non-empty and aligned")
    if not np.isfinite(dr).all() or not np.isfinite(e).all():
        raise ValueError("DR diagnostic inputs must be finite")
    weights = d / np.clip(e, 1e-8, 1.0) + (1 - d) / np.clip(1.0 - e, 1e-8, 1.0)
    effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())
    return {
        "action_id": str(action_id),
        "split": str(split),
        "n": int(len(dr)),
        "dr_mean": float(np.mean(dr)),
        "dr_std": float(np.std(dr, ddof=1)) if len(dr) > 1 else 0.0,
        "dr_median": float(np.median(dr)),
        "dr_p01": float(np.quantile(dr, 0.01)),
        "dr_p99": float(np.quantile(dr, 0.99)),
        "dr_min": float(np.min(dr)),
        "dr_max": float(np.max(dr)),
        "propensity_min": float(np.min(e)),
        "propensity_p01": float(np.quantile(e, 0.01)),
        "propensity_p99": float(np.quantile(e, 0.99)),
        "propensity_max": float(np.max(e)),
        "effective_sample_size": effective_sample_size,
    }
