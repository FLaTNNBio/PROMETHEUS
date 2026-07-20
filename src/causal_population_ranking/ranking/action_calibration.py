"""Validation-only calibration alternatives for patient-action priority scores."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..evaluation.action_policy_metrics import linear_calibration_metrics


def _arrays(score, target, action_index=None, sample_weight=None):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    action = (
        np.zeros(len(score), dtype=int)
        if action_index is None else np.asarray(action_index, dtype=int)
    )
    weight = (
        np.ones(len(score), dtype=float)
        if sample_weight is None else np.asarray(sample_weight, dtype=float)
    )
    if any(value.shape != score.shape for value in (target, action, weight)):
        raise ValueError("Calibration arrays must align")
    if len(score) < 2 or not all(np.isfinite(value).all() for value in (score, target, weight)):
        raise ValueError("Calibration requires finite aligned arrays")
    if (weight <= 0).any():
        raise ValueError("Calibration weights must be positive")
    return score, target, action, weight


def _rank_correlation(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(float)
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(float)
    if left_rank.std() <= 0 or right_rank.std() <= 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def validation_calibration_metrics(prediction, target) -> dict:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return {
        "validation_dr_mae": float(np.mean(np.abs(prediction - target))),
        "validation_dr_mse": float(np.mean(np.square(prediction - target))),
        "validation_rank_correlation": _rank_correlation(prediction, target),
        **linear_calibration_metrics(prediction, target, "validation_dr"),
    }


class PooledIsotonicActionCalibrator:
    def __init__(self, reliability_weighted: bool = False):
        self.reliability_weighted = bool(reliability_weighted)
        self.method = (
            "reliability_weighted_pooled_isotonic"
            if self.reliability_weighted else "pooled_isotonic"
        )

    def fit(self, score, target, action_index=None, sample_weight=None):
        score, target, _, weight = _arrays(score, target, action_index, sample_weight)
        used_weight = weight if self.reliability_weighted else None
        self.model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        fitted = self.model.fit_transform(score, target, sample_weight=used_weight)
        self.diagnostics = {
            "method": self.method,
            "fit_partition": "validation",
            "target": "heldout_robust_doubly_robust_signal_days",
            "oracle_target": False,
            "pooled": True,
            "reliability_weighted": self.reliability_weighted,
            "monotonicity_check": bool(
                np.all(np.diff(fitted[np.argsort(score, kind="mergesort")]) >= -1e-12)
            ),
            **validation_calibration_metrics(fitted, target),
        }
        return self

    def predict(self, score, action_index=None) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(score, dtype=float)), dtype=float)


class MonotonicBinnedActionCalibrator:
    def __init__(self, minimum_bin_size: int = 25):
        if int(minimum_bin_size) < 2:
            raise ValueError("minimum_bin_size must be at least 2")
        self.minimum_bin_size = int(minimum_bin_size)
        self.method = "monotonic_binned"

    def fit(self, score, target, action_index=None, sample_weight=None):
        score, target, _, weight = _arrays(score, target, action_index, sample_weight)
        order = np.argsort(score, kind="mergesort")
        bin_count = max(1, len(score) // self.minimum_bin_size)
        chunks = np.array_split(order, bin_count)
        bin_score = np.asarray([
            np.average(score[index], weights=weight[index]) for index in chunks
        ])
        bin_target = np.asarray([
            np.average(target[index], weights=weight[index]) for index in chunks
        ])
        bin_weight = np.asarray([weight[index].sum() for index in chunks])
        if len(chunks) == 1:
            self.constant_ = float(bin_target[0])
            fitted = np.full(len(score), self.constant_, dtype=float)
            self.model = None
        else:
            self.model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            self.model.fit(bin_score, bin_target, sample_weight=bin_weight)
            fitted = self.model.predict(score)
        self.diagnostics = {
            "method": self.method,
            "fit_partition": "validation",
            "target": "heldout_robust_doubly_robust_signal_days",
            "oracle_target": False,
            "pooled": True,
            "minimum_bin_size": self.minimum_bin_size,
            "bins": int(len(chunks)),
            "monotonicity_check": bool(
                np.all(np.diff(fitted[np.argsort(score, kind="mergesort")]) >= -1e-12)
            ),
            **validation_calibration_metrics(fitted, target),
        }
        return self

    def predict(self, score, action_index=None) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        if self.model is None:
            return np.full(len(score), self.constant_, dtype=float)
        return np.asarray(self.model.predict(score), dtype=float)


class ActionMeanShrinkageCalibrator:
    def __init__(self, pooled_weight: float = 0.70):
        if not 0.0 <= float(pooled_weight) <= 1.0:
            raise ValueError("pooled_weight must be in [0,1]")
        self.pooled_weight = float(pooled_weight)
        self.method = "action_mean_shrinkage"

    def fit(self, score, target, action_index=None, sample_weight=None):
        score, target, action, weight = _arrays(score, target, action_index, sample_weight)
        self.pooled = PooledIsotonicActionCalibrator(reliability_weighted=True).fit(
            score, target, action, weight
        )
        self.global_mean = float(np.average(target, weights=weight))
        self.action_means = {
            int(value): float(np.average(target[action == value], weights=weight[action == value]))
            for value in np.unique(action)
        }
        fitted = self.predict(score, action)
        self.diagnostics = {
            "method": self.method,
            "fit_partition": "validation",
            "target": "heldout_robust_doubly_robust_signal_days",
            "oracle_target": False,
            "pooled": False,
            "pooled_weight": self.pooled_weight,
            "action_means": self.action_means,
            "monotonicity_check": "within_action",
            **validation_calibration_metrics(fitted, target),
        }
        return self

    def predict(self, score, action_index=None) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        action = np.asarray(action_index, dtype=int)
        if action.shape != score.shape:
            raise ValueError("Shrinkage prediction requires aligned action indices")
        pooled = self.pooled.predict(score)
        action_mean = np.asarray([
            self.action_means.get(int(value), self.global_mean) for value in action
        ])
        return self.pooled_weight * pooled + (1.0 - self.pooled_weight) * action_mean


def fit_action_calibrators(
    score,
    target,
    action_index,
    reliability_weight,
    methods,
    minimum_bin_size: int = 25,
    shrinkage_pooled_weight: float = 0.70,
) -> dict[str, object]:
    constructors = {
        "pooled_isotonic": lambda: PooledIsotonicActionCalibrator(False),
        "reliability_weighted_pooled_isotonic": lambda: PooledIsotonicActionCalibrator(True),
        "monotonic_binned": lambda: MonotonicBinnedActionCalibrator(minimum_bin_size),
        "action_mean_shrinkage": lambda: ActionMeanShrinkageCalibrator(shrinkage_pooled_weight),
    }
    unknown = set(methods).difference(constructors)
    if unknown:
        raise ValueError(f"Unknown action calibration methods: {sorted(unknown)}")
    return {
        method: constructors[method]().fit(
            score, target, action_index, reliability_weight
        )
        for method in methods
    }
