from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class MonotoneScoreCalibrator:
    """Validation-only isotonic map from ordinal score to approximate benefit days."""

    def __init__(self, out_of_bounds: str = "clip"):
        if out_of_bounds != "clip":
            raise ValueError("PROMETHEUS calibration requires out_of_bounds='clip'")
        self.out_of_bounds = out_of_bounds

    def fit(self, score, heldout_dr_signal) -> "MonotoneScoreCalibrator":
        score = np.asarray(score, dtype=float)
        target = np.asarray(heldout_dr_signal, dtype=float)
        if score.ndim != 1 or target.shape != score.shape or len(score) < 2:
            raise ValueError("Calibration score and held-out DR signal must align")
        if not np.isfinite(score).all() or not np.isfinite(target).all():
            raise ValueError("Calibration inputs must be finite")
        self.model = IsotonicRegression(increasing=True, out_of_bounds=self.out_of_bounds)
        fitted = self.model.fit_transform(score, target)
        ordered = np.argsort(score, kind="mergesort")
        monotone = bool(np.all(np.diff(fitted[ordered]) >= -1e-12))
        if not monotone:
            raise AssertionError("Isotonic score calibration is not monotone")
        self.diagnostics = {
            "method": "isotonic_regression",
            "fit_partition": "validation",
            "target": "heldout_doubly_robust_signal_days",
            "oracle_target": False,
            "pooled": True,
            "out_of_bounds": self.out_of_bounds,
            "score_range": [float(score.min()), float(score.max())],
            "calibrated_range": [float(fitted.min()), float(fitted.max())],
            "mae": float(np.mean(np.abs(fitted - target))),
            "mse": float(np.mean((fitted - target) ** 2)),
            "monotonicity_check": monotone,
            "unique_score_points": int(np.unique(score).size),
            "x_thresholds": self.model.X_thresholds_.tolist(),
            "y_thresholds": self.model.y_thresholds_.tolist(),
        }
        return self

    def predict(self, score) -> np.ndarray:
        if not hasattr(self, "model"):
            raise RuntimeError("Calibrator has not been fitted")
        score = np.asarray(score, dtype=float)
        if not np.isfinite(score).all():
            raise ValueError("Calibration scores must be finite")
        return np.asarray(self.model.predict(score), dtype=float)


class TransitionwiseCalibrator:
    """Separate validation calibrators for local-ranker allocation baselines."""

    def __init__(self, transition_names: tuple[str, ...]):
        self.transition_names = tuple(transition_names)
        self.models = {name: MonotoneScoreCalibrator() for name in self.transition_names}

    def fit(self, score, signal, transition_index) -> "TransitionwiseCalibrator":
        score = np.asarray(score, dtype=float)
        signal = np.asarray(signal, dtype=float)
        transition = np.asarray(transition_index, dtype=np.int64)
        for index, name in enumerate(self.transition_names):
            mask = transition == index
            if mask.sum() < 2:
                raise ValueError(f"Insufficient validation opportunities to calibrate {name}")
            self.models[name].fit(score[mask], signal[mask])
            self.models[name].diagnostics["pooled"] = False
            self.models[name].diagnostics["transition"] = name
        self.diagnostics = {
            "method": "transitionwise_isotonic_regression",
            "pooled": False,
            "raw_scores_globally_comparable": False,
            "per_transition": {name: model.diagnostics for name, model in self.models.items()},
        }
        return self

    def predict(self, score, transition_index) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        transition = np.asarray(transition_index, dtype=np.int64)
        if score.shape != transition.shape:
            raise ValueError("Local calibration arrays must align")
        result = np.empty(len(score), dtype=float)
        for index, name in enumerate(self.transition_names):
            mask = transition == index
            result[mask] = self.models[name].predict(score[mask])
        return result
