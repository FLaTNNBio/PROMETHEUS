from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class TransitionSupportOutcomePredictor:
    """Non-test transition nuisance extrapolator for support and CATE baselines."""

    def __init__(self, model: str, seed: int):
        if model not in {"linear", "gbm", "random_forest"}:
            raise ValueError("Unsupported transition support model")
        self.model_kind = model
        self.seed = int(seed)

    def _models(self):
        if self.model_kind == "linear":
            return (
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=self.seed)),
                make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            )
        # The support/export predictor stays lightweight even when the repeated
        # nuisance protocol uses the legacy random-forest option.
        return (
            HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=self.seed),
            HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=self.seed),
        )

    def fit(self, learner: pd.DataFrame, features: list[str] | tuple[str, ...]):
        fit = learner.loc[learner.split.astype(str) != "test"].copy()
        treatment = fit.transition_treatment.to_numpy(int)
        if set(np.unique(treatment)) != {0, 1}:
            raise ValueError("Non-test transition support fit requires both adjacent treatment arms")
        x = fit.loc[:, features].to_numpy(float)
        y = fit.observed_outcome.to_numpy(float)
        self.propensity_model, outcome_template = self._models()
        self.propensity_model.fit(x, treatment)
        self.outcome_models = {}
        for arm in (0, 1):
            _, model = self._models()
            model.fit(x[treatment == arm], y[treatment == arm])
            self.outcome_models[arm] = model
        self.training_patient_ids = frozenset(fit.patient_id.astype(str))
        self.diagnostics = {
            "fit_partition": "all_non_test",
            "test_outcomes_used": False,
            "training_n": int(len(fit)),
            "model": self.model_kind,
            "seed": self.seed,
        }
        return self

    def predict(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        propensity = self.propensity_model.predict_proba(x)[:, 1]
        mu0 = self.outcome_models[0].predict(x)
        mu1 = self.outcome_models[1].predict(x)
        return np.asarray(propensity), np.asarray(mu0), np.asarray(mu1)
