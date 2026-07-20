from __future__ import annotations

import numpy as np
import pandas as pd


BASELINE_LEVEL_COLUMN = "baseline_care_level"
CURRENT_LEVEL_COLUMN = "current_care_level"


def _z(values) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.mean()) / (values.std() + 1e-8)


def validate_current_care_level(values, n: int | None = None) -> np.ndarray:
    """Validate an externally supplied ordered current-care state."""
    raw = np.asarray(values)
    if n is not None and len(raw) != n:
        raise ValueError(f"Expected {n} current care levels, received {len(raw)}")
    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("current_care_level must contain finite integers")
    levels = numeric.astype(int)
    if not np.isin(levels, np.arange(1, 7)).all():
        raise ValueError("current_care_level must be in {1, 2, 3, 4, 5, 6}")
    return levels


def simulate_external_current_care_level(cohort: pd.DataFrame, seed: int) -> np.ndarray:
    """Simulate an observed external stratification for semi-synthetic studies.

    This adapter is not part of the causal allocation method and is not a
    clinical-need or safety-floor model. It creates the baseline state A from
    pre-index covariates only so the core can be exercised when no source
    system supplies a current care level.
    """
    required = {
        "age", "condition_distinct", "prior_inpatient", "prior_emergency",
        "medication_distinct", "recent_utilization_trend",
    }
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise ValueError(f"Missing current-level adapter columns: {missing}")
    rng = np.random.default_rng(seed)
    score = (
        0.25 * _z(cohort.age)
        + 0.65 * _z(np.log1p(cohort.condition_distinct))
        + 0.55 * _z(np.log1p(cohort.prior_inpatient))
        + 0.40 * _z(np.log1p(cohort.prior_emergency))
        + 0.25 * _z(np.log1p(cohort.medication_distinct))
        + 0.15 * _z(cohort.recent_utilization_trend)
        + rng.normal(0.0, 0.35, len(cohort))
    )
    cutpoints = np.quantile(score, np.arange(1, 6) / 6)
    return np.digitize(score, cutpoints, right=True).astype(int) + 1


def attach_current_care_level(cohort: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Use a supplied current level or attach the separate simulation adapter."""
    out = cohort.reset_index(drop=True).copy()
    if BASELINE_LEVEL_COLUMN in out:
        levels = validate_current_care_level(out[BASELINE_LEVEL_COLUMN], len(out))
        source = "provided_by_input_system"
    elif CURRENT_LEVEL_COLUMN in out:
        levels = validate_current_care_level(out[CURRENT_LEVEL_COLUMN], len(out))
        source = "provided_by_input_system"
    else:
        levels = simulate_external_current_care_level(out, seed)
        source = "semi_synthetic_external_stratification_adapter"
    out[BASELINE_LEVEL_COLUMN] = levels
    out[CURRENT_LEVEL_COLUMN] = levels
    shares = {str(level): float(np.mean(levels == level)) for level in range(1, 7)}
    return out, {
        "source": source,
        "column": BASELINE_LEVEL_COLUMN,
        "seed": int(seed),
        "level_shares": shares,
        "causal_method_component": False,
        "clinical_floor": False,
    }
