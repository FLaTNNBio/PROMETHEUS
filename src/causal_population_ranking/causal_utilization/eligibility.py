from __future__ import annotations

import numpy as np
import pandas as pd

from .initial_state import BASELINE_LEVEL_COLUMN, validate_current_care_level


TRANSITION_LOWERS = {
    "1_to_2": 1,
    "2_to_3": 2,
    "3_to_4": 3,
    "4_to_5": 4,
    "5_to_6": 5,
}


def derive_transition_eligibility(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build pathway eligibility without inferring a minimum care level.

    Optional input columns named ``eligible_<transition>`` may encode external,
    clinically governed eligibility. The primary one-step decision population
    additionally requires ``baseline_care_level == lower_level`` exactly.
    """
    if BASELINE_LEVEL_COLUMN not in cohort:
        raise ValueError(f"Missing required column {BASELINE_LEVEL_COLUMN!r}")
    levels = validate_current_care_level(cohort[BASELINE_LEVEL_COLUMN], len(cohort))
    out = pd.DataFrame(
        {
            "patient_id": cohort.patient_id.astype(str),
            BASELINE_LEVEL_COLUMN: levels,
            "current_care_level": levels,
        }
    )
    for name, lower in TRANSITION_LOWERS.items():
        reachable = levels == lower
        column = f"eligible_{name}"
        if column in cohort:
            external = cohort[column]
            if external.isna().any():
                raise ValueError(f"External eligibility column {column!r} contains missing values")
            external = external.astype(bool).to_numpy()
        else:
            external = np.ones(len(cohort), dtype=bool)
        out[column] = reachable & external
    return out
