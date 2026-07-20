"""Pre-index current-care state derivation, separate from DM 77 need."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .action_models import CareActionCatalog


PROTECTED_STATE = "protected_palliative_care"


def _z(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-8)


def derive_current_care_states(
    patients: pd.DataFrame,
    catalog: CareActionCatalog,
    seed: int,
) -> pd.DataFrame:
    """Simulate/load services already active before the action index date.

    This model deliberately does not read ``dm77_need_level``. Need and care state
    can be associated through common clinical history but are not identified.
    """

    required = {
        "patient_id", "age", "condition_distinct", "prior_inpatient",
        "prior_emergency", "medication_distinct", "encounter_count",
    }
    missing = sorted(required.difference(patients.columns))
    if missing:
        raise ValueError(f"Current-care state inputs are missing: {missing}")
    regular_states = tuple(state for state in catalog.care_states if state != PROTECTED_STATE)
    if len(regular_states) != 6:
        raise ValueError("Integrated v1 expects six ordinary current-care states")
    rng = np.random.default_rng(int(seed))
    intensity = _z(
        0.25 * _z(patients.age)
        + 0.65 * _z(np.log1p(patients.condition_distinct))
        + 0.55 * _z(np.log1p(patients.prior_inpatient))
        + 0.30 * _z(np.log1p(patients.prior_emergency))
        + 0.25 * _z(np.log1p(patients.medication_distinct))
        + 0.15 * _z(np.log1p(patients.encounter_count))
        + rng.normal(0.0, 0.85, len(patients))
    )
    state_index = np.digitize(intensity, (-1.05, -0.45, 0.15, 0.75, 1.35)).astype(int)
    states = np.asarray(regular_states, dtype=object)[state_index]
    return pd.DataFrame({
        "patient_id": patients.patient_id.astype(str).to_numpy(),
        "current_care_state": states,
        "current_care_state_index": state_index,
        "active_services": [
            "|".join(catalog.active_services_by_state[str(state)]) for state in states
        ],
        "current_care_state_reason_codes": "SYNTHETIC_PREINDEX_SERVICE_HISTORY_MODEL",
        "current_care_state_source": "seeded_synthetic_preindex_model_v1",
    })


def attach_current_care_state_features(
    patients: pd.DataFrame,
    states: pd.DataFrame,
    catalog: CareActionCatalog,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    merged = patients.merge(states, on="patient_id", validate="one_to_one")
    feature_columns = []
    for state in catalog.care_states:
        if state == PROTECTED_STATE:
            continue
        column = f"current_state__{state}"
        merged[column] = (merged.current_care_state == state).astype(float)
        feature_columns.append(column)
    return merged, tuple(feature_columns)
