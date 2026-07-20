from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_cohort(path, sample_size, seed):
    raw = pd.read_csv(path, low_memory=False)
    raw = raw.sample(min(sample_size, len(raw)), random_state=seed).reset_index(drop=True)
    out = pd.DataFrame({
        "patient_id": raw.patient_id.astype(str),
        "age": raw.age.astype(float),
        "female": raw.female.astype(float),
        "condition_distinct": raw.condition_count.astype(float),
        "prior_inpatient": raw.prior_inpatient.astype(float),
        "prior_emergency": raw.prior_emergency.astype(float),
        "medication_distinct": raw.medication_count.astype(float),
        "recent_utilization_trend": raw.encounter_count.astype(float) / 12,
        "multimorbidity": raw.multimorbidity.astype(float),
        "polypharmacy": raw.polypharmacy.astype(float),
        "encounter_count": raw.encounter_count.astype(float),
        "days_since_encounter": 0.0,
        "procedure_count": 0.0,
        "careplan_count": 0.0,
    })
    for column in (
        "baseline_care_level", "current_care_level",
        "eligible_1_to_2", "eligible_2_to_3", "eligible_3_to_4",
        "eligible_4_to_5", "eligible_5_to_6",
    ):
        if column in raw:
            out[column] = raw[column].to_numpy()
    return out


def _align(frame, truth):
    out = truth.set_index("patient_id").loc[frame.patient_id.astype(str)].reset_index()
    if out.patient_id.astype(str).tolist() != frame.patient_id.astype(str).tolist():
        raise RuntimeError("patient alignment failed")
    return out


def _nx(frame):
    return frame[["e_hat", "mu0_hat", "mu1_hat"]].to_numpy(float)


def _percentile(validation_score, test_score):
    ordered = np.sort(np.asarray(validation_score, float))
    return np.searchsorted(ordered, np.asarray(test_score, float), side="right") / len(ordered)


def _policy_value(assignments, truth):
    aligned = _align(assignments, truth)
    levels = assignments.assigned_causal_level.to_numpy(int)
    baseline_column = (
        "baseline_care_level" if "baseline_care_level" in assignments else "current_care_level"
    )
    current = assignments[baseline_column].to_numpy(int)
    actionable = (current >= 1) & (current <= 5)

    def gather(chosen):
        values = np.empty(len(chosen), float)
        for level in range(1, 7):
            mask = chosen == level
            values[mask] = aligned[f"potential_outcome_{level}"].to_numpy()[mask]
        return values

    return (
        float(gather(levels)[actionable].mean()),
        float(gather(current)[actionable].mean()),
        int(actionable.sum()),
    )


def _risk_proxy(frame):
    raw = (
        0.25 * frame.age.to_numpy(float)
        + 6 * frame.condition_distinct.to_numpy(float)
        + 10 * frame.prior_inpatient.to_numpy(float)
        + 6 * frame.prior_emergency.to_numpy(float)
        + 2 * frame.medication_distinct.to_numpy(float)
    )
    return (raw - raw.mean()) / (raw.std() + 1e-8)
