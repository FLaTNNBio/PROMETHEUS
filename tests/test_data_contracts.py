import numpy as np
import pandas as pd
import pytest

from causal_population_ranking.causal_utilization.simulation import simulate_care_intensity
from causal_population_ranking.data.validation import (
    assert_no_oracle_columns,
    assert_patient_level_split_integrity,
)


def cohort(n=200):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "age": rng.uniform(50, 100, n),
        "female": rng.binomial(1, 0.5, n),
        "condition_distinct": rng.poisson(3, n),
        "prior_inpatient": rng.poisson(0.5, n),
        "prior_emergency": rng.poisson(1, n),
        "medication_distinct": rng.poisson(4, n),
        "recent_utilization_trend": rng.normal(size=n),
        "multimorbidity": rng.binomial(1, 0.7, n),
        "polypharmacy": rng.binomial(1, 0.4, n),
        "encounter_count": rng.poisson(3, n),
        "days_since_encounter": np.zeros(n),
        "procedure_count": np.zeros(n),
        "careplan_count": np.zeros(n),
    })


def test_current_simulation_is_reproducible_and_keeps_truth_separate():
    first = simulate_care_intensity(cohort(), 9)
    second = simulate_care_intensity(cohort(), 9)
    for transition in first[0]:
        assert_no_oracle_columns(first[0][transition])
        assert first[0][transition].equals(second[0][transition])
        assert first[1][transition].equals(second[1][transition])
    assert first[2].equals(second[2])


def test_leakage_guard_rejects_ground_truth_columns():
    with pytest.raises(ValueError):
        assert_no_oracle_columns(pd.DataFrame({"x": [1], "true_benefit": [2]}))


def test_cross_transition_patient_split_leakage_is_detected():
    frames = {
        "2_to_3": pd.DataFrame({"patient_id": ["p1"], "split": ["rank_train"]}),
        "3_to_4": pd.DataFrame({"patient_id": ["p1"], "split": ["validation"]}),
    }
    with pytest.raises(ValueError, match="Patient-level train/validation/test leakage"):
        assert_patient_level_split_integrity(frames)
