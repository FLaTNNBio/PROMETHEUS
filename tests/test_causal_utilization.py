import numpy as np
import pandas as pd
import pytest

from causal_population_ranking.causal_utilization.capacity_thresholds import fit_capacity_threshold
from causal_population_ranking.causal_utilization.eligibility import derive_transition_eligibility
from causal_population_ranking.causal_utilization.initial_state import attach_current_care_level
from causal_population_ranking.causal_utilization.simulation import (
    DEFAULT_CAPACITIES,
    simulate_care_intensity,
)
from causal_population_ranking.causal_utilization.stratifier import HierarchicalCausalStratifier
from causal_population_ranking.causal_utilization.support import support_label, supported
from causal_population_ranking.data.validation import (
    assert_no_oracle_columns,
    assert_patient_level_split_integrity,
)


TRANSITIONS = tuple(DEFAULT_CAPACITIES)


def cohort(n=1200):
    rng = np.random.default_rng(8)
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "age": rng.uniform(50, 100, n),
        "female": rng.binomial(1, 0.5, n),
        "condition_distinct": rng.poisson(4, n),
        "prior_inpatient": rng.poisson(0.4, n),
        "prior_emergency": rng.poisson(0.8, n),
        "medication_distinct": rng.poisson(4, n),
        "recent_utilization_trend": rng.normal(size=n),
        "multimorbidity": rng.binomial(1, 0.65, n),
        "polypharmacy": rng.binomial(1, 0.4, n),
        "encounter_count": rng.poisson(3, n),
        "days_since_encounter": np.zeros(n),
        "procedure_count": np.zeros(n),
        "careplan_count": np.zeros(n),
    })


def test_all_five_transitions_are_generated_with_separate_oracle_truth():
    learners, truths, all_truth, _, _, features, _ = simulate_care_intensity(cohort(), 11)
    assert tuple(learners) == TRANSITIONS
    assert tuple(truths) == TRANSITIONS
    assert set(features).isdisjoint(all_truth.columns)
    for name, learner in learners.items():
        assert_no_oracle_columns(learner)
        assert len(learner) == len(truths[name])


def test_recursive_potential_outcomes_equal_transition_effects():
    _, _, truth, _, _, _, _ = simulate_care_intensity(cohort(), 12)
    for level, name in enumerate(TRANSITIONS, start=1):
        np.testing.assert_allclose(
            truth[f"potential_outcome_{level + 1}"] - truth[f"potential_outcome_{level}"],
            truth[f"true_benefit_{name}"],
            rtol=0,
            atol=1e-12,
        )


def test_received_level_equals_baseline_plus_binary_transition_treatment():
    learners, _, _, _, _, _, _ = simulate_care_intensity(cohort(), 13)
    for level, (name, learner) in enumerate(learners.items(), start=1):
        assert set(learner.transition_treatment).issubset({0, 1})
        assert learner.baseline_care_level.eq(level).all()
        np.testing.assert_array_equal(
            learner.received_care_level,
            learner.baseline_care_level + learner.transition_treatment,
        )


def test_decision_populations_are_exactly_baseline_compatible_and_disjoint():
    learners, _, _, clinical, _, _, _ = simulate_care_intensity(cohort(), 14)
    seen = set()
    clinical_index = clinical.set_index("patient_id")
    for level, (name, learner) in enumerate(learners.items(), start=1):
        identifiers = set(learner.patient_id.astype(str))
        assert seen.isdisjoint(identifiers)
        seen.update(identifiers)
        assert clinical_index.loc[list(identifiers), "baseline_care_level"].eq(level).all()
        assert clinical_index.loc[list(identifiers), f"eligible_{name}"].all()
        for other in TRANSITIONS:
            if other != name:
                assert not clinical_index.loc[list(identifiers), f"eligible_{other}"].any()


def test_shared_component_endpoints_remove_the_requested_component():
    zero = simulate_care_intensity(cohort(), 15, shared_effect_correlation=0.0)
    one = simulate_care_intensity(cohort(), 15, shared_effect_correlation=1.0)
    for name in TRANSITIONS:
        assert np.allclose(zero[2][f"weighted_shared_component_{name}"], 0.0)
        assert np.allclose(one[2][f"weighted_specific_component_{name}"], 0.0)


def test_default_dgp_contains_negative_near_zero_and_positive_effects():
    _, _, truth, _, metadata, _, _ = simulate_care_intensity(cohort(2400), 16)
    effects = truth[[f"true_benefit_{name}" for name in TRANSITIONS]].to_numpy().ravel()
    assert np.any(effects < -0.01)
    assert np.any(np.abs(effects) <= 0.01)
    assert np.any(effects > 0.01)
    assert all(value["true_effect_sd"] > 0 for value in metadata.values())


def test_patient_splits_are_reproducible_and_leakage_free():
    first = simulate_care_intensity(cohort(), 17)
    second = simulate_care_intensity(cohort(), 17)
    assert_patient_level_split_integrity(first[0])
    for name in TRANSITIONS:
        assert first[0][name].equals(second[0][name])
        assert first[1][name].equals(second[1][name])


def test_capacities_and_transition_imbalance_targets_are_respected():
    rho = 0.75
    learners, _, _, _, metadata, _, _ = simulate_care_intensity(
        cohort(3000), 18, transition_imbalance_rho=rho
    )
    first_n = len(learners["1_to_2"])
    for index, name in enumerate(TRANSITIONS):
        expected = int(np.floor(first_n * rho**index))
        assert len(learners[name]) == expected
        assert metadata[name]["target_sample_n"] == expected
        assert metadata[name]["realized_sample_n"] == expected
        assert metadata[name]["capacity"] == DEFAULT_CAPACITIES[name]


def test_overlap_and_hidden_confounding_are_explicit_stress_parameters():
    good = simulate_care_intensity(cohort(), 19, overlap_strength=0.5)
    poor = simulate_care_intensity(cohort(), 19, overlap_strength=3.0)
    hidden = simulate_care_intensity(cohort(), 19, scenario="hidden_confounding")
    good_low = np.mean([value["true_low_support_rate"] for value in good[4].values()])
    poor_low = np.mean([value["true_low_support_rate"] for value in poor[4].values()])
    assert poor_low > good_low
    assert all(value["conditional_exchangeability_by_construction"] for value in good[4].values())
    assert not any(value["conditional_exchangeability_by_construction"] for value in hidden[4].values())
    assert "unobserved_confounder_truth" not in hidden[0]["1_to_2"].columns


def test_baseline_level_adapter_and_exact_eligibility():
    supplied = cohort(6)
    supplied["baseline_care_level"] = [1, 2, 3, 4, 5, 6]
    attached, metadata = attach_current_care_level(supplied, 99)
    assert attached.baseline_care_level.tolist() == [1, 2, 3, 4, 5, 6]
    assert metadata["source"] == "provided_by_input_system"
    state = derive_transition_eligibility(attached)
    for level, name in enumerate(TRANSITIONS, start=1):
        assert state[f"eligible_{name}"].tolist() == [value == level for value in range(1, 7)]


def test_capacity_support_and_one_step_assignment():
    score = np.arange(100, dtype=float)
    threshold = fit_capacity_threshold(score, 0.10)
    assert np.sum(score >= threshold) == 10
    assert support_label([0.01, 0.06, 0.5, 0.94, 0.99]).tolist() == [
        "low", "moderate", "high", "moderate", "low"
    ]
    assert supported([0.01, 0.06, 0.5]).tolist() == [False, True, True]

    clinical = pd.DataFrame({
        "patient_id": [f"p{level}" for level in range(1, 6)],
        "baseline_care_level": np.arange(1, 6),
        "current_care_level": np.arange(1, 6),
        **{
            f"eligible_{name}": [index == level for index in range(1, 6)]
            for level, name in enumerate(TRANSITIONS, start=1)
        },
    })
    predictions = {
        name: pd.DataFrame({
            "patient_id": [f"p{level}"], "score": [1.0],
            "support_label": ["high"], "supported": [True], "selected": [True],
        })
        for level, name in enumerate(TRANSITIONS, start=1)
    }
    assigned = HierarchicalCausalStratifier({name: 0.5 for name in TRANSITIONS}).assign(
        clinical, predictions
    )
    assert assigned.assigned_causal_level.tolist() == [2, 3, 4, 5, 6]


def test_invalid_dgp_parameters_are_rejected():
    with pytest.raises(ValueError, match="shared_effect_correlation"):
        simulate_care_intensity(cohort(), 20, shared_effect_correlation=1.1)
    with pytest.raises(ValueError, match="risk_benefit_alignment"):
        simulate_care_intensity(cohort(), 20, risk_benefit_alignment=-1.1)
