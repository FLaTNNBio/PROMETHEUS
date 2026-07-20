"""Tests for the synthetic care-profile data-generating process."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from causal_population_ranking.dm77 import (
    BaselineNeedStratifier,
    load_care_profile_catalog,
)
from causal_population_ranking.synthetic import (
    NO_NEW_PROFILE,
    PROFILE_DGP_SCENARIOS,
    generate_exact_current_care_profiles,
    generate_synthetic_population,
    generate_synthetic_profile_dgp,
    validate_synthetic_profile_dgp,
)


SEEDS = {
    "reference_seed": 2111,
    "need_model_seed": 2129,
    "need_split_seed": 2141,
    "current_care_seed": 2153,
    "effect_seed": 2161,
    "assignment_seed": 2179,
    "outcome_seed": 2203,
}


@pytest.fixture(scope="module")
def population():
    return generate_synthetic_population(360, seed=2101).patients


@pytest.fixture(scope="module")
def catalog():
    return load_care_profile_catalog(
        "configs/care_catalog.yaml",
        "configs/care_catalog.yaml",
    )


def _generate(population, catalog, scenario):
    return generate_synthetic_profile_dgp(
        population,
        catalog,
        scenario=scenario,
        need_mode="rules",
        **SEEDS,
    )


def test_profile_dgp_is_deterministic_and_truth_is_physically_separate(population, catalog):
    first = _generate(population, catalog, "baseline_identifiable")
    second = _generate(population, catalog, "baseline_identifiable")
    for field in (
        "learner", "baseline_need_assessments", "need_reference_labels",
        "current_care_profiles", "profile_eligibility",
        "patient_profile_opportunities", "profile_resources",
        "resource_capacities", "need_ground_truth", "profile_ground_truth",
    ):
        pd.testing.assert_frame_equal(getattr(first, field), getattr(second, field))
    assert first.metadata == second.metadata
    assert first.audit == second.audit
    assert not any(
        column.startswith(("true_", "oracle_", "potential_outcome", "latent_"))
        for column in first.learner.columns
    )
    assert not any(
        column.startswith(("true_", "oracle_", "potential_outcome", "latent_"))
        for column in first.profile_eligibility.columns
    )
    assert {"true_baseline_need_level", "true_need_latent_score"} <= set(
        first.need_ground_truth.columns
    )
    assert {
        "true_profile_benefit", "potential_outcome_no_new_profile",
        "potential_outcome_profile", "oracle_profile_rank",
    } <= set(first.profile_ground_truth.columns)


def test_profile_dgp_structural_validation_and_exact_comparator(population, catalog):
    result = _generate(population, catalog, "baseline_identifiable")
    checks, report = validate_synthetic_profile_dgp(result, catalog)
    assert report["critical_failures"] == 0
    assert checks.status.eq("pass").all()
    assert result.learner.current_care_profile.notna().all()
    assert result.current_care_profiles.current_care_profile_status.eq(
        "generated_exact_profile"
    ).all()
    observed = set(result.learner.observed_treatment_profile)
    assert NO_NEW_PROFILE in observed
    assert observed.difference({NO_NEW_PROFILE})
    assert "care_action_id" not in result.learner.columns
    assert result.metadata["comparator"] == NO_NEW_PROFILE
    assert result.metadata["comparison_semantics"] == (
        "exact_profile_assignment_versus_maintenance_only"
    )
    assert result.audit["observed_assignment_uses_only_eligible_profiles"] is True


def test_current_profile_scenarios_have_prespecified_directions(population, catalog):
    need = BaselineNeedStratifier("rules", seed=2129).predict(population).assessments
    alignment = generate_exact_current_care_profiles(
        population, need, catalog, "need_current_alignment", seed=2153
    )
    unmet = generate_exact_current_care_profiles(
        population, need, catalog, "unmet_need", seed=2153
    )
    over = generate_exact_current_care_profiles(
        population, need, catalog, "over_intensive_care", seed=2153
    )
    target = np.clip(need.baseline_need_level.to_numpy(float), 1, 5)
    aligned_level = alignment.current_care_profile_level.to_numpy(float)
    assert pd.Series(target).corr(pd.Series(aligned_level), method="spearman") > 0.85
    assert float(np.mean(unmet.current_care_profile_level.to_numpy(float) - target)) < -0.65
    assert float(np.mean(over.current_care_profile_level.to_numpy(float) - target)) > 0.65
    assert not alignment.current_care_profile.isna().any()


@pytest.fixture(scope="module")
def effect_scenarios(population, catalog):
    names = (
        "baseline_identifiable", "risk_benefit_aligned",
        "risk_benefit_misaligned", "no_shared_response", "poor_overlap",
        "sharp_null", "placebo_outcome", "hidden_confounding",
        "capacity_scarcity", "heterogeneous_profile_costs",
        "strong_observed_confounding",
    )
    return {name: _generate(population, catalog, name) for name in names}


def test_effect_scenarios_cover_shared_specific_alignment_null_and_placebo(effect_scenarios):
    baseline = effect_scenarios["baseline_identifiable"].profile_ground_truth
    aligned = effect_scenarios["risk_benefit_aligned"]
    misaligned = effect_scenarios["risk_benefit_misaligned"]
    no_shared = effect_scenarios["no_shared_response"].profile_ground_truth
    sharp_null = effect_scenarios["sharp_null"].profile_ground_truth
    placebo = effect_scenarios["placebo_outcome"].profile_ground_truth

    assert baseline.latent_shared_causal_response.std() > 0.1
    assert baseline.latent_profile_specific_response.std() > 0.1
    assert np.allclose(no_shared.latent_shared_causal_response, 0.0)
    assert no_shared.latent_profile_specific_response.std() > 0.1
    assert aligned.metadata["scenario_diagnostics_evaluation_only"][
        "risk_benefit_spearman"
    ] > 0.45
    assert misaligned.metadata["scenario_diagnostics_evaluation_only"][
        "risk_benefit_spearman"
    ] < -0.25
    assert np.allclose(sharp_null.true_profile_benefit, 0.0)
    assert np.allclose(placebo.true_profile_benefit, 0.0)
    assert float(np.abs(placebo.true_primary_profile_benefit).mean()) > 0.1


def test_assignment_identification_and_resource_stress_are_declared(effect_scenarios, catalog):
    baseline = effect_scenarios["baseline_identifiable"]
    poor = effect_scenarios["poor_overlap"]
    hidden = effect_scenarios["hidden_confounding"]
    strong = effect_scenarios["strong_observed_confounding"]
    scarcity = effect_scenarios["capacity_scarcity"]
    costs = effect_scenarios["heterogeneous_profile_costs"]

    baseline_max = baseline.metadata["scenario_diagnostics_evaluation_only"][
        "maximum_profile_assignment_probability"
    ]
    poor_max = poor.metadata["scenario_diagnostics_evaluation_only"][
        "maximum_profile_assignment_probability"
    ]
    assert poor_max > baseline_max
    assert poor.metadata["identification"]["positivity_stressed"] is True
    assert hidden.metadata["identification"]["hidden_confounding_present"] is True
    assert hidden.metadata["identification"][
        "conditional_exchangeability_given_learner_covariates"
    ] is False
    assert hidden.audit["latent_hidden_variable_used_by_assignment_dgp"] is True
    assert strong.metadata["identification"]["observed_confounding_strength"] == "strong"
    assert scarcity.resource_capacities.capacity_scarcity.astype(bool).all()
    assert scarcity.resource_capacities.capacity_fraction_of_population.max() < 0.05
    assert costs.profile_resources.scenario_cost_multiplier.nunique() > 3
    for result in effect_scenarios.values():
        _, report = validate_synthetic_profile_dgp(result, catalog)
        assert report["critical_failures"] == 0


def test_scenario_assignment_remains_nontrivial_and_uses_one_observed_outcome(effect_scenarios):
    for name, result in effect_scenarios.items():
        treated = result.learner.observed_treatment_profile.ne(NO_NEW_PROFILE)
        assert 0 < int(treated.sum()) < len(treated), name
        assert result.learner.outcome_name.nunique() == 1
        assert result.learner.observed_outcome.notna().all()
        assert result.profile_ground_truth[["patient_id", "care_profile_id"]].duplicated().sum() == 0
        assert result.audit["one_common_analysis_outcome"] is True
        assert result.metadata["ranker_architecture_used_to_design_dgp"] is False


def test_pipeline_configuration_freezes_all_scenarios_and_distinct_consumed_seeds():
    config = yaml.safe_load(Path(
        "configs/pipeline.yaml"
    ).read_text(encoding="utf-8"))
    registry = config["seed_registry"]
    assert tuple(config["scenarios"]) == PROFILE_DGP_SCENARIOS
    seed_keys = (
        "population_seed", "reference_seed", "need_model_seed", "need_split_seed",
        "current_care_seed", "effect_seed", "assignment_seed", "outcome_seed",
        "split_seed", "nuisance_seed",
    )
    seeds = [int(config["run"][key]) for key in seed_keys]
    consumed = {
        int(seed)
        for group in registry["consumed"].values()
        for seed in group.get("run_seeds", [])
    }
    assert len(seeds) == len(set(seeds))
    assert set(seeds) <= consumed
    assert config["gates"]["truth_tables_written_after_learner_boundary"] is True
    assert config["gates"]["oracle_used_for_model_selection"] is False
    supervision = config["causal_supervision"]
    repeat_seeds = {
        int(config["run"]["nuisance_seed"])
        + repeat * int(supervision["repeat_seed_stride"])
        for repeat in range(int(supervision["repeats"]))
    }
    assert repeat_seeds <= consumed
    assert supervision["model"] == "linear"
    assert supervision["split_fractions"] == {
        "nuisance_train": 0.40,
        "rank_train": 0.30,
        "validation": 0.15,
        "test": 0.15,
    }
