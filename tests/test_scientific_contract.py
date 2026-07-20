from __future__ import annotations

from pathlib import Path

import pytest
import yaml


PIPELINE_PATH = Path("configs/pipeline.yaml")
CARE_CATALOG_PATH = Path("configs/care_catalog.yaml")


def _yaml(source: Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    return yaml.safe_load(source.read_text(encoding="utf-8"))


PIPELINE = _yaml(PIPELINE_PATH)
CARE_CATALOG = _yaml(CARE_CATALOG_PATH)
PROFILE_CATALOG = CARE_CATALOG
ACTION_CATALOG = CARE_CATALOG
NEED_CONTRACT = PIPELINE["baseline_need"]["contract"]
SCIENTIFIC_CONTRACT = PIPELINE["scientific_contract"]
PROFILE_SEEDS = PIPELINE["seed_registry"]


def test_profile_catalogue_has_explicit_non_aliasing_bundle_contract():
    profiles_document = _yaml(PROFILE_CATALOG)
    actions_document = _yaml(ACTION_CATALOG)
    profiles = profiles_document["profiles"]
    actions = {item["action_id"]: item for item in actions_document["actions"]}
    states = {
        item["state_id"]: set(item["active_services"])
        for item in actions_document["care_states"]
    }
    required = {
        "care_profile_id", "care_profile_level", "care_profile_name",
        "resulting_care_state", "included_care_actions",
        "action_application_order", "admissible_current_profile_ids",
        "eligible_baseline_need_levels", "outcome_name", "outcome_horizon",
        "outcome_unit", "benefit_direction", "ranking_domain",
        "resource_cost", "capacity_pool", "capacity_requirements",
        "clinical_prerequisites", "mandatory", "protected",
        "maintenance_reference_only", "automatic_rank_candidate",
    }
    profile_ids = [item["care_profile_id"] for item in profiles]
    allowed_need_features = set(_yaml(NEED_CONTRACT)["allowed_preindex_features"])
    known_services = set().union(*states.values())

    assert profiles_document["component_action_catalog_version"].startswith("dm77_")
    assert len(profile_ids) == len(set(profile_ids))
    assert any(len(item["included_care_actions"]) > 1 for item in profiles)
    assert sum(item["maintenance_reference_only"] for item in profiles) == 1
    assert all(required <= set(item) for item in profiles)

    for profile in profiles:
        assert profile["care_profile_id"].startswith("profile_")
        assert int(profile["care_profile_level"]) in range(1, 7)
        assert profile["resulting_care_state"] in states
        assert profile["included_care_actions"] == profile["action_application_order"]
        assert set(profile["included_care_actions"]) <= set(actions)
        assert set(profile["clinical_prerequisites"]) <= (
            known_services | allowed_need_features
        )
        assert set(profile["admissible_current_profile_ids"]) <= set(profile_ids)
        assert profile["care_profile_id"] not in profile["admissible_current_profile_ids"]
        expected_cost = sum(
            float(actions[action_id]["cost"])
            for action_id in profile["included_care_actions"]
        )
        assert float(profile["resource_cost"]) == pytest.approx(expected_cost)
        expected_pools = {
            actions[action_id]["capacity_pool"]
            for action_id in profile["included_care_actions"]
        }
        assert set(profile["capacity_requirements"]) == expected_pools
        if expected_pools:
            assert profile["capacity_pool"] in expected_pools
        else:
            assert profile["capacity_pool"] == "none"
        for left_id in profile["included_care_actions"]:
            exclusions = set(actions[left_id]["mutually_exclusive_with"])
            assert not exclusions.intersection(profile["included_care_actions"])
            action = actions[left_id]
            assert action["outcome_name"] == profile["outcome_name"]
            assert int(action["followup_horizon"]) == int(profile["outcome_horizon"])
            assert action["outcome_unit"] == profile["outcome_unit"]
            assert action["outcome_direction"] == profile["benefit_direction"]


def test_every_declared_profile_bundle_has_a_feasible_order_from_each_source():
    profiles = _yaml(PROFILE_CATALOG)["profiles"]
    actions_document = _yaml(ACTION_CATALOG)
    actions = {item["action_id"]: item for item in actions_document["actions"]}
    states = {
        item["state_id"]: set(item["active_services"])
        for item in actions_document["care_states"]
    }
    profile_by_id = {item["care_profile_id"]: item for item in profiles}

    for profile in profiles:
        for current_profile_id in profile["admissible_current_profile_ids"]:
            current_state = profile_by_id[current_profile_id]["resulting_care_state"]
            active = set(states[current_state])
            for action_id in profile["action_application_order"]:
                action = actions[action_id]
                resulting_services = states[action["to_state"]]
                if resulting_services <= active:
                    continue
                assert current_state in action["from_states"], (
                    profile["care_profile_id"], current_profile_id, action_id
                )
                assert set(action["prerequisites"]) <= active
                current_state = action["to_state"]
                active = set(states[current_state])
            assert current_state == profile["resulting_care_state"]


def test_standard_profiles_share_one_outcome_domain_and_palliative_is_separate():
    profiles = _yaml(PROFILE_CATALOG)["profiles"]
    ranked = [item for item in profiles if item["automatic_rank_candidate"]]
    contracts = {
        (
            item["outcome_name"], int(item["outcome_horizon"]),
            item["outcome_unit"], item["benefit_direction"], item["ranking_domain"],
        )
        for item in ranked
    }
    protected = [item for item in profiles if item["protected"]]

    assert len(contracts) == 1
    assert len(protected) == 1
    assert protected[0]["automatic_rank_candidate"] is False
    assert protected[0]["ranking_domain"] != ranked[0]["ranking_domain"]
    assert int(protected[0]["outcome_horizon"]) != int(ranked[0]["outcome_horizon"])


def test_baseline_need_contract_forbids_every_downstream_and_oracle_input():
    contract = _yaml(NEED_CONTRACT)
    allowed = set(contract["allowed_preindex_features"])
    forbidden = set(contract["forbidden_exact_fields"])
    modes = contract["modes"]

    assert allowed.isdisjoint(forbidden)
    assert set(modes) == {"rules", "supervised"}
    assert modes["supervised"]["reference_label"] == "clinician_assigned_need_level"
    assert modes["supervised"]["implementation"].startswith("cumulative_binary_ordinal")
    assert modes["supervised"]["capacity_inputs_allowed"] is False
    assert modes["supervised"]["causal_ranking_inputs_allowed"] is False
    assert contract["synthetic_label_contract"]["learner_may_read_truth_field"] is False
    assert {
        "recommended_actionable_level", "allocated_care_level",
        "raw_priority_score", "calibrated_incremental_benefit",
        "dr_pseudo_outcome", "observed_outcome",
    } <= forbidden
    assert {"true_", "oracle_", "potential_outcome", "latent_"} <= set(
        contract["forbidden_prefixes"]
    )


def test_scientific_contract_separates_need_recommendation_and_allocation():
    contract = _yaml(SCIENTIFIC_CONTRACT)
    outputs = contract["patient_level_outputs"]
    recommendation = contract["recommendation_contract"]
    allocation = contract["allocation_contract"]

    assert contract["decision_unit"]["primary"] == "care_profile"
    assert contract["decision_unit"]["only_supported_runtime"] == "care_profile"
    assert contract["decision_unit"]["silent_action_to_profile_conversion_allowed"] is False
    assert set(outputs) >= {
        "baseline_need_level", "recommended_actionable_level", "allocated_care_level",
    }
    assert outputs["must_remain_distinct"] is True
    assert contract["treatment_contract"]["treatment_identifier"] == "care_profile_id"
    assert contract["treatment_contract"]["comparator_identifier"] == "no_new_profile"
    assert contract["ranking_contract"]["primary_objective"] == "direct_pairwise_causal_ranking"
    assert contract["ranking_contract"]["score_has_causal_zero"] is False
    assert recommendation["selector"] == "argmax_calibrated_incremental_benefit"
    assert float(recommendation["primary_minimum_benefit_threshold_days"]) > 0.0
    assert recommendation["minimum_benefit_operator"] == "strictly_greater_than"
    assert recommendation["capacity_inputs_allowed"] is False
    assert recommendation["record_written_before_allocation"] is True
    assert allocation["stage"] == "after_recommendation_freeze"
    assert allocation["alternative_profile_substitution_allowed"] is False
    assert allocation["recommendation_may_be_overwritten"] is False
    assert allocation["existing_components_charged_again"] is False


def test_oracle_boundary_opens_only_after_recommendation_and_allocation_freeze():
    boundary = _yaml(SCIENTIFIC_CONTRACT)["oracle_boundary"]
    stages = set(boundary["forbidden_stages"])
    open_conditions = set(boundary["evaluation_open_condition"])

    assert {"calibration", "recommendation", "allocation", "eligibility"} <= stages
    assert {
        "recommendations_written", "allocations_written", "ranker_frozen",
        "calibrator_frozen", "recommendation_policy_frozen", "allocation_policy_frozen",
    } <= open_conditions
    assert boundary["oracle_join_stage"] == "final_synthetic_evaluation_only"


def test_profile_seed_partitions_are_disjoint_and_confirmation_is_lock_guarded():
    profile = _yaml(PROFILE_SEEDS)
    groups = {
        name: set(map(int, values.get("run_seeds", [])))
        for partition in ("consumed", "reserved")
        for name, values in profile.get(partition, {}).items()
    }
    names = list(groups)
    for index, left in enumerate(names):
        assert groups[left]
        for right in names[index + 1:]:
            assert groups[left].isdisjoint(groups[right])

    assert profile["reserved"] == {}
    assert "do_not_reuse" in profile["consumed"]["confirmation"]["status"]
    assert "do_not_reuse" in profile["consumed"]["fixed_dataset_stability"]["status"]
    assert "only_after" in profile["consumed"]["stabilized_confirmation"]["status"]
    assert "only_after" in profile["consumed"]["stabilized_fixed_dataset"]["status"]
    assert profile["policy"]["superseded_seed_registries_reused_automatically"] is False


def test_phase9b_stabilization_protocol_is_prespecified_and_oracle_safe():
    experiments = PIPELINE["experiments"]
    contract = SCIENTIFIC_CONTRACT["experiment_contract"]
    assert experiments["candidate_variants"] == [
        "global_rank_only", "global_rank_plus_contrastive"
    ]
    assert experiments["protocol_version"] == "prometheus_ranker_stabilization_v7"
    stability = experiments["stability_diagnostics"]
    assert stability["registry_group"] == "stability_diagnostics_v7"
    assert experiments["population_size"] >= 5000
    assert stability["oracle_allowed"] is False
    assert stability["outcome_allowed_in_stability_metric"] is False
    assert stability["observed_outcome_allowed_for_causal_supervision"] is True
    assert stability["baseline_comparison_role"] == (
        "descriptive_only_not_an_acceptance_gate"
    )
    assert not any("improvement" in key for key in stability["acceptance"])
    assert set(stability["sources"]) == {
        "model_initialization", "pair_sampling", "nuisance_fitting",
        "split_and_nuisance",
    }
    assert PIPELINE["ranker"]["model_restarts"] == 3
    assert PIPELINE["ranker"]["training_pairs"] == 4096
    assert PIPELINE["ranker"]["epochs"] == 32
    assert experiments["discovery"]["selection_partition"] == "validation"
    assert experiments["discovery"]["oracle_allowed"] is False
    assert experiments["discovery"]["test_partition_allowed"] is False
    assert len(experiments["discovery"]["run_seeds"]) == 5
    assert len(experiments["confirmation"]["run_seeds"]) == 10
    assert len(experiments["fixed_dataset_stability"]["run_seeds"]) == 5
    assert experiments["reporting"]["bootstrap_samples"] >= 500
    assert experiments["reporting"]["oracle_solver_time_limit_seconds"] >= (
        PIPELINE["allocation"]["solver"]["time_limit_seconds"]
    )
    assert contract["discovery_oracle_allowed"] is False
    assert contract["stabilization_diagnostics_oracle_allowed"] is False
    assert contract["stabilization_diagnostics_outcome_metric_allowed"] is False
    assert contract["stabilization_thresholds_frozen_before_run"] is True
    assert contract["candidate_may_change_after_confirmation_unblinding"] is False
    assert PIPELINE["gates"]["require_phase9_protocol_exit_gates"] is True


def test_phase8_diagnostics_are_heldout_nonoracle_and_consumed_before_discovery():
    diagnostics = PIPELINE["diagnostics"]
    contract = SCIENTIFIC_CONTRACT["diagnostic_contract"]
    consumed = PROFILE_SEEDS["consumed"]["negative_control_diagnostics"]

    assert diagnostics["evaluation_partition"] == "test"
    assert diagnostics["evaluation_partition_used_for_fitting_or_early_stopping"] is False
    assert diagnostics["eligible_for_model_selection"] is False
    assert diagnostics["base_seeds"] == [1201, 1223, 1249, 1277, 1301]
    assert diagnostics["base_seeds"] == consumed["run_seeds"]
    assert tuple(map(float, diagnostics["threshold_sensitivity_days"])) == (0.0, 2.0, 5.0)
    assert diagnostics["supervised_need_ablation"]["population_size"] == 5000
    assert contract["oracle_targets_allowed"] is False
    assert contract["diagnostic_results_select_primary_model"] is False
    assert contract["hidden_confounding_eligible_for_optimization"] is False
    assert PIPELINE["gates"]["require_phase8_diagnostic_exit_gates"] is True


def test_current_documents_make_pipeline_scope_and_decisions_explicit():
    active = Path("docs/prometheus_causal_stratification_plan.md").read_text(
        encoding="utf-8"
    )
    scientific = Path("docs/prometheus_profile_scientific_contract.md").read_text(
        encoding="utf-8"
    )
    decisions = Path("docs/prometheus_decisions.md").read_text(
        encoding="utf-8"
    )

    assert "one pipeline command" in active
    assert "profile-specific target-trial template" in scientific.lower()
    assert "P0-22" in decisions
    assert "current pipeline" in decisions.lower()


def test_repository_exposes_one_pipeline_runner_without_phase_entrypoints():
    from causal_population_ranking.cli import main

    scripts = sorted(path.name for path in Path("src/scripts").glob("*.py"))
    configs = sorted(path.name for path in Path("configs").glob("*.yaml"))
    nested_configs = list(Path("configs").glob("*/*.yaml"))
    phase_documents = list(Path("docs").glob("*phase*.md"))

    assert callable(main)
    assert scripts == ["run_pipeline.py"]
    assert configs == ["care_catalog.yaml", "pipeline.yaml"]
    assert nested_configs == []
    assert phase_documents == []
