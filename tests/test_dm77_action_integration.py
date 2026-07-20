from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from causal_population_ranking.allocation.action_allocator import (
    allocate_care_actions,
    allocate_care_actions_greedy,
    apply_action,
)
from causal_population_ranking.causal_utilization.action_simulation import (
    ACTION_DGP_SCENARIOS,
    NO_ACTION,
    simulate_observational_care_actions,
)
from causal_population_ranking.causal_utilization.dm77_integrated_runner import (
    _allocate_action_policy,
)
from causal_population_ranking.causal_utilization.global_prometheus_runner import (
    run_global_prometheus,
)
from causal_population_ranking.data.validation import assert_no_oracle_columns
from causal_population_ranking.evaluation.action_policy_metrics import (
    actionwise_off_policy_metrics,
    allocation_budget_metrics,
    budget_curve_area,
    normalized_oracle_efficiency,
    rank_weighted_effect_metrics,
    ranking_stability_metrics,
    score_group_diagnostics,
    synthetic_top_q_metrics,
)
from causal_population_ranking.evaluation.dm77_clinical_validation import (
    evaluate_dm77_against_adjudicated_panel,
    weighted_fleiss_kappa,
)
from causal_population_ranking.dm77 import (
    assess_dm77_population,
    attach_current_care_state_features,
    build_patient_action_opportunities,
    derive_current_care_states,
    derive_dm77_action_eligibility,
    load_care_action_catalog,
    load_dm77_settings,
    validate_ranked_action_comparability,
)
from causal_population_ranking.ranking.global_pairs import sample_global_ranking_pairs
from causal_population_ranking.ranking.action_calibration import fit_action_calibrators
from causal_population_ranking.ranking.causal_signals import (
    dr_signal_diagnostics,
    repeated_causal_signals,
    robustify_repeated_signals,
)
from causal_population_ranking.ranking.opportunities import OpportunityArrays
from causal_population_ranking.ranking.tree_baselines import (
    DirectPairwiseGBDTRanker,
    DRGBDTPriority,
    DRPolicyTreePriority,
    DRRandomForestPriority,
)
from causal_population_ranking.synthetic import generate_synthetic_population


CATALOG_PATH = "configs/dm77/care_action_catalog.yaml"


def _patient(patient_id="p1"):
    return pd.DataFrame([{
        "patient_id": patient_id,
        "age": 76.0,
        "condition_distinct": 4.0,
        "prior_inpatient": 1.0,
        "prior_emergency": 1.0,
        "medication_distinct": 5.0,
        "frailty_index": 0.45,
        "functional_limitation_score": 0.45,
        "social_fragility_score": 0.35,
        "cognitive_impairment": 0.0,
        "non_self_sufficiency": 0.0,
        "caregiver_available": 1.0,
        "housing_instability": 0.0,
        "palliative_need": 0.0,
    }])


def _assessment(patient_id="p1", level=4, status="assessed", protected=False):
    return pd.DataFrame([{
        "patient_id": patient_id,
        "dm77_need_level": level if level is not None else pd.NA,
        "dm77_need_label": "test",
        "dm77_assessment_status": status,
        "dm77_reason_codes": "TEST_REASON",
        "dm77_protected_pathway": protected,
    }]).astype({"dm77_need_level": "Int64"})


def _state(patient_id="p1", state="structured_followup"):
    catalog = load_care_action_catalog(CATALOG_PATH)
    return pd.DataFrame([{
        "patient_id": patient_id,
        "current_care_state": state,
        "active_services": "|".join(catalog.active_services_by_state[state]),
    }])


def test_catalog_is_computable_and_ranked_actions_share_outcome_semantics():
    catalog = load_care_action_catalog(CATALOG_PATH)
    assert len(catalog.ranked_actions) == 6
    assert catalog.actions[-1].protected and catalog.actions[-1].mandatory
    diagnostics = validate_ranked_action_comparability(catalog)
    assert diagnostics["all_ranked_actions_comparable"] is True
    assert diagnostics["outcome_unit"] == "days"
    assert {action.action_id: action.action_index for action in catalog.ranked_actions} == {
        "add_structured_followup": 0,
        "start_chronic_disease_management": 1,
        "add_proactive_case_management": 2,
        "add_remote_monitoring": 3,
        "add_multidisciplinary_coordination": 4,
        "add_high_intensity_home_support": 5,
    }


def test_need_level_and_current_state_jointly_control_authoritative_actions():
    catalog = load_care_action_catalog(CATALOG_PATH)
    eligibility = derive_dm77_action_eligibility(
        _patient(), _assessment(level=4), _state(state="structured_followup"), catalog
    )
    candidates = build_patient_action_opportunities(eligibility)
    assert set(candidates.action_id) == {
        "start_chronic_disease_management", "add_proactive_case_management",
    }
    assert candidates.dm77_need_level.eq(4).all()
    assert candidates.current_care_state.eq("structured_followup").all()


def test_level_vi_and_manual_review_never_enter_discretionary_ranking():
    catalog = load_care_action_catalog(CATALOG_PATH)
    palliative_patient = _patient("p6").assign(palliative_need=1.0)
    level_vi = derive_dm77_action_eligibility(
        palliative_patient,
        _assessment("p6", 6, "protected_pathway", True),
        _state("p6", "chronic_disease_management"),
        catalog,
    )
    assert build_patient_action_opportunities(level_vi).empty
    eligible = level_vi.loc[level_vi.eligible]
    assert eligible.action_id.tolist() == ["protected_palliative_pathway"]
    assert eligible.protected_action.all()

    manual = derive_dm77_action_eligibility(
        _patient("pm"), _assessment("pm", None, "manual_review", False),
        _state("pm", "structured_followup"), catalog,
    )
    assert not manual.eligible.any()
    assert build_patient_action_opportunities(manual).empty


def test_authoritative_opportunities_ignore_legacy_numeric_eligibility_columns():
    catalog = load_care_action_catalog(CATALOG_PATH)
    patient = _patient().assign(
        eligible_1_to_2=False, eligible_2_to_3=False, eligible_3_to_4=False,
        eligible_4_to_5=False, eligible_5_to_6=False,
    )
    first = derive_dm77_action_eligibility(
        patient, _assessment(level=4), _state(), catalog
    )
    mutated = patient.copy()
    for column in [column for column in mutated if column.startswith("eligible_")]:
        mutated[column] = True
    second = derive_dm77_action_eligibility(
        mutated, _assessment(level=4), _state(), catalog
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(
        build_patient_action_opportunities(first),
        build_patient_action_opportunities(second),
    )


def _allocation_frame():
    common = {
        "dm77_need_level": 4,
        "eligible": True,
        "support_flag": True,
        "prerequisites_satisfied": True,
        "mandatory": False,
        "protected_action": False,
    }
    return pd.DataFrame([
        {**common, "patient_id": "p1", "current_care_state": "s0", "action_id": "a",
         "transition_index": 0, "raw_priority_score": 4.0,
         "calibrated_incremental_benefit": 8.0, "cost": 2.0, "capacity_pool": "pool_a",
         "from_states": "s0", "to_state": "s1", "mutually_exclusive_with": "b"},
        {**common, "patient_id": "p1", "current_care_state": "s0", "action_id": "b",
         "transition_index": 1, "raw_priority_score": 3.0,
         "calibrated_incremental_benefit": 7.0, "cost": 1.0, "capacity_pool": "pool_b",
         "from_states": "s0", "to_state": "s2", "mutually_exclusive_with": "a"},
        {**common, "patient_id": "p2", "current_care_state": "s0", "action_id": "a",
         "transition_index": 0, "raw_priority_score": 2.0,
         "calibrated_incremental_benefit": 6.0, "cost": 2.0, "capacity_pool": "pool_a",
         "from_states": "s0", "to_state": "s1", "mutually_exclusive_with": "b"},
        {**common, "patient_id": "p3", "current_care_state": "s0", "action_id": "b",
         "transition_index": 1, "raw_priority_score": 1.0,
         "calibrated_incremental_benefit": 100.0, "cost": 1.0, "capacity_pool": "pool_b",
         "from_states": "s0", "to_state": "s2", "mutually_exclusive_with": "a",
         "prerequisites_satisfied": False},
    ])


def test_action_allocator_enforces_exclusion_prerequisite_budget_pools_and_final_state():
    frame = _allocation_frame()
    result = allocate_care_actions(
        frame, shared_budget=3.0, capacity_pools={"pool_a": 1, "pool_b": 1},
        max_primary_actions_per_patient=1,
    )
    selected = frame.assign(selected=result.selected).loc[lambda value: value.selected]
    assert selected.groupby("patient_id").size().max() == 1
    assert not ((selected.patient_id == "p3") & (selected.action_id == "b")).any()
    assert selected.cost.sum() <= 3.0
    assert selected.groupby("capacity_pool").size().max() <= 1
    assert result.diagnostics["prerequisite_violations"] == 0
    assert result.diagnostics["mutual_exclusion_violations"] == 0
    p1 = result.decisions.set_index("patient_id").loc["p1"]
    assert p1.resulting_care_state in {"s1", "s2"}
    assert result.diagnostics["result_uses_final_package_arithmetic"] is False
    assert apply_action("s0", "b", {"b": "s2"}) == "s2"


@pytest.mark.parametrize("strategy", ["score", "score_per_cost"])
def test_greedy_allocator_is_deterministic_and_feasible(strategy):
    frame = _allocation_frame()
    first = allocate_care_actions_greedy(
        frame, shared_budget=3.0, capacity_pools={"pool_a": 1, "pool_b": 1},
        strategy=strategy,
    )
    second = allocate_care_actions_greedy(
        frame, shared_budget=3.0, capacity_pools={"pool_a": 1, "pool_b": 1},
        strategy=strategy,
    )
    np.testing.assert_array_equal(first.selected, second.selected)
    assert first.diagnostics["budget_violation"] == 0.0
    assert first.diagnostics["prerequisite_violations"] == 0
    assert first.diagnostics["capacity_pool_violations"] == {
        "pool_a": 0, "pool_b": 0,
    }


def test_action_allocator_can_require_an_explicit_joint_prerequisite_action():
    common = {
        "patient_id": "p_joint", "dm77_need_level": 4,
        "current_care_state": "s0", "eligible": True, "support_flag": True,
        "mandatory": False, "protected_action": False,
        "from_states": "s0", "mutually_exclusive_with": "",
        "capacity_pool": "joint_pool", "cost": 1.0,
    }
    frame = pd.DataFrame([
        {
            **common, "action_id": "base", "transition_index": 0,
            "raw_priority_score": 1.0, "calibrated_incremental_benefit": 4.0,
            "to_state": "s1", "prerequisites_satisfied": True,
            "joint_prerequisite_actions": "",
        },
        {
            **common, "action_id": "complement", "transition_index": 1,
            "raw_priority_score": 2.0, "calibrated_incremental_benefit": 8.0,
            "to_state": "s2", "prerequisites_satisfied": False,
            "joint_prerequisite_actions": "base",
        },
    ])
    result = allocate_care_actions(
        frame, shared_budget=2.0, capacity_pools={"joint_pool": 2},
        max_primary_actions_per_patient=2,
    )
    assert result.selected.tolist() == [True, True]
    assert result.diagnostics["prerequisite_violations"] == 0


def test_cross_action_pairs_exist_and_balance_flag_changes_imbalanced_budgets():
    rng = np.random.default_rng(9)
    transition = np.concatenate([np.zeros(60, dtype=int), np.ones(20, dtype=int), np.full(8, 2)])
    n = len(transition)
    signal = np.arange(n, dtype=float)
    repeated = np.column_stack([signal, signal + 0.01, signal - 0.01])
    opportunities = OpportunityArrays(
        rng.normal(size=(n, 4)), transition,
        np.asarray([f"p{i}" for i in range(n)]), signal, repeated,
    )
    balanced = sample_global_ranking_pairs(
        opportunities, 90, 90, 0.1, 0.66, 17, balance_transition_pairs=True
    )
    proportional = sample_global_ranking_pairs(
        opportunities, 90, 90, 0.1, 0.66, 17, balance_transition_pairs=False
    )
    assert balanced.is_cross.any()
    assert np.any(
        opportunities.transition_index[balanced.left[balanced.is_cross]]
        != opportunities.transition_index[balanced.right[balanced.is_cross]]
    )
    assert (
        balanced.diagnostics["pair_counts_per_transition_pair"]
        != proportional.diagnostics["pair_counts_per_transition_pair"]
    )


def test_cross_action_pairs_exclude_same_patient_and_use_robust_signal_labels():
    rng = np.random.default_rng(29)
    patients = np.asarray([f"p{i}" for i in range(24)] * 2)
    action = np.repeat([0, 1], 24)
    robust_signal = np.concatenate([np.arange(24), np.arange(24) + 3.5]).astype(float)
    repeated = np.column_stack([
        robust_signal - 0.05, robust_signal, robust_signal + 0.05,
    ])
    opportunities = OpportunityArrays(
        rng.normal(size=(48, 5)), action, patients, robust_signal, repeated,
    )
    pairs = sample_global_ranking_pairs(
        opportunities, 30, 60, 0.10, 0.80, 31,
        balance_transition_pairs=True, allow_same_patient_cross_pairs=False,
    )
    cross = pairs.is_cross
    assert cross.any()
    assert np.all(action[pairs.left[cross]] != action[pairs.right[cross]])
    assert np.all(patients[pairs.left[cross]] != patients[pairs.right[cross]])
    np.testing.assert_array_equal(
        pairs.target,
        np.sign(robust_signal[pairs.left] - robust_signal[pairs.right]),
    )


def test_robust_dr_signals_preserve_raw_audit_and_emit_required_diagnostics():
    raw = np.asarray([
        [1.0, 1.2, 1000.0], [2.0, 2.2, 2.4],
        [-900.0, 3.0, 3.2], [4.0, 4.2, 4.4],
    ])
    result = robustify_repeated_signals(
        raw, ["rank_train", "rank_train", "validation", "validation"],
        aggregation="median", winsorize=True, winsorize_quantiles=(0.10, 0.90),
    )
    assert result.raw_aggregate.shape == (4,)
    assert np.max(result.robust_repeated) < np.max(raw)
    assert np.min(result.robust_repeated) > np.min(raw)
    diagnostic = dr_signal_diagnostics(
        result.robust_aggregate, [0.2, 0.4, 0.6, 0.8], [0, 1, 0, 1], "a", "validation"
    )
    required = {
        "dr_mean", "dr_std", "dr_median", "dr_p01", "dr_p99", "dr_min", "dr_max",
        "propensity_min", "propensity_p01", "propensity_p99", "propensity_max",
        "effective_sample_size",
    }
    assert required <= set(diagnostic)


def test_causal_signal_ablation_estimators_use_only_observed_nuisance_inputs():
    treatment = np.asarray([0.0, 1.0, 0.0, 1.0])
    outcome = np.asarray([2.0, 7.0, 3.0, 9.0])
    nuisance = np.asarray([
        [[0.4, 2.0, 6.0], [0.5, 2.2, 6.2]],
        [[0.6, 2.5, 7.0], [0.5, 2.6, 7.1]],
        [[0.3, 3.0, 8.0], [0.4, 3.2, 8.2]],
        [[0.7, 4.0, 9.0], [0.6, 4.1, 9.1]],
    ])
    outputs = {
        method: repeated_causal_signals(
            treatment, outcome, nuisance, estimator=method
        ) for method in ("dr", "outcome_regression", "ipw", "naive_outcome")
    }
    assert all(value.shape == (4, 2) for value in outputs.values())
    assert all(np.isfinite(value).all() for value in outputs.values())
    assert not np.allclose(outputs["dr"], outputs["outcome_regression"])


def test_tree_baselines_are_seeded_non_oracle_and_return_global_scores():
    rng = np.random.default_rng(991)
    patients = np.asarray([f"p{i}" for i in range(80)])
    x_patient = rng.normal(size=(80, 5))
    x = np.repeat(x_patient, 2, axis=0)
    action = np.tile(np.arange(2), 80)
    signal = x[:, 0] - 0.4 * x[:, 1] + 1.2 * action
    repeated = np.column_stack((signal - 0.05, signal, signal + 0.05))
    opportunities = OpportunityArrays(
        x, action, np.repeat(patients, 2), signal, repeated
    )
    models = [
        DirectPairwiseGBDTRanker(
            2, within_pairs=120, cross_pairs=120, anchor_count=20,
            max_iter=15, min_signal_gap_days=0.05, seed=7,
        ),
        DRGBDTPriority(2, pooled=True, max_iter=15, seed=8),
        DRGBDTPriority(2, pooled=False, max_iter=15, seed=9),
        DRRandomForestPriority(2, estimators=20, min_samples_leaf=5, seed=10),
        DRPolicyTreePriority(2, max_depth=2, min_samples_leaf=10, seed=11),
    ]
    for model in models:
        score = model.fit(opportunities).predict_opportunity_scores(x, action)
        assert score.shape == (160,)
        assert np.isfinite(score).all()
        assert model.diagnostics["oracle_supervision"] is False


def test_rate_ope_budget_and_oracle_efficiency_metrics_have_expected_direction():
    signal = np.linspace(-2.0, 3.0, 100)
    perfect = rank_weighted_effect_metrics(signal, signal)
    inverse = rank_weighted_effect_metrics(-signal, signal)
    assert perfect["heldout_dr_rate_autoc"] > 0
    assert inverse["heldout_dr_rate_autoc"] < 0
    assert normalized_oracle_efficiency(8.0, 2.0, 10.0) == pytest.approx(0.75)
    assert budget_curve_area([0.5, 1.0], [4.0, 10.0]) == pytest.approx(3.5)
    treatment = np.tile([0.0, 1.0], 50)
    selected = signal > 0
    ope = actionwise_off_policy_metrics(
        treatment, 5.0 + treatment, np.full(100, 0.5),
        np.full(100, 5.0), np.full(100, 6.0), np.ones(100), selected,
    )
    assert ope["dr_incremental_policy_value"] == pytest.approx(selected.mean())
    assert ope["effective_sample_size"] > 0


def test_all_action_calibration_methods_fit_only_heldout_dr_targets():
    score = np.linspace(-2, 2, 80)
    action = np.repeat(np.arange(4), 20)
    target = 1.5 * score + 0.25 * action + np.sin(score)
    weights = np.linspace(0.5, 1.5, len(score))
    methods = (
        "pooled_isotonic", "reliability_weighted_pooled_isotonic",
        "monotonic_binned", "action_mean_shrinkage",
    )
    fitted = fit_action_calibrators(
        score, target, action, weights, methods, minimum_bin_size=10,
    )
    assert set(fitted) == set(methods)
    for model in fitted.values():
        prediction = model.predict(score, action)
        assert np.isfinite(prediction).all()
        assert model.diagnostics["oracle_target"] is False
        assert {"validation_dr_mae", "validation_dr_mse", "validation_rank_correlation"} <= set(
            model.diagnostics
        )


def test_top_q_budget_group_and_stability_metrics_have_explicit_targets():
    cate = np.linspace(-2.0, 8.0, 40)
    oracle_score = cate.copy()
    top = synthetic_top_q_metrics(oracle_score, cate)
    for quantile in (5, 10, 20):
        assert top[f"oracle_top_{quantile}pct_recovery"] == 1.0
        assert top[f"regret_at_{quantile}pct"] == pytest.approx(0.0)
    selected = np.zeros(len(cate), dtype=bool)
    selected[-5:] = True
    budget = allocation_budget_metrics(cate, selected, np.ones(len(cate)), selected)
    assert budget["value_at_budget"] == pytest.approx(cate[-5:].sum())
    assert budget["regret_at_budget"] == pytest.approx(0.0)

    groups = score_group_diagnostics(
        oracle_score, oracle_score, cate,
        method="identity", partition="test", target_name="true_cate",
        oracle_target=True, groups=5,
    )
    assert len(groups) == 5
    assert groups.target_benefit_mean.is_monotonic_increasing
    assert groups.oracle_target.all()

    slightly_perturbed = oracle_score.copy()
    slightly_perturbed[-2:] = slightly_perturbed[-2:][::-1]
    stability = ranking_stability_metrics(
        oracle_score, slightly_perturbed, selected, selected, cate
    )
    assert stability["ranking_spearman"] > 0.99
    assert stability["allocation_identical_fraction"] == 1.0
    assert stability["policy_value_absolute_difference"] == 0.0


def test_dm77_panel_metrics_are_separate_from_causal_and_effectiveness_claims():
    rules = pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "dm77_need_level": [2, 3, 5, 6],
        "dm77_assessment_status": ["assessed", "assessed", "assessed", "protected_pathway"],
        "dm77_protected_pathway": [False, False, False, True],
        "eligible_action_ids": ["x", "x|y", "z", ""],
    })
    reference = pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "reference_dm77_need_level": [2, 4, 5, 6],
        "reference_manual_review": [False, False, True, False],
        "reference_protected_pathway": [False, False, False, True],
        "reference_eligible_action_ids": ["x", "x|y", "z", ""],
    })
    result = evaluate_dm77_against_adjudicated_panel(rules, reference)
    assert result.summary["raw_level_agreement"] == 0.75
    assert result.summary["ordinal_mean_absolute_error"] == 0.25
    assert result.summary["protected_pathway_false_negative_rate"] == 0.0
    assert result.summary["manual_review_false_negative_rate"] == 1.0
    assert result.summary["eligible_action_exact_agreement"] == 1.0
    assert result.summary["causal_ranker_evaluated"] is False
    assert result.confusion_matrix.shape == (6, 6)
    assert weighted_fleiss_kappa(np.asarray([
        [2, 2, 2], [3, 3, 3], [5, 5, 5],
    ])) == pytest.approx(1.0)


@pytest.mark.parametrize(("field", "value"), [
    ("outcome_name", "different_outcome"),
    ("followup_horizon", 180),
    ("outcome_unit", "events"),
    ("outcome_direction", "lower_is_better"),
])
def test_incompatible_action_outcomes_block_cross_action_ranking(field, value):
    document = yaml.safe_load(Path(CATALOG_PATH).read_text(encoding="utf-8"))
    document = copy.deepcopy(document)
    document["actions"][1][field] = value
    catalog = load_care_action_catalog(document)
    with pytest.raises(ValueError, match="common outcome semantics"):
        validate_ranked_action_comparability(catalog)


def test_action_dgp_uses_catalog_candidates_and_keeps_oracle_separate():
    source = generate_synthetic_population(700, seed=811, history_months=24).patients
    settings = load_dm77_settings({"config_path": "configs/dm77/default.yaml"})
    assessments, _ = assess_dm77_population(source, settings)
    catalog = load_care_action_catalog(CATALOG_PATH)
    states = derive_current_care_states(source, catalog, seed=812)
    context, _ = attach_current_care_state_features(source, states, catalog)
    eligibility = derive_dm77_action_eligibility(source, assessments, states, catalog)
    candidates = build_patient_action_opportunities(eligibility)
    simulation = simulate_observational_care_actions(
        context, eligibility, catalog, seed=813, minimum_action_assignments_per_split=3
    )
    assert_no_oracle_columns(simulation.learner)
    assert "dm77_need_level" not in simulation.learner
    candidate_keys = set(zip(candidates.patient_id.astype(str), candidates.action_id.astype(str)))
    treated = simulation.learner.loc[simulation.learner.observed_action_id != NO_ACTION]
    assert all(
        (str(patient), str(action)) in candidate_keys
        for patient, action in zip(treated.patient_id, treated.observed_action_id)
    )
    assert any(column.startswith("true_cate__") for column in simulation.ground_truth)


def test_identifiable_baseline_has_deterministic_cate_and_zero_latent_effect():
    source = generate_synthetic_population(1600, seed=911, history_months=24).patients
    settings = load_dm77_settings({"config_path": "configs/dm77/default.yaml"})
    assessments, _ = assess_dm77_population(source, settings)
    catalog = load_care_action_catalog(CATALOG_PATH)
    states = derive_current_care_states(source, catalog, seed=912)
    context, _ = attach_current_care_state_features(source, states, catalog)
    eligibility = derive_dm77_action_eligibility(source, assessments, states, catalog)
    first = simulate_observational_care_actions(
        context, eligibility, catalog, seed=913,
        minimum_action_assignments_per_split=3, scenario="baseline_identifiable",
    )
    second = simulate_observational_care_actions(
        context, eligibility, catalog, seed=914,
        minimum_action_assignments_per_split=3, scenario="baseline_identifiable",
    )
    cate_columns = [column for column in first.ground_truth if column.startswith("true_cate__")]
    latent_columns = [
        column for column in first.ground_truth
        if column.startswith("latent_individual_effect__")
    ]
    np.testing.assert_allclose(first.ground_truth[cate_columns], second.ground_truth[cate_columns])
    assert np.allclose(first.ground_truth[latent_columns], 0.0)
    assert first.metadata["conditional_exchangeability_by_construction"] is True
    assert first.metadata["treatment_assignment_uses_oracle"] is False

    hidden = simulate_observational_care_actions(
        context, eligibility, catalog, seed=915,
        minimum_action_assignments_per_split=3, scenario="hidden_confounding",
    )
    assert not np.allclose(hidden.ground_truth[latent_columns], 0.0)
    assert "latent_confounder_u" in hidden.ground_truth
    assert "latent_confounder_u" not in hidden.learner
    assert hidden.metadata["conditional_exchangeability_by_construction"] is False
    strong_observed = simulate_observational_care_actions(
        context, eligibility, catalog, seed=916,
        minimum_action_assignments_per_split=3,
        scenario="strong_observed_confounding",
    )
    assert strong_observed.metadata["conditional_exchangeability_by_construction"] is True
    assert strong_observed.metadata["observed_confounding_strength"] == "strong"
    assert np.allclose(strong_observed.ground_truth[latent_columns], 0.0)
    assert set(ACTION_DGP_SCENARIOS) == {
        "baseline_identifiable", "strong_observed_confounding", "poor_overlap",
        "risk_benefit_misalignment",
        "targeted_selection_observed", "hidden_confounding", "combined_stress",
        "null_treatment_effect", "placebo_outcome",
    }
    for scenario in ("null_treatment_effect", "placebo_outcome"):
        control = simulate_observational_care_actions(
            context, eligibility, catalog, seed=917,
            minimum_action_assignments_per_split=3, scenario=scenario,
        )
        assert np.allclose(control.ground_truth[cate_columns], 0.0)
        assert control.metadata["sharp_null_treatment_effect"] is True
        assert control.metadata["negative_control"] is not None


def test_equal_cost_rank_only_allocation_does_not_use_cardinal_calibration():
    frame = _allocation_frame()
    config = {
        "allocation": {
            "value_mode": "equal_cost_rank_only", "shared_budget": 2.0,
            "capacity_pools": {"pool_a": 2, "pool_b": 2},
            "max_primary_actions_per_patient": 1,
            "require_empirical_support": True,
            "allocate_negative_predicted_benefit": False,
            "solver_time_limit_seconds": 10.0,
        }
    }
    result = _allocate_action_policy(
        frame, frame.raw_priority_score, np.full(len(frame), -9999.0), config,
    )
    assert result.diagnostics["value_mode"] == "equal_cost_rank_only"
    assert result.diagnostics["budget_used"] <= 2.0


def test_mandatory_action_is_selected_before_discretionary_value_filter():
    frame = _allocation_frame().iloc[[2]].copy()
    frame["mandatory"] = True
    frame["calibrated_incremental_benefit"] = -100.0
    result = allocate_care_actions(
        frame, shared_budget=2.0, capacity_pools={"pool_a": 1},
        max_primary_actions_per_patient=1,
        allocate_negative_predicted_benefit=False,
    )
    assert result.selected.tolist() == [True]


def test_dm77_integrated_smoke_writes_action_contract_and_no_oracle_operationally(tmp_path):
    config = yaml.safe_load(
        Path("configs/prometheus/dm77_integrated_smoke.yaml").read_text(encoding="utf-8")
    )
    config["run"]["output_root"] = str(tmp_path)
    config["run"]["sample_size"] = 900
    output = run_global_prometheus(config)
    assert output.name.endswith("_dm77_bi_s17")
    assert len(output.name) < 70
    required = {
        "dm77_need_assessment.csv", "patient_current_care_state.csv",
        "dm77_action_eligibility.csv", "patient_action_opportunities.csv",
        "action_specific_dr_signals.csv", "global_ranking.csv",
        "calibration_diagnostics.csv", "allocation_decisions.csv",
        "protected_and_manual_review_cases.csv", "run_manifest.json",
        "prometheus_dm77_action_ranker.pt",
        "action_specific_dr_diagnostics.csv", "baseline_comparison.csv",
        "calibration_score_group_diagnostics.csv",
        "observed_test_rate_metrics.csv", "actionwise_ope_metrics.csv",
        "allocator_ablation.csv", "budget_value_curve.csv",
        "subgroup_policy_metrics.csv", "risk_causal_policy_discordance.csv",
    }
    assert required <= {path.name for path in output.iterdir()}
    eligibility = pd.read_csv(output / "dm77_action_eligibility.csv")
    candidates = pd.read_csv(output / "patient_action_opportunities.csv")
    authoritative = eligibility.loc[eligibility.discretionary_rank_candidate]
    assert set(map(tuple, candidates[["patient_id", "action_id"]].to_numpy())) == set(
        map(tuple, authoritative[["patient_id", "action_id"]].to_numpy())
    )
    ranking = pd.read_csv(output / "global_ranking.csv")
    decisions = pd.read_csv(output / "allocation_decisions.csv")
    assert_no_oracle_columns(ranking)
    assert_no_oracle_columns(decisions)
    assert not ranking.action_id.str.match(r"^[1-5]_to_[2-6]$").any()
    protected = decisions.protected_pathway.astype(bool)
    assert not decisions.loc[protected, "allocation_status"].eq("allocated").any()
    manual = decisions.manual_review.astype(bool)
    assert decisions.loc[manual, "allocation_status"].eq("manual_review").all()
    assert decisions.allocation_status.eq("no_admissible_action").any()
    recommended = decisions.recommended_action.notna()
    assert decisions.loc[recommended, "recommended_or_allocated_action"].notna().all()
    discretionary_recommendation = recommended & ~protected
    assert decisions.loc[
        discretionary_recommendation, "calibrated_incremental_benefit"
    ].notna().all()
    ranked_decisions = decisions.candidate_action_id.notna()
    assert decisions.loc[ranked_decisions, "causal_priority_score"].notna().all()
    manifest = yaml.safe_load((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method_version"] == "prometheus_dm77_integrated_global_v3"
    assert manifest["primary_ranking_method"] == "unified_global_ranker"
    assert manifest["need_level_semantics"] == "dm77_multidimensional_need_not_treatment"
    assert manifest["resulting_state_uses_1_plus_sum"] is False
    assert manifest["cross_action_pairs"] > 0
    assert manifest["action_dgp_scenario"] == "baseline_identifiable"
    assert manifest["primary_oracle_target"] == "true_cate_f_of_observed_x"
    assert manifest["latent_individual_effect_zero"] is True
    assert manifest["fixed_validation_pair_set"] is True
    assert manifest["oracle_used_for_calibration"] is False
    assert manifest["dataset_seed"] == manifest["seed"]
    training = yaml.safe_load(
        (output / "prometheus_training_diagnostics.json").read_text(encoding="utf-8")
    )
    assert training["fixed_validation_pairs"] is True
    assert training["fixed_validation_pair_diagnostics"]["same_patient_cross_pair_fraction"] == 0
    assert {
        "initial_validation_metric", "best_epoch", "best_validation_within",
        "best_validation_cross", "best_validation_global", "validation_policy_value_dr",
    } <= set(training)
    baseline_table = pd.read_csv(output / "baseline_comparison.csv")
    assert {
        "random_priority", "risk_priority", "action_mean_priority",
        "dm77_need_fixed_action_rule", "historical_action_propensity",
        "cost_normalized_risk_priority",
            "independent_action_rankers", "unified_local_ranker",
            "unified_global_ranker", "direct_pairwise_gbdt_ranker",
            "pooled_dr_gbdt", "independent_dr_gbdt",
            "dr_random_forest_priority", "dr_policy_tree", "oracle_priority",
        } == set(baseline_table.method)
    global_metrics = yaml.safe_load(
        (output / "global_metrics.json").read_text(encoding="utf-8")
    )
    unified_global = baseline_table.set_index("method").loc["unified_global_ranker"]
    assert global_metrics["global_concordance"] == pytest.approx(
        unified_global.global_concordance
    )
    assert global_metrics["hard_cross_action_concordance"] == pytest.approx(
        unified_global.hard_cross_action_concordance
    )
    assert global_metrics["global_allocation_value"] == pytest.approx(
        unified_global.global_allocation_value
    )
    assert {
        "benefit_at_5pct", "oracle_top_5pct_recovery", "value_at_5pct",
        "regret_at_5pct", "benefit_at_budget", "value_at_budget",
        "regret_at_budget",
    } <= set(global_metrics)
    calibration_table = pd.read_csv(output / "calibration_diagnostics.csv")
    assert {
        "validation_dr_mae", "validation_dr_mse", "validation_rank_correlation",
        "synthetic_oracle_mae", "synthetic_oracle_bias", "selected_benefit_bias",
        "validation_dr_calibration_slope", "validation_dr_calibration_intercept",
        "synthetic_oracle_calibration_slope", "synthetic_oracle_calibration_intercept",
    } <= set(calibration_table)
    calibration_groups = pd.read_csv(output / "calibration_score_group_diagnostics.csv")
    assert set(calibration_groups.partition) == {
        "validation", "test_synthetic_evaluation_after_allocation_fixed",
    }
    assert not calibration_groups.loc[
        calibration_groups.partition.eq("validation"), "oracle_target"
    ].astype(bool).any()
    checkpoint = torch.load(
        output / "prometheus_dm77_action_ranker.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["method_version"] == manifest["method_version"]
    diagnostics = yaml.safe_load((output / "allocation_diagnostics.json").read_text(encoding="utf-8"))
    for key in (
        "budget_violation", "eligibility_violations", "support_violations",
        "state_compatibility_violations", "prerequisite_violations",
        "mutual_exclusion_violations", "max_actions_per_patient_violations",
    ):
        assert diagnostics[key] == 0


def test_dm77_integrated_causal_contrastive_v2_is_mixed_and_separate(tmp_path):
    config = yaml.safe_load(
        Path(
            "configs/prometheus/dm77_integrated_contrastive_v2_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    config["run"]["output_root"] = str(tmp_path)
    config["run"]["sample_size"] = 900
    output = run_global_prometheus(config)

    manifest = yaml.safe_load((output / "run_manifest.json").read_text(encoding="utf-8"))
    primary = "prometheus_global_causal_contrastive_v2"
    assert manifest["primary_ranking_method"] == primary
    assert manifest["contrastive_configuration"]["pair_scope"] == "balanced_mixed"
    assert manifest["contrastive_configuration"]["lambda_con"] == pytest.approx(0.01)
    assert manifest["contrastive_configuration"]["reliability_weighting"] is True
    assert manifest["contrastive_configuration"]["separate_contrastive_head"] is True
    assert manifest["oracle_used_for_ranking"] is False

    training = yaml.safe_load(
        (output / "prometheus_training_diagnostics.json").read_text(encoding="utf-8")
    )
    contrastive = training["best_contrastive_diagnostics"]
    assert training["training_mode"] == primary
    assert training["separate_contrastive_head"] is True
    assert contrastive["pair_scope"] == "balanced_mixed"
    assert contrastive["within_pairs"] > 0
    assert contrastive["cross_pairs"] > 0
    assert contrastive["reliability_weighting"] is True
    assert contrastive["positive_pairs"] > 0
    assert contrastive["negative_pairs"] > 0

    evaluation = pd.read_csv(output / "evaluation_ranked_actions.csv")
    assert f"score__{primary}" in evaluation
    assert "score__unified_global_ranker" in evaluation
    assert np.allclose(evaluation.raw_priority_score, evaluation[f"score__{primary}"])
    comparison = pd.read_csv(output / "baseline_comparison.csv").set_index("method")
    assert primary in comparison.index
    assert "unified_global_ranker" in comparison.index
