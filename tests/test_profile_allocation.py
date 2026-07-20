from __future__ import annotations

import pandas as pd
import pytest

from causal_population_ranking.allocation import allocate_recommended_profiles
from causal_population_ranking.evaluation import evaluate_profile_allocation


def _settings():
    return {
        "method": "cardinal_milp_over_frozen_recommendations_v1",
        "shared_budget_units_per_patient": 0.8,
        "tie_break_epsilon": 1e-9,
        "budget_curve_multipliers": [0.5, 1.0],
        "fixed_capacity_fractions": [0.4],
        "solver": {
            "presolve": True,
            "mip_relative_gap": 0.0,
            "time_limit_seconds": 5.0,
        },
    }


def _contract():
    return {
        "candidate_profiles": "recommended_profile_only",
        "alternative_profile_substitution_allowed": False,
        "resource_accounting": "missing_profile_components_relative_to_current_care",
    }


def _inputs():
    patients = ["p1", "p2", "p3", "p4", "p5"]
    profile = ["profile_a", "profile_a", "profile_b", "profile_b", None]
    benefit = [10.0, 9.0, 8.0, 7.0, None]
    raw = [0.90, 0.80, 0.95, 0.70, None]
    recommendations = pd.DataFrame({
        "patient_id": patients,
        "split": ["test"] * 5,
        "baseline_need_level": [2] * 5,
        "current_care_profile": ["profile_current"] * 5,
        "current_care_profile_level": [1] * 5,
        "recommended_actionable_level": [3, 3, 4, 4, 2],
        "recommended_profile_id": profile,
        "recommended_raw_priority_score": raw,
        "calibrated_incremental_benefit": benefit,
        "recommendation_abstained": [False, False, False, False, True],
        "capacity_inputs_used": [False] * 5,
        "oracle_used": [False] * 5,
    })
    rows = []
    for patient, profile_id, level in zip(patients[:4], profile[:4], [3, 3, 4, 4]):
        requirements = {"pool_a": 1.0} if profile_id == "profile_a" else {
            "pool_b": 1.0
        }
        action = "add_a" if profile_id == "profile_a" else "add_b"
        rows.append({
            "patient_id": patient,
            "care_profile_id": profile_id,
            "care_profile_level": level,
            "eligibility": True,
            "discretionary_rank_candidate": True,
            "empirical_support": True,
            "protected_pathway": False,
            "protected_profile": False,
            "mandatory_care": False,
            "eligibility_reasons": "ELIGIBLE_BY_PROFILE_CATALOG",
            "missing_component_actions": action,
            "deactivated_component_actions": "",
            "incremental_capacity_requirements": requirements,
            "scenario_incremental_resource_cost": 2.0,
        })
    opportunities = pd.DataFrame(rows)
    capacities = pd.DataFrame({
        "capacity_pool": ["pool_a", "pool_b"],
        "available_units": [1.0, 2.0],
    })
    resources = pd.DataFrame({
        "care_profile_id": ["profile_a", "profile_b"],
        "primary_capacity_pool": ["pool_a", "pool_b"],
    })
    return recommendations, opportunities, capacities, resources


def _allocate(inputs=None, settings=None):
    if inputs is None:
        inputs = _inputs()
    return allocate_recommended_profiles(
        *inputs,
        allocation_contract=_contract(),
        settings=_settings() if settings is None else settings,
        tie_seed=211,
    )


def test_cardinal_allocator_only_allocates_the_frozen_recommended_profile():
    result = _allocate()
    decisions = result.decisions.set_index("patient_id")
    allocated = result.decisions.allocated_profile_id.notna()

    assert set(result.decisions.loc[allocated, "patient_id"]) == {"p1", "p3"}
    assert decisions.loc["p1", "allocated_profile_id"] == "profile_a"
    assert decisions.loc["p3", "allocated_profile_id"] == "profile_b"
    assert decisions.loc["p2", "deferred_recommendation"]
    assert decisions.loc["p2", "allocated_care_level"] == 1
    assert decisions.loc["p5", "allocation_status"] == "no_actionable_recommendation"
    assert decisions.loc["p5", "deferred_recommendation"] == False
    assert result.audit["shared_budget_used"] == pytest.approx(4.0)
    assert result.audit["constraint_violations_total"] == 0
    assert result.audit["alternative_profile_substitution"] is False
    assert result.audit["existing_components_charged_again"] is False
    assert result.audit["component_capacity_double_counting"] is False


def test_lower_capacity_only_defers_and_never_changes_the_recommendation():
    inputs = list(_inputs())
    normal = _allocate(tuple(inputs))
    scarce_inputs = list(_inputs())
    scarce_inputs[2]["available_units"] = [1.0, 0.0]
    scarce = _allocate(tuple(scarce_inputs))
    columns = [
        "patient_id", "recommended_actionable_level", "recommended_profile_id",
        "recommendation_abstained", "calibrated_incremental_benefit",
    ]
    pd.testing.assert_frame_equal(
        normal.decisions[columns], scarce.decisions[columns]
    )
    assert scarce.audit["deferred_recommendations"] > normal.audit[
        "deferred_recommendations"
    ]
    assert scarce.audit["constraint_violations_total"] == 0


def test_ordinal_diagnostic_ignores_calibration_and_monetary_cost():
    inputs = list(_inputs())
    first = _allocate(tuple(inputs))
    changed = list(_inputs())
    changed[0].loc[:3, "calibrated_incremental_benefit"] = [3.0, 100.0, 4.0, 90.0]
    changed[1].loc[:, "scenario_incremental_resource_cost"] = [50.0, 1.0, 40.0, 1.0]
    second = _allocate(tuple(changed))

    def ordinal(result):
        frame = result.diagnostic_selections
        return frame.loc[
            frame.diagnostic_type.eq("fixed_count_ordinal"),
            ["setting", "patient_id", "care_profile_id"],
        ].reset_index(drop=True)

    pd.testing.assert_frame_equal(ordinal(first), ordinal(second))
    diagnostic = first.diagnostics.loc[
        first.diagnostics.diagnostic_type.eq("fixed_count_ordinal")
    ].iloc[0]
    assert diagnostic.uses_calibration == False
    assert diagnostic.uses_monetary_cost == False


def test_allocation_is_seeded_reproducible_and_rejects_oracle_columns():
    first = _allocate()
    second = _allocate()
    pd.testing.assert_frame_equal(first.decisions, second.decisions)
    pd.testing.assert_frame_equal(first.diagnostics, second.diagnostics)

    leaked = list(_inputs())
    leaked[0]["true_profile_benefit"] = 99.0
    with pytest.raises(ValueError, match="Oracle columns"):
        _allocate(tuple(leaked))


def test_allocation_metrics_open_truth_only_after_the_decision_is_frozen():
    result = _allocate()
    truth = pd.DataFrame([
        {
            "patient_id": row.patient_id,
            "care_profile_id": row.care_profile_id,
            "true_profile_benefit": benefit,
            "evaluation_only": True,
        }
        for row, benefit in zip(
            result.allocation_candidates.itertuples(index=False),
            [6.0, 5.0, 12.0, 1.0],
        )
    ])
    metrics = evaluate_profile_allocation(
        result.decisions,
        result.allocation_candidates,
        result.diagnostic_selections,
        truth,
        shared_budget_limit=result.audit["shared_budget_limit"],
        pool_capacities=result.audit["pool_capacities"],
        profile_capacities=result.audit["profile_capacities"],
        tie_seed=result.audit["tie_seed"],
        tie_break_epsilon=result.audit["tie_break_epsilon"],
        solver_settings=result.audit["solver_settings"],
    ).set_index("metric")
    assert metrics.loc["deferred_recommendations", "value"] == 2.0
    assert metrics.loc["allocation_regret_days", "value"] >= 0.0
    assert metrics.loc["recommendation_preservation_rate", "value"] == 1.0
    assert metrics.oracle_evaluation_only.all()
