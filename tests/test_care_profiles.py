"""Tests for explicit care-profile structures and governance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from causal_population_ranking.dm77 import (
    CARE_PROFILE,
    apply_profile_bundle,
    build_patient_profile_opportunities,
    derive_care_profile_eligibility,
    load_care_profile_catalog,
    require_comparable_profile_pair,
    resolve_current_care_profiles,
    resolve_decision_unit,
    validate_ranked_profile_comparability,
)


PROFILE_PATH = "configs/care_catalog.yaml"
ACTION_PATH = PROFILE_PATH


def _document() -> dict:
    return yaml.safe_load(Path(PROFILE_PATH).read_text(encoding="utf-8"))


def _patient(patient_id: str, **updates) -> dict:
    row = {
        "patient_id": patient_id,
        "condition_distinct": 3.0,
        "functional_limitation_score": 0.4,
        "social_fragility_score": 0.2,
        "housing_instability": 0.0,
        "palliative_need": 0.0,
        "urgent_case": False,
        "mandatory_care": False,
    }
    row.update(updates)
    return row


def test_unified_care_catalog_keeps_profile_and_action_ids_distinct():
    catalog = load_care_profile_catalog(PROFILE_PATH, ACTION_PATH)
    validation = validate_ranked_profile_comparability(catalog)
    assert catalog.catalog_version == "prometheus_dm77_care_profiles_v1"
    assert len(catalog.profiles) == 8
    assert len(catalog.automatic_rank_profiles) == 6
    assert catalog.maintenance_profile.care_profile_level == 1
    assert set(catalog.profile_ids).isdisjoint(catalog.action_catalog.action_ids)
    assert all(profile.care_profile_id.startswith("profile_") for profile in catalog.profiles)
    assert any(len(profile.included_care_actions) == 1 for profile in catalog.profiles)
    assert all(
        profile.care_profile_id not in catalog.action_catalog.action_ids
        for profile in catalog.profiles if len(profile.included_care_actions) == 1
    )
    assert set(validation["ranking_domains"]) == {
        "standard_hospital_free_days_365d", "protected_palliative_90d",
    }
    assert validation["cross_domain_pairs_allowed"] is False


def test_profile_catalog_rejects_unknown_actions_bad_bundles_and_bad_prerequisites():
    unknown = _document()
    unknown["profiles"][1]["included_care_actions"] = ["unknown_action"]
    unknown["profiles"][1]["action_application_order"] = ["unknown_action"]
    with pytest.raises(ValueError, match="unknown actions"):
        load_care_profile_catalog(unknown, ACTION_PATH)

    order = _document()
    reversed_order = list(reversed(order["profiles"][4]["action_application_order"]))
    order["profiles"][4]["included_care_actions"] = reversed_order
    order["profiles"][4]["action_application_order"] = reversed_order
    with pytest.raises(ValueError, match="cannot apply"):
        load_care_profile_catalog(order, ACTION_PATH)

    prerequisite = _document()
    prerequisite["profiles"][1]["clinical_prerequisites"] = ["future_unknown_input"]
    with pytest.raises(ValueError, match="unknown prerequisites"):
        load_care_profile_catalog(prerequisite, ACTION_PATH)

    outcome = _document()
    outcome["profiles"][1]["outcome_horizon"] = 90
    with pytest.raises(ValueError, match="incompatible outcomes"):
        load_care_profile_catalog(outcome, ACTION_PATH)


def test_current_profile_resolution_abstains_on_ambiguous_legacy_state():
    catalog = load_care_profile_catalog(PROFILE_PATH, ACTION_PATH)
    states = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "current_care_state": [
            "remote_multidisciplinary_management", "routine_primary_care",
        ],
    })
    unresolved = resolve_current_care_profiles(states, catalog)
    assert unresolved.loc[0, "current_care_profile_manual_review"]
    assert pd.isna(unresolved.loc[0, "current_care_profile"])
    assert "AMBIGUOUS_LEGACY_STATE" in unresolved.loc[
        0, "current_care_profile_reason_codes"
    ]
    assert unresolved.loc[1, "current_care_profile"] == "profile_i_routine_primary_care"

    explicit = pd.DataFrame({
        "patient_id": ["p1"],
        "current_care_profile": ["profile_iv_remote_monitoring"],
    })
    resolved = resolve_current_care_profiles(states, catalog, explicit)
    assert resolved.loc[0, "current_care_profile"] == "profile_iv_remote_monitoring"
    assert resolved.loc[0, "current_care_profile_status"] == "resolved_explicit_profile"


def test_incremental_bundle_resources_charge_only_missing_components():
    catalog = load_care_profile_catalog(PROFILE_PATH, ACTION_PATH)
    patient = pd.Series(_patient("p1"))
    from_routine = apply_profile_bundle(
        patient,
        "profile_i_routine_primary_care",
        "profile_iv_remote_monitoring",
        catalog,
    )
    assert from_routine.feasible
    assert from_routine.missing_action_ids == (
        "start_chronic_disease_management", "add_remote_monitoring",
    )
    assert from_routine.incremental_cost == pytest.approx(4.0)
    assert from_routine.incremental_capacity_requirements == {
        "chronic_management_capacity": 1.0,
        "remote_monitoring_capacity": 1.0,
    }

    from_chronic = apply_profile_bundle(
        patient,
        "profile_iii_chronic_management",
        "profile_iv_remote_monitoring",
        catalog,
    )
    assert from_chronic.feasible
    assert from_chronic.missing_action_ids == ("add_remote_monitoring",)
    assert from_chronic.incremental_cost == pytest.approx(2.5)
    assert from_chronic.incremental_capacity_requirements == {
        "remote_monitoring_capacity": 1.0,
    }
    lateral = apply_profile_bundle(
        patient,
        "profile_iv_remote_monitoring",
        "profile_iv_multidisciplinary_coordination",
        catalog,
    )
    assert lateral.feasible
    assert lateral.missing_action_ids == ("add_multidisciplinary_coordination",)
    assert lateral.deactivated_action_ids == ("add_remote_monitoring",)
    assert lateral.incremental_cost == pytest.approx(2.0)
    assert lateral.incremental_capacity_requirements == {
        "multidisciplinary_capacity": 1.0,
    }


def _eligibility_inputs(catalog):
    patients = pd.DataFrame([
        _patient("standard"),
        _patient("protected", palliative_need=1.0),
        _patient("ambiguous"),
        _patient("urgent", urgent_case=True),
    ])
    assessments = pd.DataFrame({
        "patient_id": patients.patient_id,
        "baseline_need_level": pd.array([3, 6, 3, 3], dtype="Int64"),
        "baseline_need_status": ["assessed", "protected_pathway", "assessed", "assessed"],
        "baseline_need_protected_pathway": [False, True, False, False],
    })
    states = pd.DataFrame({
        "patient_id": patients.patient_id,
        "current_care_state": [
            "routine_primary_care", "routine_primary_care",
            "remote_multidisciplinary_management", "routine_primary_care",
        ],
    })
    current = resolve_current_care_profiles(states, catalog)
    return patients, assessments, current


def test_profile_eligibility_is_keyed_governed_and_invariant_to_forbidden_fields():
    catalog = load_care_profile_catalog(PROFILE_PATH, ACTION_PATH)
    patients, assessments, current = _eligibility_inputs(catalog)
    clean, clean_audit = derive_care_profile_eligibility(
        patients, assessments, current, catalog
    )
    contaminated = patients.copy()
    contaminated["true_profile_benefit"] = 999.0
    contaminated["raw_priority_score"] = -999.0
    contaminated["observed_outcome"] = 365.0
    contaminated["recommended_profile_id"] = "profile_v_integrated_home_support"
    contaminated["allocated_profile_id"] = "profile_i_routine_primary_care"
    changed, changed_audit = derive_care_profile_eligibility(
        contaminated, assessments, current, catalog
    )
    pd.testing.assert_frame_equal(clean, changed)
    assert len(clean) == len(patients) * len(catalog.profiles)
    assert not clean[["patient_id", "care_profile_id"]].duplicated().any()
    assert clean_audit["oracle_used"] is False
    assert clean_audit["implicit_action_to_profile_conversion"] is False
    assert {
        "true_profile_benefit", "raw_priority_score", "observed_outcome",
        "recommended_profile_id", "allocated_profile_id",
    } <= set(changed_audit["forbidden_columns_present_but_ignored"])

    protected = clean.loc[
        (clean.patient_id == "protected")
        & (clean.care_profile_id == "profile_vi_protected_palliative")
    ].iloc[0]
    assert protected.eligibility
    assert not protected.discretionary_rank_candidate
    assert clean.loc[clean.patient_id == "ambiguous", "eligibility"].eq(False).all()
    urgent = clean.loc[clean.patient_id == "urgent"]
    assert urgent.eligibility.any()
    assert not urgent.discretionary_rank_candidate.any()


def test_profile_opportunities_require_explicit_keys_and_support_gate():
    catalog = load_care_profile_catalog(PROFILE_PATH, ACTION_PATH)
    eligibility, _ = derive_care_profile_eligibility(
        *_eligibility_inputs(catalog), catalog
    )
    structural = build_patient_profile_opportunities(
        eligibility, require_empirical_support=False
    )
    supported = build_patient_profile_opportunities(
        eligibility, require_empirical_support=True
    )
    assert len(structural) > 0
    assert supported.empty
    assert structural.opportunity_status.eq(
        "structurally_eligible_support_pending"
    ).all()
    assert set(structural.care_profile_id).isdisjoint(catalog.action_catalog.action_ids)

    duplicated = pd.concat([eligibility, eligibility.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate patient-profile"):
        build_patient_profile_opportunities(duplicated, require_empirical_support=False)


def test_profile_pair_gate_blocks_incompatible_domains():
    catalog = load_care_profile_catalog(PROFILE_PATH, ACTION_PATH)
    require_comparable_profile_pair(
        catalog,
        "profile_ii_structured_followup",
        "profile_iv_remote_monitoring",
    )
    with pytest.raises(ValueError, match="incompatible outcome"):
        require_comparable_profile_pair(
            catalog,
            "profile_ii_structured_followup",
            "profile_vi_protected_palliative",
        )


def test_decision_unit_selector_requires_the_explicit_profile_contract():
    profile = {
        "decision_unit": "care_profile",
        "opportunity_source": "care_profile_catalog",
        "care": {"profile_catalog": PROFILE_PATH},
    }
    assert resolve_decision_unit(profile).decision_unit == CARE_PROFILE
    with pytest.raises(ValueError, match="explicit decision_unit"):
        resolve_decision_unit({"care": {"profile_catalog": PROFILE_PATH}})


def test_pipeline_config_declares_profile_structure_without_legacy_fallback():
    config = yaml.safe_load(Path(
        "configs/pipeline.yaml"
    ).read_text(encoding="utf-8"))
    assert resolve_decision_unit(config).decision_unit == CARE_PROFILE
    assert config["gates"]["implicit_action_to_profile_conversion_allowed"] is False
    assert config["gates"]["cross_domain_pairs_allowed"] is False
    plan = Path("docs/prometheus_causal_stratification_plan.md").read_text(
        encoding="utf-8"
    )
    assert "New Phase 2 | Complete" in plan
    assert "New Phase 3 | Complete" in plan
    assert "New Phase 4 | Complete" in plan
    assert "New Phase 5 | Complete" in plan
    assert "New Phase 6 | Complete" in plan
    assert "New Phase 8 | Complete" in plan
    assert "New Phase 9 | Complete" in plan
    assert "one pipeline command" in plan
