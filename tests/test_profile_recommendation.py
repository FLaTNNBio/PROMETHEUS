from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_population_ranking.recommendation import (
    CALIBRATION_METHODS,
    calibrate_and_recommend_profiles,
)


PRIMARY = "global_rank_plus_contrastive"


def _contracts():
    calibration = {"candidate_methods": list(CALIBRATION_METHODS)}
    recommendation = {
        "candidate_policy": "same_or_higher_level_than_current_care_profile",
        "deintensification_supported_in_v1": False,
        "lateral_profile_change_supported": True,
        "selector": "argmax_calibrated_incremental_benefit",
        "primary_minimum_benefit_threshold_days": 2.0,
        "support_requirements": {
            "propensity_lower": 0.05,
            "propensity_upper": 0.95,
            "minimum_profile_effective_sample_size": 10,
            "require_patient_empirical_support": True,
            "require_score_inside_validation_calibration_range": True,
        },
        "tie_breakers": [
            "maximum_raw_priority_score", "lexicographic_care_profile_id"
        ],
        "unsupported_or_below_threshold_policy": "abstain_to_baseline_need_level",
    }
    implementation = {
        "minimum_validation_rows": 8,
        "monotonic_bins": 4,
        "profile_mean_shrinkage_strength": 4.0,
        "uncertainty_residual_quantile": 0.90,
        "minimum_profile_validation_rows": 4,
        "profile_evidence_one_sided_z": 2.576,
        "profile_evidence_null_days": 0.0,
    }
    return calibration, recommendation, implementation


def _inputs():
    validation_patients = [f"v{index}" for index in range(8)]
    test_patients = ["t_good", "t_out", "t_low", "t_same", "t_tie"]
    patients = validation_patients + test_patients
    split = {patient: "validation" for patient in validation_patients}
    split.update({patient: "test" for patient in test_patients})
    validation_score = np.linspace(-1.0, 1.0, len(validation_patients))
    special = {
        "t_good": (0.55, 0.25),
        "t_out": (1.50, 0.40),
        "t_low": (-0.85, -0.75),
        "t_same": (0.20, 0.65),
        "t_tie": (0.50, 0.50),
    }
    score_rows = []
    supervision_rows = []
    opportunity_rows = []
    for patient_index, patient in enumerate(patients):
        for profile_index, profile in enumerate(("profile_a", "profile_b")):
            if patient in validation_patients:
                raw = float(validation_score[patient_index] + 0.05 * profile_index)
            else:
                raw = float(special[patient][profile_index])
            score_rows.append({
                "patient_id": patient,
                "care_profile_id": profile,
                "split": split[patient],
                "method": PRIMARY,
                "raw_priority_score": raw,
                "oracle_used": False,
            })
            supervision_rows.append({
                "patient_id": patient,
                "care_profile_id": profile,
                "split": split[patient],
                "dr_pseudo_outcome": 3.0 + 4.0 * raw,
                "dr_reliability_weight": 1.0 + 0.1 * profile_index,
                "causal_supervision_status": "supported",
                "observed_outcome": 100.0 + patient_index,
            })
            opportunity_rows.append({
                "patient_id": patient,
                "care_profile_id": profile,
                "care_profile_index": profile_index,
                "care_profile_level": 3 + profile_index,
                "current_care_profile_level": 4 if patient == "t_same" else 2,
                "eligibility": True,
                "discretionary_rank_candidate": True,
                "empirical_support": True,
                "incremental_resource_cost": 1.0 + profile_index,
                "incremental_capacity_requirements": "{}",
            })
    baseline = pd.DataFrame({
        "patient_id": patients,
        "baseline_need_level": [2] * len(patients),
    })
    current = pd.DataFrame({
        "patient_id": patients,
        "current_care_profile": ["profile_current"] * len(patients),
        "current_care_profile_level": [
            4 if patient == "t_same" else 2 for patient in patients
        ],
    })
    splits = pd.DataFrame({
        "patient_id": patients,
        "split": [split[patient] for patient in patients],
    })
    support = pd.DataFrame([
        {
            "care_profile_id": profile,
            "split": partition,
            "profile_empirical_support": True,
            "profile_effective_sample_size": 50.0,
        }
        for profile in ("profile_a", "profile_b")
        for partition in ("nuisance_train", "rank_train", "validation", "test")
    ])
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(supervision_rows),
        pd.DataFrame(opportunity_rows),
        baseline,
        current,
        splits,
        support,
    )


def _run(inputs=None):
    if inputs is None:
        inputs = _inputs()
    calibration, recommendation, implementation = _contracts()
    return calibrate_and_recommend_profiles(
        *inputs,
        primary_variant=PRIMARY,
        calibration_contract=calibration,
        recommendation_contract=recommendation,
        implementation_settings=implementation,
    )


def test_calibration_uses_only_validation_and_preserves_raw_order():
    inputs = _inputs()
    original_scores = inputs[0].copy(deep=True)
    first = _run(inputs)
    mutated = list(_inputs())
    test = mutated[1].split.eq("test")
    mutated[1].loc[test, "dr_pseudo_outcome"] = np.linspace(-1e6, 1e6, test.sum())
    mutated[1].loc[test, "observed_outcome"] = -999999.0
    second = _run(tuple(mutated))

    pd.testing.assert_frame_equal(inputs[0], original_scores)
    pd.testing.assert_frame_equal(
        first.calibration_diagnostics, second.calibration_diagnostics
    )
    pd.testing.assert_frame_equal(first.recommendations, second.recommendations)
    assert first.audit["calibration_fit_partition"] == "validation"
    assert first.audit["test_dr_used_for_calibration_or_selection"] is False
    assert first.audit["raw_priority_score_modified"] is False
    assert first.audit["raw_to_calibrated_ordering_violations"] == 0
    finite = first.calibrated_opportunities.dropna(
        subset=["calibrated_incremental_benefit"]
    ).sort_values("raw_priority_score")
    assert np.all(np.diff(finite.calibrated_incremental_benefit) >= -1e-10)
    assert first.calibration_diagnostics.selected.sum() == 1


def test_recommendation_applies_threshold_range_level_and_tie_rules():
    result = _run()
    recommendation = result.recommendations.set_index("patient_id")

    assert recommendation.loc["t_good", "recommended_profile_id"] == "profile_a"
    assert recommendation.loc["t_out", "recommendation_abstained"]
    assert recommendation.loc["t_out", "recommendation_reason"] == (
        "TOP_SCORE_OUTSIDE_VALIDATION_RANGE"
    )
    assert recommendation.loc["t_low", "recommendation_abstained"]
    assert recommendation.loc["t_low", "recommended_actionable_level"] == 2
    assert recommendation.loc["t_same", "recommended_profile_id"] == "profile_b"
    assert recommendation.loc["t_same", "recommended_actionable_level"] == 4
    assert recommendation.loc["t_tie", "recommended_profile_id"] == "profile_a"
    accepted = ~result.recommendations.recommendation_abstained
    assert (
        result.recommendations.loc[accepted, "calibrated_incremental_benefit"] > 2.0
    ).all()
    assert result.audit["deintensification_recommendations"] == 0
    assert result.audit["raw_score_has_causal_zero"] is False
    assert result.audit["calibrated_output_is_individual_cate"] is False


def test_capacity_is_ignored_and_oracle_inputs_are_rejected():
    inputs = list(_inputs())
    first = _run(tuple(inputs))
    inputs[2]["incremental_resource_cost"] = 1e12
    inputs[2]["incremental_capacity_requirements"] = "changed"
    second = _run(tuple(inputs))
    pd.testing.assert_frame_equal(first.recommendations, second.recommendations)
    assert first.audit["capacity_inputs_used"] is False

    leaked = list(_inputs())
    leaked[2]["true_profile_benefit"] = 999.0
    with pytest.raises(ValueError, match="Oracle columns"):
        _run(tuple(leaked))

    flagged = list(_inputs())
    flagged[0].loc[0, "oracle_used"] = True
    with pytest.raises(ValueError, match="marked as oracle-used"):
        _run(tuple(flagged))


def test_profile_validation_evidence_guardrail_blocks_null_profiles():
    inputs = list(_inputs())
    validation = inputs[1].split.eq("validation")
    inputs[1].loc[validation, "dr_pseudo_outcome"] = 0.0
    result = _run(tuple(inputs))
    assert result.recommendations.recommendation_abstained.all()
    assert set(result.recommendations.recommendation_reason) == {
        "PROFILE_VALIDATION_EVIDENCE_NOT_POSITIVE"
    }
    assert not any(
        row["profile_positive_validation_evidence"]
        for row in result.audit["profile_validation_evidence"]
    )
    assert result.audit["profile_validation_evidence_oracle_used"] is False


def test_missing_ranker_scores_produces_explicit_baseline_fallback():
    inputs = list(_inputs())
    inputs[0] = inputs[0].iloc[0:0].copy()
    result = _run(tuple(inputs))
    assert result.audit["status"] == "calibration_unavailable_all_patients_abstained"
    assert result.recommendations.recommendation_abstained.all()
    assert result.recommendations.recommended_profile_id.isna().all()
    assert (
        result.recommendations.recommended_actionable_level
        == result.recommendations.baseline_need_level
    ).all()
    assert set(result.recommendations.recommendation_reason) == {
        "CALIBRATION_UNAVAILABLE"
    }
