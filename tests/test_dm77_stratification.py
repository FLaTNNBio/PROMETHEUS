from __future__ import annotations

import numpy as np
import pandas as pd

from causal_population_ranking.dm77 import (
    assess_dm77_population,
    load_baseline_need_contract,
    load_dm77_settings,
    summarize_dm77_population,
)


def _settings():
    contract = load_baseline_need_contract()
    return load_dm77_settings(contract["modes"]["rules"]["settings"])


def _profiles() -> pd.DataFrame:
    base = {
        "age": 30,
        "condition_distinct": 0,
        "prior_inpatient": 0,
        "prior_emergency": 0,
        "medication_distinct": 0,
        "frailty_index": 0.0,
        "functional_limitation_score": 0.0,
        "social_fragility_score": 0.0,
        "cognitive_impairment": 0,
        "non_self_sufficiency": 0,
        "caregiver_available": 1,
        "housing_instability": 0,
        "palliative_need": 0,
    }
    overrides = {
        "level_1": {},
        "level_2": {"condition_distinct": 1, "medication_distinct": 1},
        "level_3": {"condition_distinct": 2, "medication_distinct": 2},
        "level_4": {
            "age": 85, "condition_distinct": 6, "prior_inpatient": 3,
            "prior_emergency": 2, "medication_distinct": 10,
            "frailty_index": 0.50, "functional_limitation_score": 0.50,
            "social_fragility_score": 0.70, "caregiver_available": 0,
        },
        "level_5": {
            "age": 88, "condition_distinct": 7, "prior_inpatient": 3,
            "medication_distinct": 11, "frailty_index": 0.85,
            "functional_limitation_score": 0.90, "non_self_sufficiency": 1,
        },
        "level_6": {"palliative_need": 1},
        "manual": {"functional_limitation_score": np.nan},
    }
    return pd.DataFrame([
        {"patient_id": patient_id, **base, **values}
        for patient_id, values in overrides.items()
    ])


def test_transparent_rules_cover_six_levels_and_abstain_on_missing_data():
    assessments, audit = assess_dm77_population(_profiles(), _settings())
    levels = assessments.set_index("patient_id").dm77_need_level
    assert [int(levels[f"level_{level}"]) for level in range(1, 7)] == list(range(1, 7))
    assert pd.isna(levels["manual"])
    manual = assessments.set_index("patient_id").loc["manual"]
    assert manual.dm77_assessment_status == "manual_review"
    assert "MISSING_INPUT:functional_limitation_score" in manual.dm77_reason_codes
    level_vi = assessments.set_index("patient_id").loc["level_6"]
    assert level_vi.dm77_protected_pathway
    assert level_vi.dm77_assessment_status == "protected_pathway"
    assert audit["protected_level_vi_patients"] == 1
    assert audit["manual_review_patients"] == 1


def test_treatment_outcome_oracle_and_causal_scores_are_ignored():
    frame = _profiles()
    first, first_audit = assess_dm77_population(frame, _settings())
    mutated = frame.assign(
        treatment_level=np.arange(len(frame)) % 6 + 1,
        observed_outcome=np.linspace(0, 365, len(frame)),
        true_benefit_1_to_2=np.linspace(-1000, 1000, len(frame)),
        potential_outcome_1=np.linspace(365, 0, len(frame)),
        raw_score=np.linspace(500, -500, len(frame)),
    )
    second, second_audit = assess_dm77_population(mutated, _settings())
    pd.testing.assert_frame_equal(first, second)
    assert first_audit["forbidden_columns_present_but_ignored"] == []
    assert {
        "treatment_level", "observed_outcome", "true_benefit_1_to_2",
        "potential_outcome_1", "raw_score",
    } <= set(second_audit["forbidden_columns_present_but_ignored"])
    assert second_audit["treatment_used"] is False
    assert second_audit["oracle_used"] is False


def test_summary_is_complete_and_preserves_population_total():
    assessments, _ = assess_dm77_population(_profiles(), _settings())
    summary = summarize_dm77_population(assessments)
    assert len(summary) == 7
    assert int(summary.patient_count.sum()) == len(assessments)
    assert np.isclose(float(summary.population_fraction.sum()), 1.0)


def test_dm77_configuration_explicitly_marks_research_profile():
    settings = load_baseline_need_contract()["modes"]["rules"]["settings"]
    assert settings["threshold_profile"] == "research_default_not_official"
    assert settings["missing_data_policy"] == "manual_review"
    assert settings["exclude_level_vi_from_standard_allocation"] is True
