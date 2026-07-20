from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_population_ranking.causal_utilization.global_simulation import simulate_multivalued_care
from causal_population_ranking.dm77 import (
    assess_dm77_population,
    derive_intervention_eligibility,
    load_dm77_settings,
    summarize_dm77_population,
)


def _settings():
    return load_dm77_settings({"config_path": "configs/dm77/default.yaml"})


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


def test_catalog_routes_level_vi_only_to_protected_palliative_pathway():
    assessments, _ = assess_dm77_population(_profiles(), _settings())
    eligibility = derive_intervention_eligibility(
        assessments, "configs/dm77/intervention_catalog.yaml"
    )
    level_vi = eligibility.loc[eligibility.patient_id == "level_6"]
    selected = level_vi.loc[level_vi.catalog_eligible, "intervention_id"].tolist()
    assert selected == ["palliative_care_pathway"]
    assert level_vi.loc[level_vi.catalog_eligible, "protected_pathway"].all()
    manual = eligibility.loc[eligibility.patient_id == "manual"]
    assert not manual.catalog_eligible.any()
    assert manual.requires_professional_review.all()
    assert not eligibility.causal_effectiveness_established.any()


def test_summary_is_complete_and_preserves_population_total():
    assessments, _ = assess_dm77_population(_profiles(), _settings())
    summary = summarize_dm77_population(assessments)
    assert len(summary) == 7
    assert int(summary.patient_count.sum()) == len(assessments)
    assert np.isclose(float(summary.population_fraction.sum()), 1.0)


def test_semisynthetic_dm77_covariates_are_pre_index_but_derived_level_is_not_a_feature():
    rng = np.random.default_rng(31)
    cohort = pd.DataFrame({
        "patient_id": [f"p{index:04d}" for index in range(250)],
        "age": rng.uniform(18, 92, 250),
        "condition_distinct": rng.poisson(3, 250),
        "prior_inpatient": rng.poisson(0.8, 250),
        "prior_emergency": rng.poisson(1.0, 250),
        "medication_distinct": rng.poisson(4, 250),
        "recent_utilization_trend": rng.normal(size=250),
    })
    first = simulate_multivalued_care(cohort, seed=71)
    second = simulate_multivalued_care(cohort, seed=71)
    generated = {
        "frailty_index", "functional_limitation_score", "social_fragility_score",
        "cognitive_impairment", "non_self_sufficiency", "caregiver_available",
        "housing_instability", "palliative_need",
    }
    assert generated <= set(first.feature_columns)
    assert "dm77_need_level" not in first.feature_columns
    assert not any(column.startswith("dm77_") for column in first.feature_columns)
    pd.testing.assert_frame_equal(first.learner, second.learner)


def test_dm77_configuration_explicitly_marks_research_profile():
    raw = yaml.safe_load(Path("configs/dm77/default.yaml").read_text(encoding="utf-8"))
    assert raw["threshold_profile"] == "research_default_not_official"
    assert raw["missing_data_policy"] == "manual_review"
    assert raw["exclude_level_vi_from_standard_allocation"] is True
