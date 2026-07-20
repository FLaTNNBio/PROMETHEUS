from __future__ import annotations

import pandas as pd

from causal_population_ranking.synthetic import (
    generate_synthetic_population,
    validate_synthetic_population,
)


def test_fully_synthetic_generator_is_deterministic_and_contains_no_real_record_fields():
    first = generate_synthetic_population(400, seed=151, history_months=24)
    second = generate_synthetic_population(400, seed=151, history_months=24)
    pd.testing.assert_frame_equal(first.patients, second.patients)
    pd.testing.assert_frame_equal(first.monthly_history, second.monthly_history)
    assert first.metadata == second.metadata
    assert first.privacy_audit == second.privacy_audit
    assert first.metadata["real_patient_records_used"] is False
    assert first.metadata["calibrated_to_italian_population"] is False
    assert first.privacy_audit["formal_privacy_guarantee_claimed"] is False
    forbidden = {
        "name", "first_name", "last_name", "address", "phone", "email", "ssn",
        "birthdate", "treatment", "treatment_level", "outcome", "observed_outcome",
    }
    assert not forbidden.intersection(first.patients.columns)
    assert not any(
        column.startswith(("true_", "oracle_", "potential_", "latent_rank"))
        for column in first.patients.columns
    )


def test_longitudinal_panel_reconstructs_every_12_month_snapshot_feature():
    result = generate_synthetic_population(300, seed=233, history_months=24)
    history = result.monthly_history
    patients = result.patients.set_index("patient_id")
    recent = history.loc[history.month_offset >= -11]
    for source, target in {
        "encounter_count": "encounter_count",
        "emergency_count": "prior_emergency",
        "inpatient_count": "prior_inpatient",
        "procedure_count": "procedure_count",
        "careplan_count": "careplan_count",
    }.items():
        aggregate = recent.groupby("patient_id")[source].sum().reindex(patients.index)
        pd.testing.assert_series_equal(
            aggregate.astype(float), patients[target], check_names=False
        )
    assert len(history) == 300 * 24
    assert not history[["patient_id", "month_offset"]].duplicated().any()
    assert (history.encounter_count >= history.emergency_count + history.inpatient_count).all()


def test_synthetic_validation_has_no_critical_failures_and_checks_dependencies():
    result = generate_synthetic_population(800, seed=307, history_months=24)
    checks, report = validate_synthetic_population(
        result.patients, result.monthly_history, history_months=24
    )
    assert report["critical_failures"] == 0
    assert report["italian_population_representativeness_established"] is False
    assert not checks.loc[checks.severity == "critical", "status"].eq("fail").any()
    dependency = checks.loc[checks.category == "dependency"]
    assert len(dependency) == 5
    assert dependency.status.eq("pass").all()


def test_functional_social_and_utilization_domains_are_non_degenerate():
    patients = generate_synthetic_population(600, seed=401, history_months=24).patients
    for column in (
        "frailty_index", "functional_limitation_score", "social_fragility_score",
        "condition_distinct", "encounter_count", "medication_distinct",
    ):
        assert patients[column].nunique() > 5
    assert patients.frailty_index.corr(patients.functional_limitation_score, method="spearman") > 0.30
    assert patients.condition_distinct.corr(patients.medication_distinct, method="spearman") > 0.20


def test_population_shift_scenarios_change_expected_domains_without_changing_schema():
    baseline = generate_synthetic_population(600, seed=509, history_months=24)
    older = generate_synthetic_population(
        600, seed=509, history_months=24, scenario="older_high_need"
    )
    social = generate_synthetic_population(
        600, seed=509, history_months=24, scenario="social_fragility_shift"
    )
    surge = generate_synthetic_population(
        600, seed=509, history_months=24, scenario="utilization_surge"
    )
    assert list(baseline.patients.columns) == list(older.patients.columns)
    assert list(baseline.patients.columns) == list(social.patients.columns)
    assert older.patients.age.mean() > baseline.patients.age.mean()
    assert social.patients.social_fragility_score.mean() > baseline.patients.social_fragility_score.mean()
    assert surge.patients.encounter_count.mean() > baseline.patients.encounter_count.mean()
