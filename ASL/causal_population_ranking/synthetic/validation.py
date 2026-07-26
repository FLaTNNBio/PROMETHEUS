"""Structural, temporal, and plausibility checks for fully synthetic populations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.validation import ORACLE_PREFIXES


def _correlation(left, right) -> float:
    value = float(pd.Series(left).corr(pd.Series(right), method="spearman"))
    return value if np.isfinite(value) else 0.0


def validate_synthetic_population(
    patients: pd.DataFrame,
    monthly_history: pd.DataFrame,
    history_months: int,
) -> tuple[pd.DataFrame, dict]:
    """Validate identities as failures and broad plausibility as warnings."""

    checks: list[dict] = []

    def record(name: str, category: str, passed: bool, observed, expectation: str, severity: str) -> None:
        checks.append({
            "check": name,
            "category": category,
            "status": "pass" if passed else ("fail" if severity == "critical" else "warning"),
            "severity": severity,
            "observed": observed,
            "expectation": expectation,
        })

    numeric = patients.select_dtypes(include="number")
    forbidden = [column for column in patients if column.startswith(ORACLE_PREFIXES)]
    pii = sorted(set(patients).intersection({
        "name", "first_name", "last_name", "address", "phone", "email", "ssn", "birthdate",
    }))
    record("population_size", "structure", len(patients) >= 100, int(len(patients)), ">= 100", "critical")
    record("unique_patient_id", "structure", not patients.patient_id.astype(str).duplicated().any(), int(patients.patient_id.nunique()), "one unique id per row", "critical")
    record("synthetic_identifier_namespace", "privacy", patients.patient_id.astype(str).str.startswith("syn_").all(), float(patients.patient_id.astype(str).str.startswith("syn_").mean()), "all ids start with syn_", "critical")
    record("no_direct_identifiers", "privacy", not pii, "|".join(pii), "no direct identifier fields", "critical")
    record("no_oracle_source_columns", "leakage", not forbidden, "|".join(forbidden), "no oracle/evaluation fields", "critical")
    record("complete_patient_snapshot", "structure", int(patients.isna().sum().sum()) == 0, int(patients.isna().sum().sum()), "0 missing cells", "critical")
    finite = bool(np.isfinite(numeric.to_numpy(float)).all())
    record("finite_numeric_snapshot", "structure", finite, finite, "all numeric values finite", "critical")
    expected_monthly_rows = len(patients) * int(history_months)
    record("complete_monthly_panel", "temporal", len(monthly_history) == expected_monthly_rows, int(len(monthly_history)), f"{expected_monthly_rows} rows", "critical")
    unique_months = not monthly_history[["patient_id", "month_offset"]].duplicated().any()
    record("unique_patient_month", "temporal", unique_months, unique_months, "one row per patient-month", "critical")
    month_counts = monthly_history.groupby("patient_id").month_offset.nunique()
    complete_months = bool((month_counts == history_months).all())
    record("same_history_length", "temporal", complete_months, int(month_counts.min()), f"{history_months} months for every patient", "critical")

    bounded_columns = (
        "deprivation_index", "health_literacy_score", "frailty_index",
        "functional_limitation_score", "social_fragility_score",
    )
    bounded = all(patients[column].between(0, 1).all() for column in bounded_columns)
    record("bounded_continuous_domains", "structure", bounded, bounded, "all bounded scores in [0,1]", "critical")
    count_columns = (
        "condition_distinct", "prior_inpatient", "prior_emergency", "medication_distinct",
        "encounter_count", "procedure_count", "careplan_count",
    )
    nonnegative = all((patients[column] >= 0).all() for column in count_columns)
    record("nonnegative_counts", "structure", nonnegative, nonnegative, "all count features >= 0", "critical")
    binary_columns = (
        "female", "rurality", "smoking", "diabetes", "cardiovascular_disease", "copd",
        "chronic_kidney_disease", "cancer_history", "mental_health_condition", "chronic_pain",
        "multimorbidity", "polypharmacy", "cognitive_impairment", "non_self_sufficiency",
        "caregiver_available", "housing_instability", "palliative_need",
    )
    binary = all(set(patients[column].unique()).issubset({0.0, 1.0}) for column in binary_columns)
    record("binary_indicators", "structure", binary, binary, "binary fields contain only 0/1", "critical")

    recent = monthly_history.loc[monthly_history.month_offset >= -11]
    aggregation_map = {
        "encounter_count": "encounter_count",
        "emergency_count": "prior_emergency",
        "inpatient_count": "prior_inpatient",
        "procedure_count": "procedure_count",
        "careplan_count": "careplan_count",
    }
    patient_index = patients.set_index("patient_id")
    for source, target in aggregation_map.items():
        observed = recent.groupby("patient_id")[source].sum().reindex(patient_index.index).to_numpy(float)
        expected = patient_index[target].to_numpy(float)
        exact = bool(np.array_equal(observed, expected))
        record(f"exact_12m_aggregation_{target}", "temporal", exact, float(np.max(np.abs(observed - expected))), "maximum absolute difference = 0", "critical")
    last_six = monthly_history.loc[monthly_history.month_offset >= -5].groupby("patient_id").encounter_count.mean()
    prior_six = monthly_history.loc[monthly_history.month_offset.between(-11, -6)].groupby("patient_id").encounter_count.mean()
    reconstructed_trend = (last_six - prior_six).reindex(patient_index.index).to_numpy(float)
    trend_error = float(np.max(np.abs(reconstructed_trend - patient_index.recent_utilization_trend.to_numpy(float))))
    record("exact_utilization_trend", "temporal", trend_error < 1e-12, trend_error, "maximum absolute difference < 1e-12", "critical")

    plausibility = (
        ("mean_age", float(patients.age.mean()), 35.0, 75.0),
        ("condition_free_fraction", float((patients.condition_distinct == 0).mean()), 0.03, 0.60),
        ("medication_free_fraction", float((patients.medication_distinct == 0).mean()), 0.02, 0.80),
        (
            "no_recorded_clinical_burden_fraction",
            float((
                (patients.condition_distinct == 0)
                & (patients.medication_distinct == 0)
                & (patients.prior_inpatient == 0)
                & (patients.prior_emergency == 0)
            ).mean()),
            0.01,
            0.50,
        ),
        ("multimorbidity_fraction", float(patients.multimorbidity.mean()), 0.05, 0.80),
        ("polypharmacy_fraction", float(patients.polypharmacy.mean()), 0.02, 0.65),
        ("prior_inpatient_fraction", float((patients.prior_inpatient > 0).mean()), 0.005, 0.65),
        ("palliative_need_fraction", float(patients.palliative_need.mean()), 0.001, 0.15),
        ("non_self_sufficiency_fraction", float(patients.non_self_sufficiency.mean()), 0.001, 0.35),
    )
    for name, observed, lower, upper in plausibility:
        record(name, "broad_plausibility", lower <= observed <= upper, observed, f"broad methodological range [{lower}, {upper}]", "advisory")

    dependencies = (
        ("age_condition_dependency", _correlation(patients.age, patients.condition_distinct), 0.10, "positive"),
        ("condition_medication_dependency", _correlation(patients.condition_distinct, patients.medication_distinct), 0.20, "positive"),
        ("frailty_function_dependency", _correlation(patients.frailty_index, patients.functional_limitation_score), 0.30, "positive"),
        ("clinical_utilization_dependency", _correlation(patients.condition_distinct, patients.encounter_count), 0.10, "positive"),
        ("social_caregiver_dependency", _correlation(patients.social_fragility_score, patients.caregiver_available), -0.05, "negative"),
    )
    for name, observed, threshold, direction in dependencies:
        passed = observed >= threshold if direction == "positive" else observed <= threshold
        record(name, "dependency", passed, observed, f"{direction} Spearman dependency beyond {threshold}", "advisory")

    result = pd.DataFrame(checks)
    report = {
        "validation_profile": "synthetic_structural_and_broad_plausibility_v1",
        "checks": int(len(result)),
        "passed": int((result.status == "pass").sum()),
        "warnings": int((result.status == "warning").sum()),
        "critical_failures": int(((result.status == "fail") & (result.severity == "critical")).sum()),
        "broad_plausibility_is_population_validity": False,
        "italian_population_representativeness_established": False,
    }
    return result, report
