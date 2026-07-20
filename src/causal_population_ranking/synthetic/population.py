"""Structured fully synthetic longitudinal population generator.

The generator is designed for methodological experiments without real patient data.
It encodes plausible dependency directions and temporal aggregation identities, but it
is not calibrated to, nor representative of, the Italian population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GENERATOR_VERSION = "structured_longitudinal_population_v1"
SYNTHETIC_POPULATION_SCENARIOS = (
    "baseline",
    "older_high_need",
    "social_fragility_shift",
    "utilization_surge",
    "combined_shift",
)


@dataclass(frozen=True)
class SyntheticPopulationResult:
    patients: pd.DataFrame
    monthly_history: pd.DataFrame
    metadata: dict
    privacy_audit: dict


def _sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _z(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-8)


def _bernoulli(rng: np.random.Generator, probability) -> np.ndarray:
    return rng.binomial(1, np.clip(np.asarray(probability, dtype=float), 0.0, 1.0))


def _overdispersed_count(rng: np.random.Generator, mean, dispersion: float = 2.0) -> np.ndarray:
    mean = np.maximum(np.asarray(mean, dtype=float), 0.0)
    rate = rng.gamma(shape=dispersion, scale=np.maximum(mean, 1e-10) / dispersion)
    return rng.poisson(rate).astype(np.int64)


def _age_distribution(
    rng: np.random.Generator,
    n: int,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    # Adult life-course mixture with explicit young-, middle-, older-, and
    # oldest-adult components. It is a methodological profile, not a census fit.
    component = rng.choice(4, size=n, p=(0.24, 0.31, 0.30, 0.15))
    fraction = np.empty(n, dtype=float)
    intervals = ((0.00, 0.27), (0.27, 0.52), (0.52, 0.78), (0.78, 1.00))
    for index, (lower, upper) in enumerate(intervals):
        mask = component == index
        fraction[mask] = lower + (upper - lower) * rng.beta(2.0, 2.0, int(mask.sum()))
    return np.rint(minimum + fraction * (maximum - minimum)).astype(np.int64)


def generate_synthetic_population(
    population_size: int,
    seed: int,
    history_months: int = 24,
    reference_date: str = "2025-01-01",
    minimum_age: int = 18,
    maximum_age: int = 100,
    profile: str = "methodological_adult_population_v1",
    scenario: str = "baseline",
) -> SyntheticPopulationResult:
    """Generate a fully synthetic patient snapshot and its monthly source history."""

    if population_size < 100:
        raise ValueError("Synthetic population_size must be at least 100")
    if history_months < 12:
        raise ValueError("At least 12 months of synthetic history are required")
    if minimum_age < 18 or maximum_age <= minimum_age or maximum_age > 110:
        raise ValueError("Synthetic age range must satisfy 18 <= minimum < maximum <= 110")
    if profile != "methodological_adult_population_v1":
        raise ValueError(f"Unknown synthetic population profile: {profile}")
    if scenario not in SYNTHETIC_POPULATION_SCENARIOS:
        raise ValueError(
            f"Unknown synthetic population scenario {scenario!r}; "
            f"choose from {SYNTHETIC_POPULATION_SCENARIOS}"
        )

    rng = np.random.default_rng(int(seed))
    n = int(population_size)
    patient_id = np.asarray([f"syn_{seed:08x}_{index:07d}" for index in range(n)])
    age = _age_distribution(rng, n, int(minimum_age), int(maximum_age))
    if scenario in {"older_high_need", "combined_shift"}:
        age = np.clip(age + rng.integers(5, 14, n), minimum_age, maximum_age)
    age_scaled = (age - minimum_age) / (maximum_age - minimum_age)
    female = _bernoulli(rng, 0.505 + 0.025 * age_scaled)
    deprivation = np.clip(rng.beta(2.1, 3.0, n), 0.0, 1.0)
    if scenario in {"social_fragility_shift", "combined_shift"}:
        deprivation = np.clip(deprivation + 0.18, 0.0, 1.0)
    rurality = _bernoulli(rng, _sigmoid(-1.25 + 1.10 * deprivation + rng.normal(0, 0.35, n)))
    health_literacy = np.clip(
        0.78 - 0.45 * deprivation - 0.10 * rurality + rng.normal(0.0, 0.12, n),
        0.0,
        1.0,
    )
    smoking = _bernoulli(
        rng, _sigmoid(-2.05 + 1.15 * deprivation - 0.95 * age_scaled + 0.25 * (1 - health_literacy))
    )
    bmi = np.clip(
        24.0 + 4.0 * deprivation + 1.8 * age_scaled + 1.2 * smoking
        + rng.normal(0.0, 3.8, n),
        16.0,
        48.0,
    )

    diabetes = _bernoulli(
        rng, _sigmoid(-4.25 + 3.65 * age_scaled + 0.105 * (bmi - 25.0) + 0.65 * deprivation)
    )
    cardiovascular = _bernoulli(
        rng, _sigmoid(-5.00 + 5.00 * age_scaled + 0.75 * diabetes + 0.42 * smoking)
    )
    copd = _bernoulli(
        rng, _sigmoid(-4.80 + 3.10 * age_scaled + 1.35 * smoking + 0.35 * deprivation)
    )
    chronic_kidney = _bernoulli(
        rng, _sigmoid(-5.20 + 4.10 * age_scaled + 0.95 * diabetes + 0.55 * cardiovascular)
    )
    cancer_history = _bernoulli(rng, _sigmoid(-4.85 + 3.75 * age_scaled + 0.30 * smoking))
    mental_health = _bernoulli(
        rng, _sigmoid(-2.00 + 0.28 * female + 0.95 * deprivation - 0.45 * age_scaled)
    )
    chronic_pain = _bernoulli(
        rng, _sigmoid(-3.10 + 2.65 * age_scaled + 0.60 * deprivation + 0.35 * female)
    )
    named_condition_count = (
        diabetes + cardiovascular + copd + chronic_kidney + cancer_history
        + mental_health + chronic_pain
    )
    additional_conditions = rng.poisson(
        np.clip(0.10 + 1.15 * age_scaled + 0.75 * deprivation + 0.20 * smoking, 0.0, 4.0)
    )
    condition_distinct = (named_condition_count + additional_conditions).astype(np.int64)

    base_frailty = _sigmoid(
        -3.35 + 3.20 * age_scaled + 0.34 * condition_distinct
        + 0.55 * chronic_kidney + 0.35 * cardiovascular + 0.25 * chronic_pain
        + rng.normal(0.0, 0.55, n)
    )
    base_social = _sigmoid(
        -1.65 + 2.10 * deprivation + 0.45 * rurality + 0.35 * mental_health
        + 0.18 * age_scaled + rng.normal(0.0, 0.70, n)
    )
    patient_utilization_random_effect = rng.normal(0.0, 0.42, n)
    utilization_slope = rng.normal(0.0, 0.12, n) + 0.16 * base_frailty + 0.05 * age_scaled

    patient_index = np.repeat(np.arange(n), history_months)
    month_offset = np.tile(np.arange(-history_months + 1, 1), n)
    time_scaled = (month_offset + history_months - 1) / max(history_months - 1, 1)
    season = 1.0 + 0.14 * np.cos(2.0 * np.pi * (month_offset % 12) / 12.0)
    idx = patient_index
    clinical_load = (
        0.32 * condition_distinct[idx] + 0.70 * base_frailty[idx]
        + 0.25 * base_social[idx] + 0.16 * mental_health[idx]
        + patient_utilization_random_effect[idx] + utilization_slope[idx] * time_scaled
    )
    encounter_mean = season * np.exp(-1.15 + clinical_load)
    emergency_mean = season * np.exp(
        -3.25 + 0.22 * condition_distinct[idx] + 0.80 * base_frailty[idx]
        + 0.45 * base_social[idx] + 0.28 * copd[idx]
    )
    inpatient_mean = season * np.exp(
        -4.35 + 0.25 * condition_distinct[idx] + 1.05 * base_frailty[idx]
        + 0.45 * cardiovascular[idx] + 0.35 * chronic_kidney[idx]
    )
    if scenario in {"utilization_surge", "combined_shift"}:
        encounter_mean *= 1.55
        emergency_mean *= 1.45
        inpatient_mean *= 1.35
    encounters = _overdispersed_count(rng, encounter_mean, 2.4)
    emergency = _overdispersed_count(rng, emergency_mean, 1.7)
    inpatient = _overdispersed_count(rng, inpatient_mean, 1.5)
    # Acute encounters are also encounters; enforce this temporal identity.
    encounters = np.maximum(encounters, emergency + inpatient)
    procedures = _overdispersed_count(
        rng, np.exp(-2.15 + 0.18 * condition_distinct[idx] + 0.35 * base_frailty[idx]), 2.0
    )
    careplans = _overdispersed_count(
        rng, np.exp(-2.45 + 0.22 * condition_distinct[idx] + 0.55 * base_frailty[idx]), 2.0
    )
    medication_base = (
        0.05 + 0.95 * condition_distinct + 0.80 * diabetes + 0.70 * cardiovascular
        + 0.75 * chronic_kidney + 0.45 * mental_health
    )
    patient_medication_count = rng.poisson(np.maximum(medication_base, 0.0)).astype(np.int64)
    monthly_start_probability = np.clip(
        0.008 + 0.018 * condition_distinct + 0.015 * base_frailty,
        0.0,
        0.18,
    )
    monthly_stop_probability = np.clip(
        0.006 + 0.004 * patient_medication_count,
        0.0,
        0.08,
    )
    active_medications = np.maximum(
        0,
        patient_medication_count[idx]
        + rng.binomial(1, monthly_start_probability[idx])
        - rng.binomial(1, monthly_stop_probability[idx]),
    ).astype(np.int64)

    reference = pd.Timestamp(reference_date)
    month_date = pd.PeriodIndex(
        [reference.to_period("M") + int(offset) for offset in month_offset], freq="M"
    ).astype(str)
    monthly = pd.DataFrame({
        "patient_id": patient_id[idx],
        "month_offset": month_offset.astype(np.int64),
        "calendar_month": month_date,
        "encounter_count": encounters,
        "emergency_count": emergency,
        "inpatient_count": inpatient,
        "procedure_count": procedures,
        "careplan_count": careplans,
        "active_medication_count": active_medications,
    })

    recent = monthly.month_offset >= -11
    last_six = monthly.month_offset >= -5
    previous_six = monthly.month_offset.between(-11, -6)
    recent_frame = monthly.loc[recent]
    aggregates = recent_frame.groupby("patient_id", sort=False).agg(
        encounter_count=("encounter_count", "sum"),
        prior_emergency=("emergency_count", "sum"),
        prior_inpatient=("inpatient_count", "sum"),
        procedure_count=("procedure_count", "sum"),
        careplan_count=("careplan_count", "sum"),
        medication_distinct=("active_medication_count", "max"),
    )
    last_six_mean = monthly.loc[last_six].groupby("patient_id", sort=False).encounter_count.mean()
    previous_six_mean = monthly.loc[previous_six].groupby("patient_id", sort=False).encounter_count.mean()
    trend = last_six_mean - previous_six_mean

    encounter_matrix = encounters.reshape(n, history_months)
    days_since = np.empty(n, dtype=float)
    for row in range(n):
        months_with_encounter = np.flatnonzero(encounter_matrix[row] > 0)
        if len(months_with_encounter):
            months_ago = history_months - 1 - int(months_with_encounter[-1])
            days_since[row] = min(730.0, months_ago * 30.4 + rng.uniform(0.0, 30.4))
        else:
            days_since[row] = 730.0

    prior_inpatient = aggregates.loc[patient_id, "prior_inpatient"].to_numpy(np.int64)
    prior_emergency = aggregates.loc[patient_id, "prior_emergency"].to_numpy(np.int64)
    frailty = _sigmoid(
        -3.45 + 3.10 * age_scaled + 0.30 * condition_distinct
        + 0.18 * np.log1p(prior_inpatient) + 0.50 * chronic_kidney
        + 0.30 * cardiovascular + rng.normal(0.0, 0.45, n)
    )
    cognitive = _bernoulli(
        rng, _sigmoid(-4.10 + 4.25 * age_scaled + 1.20 * frailty + 0.45 * cardiovascular)
    )
    functional = _sigmoid(
        -2.95 + 3.25 * frailty + 1.00 * cognitive + 0.50 * chronic_pain
        + 0.18 * np.log1p(prior_inpatient) + rng.normal(0.0, 0.50, n)
    )
    social = np.clip(
        0.65 * base_social + 0.20 * deprivation + 0.15 * _sigmoid(mental_health + rurality - 1),
        0.0,
        1.0,
    )
    housing_instability = _bernoulli(
        rng, _sigmoid(-3.65 + 3.00 * deprivation + 1.10 * mental_health - 0.70 * age_scaled)
    )
    caregiver_available = _bernoulli(
        rng, _sigmoid(1.70 - 1.70 * social - 0.85 * functional + 0.35 * age_scaled)
    )
    non_self_sufficiency = _bernoulli(
        rng, _sigmoid(-5.00 + 4.10 * functional + 1.50 * frailty + 0.85 * cognitive)
    )
    palliative_need = _bernoulli(
        rng,
        _sigmoid(
            -7.20 + 2.25 * cancer_history + 1.25 * chronic_kidney
            + 1.75 * frailty + 1.45 * functional + 0.25 * np.log1p(prior_inpatient)
        ),
    )

    patients = pd.DataFrame({
        "patient_id": patient_id,
        "age": age.astype(float),
        "female": female.astype(float),
        "deprivation_index": deprivation,
        "rurality": rurality.astype(float),
        "health_literacy_score": health_literacy,
        "smoking": smoking.astype(float),
        "bmi": bmi,
        "diabetes": diabetes.astype(float),
        "cardiovascular_disease": cardiovascular.astype(float),
        "copd": copd.astype(float),
        "chronic_kidney_disease": chronic_kidney.astype(float),
        "cancer_history": cancer_history.astype(float),
        "mental_health_condition": mental_health.astype(float),
        "chronic_pain": chronic_pain.astype(float),
        "condition_distinct": condition_distinct.astype(float),
        "prior_inpatient": prior_inpatient.astype(float),
        "prior_emergency": prior_emergency.astype(float),
        "medication_distinct": aggregates.loc[patient_id, "medication_distinct"].to_numpy(float),
        "recent_utilization_trend": trend.loc[patient_id].to_numpy(float),
        "multimorbidity": (condition_distinct >= 2).astype(float),
        "polypharmacy": (aggregates.loc[patient_id, "medication_distinct"].to_numpy() >= 5).astype(float),
        "encounter_count": aggregates.loc[patient_id, "encounter_count"].to_numpy(float),
        "days_since_encounter": days_since,
        "procedure_count": aggregates.loc[patient_id, "procedure_count"].to_numpy(float),
        "careplan_count": aggregates.loc[patient_id, "careplan_count"].to_numpy(float),
        "frailty_index": frailty,
        "functional_limitation_score": functional,
        "social_fragility_score": social,
        "cognitive_impairment": cognitive.astype(float),
        "non_self_sufficiency": non_self_sufficiency.astype(float),
        "caregiver_available": caregiver_available.astype(float),
        "housing_instability": housing_instability.astype(float),
        "palliative_need": palliative_need.astype(float),
    })

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "population_profile": profile,
        "synthetic_population_scenario": scenario,
        "population_size": n,
        "seed": int(seed),
        "reference_date": str(reference.date()),
        "history_months": int(history_months),
        "age_range": [int(minimum_age), int(maximum_age)],
        "data_source": "fully_synthetic_structural_generator",
        "real_patient_records_used": False,
        "synthea_records_used": False,
        "calibrated_to_italian_population": False,
        "dependency_design": [
            "age_behaviour_social_to_chronicity",
            "chronicity_frailty_social_to_longitudinal_utilization",
            "clinical_functional_social_to_dm77_inputs",
        ],
        "temporal_aggregation_window_months": 12,
    }
    privacy_audit = {
        "population_is_fully_synthetic": True,
        "real_person_source_records": 0,
        "direct_identifiers_present": False,
        "identifier_semantics": "deterministic_synthetic_run_identifier",
        "memorization_from_real_records_possible": False,
        "differential_privacy_applied": False,
        "formal_privacy_guarantee_claimed": False,
        "reidentification_risk_claimed_zero": False,
        "note": (
            "No real records are inputs to this generator. This is a lineage statement, "
            "not a formal differential-privacy guarantee."
        ),
    }
    return SyntheticPopulationResult(patients, monthly, metadata, privacy_audit)
