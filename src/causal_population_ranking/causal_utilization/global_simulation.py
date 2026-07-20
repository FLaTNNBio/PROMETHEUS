from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..data.validation import assert_no_oracle_columns


TRANSITION_NAMES = ("1_to_2", "2_to_3", "3_to_4", "4_to_5", "5_to_6")
TREATMENT_LEVELS = tuple(range(1, 7))
SCENARIOS = (
    "baseline",
    "poor_overlap",
    "hidden_confounding",
    "risk_benefit_misalignment",
    "combined_stress",
    "cross_transition_scale_imbalance",
    "transition_sample_imbalance",
    "heterogeneous_transition_costs",
)


@dataclass(frozen=True)
class MultivaluedSimulationResult:
    """Physically separated learner and evaluation data from one multivalued DGP."""

    learner: pd.DataFrame
    ground_truth: pd.DataFrame
    transition_learners: dict[str, pd.DataFrame]
    feature_columns: tuple[str, ...]
    metadata: dict


def _z(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-8)


def _remove_linear_component(values, reference) -> np.ndarray:
    """Remove the pooled linear component in ``reference`` and restore unit scale."""

    values_z = _z(values)
    reference_z = _z(reference)
    coefficient = float(np.mean(values_z * reference_z))
    return _z(values_z - coefficient * reference_z)


def _patient_splits(n: int, seed: int) -> np.ndarray:
    if n < 20:
        raise ValueError("At least 20 patients are required for four patient-level splits")
    order = np.random.default_rng(seed).permutation(n)
    cuts = (int(0.35 * n), int(0.65 * n), int(0.80 * n))
    split = np.empty(n, dtype=object)
    split[order[: cuts[0]]] = "nuisance_train"
    split[order[cuts[0] : cuts[1]]] = "rank_train"
    split[order[cuts[1] : cuts[2]]] = "validation"
    split[order[cuts[2] :]] = "test"
    return split


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -40.0, 40.0))
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _augment_semisynthetic_dm77_covariates(cohort: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add pre-index functional and social covariates absent from the source cohort.

    These variables belong to the semi-synthetic data-generating process. Existing
    source columns are preserved, and every random draw uses the explicit run seed.
    """

    result = cohort.reset_index(drop=True).copy()
    required = {
        "age", "condition_distinct", "prior_inpatient", "prior_emergency",
        "medication_distinct", "recent_utilization_trend",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"Cohort is missing pre-treatment columns: {missing}")
    rng = np.random.default_rng(seed + 7_711)
    n = len(result)
    age = _z(result.age)
    burden = _z(np.log1p(result.condition_distinct))
    inpatient = _z(np.log1p(result.prior_inpatient))
    emergency = _z(np.log1p(result.prior_emergency))
    medications = _z(np.log1p(result.medication_distinct))
    trend = _z(result.recent_utilization_trend)

    frailty = _sigmoid(
        -1.65 + 0.80 * age + 0.70 * burden + 0.45 * inpatient
        + 0.18 * medications + rng.normal(0.0, 0.70, n)
    )
    functional = _sigmoid(
        -1.45 + 2.20 * frailty + 0.40 * inpatient + 0.20 * emergency
        + rng.normal(0.0, 0.65, n)
    )
    social = _sigmoid(
        -1.25 + 0.25 * age + 0.22 * emergency + 0.18 * trend
        + rng.normal(0.0, 0.90, n)
    )
    cognitive_probability = _sigmoid(-3.10 + 1.20 * age + 1.40 * frailty + 0.25 * burden)
    non_self_probability = _sigmoid(
        -4.25 + 2.80 * functional + 1.35 * frailty + 0.35 * inpatient
    )
    housing_probability = _sigmoid(-3.00 + 2.10 * social + 0.22 * emergency)
    caregiver_probability = _sigmoid(1.35 - 1.55 * social - 0.45 * functional + 0.15 * age)
    cognitive = rng.binomial(1, cognitive_probability, n)
    non_self_sufficiency = rng.binomial(1, non_self_probability, n)
    housing_instability = rng.binomial(1, housing_probability, n)
    caregiver_available = rng.binomial(1, caregiver_probability, n)
    palliative_probability = _sigmoid(
        -6.20 + 1.55 * burden + 1.35 * inpatient + 1.10 * frailty
        + 0.85 * functional + 0.30 * age
    )
    palliative_need = rng.binomial(1, palliative_probability, n)

    generated = {
        "frailty_index": frailty,
        "functional_limitation_score": functional,
        "social_fragility_score": social,
        "cognitive_impairment": cognitive,
        "non_self_sufficiency": non_self_sufficiency,
        "caregiver_available": caregiver_available,
        "housing_instability": housing_instability,
        "palliative_need": palliative_need,
    }
    for column, values in generated.items():
        if column not in result:
            result[column] = values
    return result


def _eligibility(
    cohort: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    """Return nested pre-treatment eligibility with shape ``[patients, 5]``."""

    if mode == "all":
        return np.ones((len(cohort), 5), dtype=bool)
    if mode != "synthetic_clinical":
        raise ValueError("simulation.eligibility.mode must be 'all' or 'synthetic_clinical'")
    need = _z(
        0.25 * _z(cohort.age)
        + 0.60 * _z(np.log1p(cohort.condition_distinct))
        + 0.50 * _z(np.log1p(cohort.prior_inpatient))
        + 0.35 * _z(np.log1p(cohort.prior_emergency))
        + 0.25 * _z(np.log1p(cohort.medication_distinct))
        + 0.20 * _z(cohort.recent_utilization_trend)
    )
    # Higher increments are progressively more selective; nesting follows by
    # construction because every later threshold is at least as large.
    thresholds = np.quantile(need, (0.00, 0.10, 0.25, 0.40, 0.55))
    eligibility = need[:, None] >= thresholds[None, :]
    eligibility[:, 0] = True
    if np.any(eligibility[:, 1:] & ~eligibility[:, :-1]):
        raise AssertionError("Synthetic clinical eligibility is not nested")
    return eligibility


def _effect_scales(values: Mapping[str, float] | list[float] | tuple[float, ...] | None) -> np.ndarray:
    if values is None:
        return np.ones(5, dtype=float)
    if isinstance(values, Mapping):
        unknown = set(values).difference(TRANSITION_NAMES)
        if unknown or set(values) != set(TRANSITION_NAMES):
            raise ValueError("transition_effect_scales must define all five transitions")
        result = np.asarray([values[name] for name in TRANSITION_NAMES], dtype=float)
    else:
        result = np.asarray(values, dtype=float)
    if result.shape != (5,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("transition_effect_scales must contain five positive finite values")
    return result


def _validate_cohort(cohort: pd.DataFrame) -> tuple[str, ...]:
    required = {
        "patient_id",
        "age",
        "condition_distinct",
        "prior_inpatient",
        "prior_emergency",
        "medication_distinct",
        "recent_utilization_trend",
    }
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise ValueError(f"Cohort is missing pre-treatment columns: {missing}")
    if cohort.patient_id.astype(str).duplicated().any():
        raise ValueError("The multivalued DGP requires one row per patient")
    approved_pre_treatment = (
        "age", "female", "condition_distinct", "prior_inpatient", "prior_emergency",
        "medication_distinct", "recent_utilization_trend", "multimorbidity",
        "polypharmacy", "encounter_count", "days_since_encounter",
        "procedure_count", "careplan_count", "frailty_index",
        "functional_limitation_score", "social_fragility_score",
        "cognitive_impairment", "non_self_sufficiency", "caregiver_available",
        "housing_instability", "palliative_need", "deprivation_index", "rurality",
        "health_literacy_score", "smoking", "bmi", "diabetes",
        "cardiovascular_disease", "copd", "chronic_kidney_disease",
        "cancer_history", "mental_health_condition", "chronic_pain",
    )
    feature_columns = tuple(
        column for column in approved_pre_treatment
        if column in cohort.columns and pd.api.types.is_numeric_dtype(cohort[column])
    )
    if not feature_columns:
        raise ValueError("No numeric pre-treatment features are available")
    feature_values = cohort.loc[:, feature_columns].to_numpy(float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Pre-treatment features must be finite")
    return feature_columns


def simulate_multivalued_care(
    cohort: pd.DataFrame,
    seed: int = 42,
    scenario: str = "baseline",
    outcome_horizon_days: int = 365,
    treatment_levels: int = 6,
    shared_effect_correlation: float = 0.50,
    overlap_strength: float = 1.50,
    risk_benefit_alignment: float = 0.0,
    transition_sample_imbalance: float = 1.0,
    transition_effect_scales: Mapping[str, float] | list[float] | None = None,
    eligibility_mode: str = "synthetic_clinical",
    outcome_noise_sd: float = 6.0,
) -> MultivaluedSimulationResult:
    """Generate ``X -> {Y(1),...,Y(6)} -> T -> Y(T)`` on a common day scale.

    The learner table contains exactly one historical treatment per patient.
    Potential outcomes, effects, assignment probabilities, and generated noise
    are returned only in ``ground_truth`` and are never merged into learner data.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    if treatment_levels != 6:
        raise ValueError("PROMETHEUS global v1 requires exactly six treatment levels")
    if outcome_horizon_days != 365:
        raise ValueError("PROMETHEUS global v1 uses a 365-day common outcome horizon")
    if not 0.0 <= shared_effect_correlation <= 1.0:
        raise ValueError("shared_effect_correlation must be in [0, 1]")
    if overlap_strength <= 0 or transition_sample_imbalance <= 0 or outcome_noise_sd < 0:
        raise ValueError("Overlap, sample-imbalance, and outcome-noise parameters are invalid")
    if not -1.0 <= risk_benefit_alignment <= 1.0:
        raise ValueError("risk_benefit_alignment must be in [-1, 1]")

    cohort = _augment_semisynthetic_dm77_covariates(cohort, seed)
    feature_columns = _validate_cohort(cohort)
    n = len(cohort)
    rng = np.random.default_rng(seed)
    split = _patient_splits(n, seed + 17)

    age = _z(cohort.age)
    burden = _z(np.log1p(cohort.condition_distinct))
    inpatient = _z(np.log1p(cohort.prior_inpatient))
    emergency = _z(np.log1p(cohort.prior_emergency))
    medications = _z(np.log1p(cohort.medication_distinct))
    trend = _z(cohort.recent_utilization_trend)
    frailty = _z(cohort.frailty_index)
    functional = _z(cohort.functional_limitation_score)
    social = _z(cohort.social_fragility_score)
    risk = _z(
        0.30 * age
        + 0.65 * burden
        + 0.55 * inpatient
        + 0.40 * emergency
        + 0.25 * medications
        + 0.18 * trend
        + 0.22 * frailty
        + 0.16 * functional
        + 0.10 * social
        + 0.12 * burden * inpatient
    )
    hidden_confounding = scenario in {"hidden_confounding", "combined_stress"}
    hidden = _z(0.45 * risk + rng.normal(0.0, 0.9, n)) if hidden_confounding else np.zeros(n)

    # Expected days alive and outside acute-care hospitalization. This baseline
    # is intentionally well inside [0, 365], leaving room for heterogeneous sums.
    mu1 = 278.0 - 31.0 * np.tanh(risk / 1.7) + 4.0 * np.sin(age) - 2.5 * np.tanh(burden**2)
    if hidden_confounding:
        mu1 -= 3.0 * hidden

    shared = _z(
        0.50 * np.sin(age)
        + 0.45 * burden
        - 0.30 * inpatient
        + 0.28 * medications
        + 0.20 * trend * burden
    )
    specific = np.column_stack(
        (
            _z(0.70 * trend - 0.40 * emergency + 0.30 * np.sin(medications)),
            _z(0.65 * burden - 0.25 * burden**2 + 0.30 * medications),
            _z(0.55 * emergency + 0.45 * inpatient - 0.25 * age * burden),
            _z(0.55 * inpatient + 0.40 * medications + 0.25 * np.tanh(trend)),
            _z(-0.50 * age + 0.45 * emergency - 0.30 * medications * age),
        )
    )
    means = np.asarray((3.5, 5.0, 4.0, 2.5, 1.0), dtype=float)
    scales = np.asarray((5.0, 6.5, 7.0, 6.0, 5.5), dtype=float) * _effect_scales(
        transition_effect_scales
    )
    if scenario == "cross_transition_scale_imbalance":
        means = means * np.asarray((0.4, 0.8, 1.2, 1.7, 2.2))
        scales = scales * np.asarray((0.45, 0.75, 1.0, 1.35, 1.70))
    if scenario == "risk_benefit_misalignment" and risk_benefit_alignment == 0.0:
        risk_benefit_alignment = -0.65

    effects = np.empty((n, 5), dtype=float)
    correlation = float(shared_effect_correlation)
    for transition_index in range(5):
        response = correlation * shared + np.sqrt(max(0.0, 1.0 - correlation**2)) * specific[:, transition_index]
        residual_response = _remove_linear_component(response, risk)
        aligned_response = (
            risk_benefit_alignment * risk
            + np.sqrt(max(0.0, 1.0 - risk_benefit_alignment**2)) * residual_response
        )
        effects[:, transition_index] = means[transition_index] + scales[transition_index] * np.tanh(
            aligned_response / 1.35
        )
        if hidden_confounding:
            effects[:, transition_index] += 1.2 * hidden

    potential = np.empty((n, 6), dtype=float)
    potential[:, 0] = mu1
    potential[:, 1:] = mu1[:, None] + np.cumsum(effects, axis=1)
    if not np.allclose(np.diff(potential, axis=1), effects, atol=1e-10, rtol=0.0):
        raise AssertionError("Adjacent potential-outcome differences do not equal transition effects")
    if potential.min() < 0.0 or potential.max() > float(outcome_horizon_days):
        raise RuntimeError(
            "DGP parameters place expected potential outcomes outside [0, 365]; "
            "reduce transition_effect_scales instead of clipping effects"
        )

    assignment_overlap = max(float(overlap_strength), 3.2) if scenario in {
        "poor_overlap", "combined_stress"
    } else float(overlap_strength)
    sample_imbalance = float(transition_sample_imbalance)
    if scenario == "transition_sample_imbalance":
        sample_imbalance = max(sample_imbalance, 1.8)
    intensity = np.linspace(-1.0, 1.0, 6)
    intercept = np.asarray((0.55, 0.35, 0.10, -0.15, -0.45, -0.80)) * sample_imbalance
    logits = np.empty((n, 6), dtype=float)
    for level_index in range(6):
        nonlinear = (
            0.25 * np.sin((level_index + 1) * age / 2.0)
            + 0.16 * burden * trend
            - 0.10 * emergency**2 * intensity[level_index]
        )
        logits[:, level_index] = (
            intercept[level_index]
            + assignment_overlap * (0.42 * intensity[level_index] * risk + nonlinear)
            + 0.08 * intensity[level_index] * effects[:, min(level_index, 4)]
        )
        if hidden_confounding:
            logits[:, level_index] += 0.70 * intensity[level_index] * hidden
    assignment_probability = _softmax(logits)
    treatment = np.asarray(
        [rng.choice(TREATMENT_LEVELS, p=assignment_probability[row]) for row in range(n)],
        dtype=np.int64,
    )
    raw_noise = rng.normal(0.0, float(outcome_noise_sd), n)
    selected_mean = potential[np.arange(n), treatment - 1]
    observed = np.clip(selected_mean + raw_noise, 0.0, float(outcome_horizon_days))
    realized_noise = observed - selected_mean

    eligibility = _eligibility(cohort, eligibility_mode)
    learner_columns: dict[str, object] = {
        "patient_id": cohort.patient_id.astype(str).to_numpy(),
    }
    for column in feature_columns:
        learner_columns[column] = cohort[column].to_numpy()
    learner_columns.update(
        {
            "treatment_level": treatment,
            "observed_outcome": observed,
            "split": split,
        }
    )
    for transition_index, name in enumerate(TRANSITION_NAMES):
        learner_columns[f"eligible_{name}"] = eligibility[:, transition_index]
    learner = pd.DataFrame(learner_columns)
    assert_no_oracle_columns(learner)
    forbidden_learner = {"baseline_care_level", "current_care_level", "received_care_level"}
    if forbidden_learner.intersection(learner.columns):
        raise AssertionError("A baseline/current care level entered the global learner table")

    truth_columns: dict[str, object] = {
        "patient_id": cohort.patient_id.astype(str).to_numpy(),
        "scenario": scenario,
        "prognostic_risk_truth": risk,
        "unobserved_confounder_truth": hidden,
        "observed_outcome_noise": realized_noise,
    }
    for level_index in range(6):
        truth_columns[f"potential_outcome_{level_index + 1}"] = potential[:, level_index]
        truth_columns[f"oracle_treatment_probability_{level_index + 1}"] = assignment_probability[:, level_index]
    for transition_index, name in enumerate(TRANSITION_NAMES):
        truth_columns[f"true_benefit_{name}"] = effects[:, transition_index]
    ground_truth = pd.DataFrame(truth_columns)

    transition_learners: dict[str, pd.DataFrame] = {}
    transition_metadata = {}
    for transition_index, name in enumerate(TRANSITION_NAMES):
        lower = transition_index + 1
        eligible = learner[f"eligible_{name}"].to_numpy(bool)
        adjacent = learner.treatment_level.isin((lower, lower + 1)).to_numpy()
        frame = learner.loc[eligible & adjacent].copy()
        frame["transition"] = name
        frame["transition_index"] = transition_index
        frame["transition_treatment"] = (frame.treatment_level.to_numpy(int) == lower + 1).astype(int)
        frame.reset_index(drop=True, inplace=True)
        assert_no_oracle_columns(frame)
        transition_learners[name] = frame
        transition_metadata[name] = {
            "lower_level": lower,
            "upper_level": lower + 1,
            "observed_adjacent_n": int(len(frame)),
            "eligible_n": int(eligible.sum()),
            "upper_arm_rate": float(frame.transition_treatment.mean()) if len(frame) else float("nan"),
            "split_counts": frame.split.value_counts().to_dict(),
        }

    metadata = {
        "dgp": "single_multivalued_treatment_recursive_potential_outcomes_v1",
        "outcome": "365_day_hospital_free_days",
        "outcome_horizon_days": 365,
        "treatment_levels": 6,
        "treatment_counts": learner.treatment_level.value_counts().sort_index().to_dict(),
        "eligibility_mode": eligibility_mode,
        "eligibility_treatment_independent": True,
        "nested_eligibility": True,
        "scenario": scenario,
        "hidden_confounding": hidden_confounding,
        "shared_effect_correlation": correlation,
        "risk_benefit_alignment": float(risk_benefit_alignment),
        "transition_sample_imbalance": sample_imbalance,
        "transition_effect_scales": dict(zip(TRANSITION_NAMES, scales.tolist())),
        "dm77_covariates": "semi_synthetic_pre_index_v1",
        "dm77_need_level_generated": False,
        "dm77_need_level_used_as_treatment": False,
        "potential_outcome_range": [float(potential.min()), float(potential.max())],
        "observed_outcome_range": [float(observed.min()), float(observed.max())],
        "transition_populations": transition_metadata,
        "oracle_physically_separate": True,
    }
    return MultivaluedSimulationResult(
        learner=learner,
        ground_truth=ground_truth,
        transition_learners=transition_learners,
        feature_columns=feature_columns,
        metadata=metadata,
    )
