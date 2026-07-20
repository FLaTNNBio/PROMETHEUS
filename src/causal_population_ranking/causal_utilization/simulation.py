from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .eligibility import derive_transition_eligibility
from .initial_state import attach_current_care_level


DEFAULT_CAPACITIES = {
    "1_to_2": 0.40,
    "2_to_3": 0.30,
    "3_to_4": 0.20,
    "4_to_5": 0.10,
    "5_to_6": 0.05,
}
TRANSITION_LEVELS = {
    name: (level, level + 1) for level, name in enumerate(DEFAULT_CAPACITIES, start=1)
}
TRANSITIONS = {
    name: (*TRANSITION_LEVELS[name], capacity)
    for name, capacity in DEFAULT_CAPACITIES.items()
}
SCENARIOS = (
    "baseline",
    "poor_overlap",
    "hidden_confounding",
    "risk_benefit_misalignment",
    "combined_stress",
)


def _z(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-8)


def _remove_linear_component(values, reference) -> np.ndarray:
    values = _z(values)
    reference = _z(reference)
    coefficient = float(np.mean(values * reference))
    return _z(values - coefficient * reference)


def _sigmoid(values) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))


def _splits(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cuts = [int(0.35 * n), int(0.65 * n), int(0.80 * n)]
    split = np.empty(n, dtype=object)
    split[order[: cuts[0]]] = "nuisance_train"
    split[order[cuts[0] : cuts[1]]] = "rank_train"
    split[order[cuts[1] : cuts[2]]] = "validation"
    split[order[cuts[2] :]] = "test"
    return split


def _resolve_capacities(capacities: Mapping[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_CAPACITIES)
    if capacities is not None:
        unknown = set(capacities).difference(resolved)
        if unknown:
            raise ValueError(f"Unknown transition capacities: {sorted(unknown)}")
        resolved.update({name: float(value) for name, value in capacities.items()})
    if any(not 0.0 < value <= 1.0 for value in resolved.values()):
        raise ValueError("Every transition capacity must be in (0, 1]")
    return resolved


def _transition_sample_targets(levels: np.ndarray, rho: float) -> dict[str, int]:
    if not 0.0 < rho <= 1.0:
        raise ValueError("transition_imbalance_rho must be in (0, 1]")
    available = [int(np.sum(levels == level)) for level in range(1, 6)]
    if min(available) < 1:
        raise ValueError("Every baseline decision level 1..5 needs at least one patient")
    feasible_base = min(
        int(np.floor(available[index] / (rho**index))) for index in range(5)
    )
    if feasible_base < 1:
        raise ValueError("Transition imbalance leaves an empty decision population")
    return {
        name: int(np.floor(feasible_base * rho**index))
        for index, name in enumerate(DEFAULT_CAPACITIES)
    }


def simulate_care_intensity(
    cohort: pd.DataFrame,
    seed: int = 42,
    scenario: str = "baseline",
    shared_effect_correlation: float = 0.50,
    overlap_strength: float = 1.50,
    risk_benefit_alignment: float = 0.0,
    transition_imbalance_rho: float = 1.0,
    capacities: Mapping[str, float] | None = None,
):
    """Generate five coherent one-step causal-ranking decision problems.

    Oracle potential outcomes, transition effects, propensities, and response
    components are returned in physically separate frames. They are evaluation
    only and are never included in learner features or model inputs.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    if not 0.0 <= shared_effect_correlation <= 1.0:
        raise ValueError("shared_effect_correlation must be in [0, 1]")
    if overlap_strength <= 0:
        raise ValueError("overlap_strength must be positive")
    if not -1.0 <= risk_benefit_alignment <= 1.0:
        raise ValueError("risk_benefit_alignment must be in [-1, 1]")

    if scenario in {"poor_overlap", "combined_stress"}:
        overlap_strength = max(float(overlap_strength), 3.0)
    if scenario == "risk_benefit_misalignment" and risk_benefit_alignment == 0.0:
        risk_benefit_alignment = -0.5
    hidden_confounding = scenario in {"hidden_confounding", "combined_stress"}

    capacity_by_transition = _resolve_capacities(capacities)
    rng = np.random.default_rng(seed)
    cohort_with_level, initial_state_metadata = attach_current_care_level(cohort, seed + 7000)
    clinical = derive_transition_eligibility(cohort_with_level)
    n = len(cohort_with_level)

    age = _z(cohort_with_level.age)
    burden = _z(np.log1p(cohort_with_level.condition_distinct))
    inpatient = _z(np.log1p(cohort_with_level.prior_inpatient))
    emergency = _z(np.log1p(cohort_with_level.prior_emergency))
    medications = _z(np.log1p(cohort_with_level.medication_distinct))
    trend = _z(cohort_with_level.recent_utilization_trend)
    risk = _z(
        0.35 * age
        + 0.65 * burden
        + 0.55 * inpatient
        + 0.40 * emergency
        + 0.25 * medications
        + 0.15 * trend
        + 0.15 * burden * inpatient
    )
    hidden = _z(0.45 * risk + rng.normal(0.0, 0.90, n)) if hidden_confounding else np.zeros(n)
    mu = 1.5 - 0.55 * risk + 0.12 * np.sin(age) - 0.08 * burden**2
    if hidden_confounding:
        mu = mu - 0.30 * hidden - 0.10 * hidden * age

    shared_response = _z(
        0.55 * np.sin(age)
        + 0.45 * burden
        + 0.30 * medications
        - 0.25 * inpatient
        + 0.20 * trend * burden
    )
    transition_specific = {
        "1_to_2": _z(0.70 * trend - 0.40 * emergency + 0.30 * np.sin(medications)),
        "2_to_3": _z(0.65 * burden - 0.35 * burden**2 + 0.30 * medications),
        "3_to_4": _z(0.60 * emergency + 0.45 * inpatient - 0.30 * age * burden),
        "4_to_5": _z(0.60 * inpatient + 0.45 * medications + 0.30 * np.tanh(trend)),
        "5_to_6": _z(-0.55 * age + 0.45 * emergency - 0.35 * medications * age),
    }
    locations = dict(zip(DEFAULT_CAPACITIES, (-0.02, -0.01, 0.0, 0.01, 0.02)))
    scales = dict(zip(DEFAULT_CAPACITIES, (0.16, 0.18, 0.20, 0.18, 0.16)))

    effects = {}
    pre_alignment_effects = {}
    weighted_shared = {}
    weighted_specific = {}
    for name in DEFAULT_CAPACITIES:
        shared_part = shared_effect_correlation * shared_response
        specific_part = np.sqrt(1.0 - shared_effect_correlation**2) * transition_specific[name]
        response = shared_part + specific_part
        pre_alignment = locations[name] + scales[name] * np.tanh(response)
        residual_response = _remove_linear_component(pre_alignment, risk)
        aligned = (
            risk_benefit_alignment * risk
            + np.sqrt(1.0 - risk_benefit_alignment**2) * residual_response
        )
        effect = float(np.mean(pre_alignment)) + float(np.std(pre_alignment)) * aligned
        if hidden_confounding:
            effect = effect + 0.05 * hidden
        effects[name] = effect
        pre_alignment_effects[name] = pre_alignment
        weighted_shared[name] = shared_part
        weighted_specific[name] = specific_part

    potential_outcomes = {1: mu}
    for level, name in enumerate(DEFAULT_CAPACITIES, start=1):
        potential_outcomes[level + 1] = potential_outcomes[level] + effects[name]

    truth_columns = {
        "patient_id": cohort_with_level.patient_id.astype(str),
        "prognostic_risk_truth": risk,
        "unobserved_confounder_truth": hidden,
        "scenario": scenario,
    }
    for level in range(1, 7):
        truth_columns[f"potential_outcome_{level}"] = potential_outcomes[level]
    for name in DEFAULT_CAPACITIES:
        truth_columns[f"true_benefit_{name}"] = effects[name]
        truth_columns[f"pre_alignment_effect_{name}"] = pre_alignment_effects[name]
        truth_columns[f"weighted_shared_component_{name}"] = weighted_shared[name]
        truth_columns[f"weighted_specific_component_{name}"] = weighted_specific[name]
    truth_all = pd.DataFrame(truth_columns)

    split = _splits(n, seed)
    baseline_levels = clinical.baseline_care_level.to_numpy(int)
    sample_targets = _transition_sample_targets(baseline_levels, float(transition_imbalance_rho))
    excluded = {
        "patient_id", "current_care_level", "baseline_care_level",
        *[f"eligible_{name}" for name in DEFAULT_CAPACITIES],
    }
    feature_columns = [
        column for column in cohort_with_level.columns
        if column not in excluded and column != "patient_id"
    ] + ["baseline_care_level"]

    covariates = np.column_stack((age, burden, inpatient, emergency, medications, trend))
    beta = np.array([
        [0.18, 0.12, -0.10, 0.08, 0.05, 0.10],
        [-0.12, 0.20, 0.10, 0.08, -0.05, 0.12],
        [0.08, -0.10, 0.22, 0.18, 0.08, -0.05],
        [-0.10, 0.08, 0.20, 0.12, 0.16, 0.06],
        [0.12, 0.10, -0.08, 0.22, -0.14, 0.05],
    ])

    learners: dict[str, pd.DataFrame] = {}
    truths: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    for transition_index, (name, capacity) in enumerate(capacity_by_transition.items()):
        lower, upper = TRANSITION_LEVELS[name]
        candidates = np.flatnonzero(baseline_levels == lower)
        target = sample_targets[name]
        selected_population = np.random.default_rng(seed + 3000 + transition_index).choice(
            candidates, size=target, replace=False
        )
        selected_population = np.sort(selected_population)
        clinical[f"eligible_{name}"] = False
        clinical.loc[selected_population, f"eligible_{name}"] = True

        nonlinear_assignment = _z(
            0.55 * np.sin(covariates[:, transition_index % covariates.shape[1]])
            + 0.35 * covariates[:, (transition_index + 2) % covariates.shape[1]] ** 2
            - 0.20 * covariates[:, (transition_index + 4) % covariates.shape[1]]
        )
        logit = covariates @ beta[transition_index] + overlap_strength * nonlinear_assignment
        if hidden_confounding:
            logit = logit + 1.25 * hidden
        propensity = np.clip(_sigmoid(logit), 0.005, 0.995)
        treatment = rng.binomial(1, propensity)
        received_level = lower + treatment
        observed = (
            np.where(treatment == 1, potential_outcomes[upper], potential_outcomes[lower])
            + rng.normal(0.0, 0.45, n)
        )

        learner = cohort_with_level.loc[
            selected_population, ["patient_id", *feature_columns]
        ].copy()
        learner["baseline_care_level"] = lower
        learner["transition_treatment"] = treatment[selected_population]
        learner["treatment"] = treatment[selected_population]
        learner["received_care_level"] = received_level[selected_population]
        learner["observed_outcome"] = observed[selected_population]
        learner["split"] = split[selected_population]

        truth = pd.DataFrame({
            "patient_id": cohort_with_level.patient_id.astype(str).to_numpy()[selected_population],
            "true_benefit": effects[name][selected_population],
            "true_latent_rank": effects[name][selected_population],
            "true_propensity": propensity[selected_population],
            "potential_outcome_lower": potential_outcomes[lower][selected_population],
            "potential_outcome_upper": potential_outcomes[upper][selected_population],
            "weighted_shared_component": weighted_shared[name][selected_population],
            "weighted_specific_component": weighted_specific[name][selected_population],
            "transition": name,
        })
        learners[name] = learner.reset_index(drop=True)
        truths[name] = truth.reset_index(drop=True)
        effect = effects[name][selected_population]
        metadata[name] = {
            "lower_level": lower,
            "upper_level": upper,
            "capacity": capacity,
            "available_baseline_n": int(len(candidates)),
            "target_sample_n": int(target),
            "realized_sample_n": int(len(selected_population)),
            "treatment_rate": float(treatment[selected_population].mean()),
            "true_effect_mean": float(effect.mean()),
            "true_effect_sd": float(effect.std()),
            "true_effect_negative_fraction": float(np.mean(effect < -1e-8)),
            "true_effect_near_zero_fraction": float(np.mean(np.abs(effect) <= 0.01)),
            "true_effect_positive_fraction": float(np.mean(effect > 1e-8)),
            "true_low_support_rate": float(np.mean(
                (propensity[selected_population] < 0.05)
                | (propensity[selected_population] > 0.95)
            )),
            "scenario": scenario,
            "shared_effect_correlation": float(shared_effect_correlation),
            "overlap_strength": float(overlap_strength),
            "risk_benefit_alignment": float(risk_benefit_alignment),
            "transition_imbalance_rho": float(transition_imbalance_rho),
            "conditional_exchangeability_by_construction": not hidden_confounding,
            "splits": learner.split.value_counts().to_dict(),
            "initial_state": initial_state_metadata,
        }

    return learners, truths, truth_all, clinical, metadata, feature_columns, split
