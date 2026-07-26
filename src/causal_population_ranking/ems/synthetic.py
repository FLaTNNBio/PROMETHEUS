"""Semi-synthetic mission-level DGP calibrated to aggregate EMS margins."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .calibration import EMSCalibration


LEARNER_FEATURE_COLUMNS = (
    "municipality",
    "pathology_code",
    "dispatch_severity_code",
    "severity_score",
    "event_type",
    "simulated_hour",
    "simulated_weekend",
    "simulated_age",
    "simulated_female",
    "simulated_frailty",
    "simulated_distance_km",
    "simulated_system_load",
    "simulated_medicalized_available",
)

RESPONSE_SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "risk_benefit_aligned": {
        "description": "Risk and incremental response are strongly aligned.",
        "optimal_risk": 0.72,
        "risk_window_width": 0.34,
        "nurse": {
            "intercept": 0.10, "severity": 0.22, "frailty": 0.34,
            "distance": 0.004, "pathology": 0.10, "latent": 0.12,
            "risk_linear": 1.15, "risk_window": 0.10,
            "high_risk_penalty": 0.00, "high_risk_threshold": 0.80,
            "minimum_effect": 0.00, "maximum_effect": 3.25,
            "non_responder_fraction": 0.00, "negative_fraction": 0.00,
            "null_sd": 0.03, "negative_penalty_min": 0.25,
            "negative_penalty_max": 0.75,
        },
        "medicalized": {
            "intercept": -0.15, "severity": 0.34, "frailty": 0.42,
            "distance": 0.006, "pathology": 0.18, "latent": 0.20,
            "risk_linear": 1.55, "risk_window": 0.15,
            "high_risk_penalty": 0.00, "high_risk_threshold": 0.84,
            "minimum_effect": 0.00, "maximum_effect": 4.50,
            "non_responder_fraction": 0.00, "negative_fraction": 0.00,
            "null_sd": 0.03, "negative_penalty_min": 0.35,
            "negative_penalty_max": 1.00,
        },
    },
    "partially_aligned": {
        "description": "Risk is informative but not sufficient for response.",
        "optimal_risk": 0.62,
        "risk_window_width": 0.24,
        "nurse": {
            "intercept": 0.12, "severity": 0.12, "frailty": 0.18,
            "distance": 0.003, "pathology": 0.17, "latent": 0.28,
            "risk_linear": 0.35, "risk_window": 0.80,
            "high_risk_penalty": 0.35, "high_risk_threshold": 0.82,
            "minimum_effect": -0.50, "maximum_effect": 3.00,
            "non_responder_fraction": 0.10, "negative_fraction": 0.03,
            "null_sd": 0.05, "negative_penalty_min": 0.20,
            "negative_penalty_max": 0.65,
        },
        "medicalized": {
            "intercept": -0.05, "severity": 0.18, "frailty": 0.20,
            "distance": 0.004, "pathology": 0.26, "latent": 0.38,
            "risk_linear": 0.45, "risk_window": 1.10,
            "high_risk_penalty": 0.55, "high_risk_threshold": 0.82,
            "minimum_effect": -0.75, "maximum_effect": 4.00,
            "non_responder_fraction": 0.12, "negative_fraction": 0.05,
            "null_sd": 0.06, "negative_penalty_min": 0.30,
            "negative_penalty_max": 0.90,
        },
    },
    "risk_benefit_misaligned": {
        "description": "Maximum response occurs at intermediate risk; extreme risk is less modifiable.",
        "optimal_risk": 0.38,
        "risk_window_width": 0.12,
        "nurse": {
            "intercept": 0.18, "severity": -0.04, "frailty": 0.02,
            "distance": 0.001, "pathology": 0.24, "latent": 0.42,
            "risk_linear": -1.25, "risk_window": 1.75,
            "high_risk_penalty": 7.50, "high_risk_threshold": 0.46,
            "minimum_effect": -1.10, "maximum_effect": 3.25,
            "non_responder_fraction": 0.12, "negative_fraction": 0.08,
            "null_sd": 0.06, "negative_penalty_min": 0.30,
            "negative_penalty_max": 1.00,
        },
        "medicalized": {
            "intercept": 0.12, "severity": -0.06, "frailty": 0.02,
            "distance": 0.001, "pathology": 0.34, "latent": 0.55,
            "risk_linear": -1.55, "risk_window": 2.25,
            "high_risk_penalty": 9.00, "high_risk_threshold": 0.46,
            "minimum_effect": -1.50, "maximum_effect": 4.25,
            "non_responder_fraction": 0.15, "negative_fraction": 0.10,
            "null_sd": 0.07, "negative_penalty_min": 0.40,
            "negative_penalty_max": 1.20,
        },
    },
    "mixed_response": {
        "description": "Mixture of responders, near-null missions and negative net effects.",
        "optimal_risk": 0.60,
        "risk_window_width": 0.22,
        "nurse": {
            "intercept": 0.10, "severity": 0.08, "frailty": 0.12,
            "distance": 0.002, "pathology": 0.22, "latent": 0.35,
            "risk_linear": 0.10, "risk_window": 1.00,
            "high_risk_penalty": 0.80, "high_risk_threshold": 0.78,
            "minimum_effect": -1.20, "maximum_effect": 3.00,
            "non_responder_fraction": 0.35, "negative_fraction": 0.12,
            "null_sd": 0.05, "negative_penalty_min": 0.30,
            "negative_penalty_max": 1.00,
        },
        "medicalized": {
            "intercept": -0.05, "severity": 0.10, "frailty": 0.14,
            "distance": 0.003, "pathology": 0.30, "latent": 0.45,
            "risk_linear": 0.10, "risk_window": 1.35,
            "high_risk_penalty": 1.10, "high_risk_threshold": 0.78,
            "minimum_effect": -1.60, "maximum_effect": 4.00,
            "non_responder_fraction": 0.40, "negative_fraction": 0.15,
            "null_sd": 0.06, "negative_penalty_min": 0.40,
            "negative_penalty_max": 1.20,
        },
    },
}


def _deep_update(target: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _response_scenario(simulation: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(simulation.get("response_scenario", "risk_benefit_aligned"))
    if name not in RESPONSE_SCENARIO_PRESETS:
        raise ValueError(
            f"Unknown EMS response scenario {name!r}; expected one of "
            f"{sorted(RESPONSE_SCENARIO_PRESETS)}"
        )
    settings = deepcopy(RESPONSE_SCENARIO_PRESETS[name])
    overrides = simulation.get("response_effect_overrides", {})
    if overrides:
        _deep_update(settings, overrides)
    return name, settings


def _generate_increment(
    *,
    tier_settings: Mapping[str, Any],
    scenario_settings: Mapping[str, Any],
    severity: np.ndarray,
    frailty: np.ndarray,
    distance: np.ndarray,
    pathology: np.ndarray,
    latent_response: np.ndarray,
    acute_risk: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    optimal_risk = float(scenario_settings["optimal_risk"])
    width = max(float(scenario_settings["risk_window_width"]), 1e-3)
    response_window = np.exp(
        -0.5 * ((acute_risk - optimal_risk) / width) ** 2
    )
    high_excess = np.maximum(
        acute_risk - float(tier_settings["high_risk_threshold"]),
        0.0,
    )
    raw = (
        float(tier_settings["intercept"])
        + float(tier_settings["severity"]) * severity
        + float(tier_settings["frailty"]) * frailty
        + float(tier_settings["distance"]) * distance
        + float(tier_settings["pathology"]) * pathology
        + float(tier_settings["latent"]) * latent_response
        + float(tier_settings["risk_linear"]) * (acute_risk - 0.5)
        + float(tier_settings["risk_window"]) * response_window
        - float(tier_settings["high_risk_penalty"]) * high_excess ** 2
    )
    effect = np.clip(
        raw,
        float(tier_settings["minimum_effect"]),
        float(tier_settings["maximum_effect"]),
    )
    non_responder = rng.random(len(effect)) < float(
        tier_settings["non_responder_fraction"]
    )
    if non_responder.any():
        effect[non_responder] = rng.normal(
            0.0,
            float(tier_settings["null_sd"]),
            int(non_responder.sum()),
        )
    negative = (
        ~non_responder
        & (rng.random(len(effect)) < float(tier_settings["negative_fraction"]))
    )
    if negative.any():
        effect[negative] -= rng.uniform(
            float(tier_settings["negative_penalty_min"]),
            float(tier_settings["negative_penalty_max"]),
            int(negative.sum()),
        )
    return np.clip(
        effect,
        float(tier_settings["minimum_effect"]),
        float(tier_settings["maximum_effect"]),
    )


def _effect_distribution(effect: np.ndarray, *, tolerance: float = 0.10) -> dict[str, float]:
    return {
        "mean": float(np.mean(effect)),
        "std": float(np.std(effect)),
        "positive_fraction": float(np.mean(effect > tolerance)),
        "near_zero_fraction": float(np.mean(np.abs(effect) <= tolerance)),
        "negative_fraction": float(np.mean(effect < -tolerance)),
        "minimum": float(np.min(effect)),
        "maximum": float(np.max(effect)),
    }



@dataclass(frozen=True)
class EMSSyntheticCohort:
    """Learner-safe observed data and physically separate simulated truth."""

    learner_data: pd.DataFrame
    evaluation_only: pd.DataFrame
    audit: dict[str, Any]


def _required_distinct_seeds(config: Mapping[str, Any]) -> dict[str, int]:
    raw = config["simulation"]["seeds"]
    seeds = {str(name): int(value) for name, value in raw.items()}
    if len(seeds) != len(set(seeds.values())):
        raise ValueError("Every EMS simulation component requires a distinct seed")
    return seeds


def _sample_counted_categories(
    counts: pd.Series,
    *,
    size: int,
    seed: int,
) -> np.ndarray:
    labels = counts.index.astype(str).to_numpy()
    values = counts.astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    if size == int(values.sum()):
        sampled = np.repeat(labels, values)
        rng.shuffle(sampled)
        return sampled
    probabilities = values / values.sum()
    return rng.choice(labels, size=size, replace=True, p=probabilities)


def _sample_joint_demand(
    calibration: EMSCalibration,
    *,
    size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = calibration.municipality_pathology
    pathology_columns = [
        column for column in frame.columns
        if column.startswith("C") and column != "COMUNE"
    ]
    labels = []
    counts = []
    for row in frame.itertuples(index=False):
        municipality = str(row.municipality)
        for pathology in pathology_columns:
            count = int(getattr(row, pathology))
            if count:
                labels.append((municipality, pathology))
                counts.append(count)
    rng = np.random.default_rng(seed)
    if size == sum(counts):
        sampled = np.repeat(np.arange(len(labels)), counts)
        rng.shuffle(sampled)
    else:
        probabilities = np.asarray(counts, dtype=float)
        probabilities /= probabilities.sum()
        sampled = rng.choice(
            len(labels),
            size=size,
            replace=True,
            p=probabilities,
        )
    municipalities = np.asarray([labels[index][0] for index in sampled])
    pathologies = np.asarray([labels[index][1] for index in sampled])
    return municipalities, pathologies


def _softplus(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return np.log1p(np.exp(clipped))


def _sample_treatment(
    features: pd.DataFrame,
    *,
    pathology_assignment: dict[str, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    severity = features["severity_score"].to_numpy(float)
    load = features["simulated_system_load"].to_numpy(float)
    distance = features["simulated_distance_km"].to_numpy(float)
    available = features["simulated_medicalized_available"].to_numpy(bool)
    pathology = features["pathology_code"].map(pathology_assignment).to_numpy(float)
    frailty = features["simulated_frailty"].to_numpy(float)

    logits = np.column_stack([
        np.zeros(len(features)),
        -0.75 + 0.52 * severity + 0.24 * frailty + 0.012 * distance
        - 0.18 * load,
        -2.25 + 0.90 * severity + 0.32 * frailty + 0.22 * pathology
        + 0.010 * distance - 0.38 * load + np.where(available, 0.55, -3.5),
    ])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    uniforms = rng.random(len(features))
    treatment = (uniforms[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    treatment = np.minimum(treatment, 2).astype("int64")
    return treatment, probabilities


def _assign_splits(
    mission_ids: np.ndarray,
    fractions: Mapping[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    names = (
        "nuisance_train",
        "rank_train",
        "validation",
        "calibration",
        "test",
    )
    values = np.asarray([float(fractions[name]) for name in names])
    if not np.isclose(values.sum(), 1.0) or np.any(values <= 0.0):
        raise ValueError("EMS split fractions must be positive and sum to one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mission_ids))
    boundaries = np.rint(np.cumsum(values) * len(mission_ids)).astype(int)
    boundaries[-1] = len(mission_ids)
    split = np.empty(len(mission_ids), dtype=object)
    start = 0
    for name, end in zip(names, boundaries):
        split[order[start:end]] = name
        start = end
    return split.astype(str)


def generate_ems_semisynthetic_cohort(
    calibration: EMSCalibration,
    config: Mapping[str, Any],
    *,
    missions_override: int | None = None,
) -> EMSSyntheticCohort:
    """Generate learner-safe EMS missions plus isolated simulated truth."""

    seeds = _required_distinct_seeds(config)
    simulation = config["simulation"]
    size = (
        int(missions_override)
        if missions_override is not None
        else int(simulation["missions"])
    )
    if size < 400:
        raise ValueError("EMS semi-synthetic cohort requires at least 400 missions")

    municipality, pathology = _sample_joint_demand(
        calibration,
        size=size,
        seed=seeds["joint_demand_seed"],
    )
    severity_code = _sample_counted_categories(
        calibration.severity_counts,
        size=size,
        seed=seeds["severity_seed"],
    )
    event_type = _sample_counted_categories(
        calibration.event_counts,
        size=size,
        seed=seeds["event_type_seed"],
    )
    severity_order = {
        str(code): index
        for index, code in enumerate(simulation["severity_order"])
    }
    if set(severity_order) != set(calibration.severity_counts.index):
        raise ValueError("Configured severity order disagrees with source codes")
    severity_score = np.asarray(
        [severity_order[code] for code in severity_code],
        dtype=float,
    )

    clock_rng = np.random.default_rng(seeds["clock_seed"])
    hour_probabilities = np.asarray([
        0.022, 0.018, 0.015, 0.013, 0.012, 0.014,
        0.022, 0.035, 0.047, 0.052, 0.054, 0.055,
        0.055, 0.055, 0.057, 0.060, 0.061, 0.061,
        0.058, 0.053, 0.046, 0.040, 0.033, 0.027,
    ])
    hour_probabilities /= hour_probabilities.sum()
    simulated_hour = clock_rng.choice(
        np.arange(24),
        size=size,
        p=hour_probabilities,
    )
    simulated_weekend = clock_rng.random(size) < (2.0 / 7.0)
    simulated_arrival_order = clock_rng.permutation(size)

    patient_rng = np.random.default_rng(seeds["patient_feature_seed"])
    simulated_age = np.clip(
        patient_rng.normal(64.0 + 3.0 * (severity_score >= 2), 21.0, size),
        0.0,
        100.0,
    )
    simulated_female = patient_rng.random(size) < 0.53
    frailty_logit = (
        -2.1 + 0.035 * (simulated_age - 50.0)
        + 0.35 * (severity_score >= 2)
    )
    simulated_frailty = 1.0 / (1.0 + np.exp(-frailty_logit))

    municipality_volume = calibration.municipality_pathology.set_index(
        "municipality"
    )["total"]
    volume = pd.Series(municipality).map(municipality_volume).to_numpy(float)
    remoteness = 1.0 - np.log1p(volume) / np.log1p(volume.max())
    system_rng = np.random.default_rng(seeds["system_state_seed"])
    simulated_distance_km = np.clip(
        3.0 + 29.0 * remoteness + system_rng.gamma(2.0, 1.7, size),
        1.0,
        55.0,
    )
    peak = np.isin(simulated_hour, [8, 9, 10, 17, 18, 19]).astype(float)
    simulated_system_load = np.clip(
        system_rng.lognormal(mean=-0.25 + 0.28 * peak, sigma=0.45, size=size),
        0.2,
        4.5,
    )
    simulated_medicalized_available = (
        system_rng.random(size)
        < np.clip(0.82 - 0.12 * simulated_system_load, 0.25, 0.85)
    )

    mission_ids = np.asarray([f"ems_syn_{index:06d}" for index in range(size)])
    learner = pd.DataFrame({
        "mission_id": mission_ids,
        "municipality": municipality,
        "pathology_code": pathology,
        "dispatch_severity_code": severity_code,
        "severity_score": severity_score.astype("int64"),
        "event_type": event_type,
        "simulated_hour": simulated_hour,
        "simulated_weekend": simulated_weekend,
        "simulated_arrival_order": simulated_arrival_order,
        "simulated_age": simulated_age,
        "simulated_female": simulated_female,
        "simulated_frailty": simulated_frailty,
        "simulated_distance_km": simulated_distance_km,
        "simulated_system_load": simulated_system_load,
        "simulated_medicalized_available": simulated_medicalized_available,
    })

    pathology_codes = sorted(calibration.municipality_pathology.columns[
        calibration.municipality_pathology.columns.str.startswith("C")
    ])
    effect_rng = np.random.default_rng(seeds["effect_seed"])
    pathology_effect = dict(zip(
        pathology_codes,
        effect_rng.normal(0.0, 0.38, len(pathology_codes)),
    ))
    pathology_assignment = dict(zip(
        pathology_codes,
        effect_rng.normal(0.0, 0.30, len(pathology_codes)),
    ))
    pathology_term = learner.pathology_code.map(pathology_effect).to_numpy(float)
    latent_response = effect_rng.normal(0.0, 0.30, size)
    risk_logit = (
        -2.45 + 0.92 * severity_score + 1.05 * simulated_frailty
        + 0.018 * simulated_distance_km + 0.18 * pathology_term
    )
    acute_risk = 1.0 / (1.0 + np.exp(-risk_logit))
    common_noise = effect_rng.normal(0.0, 0.65, size)
    y0 = np.clip(30.0 - 8.5 * acute_risk + common_noise, 0.0, 30.0)
    response_scenario_name, response_settings = _response_scenario(simulation)
    india_increment_requested = _generate_increment(
        tier_settings=response_settings["nurse"],
        scenario_settings=response_settings,
        severity=severity_score,
        frailty=simulated_frailty,
        distance=simulated_distance_km,
        pathology=pathology_term,
        latent_response=latent_response,
        acute_risk=acute_risk,
        rng=effect_rng,
    )
    mike_increment_requested = _generate_increment(
        tier_settings=response_settings["medicalized"],
        scenario_settings=response_settings,
        severity=severity_score,
        frailty=simulated_frailty,
        distance=simulated_distance_km,
        pathology=pathology_term,
        latent_response=latent_response,
        acute_risk=acute_risk,
        rng=effect_rng,
    )
    y1 = np.clip(y0 + india_increment_requested, 0.0, 30.0)
    y2 = np.clip(y1 + mike_increment_requested, 0.0, 30.0)
    india_increment = y1 - y0
    mike_increment = y2 - y1

    assigned_tier, propensity = _sample_treatment(
        learner,
        pathology_assignment=pathology_assignment,
        seed=seeds["assignment_seed"],
    )
    potential = np.column_stack([y0, y1, y2])
    outcome_rng = np.random.default_rng(seeds["outcome_seed"])
    observed_outcome = np.clip(
        potential[np.arange(size), assigned_tier]
        + outcome_rng.normal(0.0, 0.85, size),
        0.0,
        30.0,
    )
    learner["assigned_tier"] = assigned_tier
    learner["observed_outcome"] = observed_outcome
    learner["split"] = _assign_splits(
        mission_ids,
        simulation["split_fractions"],
        seed=seeds["split_seed"],
    )

    assessment_totals = calibration.vehicle_assessment[
        ["0", "1", "2", "3", "4", "SENZA VALUTAZIONE"]
    ].sum()
    assessment = _sample_counted_categories(
        assessment_totals,
        size=size,
        seed=seeds["assessment_seed"],
    )
    evaluation_only = pd.DataFrame({
        "mission_id": mission_ids,
        "potential_outcome_basic": y0,
        "potential_outcome_nurse_supported": y1,
        "potential_outcome_medicalized": y2,
        "true_increment_nurse_supported": y1 - y0,
        "true_increment_medicalized": y2 - y1,
        "true_total_increment_medicalized": y2 - y0,
        "response_scenario": response_scenario_name,
        "simulated_post_response_assessment": assessment,
        "latent_acute_risk": acute_risk,
        "oracle_assigned_propensity": propensity[
            np.arange(size), assigned_tier
        ],
    })

    forbidden = (
        "true_",
        "oracle_",
        "potential_outcome",
        "latent_",
        "simulated_post_response",
    )
    leaked = [
        column for column in learner.columns
        if column.startswith(forbidden)
    ]
    if leaked:
        raise AssertionError(f"Evaluation-only EMS fields leaked to learner data: {leaked}")
    split_counts = learner.split.value_counts().to_dict()
    audit = {
        "status": "generated",
        "source_data_level": "aggregate_only",
        "source_intervention_total": calibration.intervention_total,
        "missions_generated": size,
        "uses_exact_joint_municipality_pathology_counts": (
            size == calibration.intervention_total
        ),
        "severity_and_event_coupling": "seeded_independent_marginal_coupling",
        "vehicle_tier_assignment": "simulated_not_observed_mission_level",
        "arrival_order": "simulated_predispatch_fcfs_benchmark_only",
        "potential_outcomes": "simulated_evaluation_only",
        "response_scenario": response_scenario_name,
        "response_scenario_description": response_settings["description"],
        "risk_benefit_spearman": {
            "nurse_supported": float(pd.Series(acute_risk).corr(
                pd.Series(india_increment), method="spearman"
            )),
            "medicalized_total": float(pd.Series(acute_risk).corr(
                pd.Series(y2 - y0), method="spearman"
            )),
        },
        "effect_distribution": {
            "nurse_supported": _effect_distribution(india_increment),
            "medicalized_incremental": _effect_distribution(mike_increment),
            "medicalized_total": _effect_distribution(y2 - y0),
        },
        "outcome_name": simulation["outcome_name"],
        "priority_score_semantics": "ordinal_not_calibrated_effect",
        "individual_cate_estimated_then_sorted": False,
        "oracle_used_for_training": False,
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "assigned_tier_counts": {
            str(key): int(value)
            for key, value in learner.assigned_tier.value_counts().sort_index().items()
        },
        "component_seeds": seeds,
    }
    return EMSSyntheticCohort(
        learner_data=learner,
        evaluation_only=evaluation_only,
        audit=audit,
    )
