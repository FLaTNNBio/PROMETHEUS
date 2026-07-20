"""Fully synthetic care-profile DGP with physically separate evaluation truth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from causal_population_ranking.dm77.need import cross_fit_baseline_need
from causal_population_ranking.dm77.opportunities import (
    build_patient_profile_opportunities,
    derive_care_profile_eligibility,
)
from causal_population_ranking.dm77.care_profiles import CareProfileCatalog

from .need_reference import generate_synthetic_need_reference


PROFILE_DGP_VERSION = "prometheus_synthetic_care_profile_dgp_v1"
NO_NEW_PROFILE = "no_new_profile"
PROFILE_DGP_SCENARIOS = (
    "baseline_identifiable",
    "need_current_alignment",
    "unmet_need",
    "over_intensive_care",
    "risk_benefit_aligned",
    "risk_benefit_misaligned",
    "no_shared_response",
    "strong_observed_confounding",
    "poor_overlap",
    "sharp_null",
    "placebo_outcome",
    "hidden_confounding",
    "capacity_scarcity",
    "heterogeneous_profile_costs",
)


@dataclass(frozen=True)
class SyntheticProfileDGPResult:
    learner: pd.DataFrame
    baseline_need_assessments: pd.DataFrame
    need_reference_labels: pd.DataFrame
    current_care_profiles: pd.DataFrame
    profile_eligibility: pd.DataFrame
    patient_profile_opportunities: pd.DataFrame
    profile_resources: pd.DataFrame
    resource_capacities: pd.DataFrame
    need_ground_truth: pd.DataFrame
    profile_ground_truth: pd.DataFrame
    metadata: dict
    audit: dict


def _z(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (array - array.mean()) / max(float(array.std()), 1e-8)


def _sigmoid(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -30.0, 30.0)))


def _profile_for_level(
    level: int,
    row: pd.Series,
    catalog: CareProfileCatalog,
) -> str:
    by_level: dict[int, list[str]] = {}
    for profile in catalog.profiles:
        if profile.protected:
            continue
        by_level.setdefault(profile.care_profile_level, []).append(profile.care_profile_id)
    if level != 4:
        values = by_level.get(level, [])
        if len(values) != 1:
            raise ValueError(f"Synthetic current-care DGP requires one ordinary level-{level} profile")
        return values[0]
    if float(row.social_fragility_score) >= 0.48 or not bool(row.caregiver_available):
        return "profile_iv_multidisciplinary_coordination"
    if not bool(row.housing_instability) and float(row.functional_limitation_score) < 0.72:
        return "profile_iv_remote_monitoring"
    return "profile_iv_proactive_case_management"


def generate_exact_current_care_profiles(
    patients: pd.DataFrame,
    baseline_need_assessments: pd.DataFrame,
    catalog: CareProfileCatalog,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    """Generate an exact profile ID directly, never through an ambiguous state map."""

    if seed is None:
        raise ValueError("Current-profile generation requires an explicit seed")
    if scenario not in PROFILE_DGP_SCENARIOS:
        raise ValueError(f"Unknown profile DGP scenario: {scenario}")
    required = {
        "patient_id", "age", "condition_distinct", "prior_inpatient",
        "frailty_index", "functional_limitation_score", "social_fragility_score",
        "caregiver_available", "housing_instability",
    }
    missing = sorted(required.difference(patients.columns))
    if missing:
        raise ValueError(f"Current-profile DGP is missing pre-index fields: {missing}")
    need_required = {"patient_id", "baseline_need_level"}
    if not need_required.issubset(baseline_need_assessments.columns):
        raise ValueError("Current-profile DGP requires the Phase-1 baseline-need output")
    context = patients.merge(
        baseline_need_assessments[["patient_id", "baseline_need_level"]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    if len(context) != len(patients):
        raise ValueError("Current-profile DGP requires complete baseline-need coverage")
    rng = np.random.default_rng(int(seed))
    need = context.baseline_need_level.fillna(3).astype(float).to_numpy()
    need = np.clip(need, 1.0, 5.0)
    observed_risk = _z(
        0.025 * context.age.to_numpy(float)
        + 0.55 * np.log1p(context.condition_distinct.to_numpy(float))
        + 0.70 * np.log1p(context.prior_inpatient.to_numpy(float))
        + 1.10 * context.frailty_index.to_numpy(float)
        + 0.75 * context.functional_limitation_score.to_numpy(float)
    )
    risk_level = 1.0 + 4.0 * _sigmoid(observed_risk)
    if scenario == "need_current_alignment":
        latent_level = need + rng.normal(0.0, 0.28, len(context))
    elif scenario == "unmet_need":
        latent_level = need - 1.45 + rng.normal(0.0, 0.45, len(context))
    elif scenario == "over_intensive_care":
        latent_level = need + 1.45 + rng.normal(0.0, 0.45, len(context))
    else:
        latent_level = (
            0.65 * need + 0.35 * risk_level
            + rng.normal(0.0, 0.85, len(context))
        )
    current_level = np.clip(np.rint(latent_level), 1, 5).astype(int)
    rows = []
    for position, (_, row) in enumerate(context.iterrows()):
        profile_id = _profile_for_level(int(current_level[position]), row, catalog)
        profile = catalog.profile_by_id[profile_id]
        rows.append({
            "patient_id": str(row.patient_id),
            "current_care_profile": profile_id,
            "current_care_profile_level": profile.care_profile_level,
            "current_care_state": profile.resulting_care_state,
            "active_services": "|".join(
                catalog.action_catalog.active_services_by_state[
                    profile.resulting_care_state
                ]
            ),
            "current_care_profile_status": "generated_exact_profile",
            "current_care_profile_reason_codes": (
                f"SYNTHETIC_EXACT_CURRENT_PROFILE:{scenario}"
            ),
            "current_care_profile_manual_review": False,
            "profile_catalog_version": catalog.catalog_version,
        })
    result = pd.DataFrame(rows)
    result["current_care_profile_level"] = result.current_care_profile_level.astype("Int64")
    return result


def _profile_resources(
    catalog: CareProfileCatalog,
    population_size: int,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    multipliers = {}
    rows = []
    for profile in catalog.profiles:
        if scenario == "heterogeneous_profile_costs" and not profile.maintenance_reference_only:
            multiplier = 0.65 + 0.22 * profile.care_profile_index
        else:
            multiplier = 1.0
        multipliers[profile.care_profile_id] = float(multiplier)
        rows.append({
            "care_profile_id": profile.care_profile_id,
            "care_profile_level": profile.care_profile_level,
            "base_full_profile_cost": profile.resource_cost,
            "scenario_cost_multiplier": float(multiplier),
            "scenario_full_profile_cost": float(profile.resource_cost * multiplier),
            "primary_capacity_pool": profile.capacity_pool,
            "resource_scenario": scenario,
            "profile_catalog_version": catalog.catalog_version,
        })
    pools = sorted({
        pool
        for profile in catalog.profiles
        for pool in profile.capacity_requirements
    })
    fraction = 0.035 if scenario == "capacity_scarcity" else 0.22
    capacities = pd.DataFrame({
        "capacity_pool": pools,
        "available_units": [max(1, int(np.floor(population_size * fraction))) for _ in pools],
        "capacity_fraction_of_population": fraction,
        "capacity_scarcity": scenario == "capacity_scarcity",
        "resource_scenario": scenario,
    })
    return pd.DataFrame(rows), capacities, multipliers


def _profile_effects(
    patients: pd.DataFrame,
    catalog: CareProfileCatalog,
    scenario: str,
    seed: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    if seed is None:
        raise ValueError("Profile-effect generation requires an explicit seed")
    rng = np.random.default_rng(int(seed))
    n = len(patients)
    profiles = list(catalog.automatic_rank_profiles)
    profile_ids = [profile.care_profile_id for profile in profiles]
    expected_profile_ids = [
        "profile_ii_structured_followup",
        "profile_iii_chronic_management",
        "profile_iv_proactive_case_management",
        "profile_iv_remote_monitoring",
        "profile_iv_multidisciplinary_coordination",
        "profile_v_integrated_home_support",
    ]
    if profile_ids != expected_profile_ids:
        raise ValueError(
            "Profile DGP v1 equations require the frozen automatic profile order"
        )
    risk = _z(
        0.030 * patients.age.to_numpy(float)
        + 0.55 * np.log1p(patients.condition_distinct.to_numpy(float))
        + 0.85 * np.log1p(patients.prior_inpatient.to_numpy(float))
        + 0.45 * np.log1p(patients.prior_emergency.to_numpy(float))
        + 1.25 * patients.frailty_index.to_numpy(float)
    )
    chronicity = _z(np.log1p(patients.condition_distinct.to_numpy(float)))
    utilization = _z(
        np.log1p(
            patients.prior_inpatient.to_numpy(float)
            + patients.prior_emergency.to_numpy(float)
        )
    )
    functional = _z(patients.functional_limitation_score.to_numpy(float))
    social = _z(patients.social_fragility_score.to_numpy(float))
    caregiver = _z(patients.caregiver_available.to_numpy(float))
    stable_housing = _z(1.0 - patients.housing_instability.to_numpy(float))
    latent_hidden = rng.normal(0.0, 1.0, n)
    if scenario == "no_shared_response":
        shared = np.zeros(n)
    elif scenario == "risk_benefit_aligned":
        shared = 1.35 * risk
    elif scenario == "risk_benefit_misaligned":
        shared = -1.35 * risk + 0.25 * social
    else:
        shared = 0.55 * chronicity + 0.35 * utilization + 0.20 * functional
    profile_specific = np.column_stack([
        0.45 * utilization + 0.25 * caregiver,
        0.75 * chronicity + 0.20 * utilization,
        0.65 * utilization + 0.60 * social,
        0.55 * chronicity + 0.60 * stable_housing + 0.20 * caregiver,
        0.45 * chronicity + 0.75 * social - 0.15 * caregiver,
        0.75 * functional + 0.55 * social - 0.20 * caregiver,
    ])
    if profile_specific.shape[1] != len(profiles):
        raise ValueError("Profile-effect equation must match automatic catalogue profiles")
    base = np.asarray([2.4, 3.6, 4.5, 4.2, 5.0, 5.8], dtype=float)
    idiosyncratic = rng.normal(0.0, 0.38, (n, len(profiles)))
    primary_effect = base[None, :] + shared[:, None] + profile_specific + idiosyncratic
    if scenario == "hidden_confounding":
        primary_effect += 0.85 * latent_hidden[:, None]
    primary_effect = np.clip(primary_effect, -8.0, 18.0)
    if scenario == "sharp_null":
        primary_effect[:] = 0.0
    analysis_effect = (
        np.zeros_like(primary_effect)
        if scenario == "placebo_outcome"
        else primary_effect.copy()
    )
    return (
        primary_effect, analysis_effect, risk, latent_hidden,
        shared, profile_specific, profile_ids,
    )


def _assignment_probabilities(
    patients: pd.DataFrame,
    opportunities: pd.DataFrame,
    profile_ids: list[str],
    catalog: CareProfileCatalog,
    cost_multipliers: dict[str, float],
    latent_hidden: np.ndarray,
    scenario: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed is None:
        raise ValueError("Profile assignment requires an explicit seed")
    rng = np.random.default_rng(int(seed))
    n, profile_count = len(patients), len(profile_ids)
    if profile_ids != [
        profile.care_profile_id for profile in catalog.automatic_rank_profiles
    ]:
        raise ValueError("Assignment profile order must match the versioned catalogue")
    patient_position = {str(value): index for index, value in enumerate(patients.patient_id)}
    profile_position = {value: index for index, value in enumerate(profile_ids)}
    eligible = np.zeros((n, profile_count), dtype=bool)
    incremental_cost = np.zeros((n, profile_count), dtype=float)
    for row in opportunities.itertuples(index=False):
        i = patient_position[str(row.patient_id)]
        j = profile_position[str(row.care_profile_id)]
        eligible[i, j] = True
        incremental_cost[i, j] = float(row.incremental_resource_cost) * cost_multipliers[
            str(row.care_profile_id)
        ]
    severity = _z(
        0.60 * np.log1p(patients.condition_distinct.to_numpy(float))
        + 0.75 * np.log1p(patients.prior_inpatient.to_numpy(float))
        + 0.40 * np.log1p(patients.prior_emergency.to_numpy(float))
        + 0.90 * patients.frailty_index.to_numpy(float)
        + 0.45 * patients.social_fragility_score.to_numpy(float)
    )
    observed_modifier = np.column_stack([
        0.25 * _z(patients.prior_emergency.to_numpy(float)),
        0.35 * _z(patients.condition_distinct.to_numpy(float)),
        0.30 * _z(patients.prior_inpatient.to_numpy(float)) + 0.20 * _z(patients.social_fragility_score),
        0.25 * _z(patients.condition_distinct.to_numpy(float)) - 0.25 * _z(patients.housing_instability),
        0.35 * _z(patients.social_fragility_score) - 0.15 * _z(patients.caregiver_available),
        0.40 * _z(patients.functional_limitation_score) + 0.20 * _z(patients.social_fragility_score),
    ])
    if observed_modifier.shape[1] != profile_count:
        raise ValueError("Assignment equation must match automatic catalogue profiles")
    scale = 1.0
    exploration = 0.10
    if scenario == "strong_observed_confounding":
        scale, exploration = 2.8, 0.035
    elif scenario == "poor_overlap":
        scale, exploration = 5.0, 0.01
    logits = (
        -0.12 * incremental_cost
        + scale * (0.30 * severity[:, None] + observed_modifier)
    )
    if scenario == "hidden_confounding":
        hidden_direction = np.linspace(-0.8, 0.8, profile_count)
        logits += 1.65 * latent_hidden[:, None] * hidden_direction[None, :]
    if scenario == "capacity_scarcity":
        logits -= 0.25 * incremental_cost
    no_new_logit = 0.35 - scale * 0.16 * severity
    probabilities = np.zeros((n, profile_count), dtype=float)
    no_new_probability = np.ones(n, dtype=float)
    observed_index = np.full(n, -1, dtype=int)
    for i in range(n):
        available = np.flatnonzero(eligible[i])
        options = np.r_[-1, available]
        option_logits = np.r_[no_new_logit[i], logits[i, available]]
        option_logits -= option_logits.max()
        softmax = np.exp(option_logits)
        softmax /= softmax.sum()
        mixed = (1.0 - exploration) * softmax + exploration / len(options)
        no_new_probability[i] = mixed[0]
        probabilities[i, available] = mixed[1:]
        observed_index[i] = int(rng.choice(options, p=mixed))
    return probabilities, no_new_probability, observed_index


def generate_synthetic_profile_dgp(
    patients: pd.DataFrame,
    catalog: CareProfileCatalog,
    scenario: str,
    need_mode: str,
    reference_seed: int,
    need_model_seed: int,
    need_split_seed: int,
    current_care_seed: int,
    effect_seed: int,
    assignment_seed: int,
    outcome_seed: int,
    baseline_need_contract: str | Path | Mapping | None = None,
    index_date: str = "2025-01-01",
) -> SyntheticProfileDGPResult:
    """Generate one learner dataset and separate need/causal truth tables."""

    if scenario not in PROFILE_DGP_SCENARIOS:
        raise ValueError(
            f"Unknown profile DGP scenario {scenario!r}; choose from {PROFILE_DGP_SCENARIOS}"
        )
    seeds = {
        "reference_seed": reference_seed,
        "need_model_seed": need_model_seed,
        "need_split_seed": need_split_seed,
        "current_care_seed": current_care_seed,
        "effect_seed": effect_seed,
        "assignment_seed": assignment_seed,
        "outcome_seed": outcome_seed,
    }
    if any(value is None for value in seeds.values()):
        raise ValueError("Every profile DGP stochastic component requires an explicit seed")
    if len(patients) < 100:
        raise ValueError("Profile DGP requires at least 100 synthetic patients")
    identifiers = patients.patient_id.astype(str)
    if identifiers.duplicated().any():
        raise ValueError("Profile DGP requires unique patients")

    need_reference = generate_synthetic_need_reference(
        patients, seed=int(reference_seed)
    )
    need_frame = patients.merge(
        need_reference.learner_labels,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    need_prediction = cross_fit_baseline_need(
        need_frame if need_mode == "supervised" else patients,
        mode=need_mode,
        model_seed=int(need_model_seed),
        split_seed=int(need_split_seed),
        contract=baseline_need_contract,
    )
    current = generate_exact_current_care_profiles(
        patients,
        need_prediction.assessments,
        catalog,
        scenario=scenario,
        seed=int(current_care_seed),
    )
    eligibility, eligibility_audit = derive_care_profile_eligibility(
        patients, need_prediction.assessments, current, catalog
    )
    opportunities = build_patient_profile_opportunities(
        eligibility, require_empirical_support=False
    )
    profile_resources, capacities, cost_multipliers = _profile_resources(
        catalog, len(patients), scenario
    )
    multiplier_series = eligibility.care_profile_id.map(cost_multipliers).astype(float)
    eligibility = eligibility.copy()
    opportunities = opportunities.copy()
    eligibility["scenario_incremental_resource_cost"] = (
        eligibility.incremental_resource_cost * multiplier_series
    )
    opportunities["scenario_incremental_resource_cost"] = (
        opportunities.incremental_resource_cost
        * opportunities.care_profile_id.map(cost_multipliers).astype(float)
    )

    (
        primary_effect, analysis_effect, risk, latent_hidden,
        shared_response, profile_specific_response, profile_ids,
    ) = _profile_effects(patients, catalog, scenario, int(effect_seed))
    assignment_probability, no_new_probability, observed_index = _assignment_probabilities(
        patients,
        opportunities,
        profile_ids,
        catalog,
        cost_multipliers,
        latent_hidden,
        scenario,
        int(assignment_seed),
    )
    outcome_rng = np.random.default_rng(int(outcome_seed))
    baseline_primary_raw = (
        338.0 - 13.0 * risk
        - (4.5 * latent_hidden if scenario == "hidden_confounding" else 0.0)
        + outcome_rng.normal(0.0, 5.0, len(patients))
    )
    baseline_primary = np.clip(baseline_primary_raw, 0.0, 365.0)
    primary_potential = np.clip(
        baseline_primary_raw[:, None] + primary_effect, 0.0, 365.0
    )
    realized_primary_effect = primary_potential - baseline_primary[:, None]
    if scenario == "placebo_outcome":
        analysis_baseline = (
            50.0 + 2.0 * _z(patients.age.to_numpy(float))
            + 1.5 * _z(patients.social_fragility_score.to_numpy(float))
            + outcome_rng.normal(0.0, 6.0, len(patients))
        )
        analysis_potential = np.repeat(analysis_baseline[:, None], len(profile_ids), axis=1)
        analysis_name = "placebo_negative_control_measure_365d"
        analysis_unit = "synthetic_units"
    else:
        analysis_baseline = baseline_primary
        analysis_potential = np.clip(
            baseline_primary_raw[:, None] + analysis_effect, 0.0, 365.0
        )
        analysis_name = "days_alive_outside_acute_hospital_365d"
        analysis_unit = "days"
    realized_analysis_effect = analysis_potential - analysis_baseline[:, None]
    observed_profile = np.asarray([
        NO_NEW_PROFILE if index < 0 else profile_ids[index]
        for index in observed_index
    ], dtype=object)
    observed_outcome = analysis_baseline.copy()
    treated = observed_index >= 0
    observed_outcome[treated] = analysis_potential[
        np.flatnonzero(treated), observed_index[treated]
    ]

    need_columns = [
        "patient_id", "baseline_need_level", "baseline_need_score",
        "baseline_need_mode", "baseline_need_status", "baseline_need_model_version",
        "baseline_need_protected_pathway",
    ]
    current_columns = [
        "patient_id", "current_care_profile", "current_care_profile_level",
        "current_care_state",
    ]
    learner = (
        patients.copy()
        .merge(need_reference.learner_labels, on="patient_id", validate="one_to_one")
        .merge(need_prediction.assessments[need_columns], on="patient_id", validate="one_to_one")
        .merge(current[current_columns], on="patient_id", validate="one_to_one")
    )
    learner["index_date"] = str(pd.Timestamp(index_date).date())
    learner["observed_treatment_profile"] = observed_profile
    learner["protected_routing_profile"] = np.where(
        learner.baseline_need_protected_pathway.to_numpy(bool),
        "profile_vi_protected_palliative",
        pd.NA,
    )
    learner["observed_outcome"] = observed_outcome
    learner["outcome_name"] = analysis_name
    learner["outcome_horizon"] = 365
    learner["outcome_unit"] = analysis_unit
    learner["benefit_direction"] = "higher_is_better"
    learner["ranking_domain"] = (
        "placebo_negative_control_365d"
        if scenario == "placebo_outcome"
        else catalog.primary_ranking_domain
    )
    learner["profile_dgp_scenario"] = scenario
    learner["profile_dgp_version"] = PROFILE_DGP_VERSION

    truth_rows = []
    for i, patient_id in enumerate(identifiers):
        eligible_profiles = set(
            opportunities.loc[
                opportunities.patient_id.astype(str) == patient_id,
                "care_profile_id",
            ].astype(str)
        )
        eligible_indices = [j for j, value in enumerate(profile_ids) if value in eligible_profiles]
        ranks = {}
        if eligible_indices:
            ordered = sorted(
                eligible_indices,
                key=lambda j: (-realized_analysis_effect[i, j], profile_ids[j]),
            )
            ranks = {j: rank + 1 for rank, j in enumerate(ordered)}
        for j, profile_id in enumerate(profile_ids):
            truth_rows.append({
                "patient_id": patient_id,
                "care_profile_id": profile_id,
                "true_profile_benefit": float(realized_analysis_effect[i, j]),
                "true_primary_profile_benefit": float(realized_primary_effect[i, j]),
                "potential_outcome_no_new_profile": float(analysis_baseline[i]),
                "potential_outcome_profile": float(analysis_potential[i, j]),
                "potential_outcome_primary_profile": float(primary_potential[i, j]),
                "true_assignment_probability": float(assignment_probability[i, j]),
                "true_no_new_profile_probability": float(no_new_probability[i]),
                "oracle_profile_rank": ranks.get(j, pd.NA),
                "oracle_structurally_eligible": profile_id in eligible_profiles,
                "latent_observed_risk": float(risk[i]),
                "latent_hidden_confounder": float(latent_hidden[i]),
                "latent_shared_causal_response": float(shared_response[i]),
                "latent_profile_specific_response": float(
                    profile_specific_response[i, j]
                ),
                "evaluation_only": True,
            })
    profile_truth = pd.DataFrame(truth_rows)
    profile_truth["oracle_profile_rank"] = profile_truth.oracle_profile_rank.astype("Int64")

    need_truth = need_reference.ground_truth.copy()
    need_truth["evaluation_only"] = True
    need_truth["profile_dgp_scenario"] = scenario
    truth_need = need_truth.true_baseline_need_level.to_numpy(float)
    current_level = current.current_care_profile_level.to_numpy(float)
    eligible_probability = assignment_probability[assignment_probability > 0]
    patient_mean_benefit = realized_analysis_effect.mean(axis=1)
    risk_benefit_spearman = (
        None
        if float(patient_mean_benefit.std()) < 1e-12
        else float(
            pd.Series(risk).corr(
                pd.Series(patient_mean_benefit), method="spearman"
            )
        )
    )
    diagnostics = {
        "need_current_spearman": float(
            pd.Series(truth_need).corr(pd.Series(current_level), method="spearman")
        ),
        "mean_current_minus_true_need_clipped_to_five": float(
            np.mean(current_level - np.clip(truth_need, 1, 5))
        ),
        "treated_profile_fraction": float(np.mean(treated)),
        "no_new_profile_fraction": float(np.mean(~treated)),
        "mean_true_analysis_benefit": float(realized_analysis_effect.mean()),
        "std_true_analysis_benefit": float(realized_analysis_effect.std()),
        "risk_benefit_spearman": risk_benefit_spearman,
        "minimum_positive_profile_assignment_probability": (
            float(eligible_probability.min()) if len(eligible_probability) else 0.0
        ),
        "maximum_profile_assignment_probability": float(assignment_probability.max()),
        "protected_routes": int(
            learner.protected_routing_profile.notna().sum()
        ),
    }
    identification = {
        "consistency_by_construction": True,
        "conditional_exchangeability_given_learner_covariates": scenario != "hidden_confounding",
        "positivity_stressed": scenario == "poor_overlap",
        "hidden_confounding_present": scenario == "hidden_confounding",
        "analysis_outcome_is_placebo": scenario == "placebo_outcome",
        "sharp_null": scenario == "sharp_null",
        "observed_confounding_strength": (
            "strong" if scenario == "strong_observed_confounding" else "standard"
        ),
        "overlap_regime": "poor" if scenario == "poor_overlap" else "standard",
    }
    metadata = {
        "generator_version": PROFILE_DGP_VERSION,
        "scenario": scenario,
        "population_size": int(len(patients)),
        "need_mode": need_mode,
        "profile_catalog_version": catalog.catalog_version,
        "component_action_catalog_version": catalog.action_catalog.catalog_version,
        "treatment_field": "observed_treatment_profile",
        "comparator": NO_NEW_PROFILE,
        "comparison_semantics": "exact_profile_assignment_versus_maintenance_only",
        "analysis_outcome_name": analysis_name,
        "analysis_outcome_horizon": 365,
        "analysis_outcome_unit": analysis_unit,
        "benefit_direction": "higher_is_better",
        "identification": identification,
        "scenario_diagnostics_evaluation_only": diagnostics,
        "seeds": {key: int(value) for key, value in seeds.items()},
        "ranker_architecture_used_to_design_dgp": False,
        "neural_model_used_to_design_dgp": False,
        "calibrated_to_italian_population": False,
        "clinical_effectiveness_established": False,
    }
    audit = {
        "learner_rows": int(len(learner)),
        "eligibility_rows": int(len(eligibility)),
        "opportunity_rows": int(len(opportunities)),
        "profile_truth_rows": int(len(profile_truth)),
        "need_truth_rows": int(len(need_truth)),
        "learner_oracle_columns": [
            column for column in learner.columns
            if column.startswith(("true_", "oracle_", "potential_outcome", "latent_"))
        ],
        "eligibility_oracle_columns": [
            column for column in eligibility.columns
            if column.startswith(("true_", "oracle_", "potential_outcome", "latent_"))
        ],
        "truth_passed_to_need_model": False,
        "truth_passed_to_current_care_generator": False,
        "truth_passed_to_eligibility": False,
        "oracle_tables_passed_to_assignment": False,
        "latent_hidden_variable_used_by_assignment_dgp": scenario == "hidden_confounding",
        "oracle_used_for_training_or_selection": False,
        "observed_assignment_uses_only_eligible_profiles": True,
        "implicit_action_to_profile_conversion": False,
        "one_common_analysis_outcome": True,
        "eligibility_audit": eligibility_audit,
    }
    return SyntheticProfileDGPResult(
        learner=learner,
        baseline_need_assessments=need_prediction.assessments,
        need_reference_labels=need_reference.learner_labels,
        current_care_profiles=current,
        profile_eligibility=eligibility,
        patient_profile_opportunities=opportunities,
        profile_resources=profile_resources,
        resource_capacities=capacities,
        need_ground_truth=need_truth,
        profile_ground_truth=profile_truth,
        metadata=metadata,
        audit=audit,
    )
