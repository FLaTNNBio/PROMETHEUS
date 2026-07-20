"""Structural validation for the synthetic care-profile DGP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_population_ranking.dm77.care_profiles import CareProfileCatalog

from .profile_dgp import NO_NEW_PROFILE, SyntheticProfileDGPResult


FORBIDDEN_PREFIXES = ("true_", "oracle_", "potential_outcome", "latent_")


def _check(
    rows: list[dict],
    name: str,
    passed: bool,
    category: str,
    detail: str,
    severity: str = "critical",
) -> None:
    rows.append({
        "check": name,
        "category": category,
        "severity": severity,
        "status": "pass" if bool(passed) else "fail",
        "detail": detail,
    })


def validate_synthetic_profile_dgp(
    result: SyntheticProfileDGPResult,
    catalog: CareProfileCatalog,
) -> tuple[pd.DataFrame, dict]:
    """Validate isolation, treatment versions, outcomes, resources and scenario flags."""

    rows: list[dict] = []
    learner_forbidden = sorted(
        column for column in result.learner.columns
        if column.startswith(FORBIDDEN_PREFIXES)
    )
    eligibility_forbidden = sorted(
        column for column in result.profile_eligibility.columns
        if column.startswith(FORBIDDEN_PREFIXES)
    )
    _check(
        rows, "learner_oracle_isolation", not learner_forbidden, "oracle_isolation",
        f"forbidden={learner_forbidden}",
    )
    _check(
        rows, "eligibility_oracle_isolation", not eligibility_forbidden,
        "oracle_isolation", f"forbidden={eligibility_forbidden}",
    )
    _check(
        rows, "unique_learner_patients",
        not result.learner.patient_id.astype(str).duplicated().any(),
        "schema", f"rows={len(result.learner)}",
    )
    _check(
        rows, "unique_patient_profile_eligibility",
        not result.profile_eligibility[["patient_id", "care_profile_id"]].duplicated().any(),
        "schema", f"rows={len(result.profile_eligibility)}",
    )
    expected_truth_rows = len(result.learner) * len(catalog.automatic_rank_profiles)
    _check(
        rows, "complete_unique_profile_truth_grid",
        len(result.profile_ground_truth) == expected_truth_rows
        and not result.profile_ground_truth[["patient_id", "care_profile_id"]].duplicated().any(),
        "truth_schema",
        f"rows={len(result.profile_ground_truth)},expected={expected_truth_rows}",
    )
    known_current = set(catalog.profile_ids)
    current_values = set(result.learner.current_care_profile.astype(str))
    _check(
        rows, "exact_known_current_profiles",
        current_values.issubset(known_current)
        and not result.current_care_profiles.current_care_profile.isna().any()
        and result.current_care_profiles.current_care_profile_status.eq(
            "generated_exact_profile"
        ).all(),
        "current_care", f"represented={sorted(current_values)}",
    )
    known_treatments = {
        profile.care_profile_id for profile in catalog.automatic_rank_profiles
    } | {NO_NEW_PROFILE}
    observed = set(result.learner.observed_treatment_profile.astype(str))
    _check(
        rows, "observed_treatment_versions_are_explicit",
        observed.issubset(known_treatments), "treatment",
        f"represented={sorted(observed)}",
    )
    treated = result.learner.loc[
        result.learner.observed_treatment_profile != NO_NEW_PROFILE,
        ["patient_id", "observed_treatment_profile"],
    ].rename(columns={"observed_treatment_profile": "care_profile_id"})
    governed = treated.merge(
        result.profile_eligibility[[
            "patient_id", "care_profile_id", "discretionary_rank_candidate",
        ]],
        on=["patient_id", "care_profile_id"],
        how="left",
        validate="one_to_one",
    )
    _check(
        rows, "observed_profiles_are_governed_candidates",
        len(governed) == len(treated)
        and governed.discretionary_rank_candidate.fillna(False).astype(bool).all(),
        "treatment", f"treated={len(treated)}",
    )
    treated_context = result.learner.loc[
        result.learner.observed_treatment_profile != NO_NEW_PROFILE,
        ["current_care_profile", "observed_treatment_profile"],
    ]
    _check(
        rows, "observed_target_differs_from_current_profile",
        treated_context.current_care_profile.ne(
            treated_context.observed_treatment_profile
        ).all(),
        "treatment", "no already-active profile encoded as a new treatment",
    )
    outcome = pd.to_numeric(result.learner.observed_outcome, errors="coerce")
    _check(
        rows, "one_finite_common_observed_outcome",
        outcome.notna().all()
        and np.isfinite(outcome).all()
        and result.learner.outcome_name.nunique() == 1
        and result.learner.outcome_horizon.nunique() == 1
        and result.learner.outcome_unit.nunique() == 1
        and result.learner.benefit_direction.nunique() == 1,
        "outcome",
        f"name={result.learner.outcome_name.iloc[0]}",
    )
    probability = result.profile_ground_truth.pivot(
        index="patient_id", columns="care_profile_id", values="true_assignment_probability"
    ).sum(axis=1)
    no_new = result.profile_ground_truth.groupby("patient_id")[
        "true_no_new_profile_probability"
    ].first()
    _check(
        rows, "assignment_probabilities_sum_to_one",
        np.allclose(
            probability.sort_index().to_numpy() + no_new.sort_index().to_numpy(),
            1.0,
        ),
        "assignment", "profile probabilities plus no_new_profile",
    )
    treatment_fraction = float(
        (result.learner.observed_treatment_profile != NO_NEW_PROFILE).mean()
    )
    _check(
        rows, "nontrivial_observed_assignment",
        0.01 < treatment_fraction < 0.99,
        "assignment", f"treated_fraction={treatment_fraction:.6f}",
    )
    _check(
        rows, "nonnegative_resources_and_positive_capacities",
        result.profile_resources.scenario_full_profile_cost.ge(0).all()
        and result.resource_capacities.available_units.gt(0).all(),
        "resources",
        f"pools={len(result.resource_capacities)}",
    )
    scenario = str(result.metadata["scenario"])
    truth = result.profile_ground_truth
    if scenario == "sharp_null":
        _check(
            rows, "sharp_null_has_zero_analysis_benefit",
            np.allclose(truth.true_profile_benefit, 0.0),
            "scenario", "all true analysis benefits equal zero",
        )
    if scenario == "placebo_outcome":
        _check(
            rows, "placebo_is_null_but_primary_effect_is_nonzero",
            np.allclose(truth.true_profile_benefit, 0.0)
            and float(np.abs(truth.true_primary_profile_benefit).mean()) > 0.1,
            "scenario", "analysis null with preserved primary profile effects",
        )
    if scenario == "hidden_confounding":
        identification = result.metadata["identification"]
        _check(
            rows, "hidden_confounding_declares_identification_failure",
            identification["hidden_confounding_present"] is True
            and identification[
                "conditional_exchangeability_given_learner_covariates"
            ] is False,
            "identification", "exchangeability explicitly false",
        )
    if scenario == "capacity_scarcity":
        _check(
            rows, "capacity_scarcity_is_declared",
            result.resource_capacities.capacity_scarcity.astype(bool).all()
            and result.resource_capacities.capacity_fraction_of_population.max() < 0.05,
            "resources", "all pools below five percent of population",
        )
    if scenario == "heterogeneous_profile_costs":
        _check(
            rows, "profile_costs_are_heterogeneous",
            result.profile_resources.scenario_cost_multiplier.nunique() > 3,
            "resources", "multiple deterministic profile cost multipliers",
        )
    _check(
        rows, "no_model_architecture_tailoring",
        result.metadata["ranker_architecture_used_to_design_dgp"] is False
        and result.metadata["neural_model_used_to_design_dgp"] is False,
        "method", "DGP equations are model-agnostic",
    )
    checks = pd.DataFrame(rows)
    critical_failures = int(
        ((checks.severity == "critical") & (checks.status == "fail")).sum()
    )
    report = {
        "validation_version": "synthetic_profile_dgp_validation_v1",
        "scenario": scenario,
        "checks": int(len(checks)),
        "critical_failures": critical_failures,
        "warnings": int(
            ((checks.severity == "warning") & (checks.status == "fail")).sum()
        ),
        "valid": critical_failures == 0,
        "oracle_isolation_passed": not learner_forbidden and not eligibility_forbidden,
        "clinical_validity_established": False,
        "italian_population_validity_established": False,
    }
    return checks, report
