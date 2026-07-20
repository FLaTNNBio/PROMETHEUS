"""Capacity-constrained allocation over frozen profile recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


ORACLE_PREFIXES = ("true_", "oracle_", "potential_outcome", "latent_")


@dataclass(frozen=True)
class BinaryAllocationSolution:
    selected: np.ndarray
    status: str
    objective_value: float
    shared_budget_used: float
    pool_usage: dict[str, float]
    profile_usage: dict[str, int]
    solver_status: int
    solver_message: str


@dataclass(frozen=True)
class ProfileAllocationResult:
    decisions: pd.DataFrame
    diagnostics: pd.DataFrame
    diagnostic_selections: pd.DataFrame
    allocation_candidates: pd.DataFrame
    allocation_contract: dict[str, Any]
    audit: dict[str, Any]


def _oracle_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in map(str, frame.columns)
        if column != "oracle_used" and column.startswith(ORACLE_PREFIXES)
    )


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing allocation columns: {missing}")


def _stable_tie_value(seed: int, patient_id: str, profile_id: str) -> float:
    digest = hashlib.sha256(
        f"{int(seed)}|{patient_id}|{profile_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _parse_requirements(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        document = dict(value)
    elif isinstance(value, str):
        document = json.loads(value or "{}")
    elif pd.isna(value):
        document = {}
    else:
        raise ValueError("Incremental capacity requirements must be JSON or a mapping")
    result = {str(pool): float(amount) for pool, amount in document.items()}
    if any(not np.isfinite(amount) or amount <= 0.0 for amount in result.values()):
        raise ValueError("Incremental capacity requirements must be finite and positive")
    return result


def _resource_maps(
    resource_capacities: pd.DataFrame,
    profile_resources: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    _require(
        resource_capacities,
        {"capacity_pool", "available_units"},
        "resource_capacities",
    )
    _require(
        profile_resources,
        {"care_profile_id", "primary_capacity_pool"},
        "profile_resources",
    )
    if resource_capacities.capacity_pool.astype(str).duplicated().any():
        raise ValueError("Capacity pools must be unique")
    if profile_resources.care_profile_id.astype(str).duplicated().any():
        raise ValueError("Profile resource records must be unique")
    capacities = {
        str(row.capacity_pool): float(row.available_units)
        for row in resource_capacities.itertuples(index=False)
    }
    if any(not np.isfinite(value) or value < 0.0 for value in capacities.values()):
        raise ValueError("Available capacity units must be finite and nonnegative")
    primary_pool = {
        str(row.care_profile_id): str(row.primary_capacity_pool)
        for row in profile_resources.itertuples(index=False)
    }
    profile_capacities = {
        profile: capacities[pool]
        for profile, pool in primary_pool.items()
        if pool != "none" and pool in capacities
    }
    return capacities, profile_capacities


def solve_binary_allocation(
    candidates: pd.DataFrame,
    objective_values: Sequence[float],
    *,
    shared_budget_limit: float,
    pool_capacities: Mapping[str, float],
    profile_capacities: Mapping[str, float],
    tie_seed: int,
    tie_break_epsilon: float,
    solver_settings: Mapping[str, Any],
) -> BinaryAllocationSolution:
    """Solve one optional binary decision per frozen patient-profile candidate."""

    required = {
        "patient_id", "care_profile_id", "incremental_resource_cost",
        "capacity_requirements",
    }
    _require(candidates, required, "allocation candidates")
    objective = np.asarray(objective_values, dtype=float)
    if objective.ndim != 1 or len(objective) != len(candidates):
        raise ValueError("Allocation objective must align with candidates")
    if not np.isfinite(objective).all():
        raise ValueError("Allocation objective values must be finite")
    if not np.isfinite(float(shared_budget_limit)) or float(shared_budget_limit) < 0.0:
        raise ValueError("Shared budget must be finite and nonnegative")
    if float(tie_break_epsilon) <= 0.0:
        raise ValueError("Tie-break epsilon must be positive")
    if candidates.empty:
        return BinaryAllocationSolution(
            selected=np.array([], dtype=bool),
            status="no_recommended_candidates",
            objective_value=0.0,
            shared_budget_used=0.0,
            pool_usage={str(pool): 0.0 for pool in pool_capacities},
            profile_usage={str(profile): 0 for profile in profile_capacities},
            solver_status=0,
            solver_message="No recommended candidates",
        )

    cost = pd.to_numeric(
        candidates.incremental_resource_cost, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(cost).all() or (cost < 0.0).any():
        raise ValueError("Incremental allocation costs must be finite and nonnegative")
    requirements = list(candidates.capacity_requirements)
    known_pools = set(map(str, pool_capacities))
    unknown_pools = sorted({
        pool for item in requirements for pool in item if pool not in known_pools
    })
    if unknown_pools:
        raise ValueError(f"Unknown capacity pools in candidates: {unknown_pools}")
    profiles = candidates.care_profile_id.astype(str).to_numpy()
    missing_profile_capacity = sorted(set(profiles).difference(profile_capacities))
    if missing_profile_capacity:
        raise ValueError(
            f"Recommended profiles lack a profile-capacity rule: {missing_profile_capacity}"
        )

    rows = [cost]
    upper = [float(shared_budget_limit)]
    constraint_names = ["shared_budget"]
    for pool, available in sorted(pool_capacities.items()):
        rows.append(np.asarray([item.get(str(pool), 0.0) for item in requirements]))
        upper.append(float(available))
        constraint_names.append(f"capacity_pool:{pool}")
    for profile, available in sorted(profile_capacities.items()):
        rows.append((profiles == str(profile)).astype(float))
        upper.append(float(available))
        constraint_names.append(f"profile_capacity:{profile}")
    patient_ids = candidates.patient_id.astype(str).to_numpy()
    # The production candidate table contains at most one frozen recommendation
    # per patient. In that common case the binary variable bound already enforces
    # the patient limit, so thousands of redundant MILP rows only slow HiGHS.
    # Keep the explicit constraint for generic callers that supply alternatives.
    if len(set(patient_ids)) != len(patient_ids):
        for patient in sorted(set(patient_ids)):
            rows.append((patient_ids == patient).astype(float))
            upper.append(1.0)
            constraint_names.append(f"patient:{patient}")
    matrix = np.vstack(rows)
    constraint = LinearConstraint(
        matrix,
        lb=np.full(len(rows), -np.inf),
        ub=np.asarray(upper, dtype=float),
    )
    tie = np.asarray([
        _stable_tie_value(tie_seed, patient, profile)
        for patient, profile in zip(patient_ids, profiles)
    ])
    epsilon = float(tie_break_epsilon)
    augmented = np.where(
        objective > 0.0,
        objective + epsilon * tie,
        objective - epsilon * (1.0 + tie),
    )
    options = {
        "presolve": bool(solver_settings["presolve"]),
        "mip_rel_gap": float(solver_settings["mip_relative_gap"]),
        "time_limit": float(solver_settings["time_limit_seconds"]),
    }
    result = milp(
        c=-augmented,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=constraint,
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Allocation MILP failed with status {result.status}: {result.message}"
        )
    selected = np.asarray(result.x > 0.5, dtype=bool)
    lhs = matrix @ selected.astype(float)
    if np.any(lhs > np.asarray(upper) + 1e-7):
        violated = [
            constraint_names[index]
            for index in np.flatnonzero(lhs > np.asarray(upper) + 1e-7)
        ]
        raise AssertionError(f"Allocation solver returned constraint violations: {violated}")
    pool_usage = {
        str(pool): float(sum(
            item.get(str(pool), 0.0)
            for item, keep in zip(requirements, selected) if keep
        ))
        for pool in pool_capacities
    }
    profile_usage = {
        str(profile): int(np.sum(selected & (profiles == str(profile))))
        for profile in profile_capacities
    }
    return BinaryAllocationSolution(
        selected=selected,
        status="optimal",
        objective_value=float(objective[selected].sum()),
        shared_budget_used=float(cost[selected].sum()),
        pool_usage=pool_usage,
        profile_usage=profile_usage,
        solver_status=int(result.status),
        solver_message=str(result.message),
    )


def _candidate_table(
    recommendations: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    recommendation_required = {
        "patient_id", "recommended_profile_id", "recommended_actionable_level",
        "recommended_raw_priority_score", "calibrated_incremental_benefit",
        "recommendation_abstained", "current_care_profile",
        "current_care_profile_level",
    }
    opportunity_required = {
        "patient_id", "care_profile_id", "care_profile_level", "eligibility",
        "discretionary_rank_candidate", "empirical_support", "protected_pathway",
        "protected_profile", "mandatory_care", "eligibility_reasons",
        "missing_component_actions", "deactivated_component_actions",
        "incremental_capacity_requirements", "scenario_incremental_resource_cost",
    }
    _require(recommendations, recommendation_required, "recommendations")
    _require(opportunities, opportunity_required, "opportunities")
    if recommendations.patient_id.astype(str).duplicated().any():
        raise ValueError("Allocation requires one frozen recommendation per patient")
    recommended = recommendations.loc[
        ~recommendations.recommendation_abstained.astype(bool)
        & recommendations.recommended_profile_id.notna()
    ].copy()
    recommended["patient_id"] = recommended.patient_id.astype(str)
    recommended["care_profile_id"] = recommended.recommended_profile_id.astype(str)
    opportunity = opportunities[list(opportunity_required)].copy()
    opportunity["patient_id"] = opportunity.patient_id.astype(str)
    opportunity["care_profile_id"] = opportunity.care_profile_id.astype(str)
    if opportunity[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Allocation opportunities must be unique by patient and profile")
    candidates = recommended.merge(
        opportunity,
        on=["patient_id", "care_profile_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_opportunity"),
    )
    if len(candidates) != len(recommended) or candidates.eligibility.isna().any():
        raise ValueError("Every recommendation must match its exact profile opportunity")
    candidates["capacity_requirements"] = candidates[
        "incremental_capacity_requirements"
    ].map(_parse_requirements)
    candidates["incremental_resource_cost"] = pd.to_numeric(
        candidates.scenario_incremental_resource_cost, errors="coerce"
    )
    candidates["calibrated_incremental_benefit"] = pd.to_numeric(
        candidates.calibrated_incremental_benefit, errors="coerce"
    )
    candidates["recommended_raw_priority_score"] = pd.to_numeric(
        candidates.recommended_raw_priority_score, errors="coerce"
    )
    numeric = candidates[[
        "incremental_resource_cost", "calibrated_incremental_benefit",
        "recommended_raw_priority_score",
    ]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Recommended allocation candidates require finite cost and scores")
    invalid = candidates.loc[
        ~candidates.eligibility.astype(bool)
        | ~candidates.discretionary_rank_candidate.astype(bool)
        | ~candidates.empirical_support.astype(bool)
        | candidates.protected_pathway.astype(bool)
        | candidates.protected_profile.astype(bool)
        | candidates.mandatory_care.astype(bool)
    ]
    if len(invalid):
        raise ValueError("Ineligible, protected or mandatory profiles cannot be allocated")
    wrong_level = candidates.care_profile_level.astype(int).ne(
        pd.to_numeric(candidates.recommended_actionable_level, errors="coerce").astype(int)
    )
    if wrong_level.any():
        raise ValueError("Recommended profile and recommended actionable level disagree")
    return candidates.sort_values(
        ["patient_id", "care_profile_id"], kind="stable"
    ).reset_index(drop=True)


def _ordinal_selection(
    candidates: pd.DataFrame,
    slots: int,
    seed: int,
) -> np.ndarray:
    if candidates.empty or int(slots) <= 0:
        return np.array([], dtype=int)
    frame = candidates[[
        "patient_id", "care_profile_id", "recommended_raw_priority_score"
    ]].copy()
    frame["tie"] = [
        _stable_tie_value(seed, patient, profile)
        for patient, profile in zip(frame.patient_id, frame.care_profile_id)
    ]
    order = frame.sort_values(
        ["recommended_raw_priority_score", "tie", "patient_id", "care_profile_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).index.to_numpy(int)
    return order[: min(int(slots), len(order))]


def allocate_recommended_profiles(
    recommendations: pd.DataFrame,
    supported_opportunities: pd.DataFrame,
    resource_capacities: pd.DataFrame,
    profile_resources: pd.DataFrame,
    *,
    allocation_contract: Mapping[str, Any],
    settings: Mapping[str, Any],
    tie_seed: int,
) -> ProfileAllocationResult:
    """Allocate only frozen recommended profiles under budget and capacity limits."""

    frames = {
        "recommendations": recommendations,
        "supported_opportunities": supported_opportunities,
        "resource_capacities": resource_capacities,
        "profile_resources": profile_resources,
    }
    leaked = {name: _oracle_columns(frame) for name, frame in frames.items()}
    if any(leaked.values()):
        raise ValueError(f"Oracle columns cannot enter profile allocation: {leaked}")
    flagged = [
        name for name, frame in frames.items()
        if "oracle_used" in frame and frame.oracle_used.fillna(False).astype(bool).any()
    ]
    if flagged:
        raise ValueError(f"Oracle-used inputs cannot enter allocation: {flagged}")
    if allocation_contract["candidate_profiles"] != "recommended_profile_only":
        raise ValueError("Allocator may consider only the frozen recommended profile")
    if allocation_contract["alternative_profile_substitution_allowed"] is not False:
        raise ValueError("Alternative profile substitution must remain disabled")
    candidates = _candidate_table(recommendations, supported_opportunities)
    pool_capacities, profile_capacities = _resource_maps(
        resource_capacities, profile_resources
    )
    population_size = int(len(recommendations))
    shared_budget = float(settings["shared_budget_units_per_patient"]) * population_size
    solver_settings = settings["solver"]
    primary = solve_binary_allocation(
        candidates,
        candidates.calibrated_incremental_benefit.to_numpy(float),
        shared_budget_limit=shared_budget,
        pool_capacities=pool_capacities,
        profile_capacities=profile_capacities,
        tie_seed=int(tie_seed),
        tie_break_epsilon=float(settings["tie_break_epsilon"]),
        solver_settings=solver_settings,
    )
    selected_keys = {
        (str(row.patient_id), str(row.care_profile_id))
        for row in candidates.loc[primary.selected].itertuples(index=False)
    }
    candidate_by_key = {
        (str(row.patient_id), str(row.care_profile_id)): row
        for row in candidates.itertuples(index=False)
    }
    records = []
    for row in recommendations.sort_values("patient_id", kind="stable").itertuples(
        index=False
    ):
        patient = str(row.patient_id)
        recommended_profile = (
            None if pd.isna(row.recommended_profile_id) else str(row.recommended_profile_id)
        )
        has_recommendation = not bool(row.recommendation_abstained)
        key = (patient, recommended_profile) if recommended_profile is not None else None
        allocated = bool(key in selected_keys) if key is not None else False
        candidate = candidate_by_key.get(key) if key is not None else None
        if allocated:
            allocated_profile = recommended_profile
            allocated_level = int(row.recommended_actionable_level)
            allocation_status = "allocated_recommended_profile"
            allocation_reason = "SELECTED_BY_CARDINAL_RESOURCE_OPTIMIZATION"
            deferred = False
            allocated_benefit = float(row.calibrated_incremental_benefit)
            allocated_cost = float(candidate.incremental_resource_cost)
            allocated_requirements = json.dumps(
                candidate.capacity_requirements, sort_keys=True
            )
        elif has_recommendation:
            allocated_profile = None
            allocated_level = int(row.current_care_profile_level)
            allocation_status = "deferred_recommended_profile"
            allocation_reason = "NOT_SELECTED_UNDER_SHARED_BUDGET_AND_CAPACITY"
            deferred = True
            allocated_benefit = 0.0
            allocated_cost = 0.0
            allocated_requirements = "{}"
        else:
            allocated_profile = None
            allocated_level = int(row.current_care_profile_level)
            allocation_status = "no_actionable_recommendation"
            allocation_reason = "RECOMMENDATION_ABSTAINED_BEFORE_ALLOCATION"
            deferred = False
            allocated_benefit = 0.0
            allocated_cost = 0.0
            allocated_requirements = "{}"
        record = row._asdict()
        record.update({
            "allocated_profile_id": allocated_profile,
            "allocated_care_level": allocated_level,
            "allocation_status": allocation_status,
            "allocation_reason": allocation_reason,
            "deferred_recommendation": deferred,
            "allocated_calibrated_benefit": allocated_benefit,
            "allocated_incremental_resource_cost": allocated_cost,
            "allocated_incremental_capacity_requirements": allocated_requirements,
            "recommendation_allocation_level_gap": (
                int(row.recommended_actionable_level) - allocated_level
                if has_recommendation else 0
            ),
            "recommendation_preserved": True,
            "alternative_profile_substituted": False,
            "allocation_method": str(settings["method"]),
            "shared_budget_limit": shared_budget,
            "oracle_used_for_allocation": False,
        })
        records.append(record)
    decisions = pd.DataFrame(records)

    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_selections: list[dict[str, Any]] = []
    curve_solutions: dict[float, BinaryAllocationSolution] = {1.0: primary}
    for multiplier in map(float, settings["budget_curve_multipliers"]):
        solution = curve_solutions.get(multiplier)
        if solution is None:
            solution = solve_binary_allocation(
                candidates,
                candidates.calibrated_incremental_benefit.to_numpy(float),
                shared_budget_limit=shared_budget * multiplier,
                pool_capacities=pool_capacities,
                profile_capacities=profile_capacities,
                tie_seed=int(tie_seed),
                tie_break_epsilon=float(settings["tie_break_epsilon"]),
                solver_settings=solver_settings,
            )
            curve_solutions[multiplier] = solution
        diagnostic_rows.append({
            "diagnostic_type": "cardinal_budget_curve",
            "setting": multiplier,
            "budget_limit": shared_budget * multiplier,
            "selected_count": int(solution.selected.sum()),
            "calibrated_objective_value": solution.objective_value,
            "resource_cost_used": solution.shared_budget_used,
            "utilization": (
                solution.shared_budget_used / (shared_budget * multiplier)
                if shared_budget * multiplier > 0.0 else 0.0
            ),
            "uses_calibration": True,
            "uses_monetary_cost": True,
            "oracle_used": False,
        })
        for row in candidates.loc[solution.selected].itertuples(index=False):
            diagnostic_selections.append({
                "diagnostic_type": "cardinal_budget_curve",
                "setting": multiplier,
                "patient_id": str(row.patient_id),
                "care_profile_id": str(row.care_profile_id),
            })
    for pool, available in sorted(pool_capacities.items()):
        used = primary.pool_usage[pool]
        diagnostic_rows.append({
            "diagnostic_type": "capacity_pool_utilization",
            "setting": pool,
            "available_units": available,
            "used_units": used,
            "utilization": used / available if available > 0.0 else 0.0,
            "uses_calibration": True,
            "uses_monetary_cost": False,
            "oracle_used": False,
        })
    for profile, available in sorted(profile_capacities.items()):
        used = primary.profile_usage[profile]
        diagnostic_rows.append({
            "diagnostic_type": "profile_capacity_utilization",
            "setting": profile,
            "available_units": available,
            "used_units": used,
            "utilization": used / available if available > 0.0 else 0.0,
            "uses_calibration": True,
            "uses_monetary_cost": False,
            "oracle_used": False,
        })
    for fraction in map(float, settings["fixed_capacity_fractions"]):
        slots = int(np.floor(population_size * fraction))
        selected = _ordinal_selection(candidates, slots, int(tie_seed))
        diagnostic_rows.append({
            "diagnostic_type": "fixed_count_ordinal",
            "setting": fraction,
            "fixed_slots": slots,
            "selected_count": int(len(selected)),
            "uses_calibration": False,
            "uses_monetary_cost": False,
            "oracle_used": False,
        })
        for row in candidates.iloc[selected].itertuples(index=False):
            diagnostic_selections.append({
                "diagnostic_type": "fixed_count_ordinal",
                "setting": fraction,
                "patient_id": str(row.patient_id),
                "care_profile_id": str(row.care_profile_id),
            })
    same_count = _ordinal_selection(candidates, int(primary.selected.sum()), int(tie_seed))
    cardinal_keys = selected_keys
    ordinal_keys = {
        (str(row.patient_id), str(row.care_profile_id))
        for row in candidates.iloc[same_count].itertuples(index=False)
    }
    union = cardinal_keys | ordinal_keys
    intersection = cardinal_keys & ordinal_keys
    diagnostic_rows.append({
        "diagnostic_type": "cardinal_vs_ordinal_selected_set_overlap",
        "setting": "same_selected_count",
        "selected_count": int(len(cardinal_keys)),
        "comparison_selected_count": int(len(ordinal_keys)),
        "intersection_count": int(len(intersection)),
        "selected_set_jaccard": len(intersection) / len(union) if union else 1.0,
        "uses_calibration": False,
        "uses_monetary_cost": False,
        "oracle_used": False,
    })
    for row in candidates.iloc[same_count].itertuples(index=False):
        diagnostic_selections.append({
            "diagnostic_type": "fixed_count_ordinal_same_as_cardinal",
            "setting": int(primary.selected.sum()),
            "patient_id": str(row.patient_id),
            "care_profile_id": str(row.care_profile_id),
        })

    budget_violation = max(0.0, primary.shared_budget_used - shared_budget)
    pool_violations = {
        pool: max(0.0, primary.pool_usage[pool] - float(available))
        for pool, available in pool_capacities.items()
    }
    profile_violations = {
        profile: max(0.0, primary.profile_usage[profile] - float(available))
        for profile, available in profile_capacities.items()
    }
    allocated = decisions.allocated_profile_id.notna()
    substitution_violations = int((
        allocated
        & decisions.allocated_profile_id.astype(str).ne(
            decisions.recommended_profile_id.astype(str)
        )
    ).sum())
    per_patient_violations = int(
        decisions.loc[allocated].patient_id.astype(str).duplicated().sum()
    )
    selected_candidates = candidates.loc[primary.selected]
    mutual_exclusion_violations = 0
    duplicate_component_violations = 0
    for row in selected_candidates.itertuples(index=False):
        missing = [value for value in str(row.missing_component_actions).split("|") if value]
        deactivated = [
            value for value in str(row.deactivated_component_actions).split("|") if value
        ]
        duplicate_component_violations += len(missing) - len(set(missing))
        mutual_exclusion_violations += int(bool(set(missing).intersection(deactivated)))
    violation_counts = {
        "eligibility": int((~selected_candidates.eligibility.astype(bool)).sum()),
        "shared_budget": int(budget_violation > 1e-7),
        "capacity_pool": int(any(value > 1e-7 for value in pool_violations.values())),
        "profile_capacity": int(any(
            value > 1e-7 for value in profile_violations.values()
        )),
        "prerequisite": int((
            ~selected_candidates.eligibility_reasons.astype(str).eq(
                "ELIGIBLE_BY_PROFILE_CATALOG"
            )
        ).sum()),
        "protected_pathway": int(selected_candidates.protected_pathway.astype(bool).sum()),
        "mutual_exclusion": int(mutual_exclusion_violations),
        "at_most_one_profile_per_patient": per_patient_violations,
        "alternative_profile_substitution": substitution_violations,
        "duplicate_missing_component_charge": int(duplicate_component_violations),
    }
    recommendation_columns = [
        "recommended_actionable_level", "recommended_profile_id",
        "recommendation_abstained", "calibrated_incremental_benefit",
    ]
    recommendation_preservation = decisions[
        ["patient_id", *recommendation_columns]
    ].merge(
        recommendations[["patient_id", *recommendation_columns]],
        on="patient_id",
        suffixes=("_allocation", "_frozen"),
        validate="one_to_one",
    )
    recommendation_changes = 0
    for column in recommendation_columns:
        left = recommendation_preservation[f"{column}_allocation"]
        right = recommendation_preservation[f"{column}_frozen"]
        recommendation_changes += int(
            not bool((left.eq(right) | (left.isna() & right.isna())).all())
        )
    violation_counts["recommendation_overwrite"] = recommendation_changes
    audit = {
        "status": "allocated",
        "method": str(settings["method"]),
        "candidate_policy": "recommended_profile_only",
        "recommended_candidates": int(len(candidates)),
        "allocated_recommendations": int(primary.selected.sum()),
        "deferred_recommendations": int(
            decisions.deferred_recommendation.astype(bool).sum()
        ),
        "patients_without_actionable_recommendation": int(
            decisions.recommendation_abstained.astype(bool).sum()
        ),
        "shared_budget_limit": shared_budget,
        "shared_budget_used": primary.shared_budget_used,
        "pool_capacities": pool_capacities,
        "pool_usage": primary.pool_usage,
        "profile_capacities": profile_capacities,
        "profile_usage": primary.profile_usage,
        "calibrated_objective_value": primary.objective_value,
        "solver_status": primary.solver_status,
        "solver_message": primary.solver_message,
        "solver_settings": dict(solver_settings),
        "tie_seed": int(tie_seed),
        "tie_break_epsilon": float(settings["tie_break_epsilon"]),
        "resource_accounting": "missing_profile_components_relative_to_current_care",
        "existing_components_charged_again": False,
        "component_capacity_double_counting": False,
        "alternative_profile_substitution": False,
        "recommendation_records_modified": False,
        "fixed_capacity_ordinal_uses_calibration": False,
        "fixed_capacity_ordinal_uses_monetary_cost": False,
        "constraint_violations": violation_counts,
        "constraint_violations_total": int(sum(violation_counts.values())),
        "oracle_columns_detected": leaked,
        "oracle_used": False,
    }
    contract = {
        "stage": "allocation_after_recommendation_freeze_before_oracle",
        "method": str(settings["method"]),
        "solver": "scipy_optimize_milp_highs",
        "solver_settings": dict(solver_settings),
        "objective": "maximize_total_validation_calibrated_incremental_benefit",
        "shared_budget_units_per_patient": float(
            settings["shared_budget_units_per_patient"]
        ),
        "shared_budget_limit": shared_budget,
        "pool_capacities": pool_capacities,
        "profile_capacities": profile_capacities,
        "tie_seed": int(tie_seed),
        "tie_break_epsilon": float(settings["tie_break_epsilon"]),
        "budget_curve_multipliers": list(map(float, settings["budget_curve_multipliers"])),
        "fixed_capacity_fractions": list(map(float, settings["fixed_capacity_fractions"])),
        "candidate_profiles": "recommended_profile_only",
        "alternative_profile_substitution_allowed": False,
        "recommendation_may_be_overwritten": False,
        "resource_accounting": allocation_contract["resource_accounting"],
        "existing_components_charged_again": False,
        "oracle_used": False,
        "audit": audit,
    }
    return ProfileAllocationResult(
        decisions=decisions,
        diagnostics=pd.DataFrame(diagnostic_rows),
        diagnostic_selections=pd.DataFrame(diagnostic_selections),
        allocation_candidates=candidates,
        allocation_contract=contract,
        audit=audit,
    )


__all__ = [
    "BinaryAllocationSolution",
    "ProfileAllocationResult",
    "allocate_recommended_profiles",
    "solve_binary_allocation",
]
