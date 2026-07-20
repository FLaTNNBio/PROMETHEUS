"""Shared-budget allocation of explicit care actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class ActionAllocationResult:
    selected: np.ndarray
    decisions: pd.DataFrame
    diagnostics: dict


def apply_action(current_care_state: str, selected_action: str | None, action_to_state: Mapping[str, str]) -> str:
    if selected_action is None or pd.isna(selected_action):
        return str(current_care_state)
    if selected_action not in action_to_state:
        raise ValueError(f"Unknown selected care action {selected_action!r}")
    return str(action_to_state[selected_action])


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id", "dm77_need_level", "current_care_state", "action_id",
        "transition_index", "raw_priority_score", "calibrated_incremental_benefit",
        "cost", "capacity_pool", "eligible", "support_flag", "from_states",
        "to_state", "prerequisites_satisfied", "mutually_exclusive_with",
        "mandatory", "protected_action",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Action allocation is missing fields: {missing}")
    out = frame.reset_index(drop=True).copy()
    if "joint_prerequisite_actions" not in out:
        out["joint_prerequisite_actions"] = ""
    if out[["patient_id", "action_id"]].duplicated().any():
        raise ValueError("Action allocation requires unique patient-action rows")
    if (out.cost < 0).any() or not np.isfinite(
        out[["cost", "calibrated_incremental_benefit"]].to_numpy(float)
    ).all():
        raise ValueError("Action costs and calibrated benefits must be finite")
    state_compatible = [
        str(state) in set(str(from_states).split("|"))
        for state, from_states in zip(out.current_care_state, out.from_states)
    ]
    out["state_compatible"] = state_compatible
    return out


def allocate_care_actions(
    opportunities: pd.DataFrame,
    shared_budget: float,
    capacity_pools: Mapping[str, int] | None,
    max_primary_actions_per_patient: int = 1,
    require_empirical_support: bool = True,
    allocate_negative_predicted_benefit: bool = False,
    solver_time_limit_seconds: float = 120.0,
) -> ActionAllocationResult:
    """Allocate explicit actions and apply each selected action to current state."""

    if shared_budget < 0 or max_primary_actions_per_patient < 1:
        raise ValueError("Invalid action budget or per-patient action limit")
    frame = _validate(opportunities)
    if frame.protected_action.to_numpy(bool).any():
        raise ValueError("Protected actions must bypass discretionary allocation")
    n = len(frame)
    mandatory = frame.mandatory.to_numpy(bool)
    prerequisite_available = frame.prerequisites_satisfied.to_numpy(bool).copy()
    joint_prerequisites: dict[int, tuple[str, ...]] = {}
    for _, group in frame.groupby("patient_id", sort=False):
        patient_actions = set(group.action_id.astype(str))
        for index, row in group.iterrows():
            required_actions = tuple(
                value for value in str(row.joint_prerequisite_actions).split("|")
                if value
            )
            joint_prerequisites[int(index)] = required_actions
            if required_actions and set(required_actions).issubset(patient_actions):
                prerequisite_available[int(index)] = True
    upper = (
        frame.eligible.to_numpy(bool)
        & frame.state_compatible.to_numpy(bool)
        & prerequisite_available
    )
    if require_empirical_support:
        upper &= frame.support_flag.to_numpy(bool) | mandatory
    if not allocate_negative_predicted_benefit:
        upper &= (frame.calibrated_incremental_benefit.to_numpy(float) > 0.0) | mandatory
    if np.any(mandatory & ~upper):
        bad = frame.loc[mandatory & ~upper, ["patient_id", "action_id"]].to_dict("records")
        raise ValueError(f"Mandatory actions violate state/prerequisite eligibility: {bad[:5]}")

    rows: list[tuple[dict[int, float], float, float]] = []
    for _, group in frame.groupby("patient_id", sort=False):
        rows.append((
            {int(index): 1.0 for index in group.index},
            -np.inf,
            float(max_primary_actions_per_patient),
        ))
        # Explicit mutual-exclusion constraints remain active if max actions is raised.
        for index, row in group.iterrows():
            exclusions = set(str(row.mutually_exclusive_with).split("|")) - {""}
            for other_index, other in group.iterrows():
                if index < other_index and str(other.action_id) in exclusions:
                    rows.append(({int(index): 1.0, int(other_index): 1.0}, -np.inf, 1.0))
            # If a prerequisite is not already active, every explicitly named
            # complementary action must be selected in the same planning cycle.
            if not bool(row.prerequisites_satisfied):
                for prerequisite_action in joint_prerequisites[int(index)]:
                    provider = group.index[
                        group.action_id.astype(str) == prerequisite_action
                    ]
                    if len(provider) == 1:
                        rows.append((
                            {int(index): 1.0, int(provider[0]): -1.0},
                            -np.inf,
                            0.0,
                        ))
    if capacity_pools:
        unknown = set(frame.capacity_pool).difference(capacity_pools)
        if unknown:
            raise ValueError(f"Missing configured capacity pools: {sorted(unknown)}")
        for pool, capacity in capacity_pools.items():
            indices = frame.index[frame.capacity_pool == pool]
            rows.append((
                {int(index): 1.0 for index in indices}, -np.inf, float(capacity)
            ))
    rows.append((
        {int(index): float(cost) for index, cost in enumerate(frame.cost)},
        -np.inf,
        float(shared_budget),
    ))
    matrix = lil_matrix((len(rows), n), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    constraint_upper = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, minimum, maximum) in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index], constraint_upper[row_index] = minimum, maximum
    variable_lower = mandatory.astype(float)
    variable_upper = upper.astype(float)
    result = milp(
        c=-frame.calibrated_incremental_benefit.to_numpy(float),
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, constraint_upper),
        options={"time_limit": float(solver_time_limit_seconds)},
    )
    if result.x is None or not result.success:
        raise RuntimeError(f"Action allocation MILP failed: {result.status}, {result.message}")
    selected = result.x > 0.5

    recommendation = {}
    for patient, group in frame.loc[
        frame.eligible.to_numpy(bool)
        & frame.support_flag.to_numpy(bool)
        & (frame.calibrated_incremental_benefit.to_numpy(float) > 0.0)
    ].groupby("patient_id", sort=False):
        recommendation[str(patient)] = str(
            group.sort_values("raw_priority_score", ascending=False).iloc[0].action_id
        )
    action_to_state = dict(zip(frame.action_id, frame.to_state))
    decisions = []
    for patient, group in frame.assign(selected=selected).groupby("patient_id", sort=False):
        chosen = group.loc[group.selected]
        allocated_action = None if chosen.empty else str(chosen.iloc[0].action_id)
        recommended_action = recommendation.get(str(patient))
        current_state = str(group.iloc[0].current_care_state)
        selected_row = None if chosen.empty else chosen.iloc[0]
        recommended_rows = (
            group.loc[group.action_id == recommended_action]
            if recommended_action is not None else group.iloc[0:0]
        )
        recommended_row = None if recommended_rows.empty else recommended_rows.iloc[0]
        decision_row = selected_row if selected_row is not None else recommended_row
        if allocated_action is not None:
            status = "mandatory_allocated" if bool(selected_row.mandatory) else "allocated"
            reason = "SELECTED_BY_SHARED_BUDGET_OPTIMIZATION"
        elif recommended_action is not None:
            status = "recommended_not_allocated"
            reason = "NOT_SELECTED_UNDER_SHARED_BUDGET_OR_POOL_CAPACITY"
        else:
            status = "no_positive_supported_action"
            reason = "NO_POSITIVE_SUPPORTED_ELIGIBLE_ACTION"
        decisions.append({
            "patient_id": str(patient),
            "dm77_need_level": group.iloc[0].dm77_need_level,
            "current_care_state": current_state,
            "recommended_action": recommended_action,
            "allocated_action": allocated_action,
            "calibrated_incremental_benefit": (
                np.nan if decision_row is None
                else float(decision_row.calibrated_incremental_benefit)
            ),
            "cost": 0.0 if decision_row is None else float(decision_row.cost),
            "capacity_pool": None if decision_row is None else str(decision_row.capacity_pool),
            "allocation_status": status,
            "resulting_care_state": apply_action(current_state, allocated_action, action_to_state),
            "allocation_reason": reason,
        })
    decisions_frame = pd.DataFrame(decisions)
    budget_used = float(frame.loc[selected, "cost"].sum())
    pool_counts = frame.loc[selected].capacity_pool.value_counts().to_dict()
    mutual_violations = 0
    patient_limit_violations = 0
    prerequisite_violations = 0
    for _, group in frame.assign(selected=selected).groupby("patient_id"):
        patient_limit_violations += int(group.selected.sum() > max_primary_actions_per_patient)
        selected_actions = set(group.loc[group.selected, "action_id"])
        for row in group.loc[group.selected].itertuples():
            required_actions = set(joint_prerequisites[int(row.Index)])
            prerequisite_violations += int(
                not bool(row.prerequisites_satisfied)
                and not required_actions.issubset(selected_actions)
            )
            mutual_violations += int(bool(
                selected_actions.intersection(set(str(row.mutually_exclusive_with).split("|")))
            ))
    diagnostics = {
        "solver": "scipy.optimize.milp_highs",
        "solver_success": bool(result.success),
        "objective_predicted_benefit": float(
            frame.loc[selected, "calibrated_incremental_benefit"].sum()
        ),
        "shared_budget": float(shared_budget),
        "budget_used": budget_used,
        "budget_violation": max(0.0, budget_used - float(shared_budget)),
        "selected_actions": int(selected.sum()),
        "mandatory_actions": int((selected & mandatory).sum()),
        "selected_per_capacity_pool": {str(key): int(value) for key, value in pool_counts.items()},
        "capacity_pool_violations": {
            str(pool): max(0, int(pool_counts.get(pool, 0)) - int(capacity))
            for pool, capacity in (capacity_pools or {}).items()
        },
        "eligibility_violations": int((selected & ~frame.eligible.to_numpy(bool)).sum()),
        "support_violations": int((selected & ~frame.support_flag.to_numpy(bool) & ~mandatory).sum()),
        "state_compatibility_violations": int((selected & ~frame.state_compatible.to_numpy(bool)).sum()),
        "prerequisite_violations": int(prerequisite_violations),
        "mutual_exclusion_violations": int(mutual_violations),
        "max_actions_per_patient_violations": int(patient_limit_violations),
        "result_uses_final_package_arithmetic": False,
    }
    violations = (
        diagnostics["budget_violation"], diagnostics["eligibility_violations"],
        diagnostics["support_violations"], diagnostics["state_compatibility_violations"],
        diagnostics["prerequisite_violations"], diagnostics["mutual_exclusion_violations"],
        diagnostics["max_actions_per_patient_violations"],
        *diagnostics["capacity_pool_violations"].values(),
    )
    if any(value > 1e-7 for value in violations):
        raise AssertionError(f"Action allocator returned infeasible solution: {diagnostics}")
    return ActionAllocationResult(selected, decisions_frame, diagnostics)


def allocate_care_actions_greedy(
    opportunities: pd.DataFrame,
    shared_budget: float,
    capacity_pools: Mapping[str, int] | None,
    strategy: str = "score",
    max_primary_actions_per_patient: int = 1,
    require_empirical_support: bool = True,
    allocate_negative_predicted_benefit: bool = False,
) -> ActionAllocationResult:
    """Feasible deterministic greedy comparator for the MILP allocator.

    ``score`` orders by predicted benefit and ``score_per_cost`` by predicted
    benefit divided by cost. This is an allocator ablation, not a new ranking
    method; all clinical/catalog constraints are still enforced.
    """

    if strategy not in {"score", "score_per_cost"}:
        raise ValueError("Greedy strategy must be score or score_per_cost")
    if shared_budget < 0 or max_primary_actions_per_patient < 1:
        raise ValueError("Invalid greedy action budget or per-patient limit")
    frame = _validate(opportunities)
    if frame.protected_action.to_numpy(bool).any():
        raise ValueError("Protected actions must bypass discretionary allocation")
    mandatory = frame.mandatory.to_numpy(bool)
    admissible = (
        frame.eligible.to_numpy(bool)
        & frame.state_compatible.to_numpy(bool)
        & frame.prerequisites_satisfied.to_numpy(bool)
    )
    if require_empirical_support:
        admissible &= frame.support_flag.to_numpy(bool) | mandatory
    if not allocate_negative_predicted_benefit:
        admissible &= (
            frame.calibrated_incremental_benefit.to_numpy(float) > 0.0
        ) | mandatory
    if np.any(mandatory & ~admissible):
        bad = frame.loc[mandatory & ~admissible, ["patient_id", "action_id"]]
        raise ValueError(
            f"Mandatory actions violate greedy eligibility: {bad.head().to_dict('records')}"
        )

    benefit = frame.calibrated_incremental_benefit.to_numpy(float)
    cost = frame.cost.to_numpy(float)
    priority = benefit if strategy == "score" else benefit / np.maximum(cost, 1e-12)
    order = sorted(
        range(len(frame)),
        key=lambda index: (
            not bool(mandatory[index]),
            -float(priority[index]),
            -float(frame.raw_priority_score.iloc[index]),
            str(frame.patient_id.iloc[index]),
            str(frame.action_id.iloc[index]),
        ),
    )
    selected = np.zeros(len(frame), dtype=bool)
    patient_counts: dict[str, int] = {}
    patient_actions: dict[str, set[str]] = {}
    pool_counts: dict[str, int] = {}
    budget_used = 0.0
    for index in order:
        if not admissible[index]:
            continue
        row = frame.iloc[index]
        patient = str(row.patient_id)
        action = str(row.action_id)
        pool = str(row.capacity_pool)
        if patient_counts.get(patient, 0) >= max_primary_actions_per_patient:
            if mandatory[index]:
                raise ValueError("Mandatory actions exceed greedy per-patient limit")
            continue
        exclusions = set(str(row.mutually_exclusive_with).split("|")) - {""}
        if patient_actions.get(patient, set()).intersection(exclusions):
            continue
        capacity = None if capacity_pools is None else capacity_pools.get(pool)
        if capacity_pools is not None and capacity is None:
            raise ValueError(f"Missing configured capacity pool: {pool}")
        if capacity is not None and pool_counts.get(pool, 0) >= int(capacity):
            if mandatory[index]:
                raise ValueError("Mandatory actions exceed greedy pool capacity")
            continue
        if budget_used + float(cost[index]) > float(shared_budget) + 1e-10:
            if mandatory[index]:
                raise ValueError("Mandatory actions exceed greedy shared budget")
            continue
        selected[index] = True
        budget_used += float(cost[index])
        patient_counts[patient] = patient_counts.get(patient, 0) + 1
        patient_actions.setdefault(patient, set()).add(action)
        pool_counts[pool] = pool_counts.get(pool, 0) + 1

    recommendation = {}
    candidate_mask = admissible & (benefit > 0.0)
    for patient, group in frame.loc[candidate_mask].groupby("patient_id", sort=False):
        recommendation[str(patient)] = str(
            group.sort_values("raw_priority_score", ascending=False).iloc[0].action_id
        )
    action_to_state = dict(zip(frame.action_id, frame.to_state))
    decisions = []
    for patient, group in frame.assign(selected=selected).groupby("patient_id", sort=False):
        chosen = group.loc[group.selected]
        allocated = None if chosen.empty else str(chosen.iloc[0].action_id)
        recommended = recommendation.get(str(patient))
        decision_row = (
            chosen.iloc[0] if not chosen.empty else
            group.loc[group.action_id == recommended].iloc[0]
            if recommended is not None else None
        )
        current_state = str(group.iloc[0].current_care_state)
        decisions.append({
            "patient_id": str(patient),
            "dm77_need_level": group.iloc[0].dm77_need_level,
            "current_care_state": current_state,
            "recommended_action": recommended,
            "allocated_action": allocated,
            "calibrated_incremental_benefit": (
                np.nan if decision_row is None else
                float(decision_row.calibrated_incremental_benefit)
            ),
            "cost": 0.0 if decision_row is None else float(decision_row.cost),
            "capacity_pool": None if decision_row is None else str(decision_row.capacity_pool),
            "allocation_status": "allocated" if allocated else (
                "recommended_not_allocated" if recommended else "no_positive_supported_action"
            ),
            "resulting_care_state": apply_action(current_state, allocated, action_to_state),
            "allocation_reason": (
                f"SELECTED_BY_GREEDY_{strategy.upper()}" if allocated else
                "NOT_SELECTED_UNDER_GREEDY_BUDGET_OR_POOL_CAPACITY"
            ),
        })
    diagnostics = {
        "solver": f"deterministic_greedy_{strategy}",
        "solver_success": True,
        "objective_predicted_benefit": float(benefit[selected].sum()),
        "shared_budget": float(shared_budget),
        "budget_used": float(budget_used),
        "budget_violation": max(0.0, budget_used - float(shared_budget)),
        "selected_actions": int(selected.sum()),
        "mandatory_actions": int((selected & mandatory).sum()),
        "selected_per_capacity_pool": {key: int(value) for key, value in pool_counts.items()},
        "capacity_pool_violations": {
            str(pool): max(0, pool_counts.get(str(pool), 0) - int(capacity))
            for pool, capacity in (capacity_pools or {}).items()
        },
        "eligibility_violations": int((selected & ~frame.eligible.to_numpy(bool)).sum()),
        "support_violations": int((selected & ~frame.support_flag.to_numpy(bool) & ~mandatory).sum()),
        "state_compatibility_violations": int((selected & ~frame.state_compatible.to_numpy(bool)).sum()),
        "prerequisite_violations": int((selected & ~frame.prerequisites_satisfied.to_numpy(bool)).sum()),
        "mutual_exclusion_violations": 0,
        "max_actions_per_patient_violations": int(any(
            value > max_primary_actions_per_patient for value in patient_counts.values()
        )),
        "result_uses_final_package_arithmetic": False,
    }
    violations = (
        diagnostics["budget_violation"], diagnostics["eligibility_violations"],
        diagnostics["support_violations"], diagnostics["state_compatibility_violations"],
        diagnostics["prerequisite_violations"], diagnostics["mutual_exclusion_violations"],
        diagnostics["max_actions_per_patient_violations"],
        *diagnostics["capacity_pool_violations"].values(),
    )
    if any(value > 1e-7 for value in violations):
        raise AssertionError(f"Greedy allocator returned infeasible solution: {diagnostics}")
    return ActionAllocationResult(selected, pd.DataFrame(decisions), diagnostics)
