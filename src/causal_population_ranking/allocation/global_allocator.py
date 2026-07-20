from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class GlobalAllocationResult:
    selected: np.ndarray
    final_packages: pd.DataFrame
    diagnostics: dict


def _validate_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id", "transition_index", "calibrated_benefit", "cost",
        "eligible", "supported",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Allocation opportunities are missing columns: {missing}")
    out = frame.reset_index(drop=True).copy()
    out.patient_id = out.patient_id.astype(str)
    out.transition_index = out.transition_index.astype(int)
    if out.duplicated(["patient_id", "transition_index"]).any():
        raise ValueError("Allocation requires one row per patient-transition opportunity")
    if not np.isfinite(out[["calibrated_benefit", "cost"]].to_numpy(float)).all():
        raise ValueError("Allocation benefits and costs must be finite")
    if (out.cost <= 0).any():
        raise ValueError("Transition costs must be positive")
    observed = set(out.transition_index)
    if observed != set(range(5)):
        raise ValueError("Allocation requires transition indices 0..4")
    expected = set(range(5))
    for patient, group in out.groupby("patient_id", sort=False):
        if set(group.transition_index) != expected:
            raise ValueError(f"Patient {patient} does not have all five opportunity rows")
    return out


def _constraint_system(
    frame: pd.DataFrame,
    shared_budget: float,
    transition_capacities: Mapping[int | str, int] | None,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    rows, lower, upper = [], [], []
    rows.append({index: float(cost) for index, cost in enumerate(frame.cost.to_numpy(float))})
    lower.append(-np.inf)
    upper.append(float(shared_budget))
    lookup = {
        (patient, int(transition)): row
        for row, (patient, transition) in enumerate(zip(frame.patient_id, frame.transition_index))
    }
    for patient in frame.patient_id.unique():
        for transition in range(1, 5):
            rows.append({lookup[(patient, transition)]: 1.0, lookup[(patient, transition - 1)]: -1.0})
            lower.append(-np.inf)
            upper.append(0.0)
    if transition_capacities:
        for key, capacity in transition_capacities.items():
            transition = int(key) if isinstance(key, (int, np.integer)) else int(str(key).split("_to_")[0]) - 1
            if transition not in range(5) or int(capacity) < 0:
                raise ValueError("Transition capacities must be non-negative counts for transitions 0..4")
            members = np.flatnonzero(frame.transition_index.to_numpy(int) == transition)
            rows.append({int(index): 1.0 for index in members})
            lower.append(-np.inf)
            upper.append(float(capacity))
    data, row_index, column_index = [], [], []
    for constraint, values in enumerate(rows):
        for column, value in values.items():
            row_index.append(constraint)
            column_index.append(column)
            data.append(value)
    matrix = csr_matrix((data, (row_index, column_index)), shape=(len(rows), len(frame)))
    return matrix, np.asarray(lower), np.asarray(upper)


def _greedy_frontier(
    frame: pd.DataFrame,
    upper_bound: np.ndarray,
    shared_budget: float,
    transition_capacities: Mapping[int | str, int] | None,
) -> np.ndarray:
    """Deterministic feasible heuristic; it does not guarantee a global optimum."""

    selected = np.zeros(len(frame), dtype=bool)
    used = 0.0
    counts = np.zeros(5, dtype=int)
    capacity = np.full(5, np.iinfo(np.int64).max)
    if transition_capacities:
        for key, value in transition_capacities.items():
            index = int(key) if isinstance(key, (int, np.integer)) else int(str(key).split("_to_")[0]) - 1
            capacity[index] = int(value)
    patient_rows = {
        patient: group.sort_values("transition_index").index.to_numpy()
        for patient, group in frame.groupby("patient_id", sort=False)
    }
    frontier = {patient: 0 for patient in patient_rows}
    while True:
        candidates = []
        for patient, transition in frontier.items():
            if transition >= 5:
                continue
            row = int(patient_rows[patient][transition])
            if upper_bound[row] < 0.5 or counts[transition] >= capacity[transition]:
                continue
            ratio = frame.calibrated_benefit.iat[row] / frame.cost.iat[row]
            candidates.append((ratio, frame.calibrated_benefit.iat[row], patient, row, transition))
        candidates.sort(reverse=True)
        chosen = next((value for value in candidates if used + frame.cost.iat[value[3]] <= shared_budget + 1e-9), None)
        if chosen is None:
            break
        _, _, patient, row, transition = chosen
        selected[row] = True
        used += float(frame.cost.iat[row])
        counts[transition] += 1
        frontier[patient] += 1
    return selected


def _feasibility(
    frame: pd.DataFrame,
    selected: np.ndarray,
    shared_budget: float,
    transition_capacities: Mapping[int | str, int] | None,
    require_empirical_support: bool,
) -> dict:
    selected = np.asarray(selected, dtype=bool)
    budget_used = float(frame.loc[selected, "cost"].sum())
    eligibility_violations = int((selected & ~frame.eligible.to_numpy(bool)).sum())
    support_violations = int((selected & ~frame.supported.to_numpy(bool)).sum()) if require_empirical_support else 0
    precedence_violations = 0
    for _, group in frame.assign(selected=selected).groupby("patient_id", sort=False):
        decisions = group.sort_values("transition_index").selected.to_numpy(bool)
        precedence_violations += int(np.sum(decisions[1:] & ~decisions[:-1]))
    transition_violations = {}
    if transition_capacities:
        for key, capacity in transition_capacities.items():
            transition = int(key) if isinstance(key, (int, np.integer)) else int(str(key).split("_to_")[0]) - 1
            count = int(np.sum(selected & (frame.transition_index.to_numpy(int) == transition)))
            transition_violations[str(key)] = max(0, count - int(capacity))
    return {
        "budget_used": budget_used,
        "shared_budget": float(shared_budget),
        "budget_violation": max(0.0, budget_used - float(shared_budget)),
        "eligibility_violations": eligibility_violations,
        "support_violations": support_violations,
        "precedence_violations": precedence_violations,
        "transition_capacity_violations": transition_violations,
        "selected_opportunities": int(selected.sum()),
        "selected_per_transition": {
            str(index): int(np.sum(selected & (frame.transition_index.to_numpy(int) == index)))
            for index in range(5)
        },
    }


def allocate_global_opportunities(
    opportunities: pd.DataFrame,
    shared_budget: float,
    transition_capacities: Mapping[int | str, int] | None = None,
    require_empirical_support: bool = True,
    allocate_negative_predicted_benefit: bool = False,
    exact_solver: bool = True,
    solver_time_limit_seconds: float = 120.0,
) -> GlobalAllocationResult:
    """Jointly allocate nested increments under one shared capacity budget."""

    if shared_budget < 0 or solver_time_limit_seconds <= 0:
        raise ValueError("Shared budget and solver time limit are invalid")
    frame = _validate_opportunities(opportunities)
    upper_bound = frame.eligible.to_numpy(bool).astype(float)
    if require_empirical_support:
        upper_bound *= frame.supported.to_numpy(bool)
    if not allocate_negative_predicted_benefit:
        upper_bound *= frame.calibrated_benefit.to_numpy(float) > 0.0
    if exact_solver:
        matrix, lower, upper = _constraint_system(frame, shared_budget, transition_capacities)
        result = milp(
            c=-frame.calibrated_benefit.to_numpy(float),
            integrality=np.ones(len(frame), dtype=np.int8),
            bounds=Bounds(np.zeros(len(frame)), upper_bound),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": float(solver_time_limit_seconds)},
        )
        if result.x is None or not result.success:
            raise RuntimeError(f"Exact allocation MILP failed: status={result.status}, {result.message}")
        selected = result.x > 0.5
        solver_diagnostics = {
            "solver": "scipy.optimize.milp_highs",
            "exact_solver_requested": True,
            "solver_success": bool(result.success),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "objective_predicted_benefit": float(frame.loc[selected, "calibrated_benefit"].sum()),
        }
    else:
        selected = _greedy_frontier(frame, upper_bound, shared_budget, transition_capacities)
        solver_diagnostics = {
            "solver": "greedy_frontier_heuristic",
            "exact_solver_requested": False,
            "optimality_guaranteed": False,
            "objective_predicted_benefit": float(frame.loc[selected, "calibrated_benefit"].sum()),
        }
    feasibility = _feasibility(
        frame, selected, shared_budget, transition_capacities, require_empirical_support
    )
    if any((feasibility["budget_violation"] > 1e-7, feasibility["eligibility_violations"],
            feasibility["support_violations"], feasibility["precedence_violations"],
            any(feasibility["transition_capacity_violations"].values()))):
        raise AssertionError(f"Allocator returned an infeasible solution: {feasibility}")
    selected_frame = frame.assign(selected=selected)
    final_packages = selected_frame.groupby("patient_id", sort=False).selected.sum().astype(int).add(1).rename(
        "final_package"
    ).reset_index()
    feasibility["patients_per_final_level"] = {
        str(level): int(count) for level, count in final_packages.final_package.value_counts().sort_index().items()
    }
    return GlobalAllocationResult(selected, final_packages, {**solver_diagnostics, **feasibility})
