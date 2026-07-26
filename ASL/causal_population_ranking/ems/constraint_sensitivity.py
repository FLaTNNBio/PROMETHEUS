"""Allocation constraint sensitivity with a frozen EMS ranking.

The scenario catalog and every non-oracle allocation are created before simulated
truth is opened. Across scenarios only the operational regime changes: allocator
values, mission ordering, calibration and model parameters remain fixed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from .policy_benchmark import EMSPolicyInputs


SENSITIVITY_POLICY_ORDER = (
    ("Oracle", "MILP"),
    ("PROMETHEUS", "MILP"),
    ("PROMETHEUS", "Greedy benefit"),
    ("PROMETHEUS", "Greedy benefit/cost"),
    ("Risk-first", "MILP"),
    ("Need-first", "MILP"),
    ("Outcome-first", "MILP"),
    ("Profile-mean", "MILP"),
    ("FCFS", "Greedy"),
    ("Random", "Greedy"),
)


@dataclass(frozen=True)
class EMSFrozenConstraintSensitivity:
    scenarios: pd.DataFrame
    selected_decisions: pd.DataFrame
    decision_summary: pd.DataFrame
    audit: dict[str, Any]


def _label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def build_ems_constraint_scenarios(
    mission_count: int,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the prespecified, bounded catalog of operational regimes."""

    allocation = config["allocation"]
    benchmark = config["policy_benchmark"]
    sensitivity = config["constraint_sensitivity"]
    served = int(np.floor(
        mission_count * float(allocation["nurse_or_higher_fraction"])
    ))
    medicalized = int(np.floor(
        mission_count * float(allocation["medicalized_fraction"])
    ))
    nurse = served - medicalized
    nurse_cost = float(benchmark["nurse_response_cost"])
    medicalized_cost = float(benchmark["medicalized_response_cost"])
    budget = float(benchmark["budget_per_test_mission"]) * mission_count
    nursing = nurse + 0.5 * medicalized
    physician = float(medicalized)
    records: list[dict[str, Any]] = []

    def add(
        experiment: str,
        scenario: str,
        order: int,
        *,
        budget_multiplier: float = 1.0,
        capacity_multiplier: float = 1.0,
        nurse_capacity_multiplier: float | None = None,
        medicalized_capacity_multiplier: float | None = None,
        nurse_cost_multiplier: float = 1.0,
        medicalized_cost_multiplier: float = 1.0,
        nursing_capacity_multiplier: float = 1.0,
        physician_capacity_multiplier: float = 1.0,
    ) -> None:
        overall = float(capacity_multiplier)
        nurse_multiplier = (
            overall if nurse_capacity_multiplier is None
            else float(nurse_capacity_multiplier)
        )
        medical_multiplier = (
            overall if medicalized_capacity_multiplier is None
            else float(medicalized_capacity_multiplier)
        )
        nurse_cap = min(
            mission_count,
            int(np.floor(nurse * nurse_multiplier)),
        )
        medical_cap = min(
            mission_count,
            int(np.floor(medicalized * medical_multiplier)),
        )
        served_cap = min(
            mission_count,
            int(np.floor(served * overall)),
            nurse_cap + medical_cap,
        )
        records.append({
            "experiment": experiment,
            "scenario": scenario,
            "scenario_id": f"{experiment}:{scenario}",
            "scenario_order": int(order),
            "budget_multiplier": float(budget_multiplier),
            "capacity_multiplier": overall,
            "nurse_capacity_multiplier": nurse_multiplier,
            "medicalized_capacity_multiplier": medical_multiplier,
            "nurse_cost_multiplier": float(nurse_cost_multiplier),
            "medicalized_cost_multiplier": float(
                medicalized_cost_multiplier
            ),
            "nursing_capacity_multiplier": float(
                nursing_capacity_multiplier
            ),
            "physician_capacity_multiplier": float(
                physician_capacity_multiplier
            ),
            "budget": budget * float(budget_multiplier),
            "served_capacity": served_cap,
            "nurse_capacity": nurse_cap,
            "medicalized_capacity": medical_cap,
            "nurse_cost": nurse_cost * float(nurse_cost_multiplier),
            "medicalized_cost": (
                medicalized_cost * float(medicalized_cost_multiplier)
            ),
            "nursing_capacity": nursing * float(
                nursing_capacity_multiplier
            ),
            "physician_capacity": physician * float(
                physician_capacity_multiplier
            ),
            "nurse_nursing_units": 1.0,
            "medicalized_nursing_units": 0.5,
            "nurse_physician_units": 0.0,
            "medicalized_physician_units": 1.0,
        })

    for order, multiplier in enumerate(sensitivity["budget_multipliers"]):
        add(
            "budget_curve",
            f"budget_{_label(multiplier)}",
            order,
            budget_multiplier=float(multiplier),
        )
    for order, multiplier in enumerate(sensitivity["capacity_multipliers"]):
        add(
            "capacity_curve",
            f"capacity_{_label(multiplier)}",
            order,
            capacity_multiplier=float(multiplier),
        )

    add("profile_bottleneck", "nominal", 0)
    add(
        "profile_bottleneck",
        "nurse_supported_40pct",
        1,
        nurse_capacity_multiplier=0.40,
    )
    add(
        "profile_bottleneck",
        "medicalized_30pct",
        2,
        medicalized_capacity_multiplier=0.30,
    )
    add(
        "profile_bottleneck",
        "nursing_resource_50pct",
        3,
        nursing_capacity_multiplier=0.50,
    )
    add(
        "profile_bottleneck",
        "joint_profile_50pct",
        4,
        nurse_capacity_multiplier=0.50,
        medicalized_capacity_multiplier=0.50,
    )

    add(
        "cost_heterogeneity",
        "uniform_costs",
        0,
        nurse_cost_multiplier=1.0,
        medicalized_cost_multiplier=(nurse_cost / medicalized_cost),
    )
    add("cost_heterogeneity", "moderate_heterogeneity", 1)
    add(
        "cost_heterogeneity",
        "strong_heterogeneity",
        2,
        medicalized_cost_multiplier=(10.0 / medicalized_cost),
    )

    add("shared_resources", "workforce_balanced", 0)
    add(
        "shared_resources",
        "nursing_shortage_50pct",
        1,
        nursing_capacity_multiplier=0.50,
    )
    add(
        "shared_resources",
        "physician_shortage_50pct",
        2,
        physician_capacity_multiplier=0.50,
    )
    add(
        "shared_resources",
        "joint_workforce_shortage",
        3,
        nursing_capacity_multiplier=0.50,
        physician_capacity_multiplier=0.50,
    )
    add(
        "shared_resources",
        "nursing_and_budget_shortage",
        4,
        budget_multiplier=0.50,
        nursing_capacity_multiplier=0.50,
    )

    add(
        "combined_stress",
        "favorable",
        0,
        budget_multiplier=1.25,
        capacity_multiplier=1.25,
    )
    add("combined_stress", "nominal", 1)
    add(
        "combined_stress",
        "severe_scarcity",
        2,
        budget_multiplier=0.50,
        capacity_multiplier=0.50,
        nurse_cost_multiplier=1.25,
        medicalized_cost_multiplier=1.25,
    )
    add(
        "combined_stress",
        "workforce_disruption",
        3,
        nurse_capacity_multiplier=0.30,
        nursing_capacity_multiplier=0.50,
    )
    add(
        "combined_stress",
        "cost_shock",
        4,
        budget_multiplier=0.75,
        medicalized_cost_multiplier=1.50,
    )

    result = pd.DataFrame(records)
    if result.scenario_id.duplicated().any():
        raise AssertionError("EMS constraint scenario identifiers must be unique")
    expected_count = (
        len(sensitivity["budget_multipliers"])
        + len(sensitivity["capacity_multipliers"])
        + 5 + 3 + 5 + 5
    )
    if len(result) != expected_count:
        raise AssertionError("EMS constraint scenario catalog is incomplete")
    return result


def _stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}|{token}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _scenario_arrays(
    scenario: pd.Series,
    mission_count: int,
) -> dict[str, np.ndarray]:
    return {
        "cost": np.concatenate([
            np.full(mission_count, float(scenario.nurse_cost)),
            np.full(mission_count, float(scenario.medicalized_cost)),
        ]),
        "nursing": np.concatenate([
            np.full(mission_count, float(scenario.nurse_nursing_units)),
            np.full(
                mission_count,
                float(scenario.medicalized_nursing_units),
            ),
        ]),
        "physician": np.concatenate([
            np.full(mission_count, float(scenario.nurse_physician_units)),
            np.full(
                mission_count,
                float(scenario.medicalized_physician_units),
            ),
        ]),
    }


def _solve_scenario_milp(
    mission_ids: np.ndarray,
    nurse_value: np.ndarray,
    medicalized_value: np.ndarray,
    scenario: pd.Series,
    config: Mapping[str, Any],
    *,
    policy: str,
    oracle: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(mission_ids)
    arrays = _scenario_arrays(scenario, n)
    values = np.concatenate([nurse_value, medicalized_value]).astype(float)
    tie_seed = _stable_seed(
        int(config["constraint_sensitivity"]["allocator_tie_seed"]),
        policy,
    )
    tie = np.random.default_rng(tie_seed).uniform(0.0, 1e-10, 2 * n)
    objective = -(values + tie) + arrays["cost"] * 1e-11
    row = np.arange(n)
    one_profile = coo_matrix(
        (
            np.ones(2 * n),
            (
                np.concatenate([row, row]),
                np.concatenate([row, n + row]),
            ),
        ),
        shape=(n, 2 * n),
    ).tocsr()

    def constraint_row(values_: np.ndarray) -> coo_matrix:
        return coo_matrix(
            (
                values_,
                (np.zeros(2 * n, dtype=int), np.arange(2 * n)),
            ),
            shape=(1, 2 * n),
        ).tocsr()

    served = constraint_row(np.ones(2 * n))
    nurse = constraint_row(np.concatenate([np.ones(n), np.zeros(n)]))
    medicalized = constraint_row(
        np.concatenate([np.zeros(n), np.ones(n)])
    )
    matrix = vstack([
        one_profile,
        served,
        nurse,
        medicalized,
        constraint_row(arrays["cost"]),
        constraint_row(arrays["nursing"]),
        constraint_row(arrays["physician"]),
    ], format="csr")
    lower = np.full(n + 6, -np.inf)
    upper = np.concatenate([
        np.ones(n),
        [
            float(scenario.served_capacity),
            float(scenario.nurse_capacity),
            float(scenario.medicalized_capacity),
            float(scenario.budget),
            float(scenario.nursing_capacity),
            float(scenario.physician_capacity),
        ],
    ])
    result = milp(
        objective,
        integrality=np.ones(2 * n),
        bounds=Bounds(np.zeros(2 * n), np.ones(2 * n)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={
            "time_limit": float(
                config["constraint_sensitivity"]["solver_time_limit_seconds"]
            )
        },
    )
    if result.x is None:
        raise RuntimeError(
            f"EMS sensitivity MILP failed for {scenario.scenario_id}/"
            f"{policy}: {result.message}"
        )
    nurse_selected = result.x[:n] > 0.5
    medicalized_selected = result.x[n:] > 0.5
    tier = nurse_selected.astype(int) + 2 * medicalized_selected.astype(int)
    selected = tier > 0
    selected_value = np.where(
        nurse_selected,
        nurse_value,
        np.where(medicalized_selected, medicalized_value, 0.0),
    )
    selected_cost = np.where(
        nurse_selected,
        float(scenario.nurse_cost),
        np.where(
            medicalized_selected,
            float(scenario.medicalized_cost),
            0.0,
        ),
    )
    selected_nursing = np.where(
        nurse_selected,
        float(scenario.nurse_nursing_units),
        np.where(
            medicalized_selected,
            float(scenario.medicalized_nursing_units),
            0.0,
        ),
    )
    selected_physician = np.where(
        nurse_selected,
        float(scenario.nurse_physician_units),
        np.where(
            medicalized_selected,
            float(scenario.medicalized_physician_units),
            0.0,
        ),
    )
    decisions = pd.DataFrame({
        "scenario_id": str(scenario.scenario_id),
        "mission_id": mission_ids[selected].astype(str),
        "policy": policy,
        "optimizer": "MILP",
        "allocated_tier": tier[selected],
        "allocator_value": selected_value[selected],
        "allocated_cost": selected_cost[selected],
        "nursing_units": selected_nursing[selected],
        "physician_units": selected_physician[selected],
        "uses_oracle_allocator_value": bool(oracle),
    })
    positive = np.maximum(nurse_value, medicalized_value) > 0.0
    return decisions, {
        "scenario_id": str(scenario.scenario_id),
        "policy": policy,
        "optimizer": "MILP",
        "positive_recommendations": int(positive.sum()),
        "served": int(selected.sum()),
        "medicalized": int(medicalized_selected.sum()),
        "cost_used": float(selected_cost.sum()),
        "nursing_used": float(selected_nursing.sum()),
        "physician_used": float(selected_physician.sum()),
        "solver_status": int(result.status),
        "solver_success": bool(result.success),
        "tie_seed": tie_seed,
    }


def _prometheus_greedy(
    mission_ids: np.ndarray,
    nurse_value: np.ndarray,
    medicalized_value: np.ndarray,
    scenario: pd.Series,
    config: Mapping[str, Any],
    *,
    optimizer: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(mission_ids)
    arrays = _scenario_arrays(scenario, n)
    values = np.concatenate([nurse_value, medicalized_value])
    tiers = np.concatenate([np.ones(n, dtype=int), np.full(n, 2, dtype=int)])
    missions = np.concatenate([np.arange(n), np.arange(n)])
    seed = int(config["constraint_sensitivity"]["greedy_tie_seed"])
    tie = np.random.default_rng(seed).uniform(0.0, 1e-12, 2 * n)
    if optimizer == "Greedy benefit":
        criterion = values + tie
    elif optimizer == "Greedy benefit/cost":
        criterion = values / arrays["cost"] + tie
    else:
        raise ValueError(f"Unsupported PROMETHEUS greedy optimizer: {optimizer}")
    order = np.argsort(-criterion, kind="mergesort")
    tier = np.zeros(n, dtype=int)
    selected_value = np.zeros(n)
    selected_cost = np.zeros(n)
    selected_nursing = np.zeros(n)
    selected_physician = np.zeros(n)
    nurse_count = 0
    medicalized_count = 0
    served_count = 0
    cost_used = 0.0
    nursing_used = 0.0
    physician_used = 0.0
    for candidate in order:
        mission = int(missions[candidate])
        candidate_tier = int(tiers[candidate])
        if tier[mission] or values[candidate] <= 0.0:
            continue
        if served_count >= int(scenario.served_capacity):
            break
        if (
            candidate_tier == 1
            and nurse_count >= int(scenario.nurse_capacity)
        ) or (
            candidate_tier == 2
            and medicalized_count >= int(scenario.medicalized_capacity)
        ):
            continue
        cost = float(arrays["cost"][candidate])
        nursing = float(arrays["nursing"][candidate])
        physician = float(arrays["physician"][candidate])
        if (
            cost_used + cost > float(scenario.budget) + 1e-9
            or nursing_used + nursing
            > float(scenario.nursing_capacity) + 1e-9
            or physician_used + physician
            > float(scenario.physician_capacity) + 1e-9
        ):
            continue
        tier[mission] = candidate_tier
        selected_value[mission] = float(values[candidate])
        selected_cost[mission] = cost
        selected_nursing[mission] = nursing
        selected_physician[mission] = physician
        cost_used += cost
        nursing_used += nursing
        physician_used += physician
        served_count += 1
        nurse_count += int(candidate_tier == 1)
        medicalized_count += int(candidate_tier == 2)
    selected = tier > 0
    decisions = pd.DataFrame({
        "scenario_id": str(scenario.scenario_id),
        "mission_id": mission_ids[selected].astype(str),
        "policy": "PROMETHEUS",
        "optimizer": optimizer,
        "allocated_tier": tier[selected],
        "allocator_value": selected_value[selected],
        "allocated_cost": selected_cost[selected],
        "nursing_units": selected_nursing[selected],
        "physician_units": selected_physician[selected],
        "uses_oracle_allocator_value": False,
    })
    positive = np.maximum(nurse_value, medicalized_value) > 0.0
    return decisions, {
        "scenario_id": str(scenario.scenario_id),
        "policy": "PROMETHEUS",
        "optimizer": optimizer,
        "positive_recommendations": int(positive.sum()),
        "served": served_count,
        "medicalized": medicalized_count,
        "cost_used": cost_used,
        "nursing_used": nursing_used,
        "physician_used": physician_used,
        "tie_seed": seed,
    }


def _ordered_greedy(
    metadata: pd.DataFrame,
    scenario: pd.Series,
    config: Mapping[str, Any],
    *,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(metadata)
    if policy == "FCFS":
        order = np.argsort(
            metadata.simulated_arrival_order.to_numpy(int),
            kind="mergesort",
        )
        seed = None
    elif policy == "Random":
        seed = int(config["policy_benchmark"]["random_policy_seed"])
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"Unsupported sensitivity policy: {policy}")
    tier = np.zeros(n, dtype=int)
    cost = np.zeros(n)
    nursing = np.zeros(n)
    physician = np.zeros(n)
    served_count = 0
    nurse_count = 0
    medicalized_count = 0
    cost_used = 0.0
    nursing_used = 0.0
    physician_used = 0.0
    for mission in order:
        if served_count >= int(scenario.served_capacity):
            break
        chosen = 0
        for candidate_tier in (2, 1):
            if (
                candidate_tier == 1
                and nurse_count >= int(scenario.nurse_capacity)
            ) or (
                candidate_tier == 2
                and medicalized_count >= int(scenario.medicalized_capacity)
            ):
                continue
            candidate_cost = float(
                scenario.medicalized_cost
                if candidate_tier == 2 else scenario.nurse_cost
            )
            candidate_nursing = float(
                scenario.medicalized_nursing_units
                if candidate_tier == 2 else scenario.nurse_nursing_units
            )
            candidate_physician = float(
                scenario.medicalized_physician_units
                if candidate_tier == 2 else scenario.nurse_physician_units
            )
            if (
                cost_used + candidate_cost <= float(scenario.budget) + 1e-9
                and nursing_used + candidate_nursing
                <= float(scenario.nursing_capacity) + 1e-9
                and physician_used + candidate_physician
                <= float(scenario.physician_capacity) + 1e-9
            ):
                chosen = candidate_tier
                cost[mission] = candidate_cost
                nursing[mission] = candidate_nursing
                physician[mission] = candidate_physician
                break
        if not chosen:
            continue
        tier[mission] = chosen
        cost_used += cost[mission]
        nursing_used += nursing[mission]
        physician_used += physician[mission]
        served_count += 1
        nurse_count += int(chosen == 1)
        medicalized_count += int(chosen == 2)
    selected = tier > 0
    decisions = pd.DataFrame({
        "scenario_id": str(scenario.scenario_id),
        "mission_id": metadata.loc[selected, "mission_id"].astype(str),
        "policy": policy,
        "optimizer": "Greedy",
        "allocated_tier": tier[selected],
        "allocator_value": np.zeros(selected.sum()),
        "allocated_cost": cost[selected],
        "nursing_units": nursing[selected],
        "physician_units": physician[selected],
        "uses_oracle_allocator_value": False,
    })
    return decisions, {
        "scenario_id": str(scenario.scenario_id),
        "policy": policy,
        "optimizer": "Greedy",
        "positive_recommendations": n,
        "served": served_count,
        "medicalized": medicalized_count,
        "cost_used": cost_used,
        "nursing_used": nursing_used,
        "physician_used": physician_used,
        "random_seed": seed,
    }


def freeze_ems_constraint_sensitivity(
    inputs: EMSPolicyInputs,
    config: Mapping[str, Any],
) -> EMSFrozenConstraintSensitivity:
    """Allocate all non-oracle policies under every frozen constraint regime."""

    metadata = inputs.mission_metadata.sort_values(
        "mission_id"
    ).reset_index(drop=True)
    mission_ids = metadata.mission_id.to_numpy(str)
    scenarios = build_ems_constraint_scenarios(len(metadata), config)
    selected: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    value_by_policy = {
        policy: group.set_index("mission_id").loc[mission_ids]
        for policy, group in inputs.allocator_values.groupby("policy")
    }
    for scenario in scenarios.itertuples(index=False):
        scenario_series = pd.Series(scenario._asdict())
        for policy in (
            "PROMETHEUS",
            "Risk-first",
            "Need-first",
            "Outcome-first",
            "Profile-mean",
        ):
            values = value_by_policy[policy]
            decisions, summary = _solve_scenario_milp(
                mission_ids,
                values.nurse_value.to_numpy(float),
                values.medicalized_value.to_numpy(float),
                scenario_series,
                config,
                policy=policy,
                oracle=False,
            )
            selected.append(decisions)
            summaries.append(summary)
            if policy == "PROMETHEUS":
                for optimizer in (
                    "Greedy benefit",
                    "Greedy benefit/cost",
                ):
                    decisions, summary = _prometheus_greedy(
                        mission_ids,
                        values.nurse_value.to_numpy(float),
                        values.medicalized_value.to_numpy(float),
                        scenario_series,
                        config,
                        optimizer=optimizer,
                    )
                    selected.append(decisions)
                    summaries.append(summary)
        for policy in ("FCFS", "Random"):
            decisions, summary = _ordered_greedy(
                metadata,
                scenario_series,
                config,
                policy=policy,
            )
            selected.append(decisions)
            summaries.append(summary)
    decisions = pd.concat(selected, ignore_index=True)
    if decisions.uses_oracle_allocator_value.astype(bool).any():
        raise AssertionError("Oracle entered frozen EMS constraint allocations")
    return EMSFrozenConstraintSensitivity(
        scenarios=scenarios,
        selected_decisions=decisions,
        decision_summary=pd.DataFrame(summaries),
        audit={
            "status": "frozen_before_oracle_evaluation",
            "ranking_refit_per_scenario": False,
            "calibration_refit_per_scenario": False,
            "allocator_values_changed_per_scenario": False,
            "only_constraints_changed_per_scenario": True,
            "oracle_inputs_used": False,
            "scenario_count": int(len(scenarios)),
            "nonoracle_policy_optimizer_count": 9,
            "coverage_constraints_in_primary_benchmark": False,
            "coverage_constraints_reason": (
                "reserved_for_secondary_governance_analysis"
            ),
        },
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_ems_constraint_sensitivity(
    frozen: EMSFrozenConstraintSensitivity,
    inputs: EMSPolicyInputs,
    evaluation_only: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Open truth after freeze and evaluate every policy-regime allocation."""

    mission_ids = inputs.mission_metadata.sort_values(
        "mission_id"
    ).mission_id.to_numpy(str)
    truth = evaluation_only.set_index("mission_id").loc[mission_ids]
    nurse_true = truth.true_increment_nurse_supported.to_numpy(float)
    medicalized_true = (
        nurse_true + truth.true_increment_medicalized.to_numpy(float)
    )
    all_decisions = [frozen.selected_decisions]
    oracle_summaries = []
    for scenario in frozen.scenarios.itertuples(index=False):
        scenario_series = pd.Series(scenario._asdict())
        decisions, summary = _solve_scenario_milp(
            mission_ids,
            nurse_true,
            medicalized_true,
            scenario_series,
            config,
            policy="Oracle",
            oracle=True,
        )
        all_decisions.append(decisions)
        oracle_summaries.append(summary)
    decisions = pd.concat(all_decisions, ignore_index=True)
    summary = pd.concat([
        frozen.decision_summary,
        pd.DataFrame(oracle_summaries),
    ], ignore_index=True)
    scenario_by_id = frozen.scenarios.set_index("scenario_id")
    summary_key = summary.set_index(["scenario_id", "policy", "optimizer"])
    order = {value: index for index, value in enumerate(SENSITIVITY_POLICY_ORDER)}
    rows = []
    selected_sets: dict[tuple[str, str, str], set[str]] = {}
    true_by_id = pd.DataFrame({
        "mission_id": mission_ids,
        "nurse_true": nurse_true,
        "medicalized_true": medicalized_true,
    }).set_index("mission_id")
    for scenario_id, scenario_decisions in decisions.groupby("scenario_id"):
        oracle_group = scenario_decisions.loc[
            scenario_decisions.policy.eq("Oracle")
            & scenario_decisions.optimizer.eq("MILP")
        ]

        def value(group: pd.DataFrame) -> float:
            if group.empty:
                return 0.0
            aligned = true_by_id.loc[group.mission_id.astype(str)]
            tier = group.allocated_tier.to_numpy(int)
            return float(np.where(
                tier == 1,
                aligned.nurse_true.to_numpy(float),
                aligned.medicalized_true.to_numpy(float),
            ).sum())

        oracle_value = value(oracle_group)
        for policy, optimizer in SENSITIVITY_POLICY_ORDER:
            group = scenario_decisions.loc[
                scenario_decisions.policy.eq(policy)
                & scenario_decisions.optimizer.eq(optimizer)
            ]
            key = (scenario_id, policy, optimizer)
            stats = summary_key.loc[key]
            true_value = value(group)
            served = int(stats.served)
            medicalized = int(stats.medicalized)
            nurse = served - medicalized
            positive = int(stats.positive_recommendations)
            selected_sets[key] = set(group.mission_id.astype(str))
            scenario = scenario_by_id.loc[scenario_id]
            rows.append({
                "experiment": scenario.experiment,
                "scenario": scenario.scenario,
                "scenario_id": scenario_id,
                "scenario_order": int(scenario.scenario_order),
                "policy": policy,
                "optimizer": optimizer,
                "true_value": true_value,
                "oracle_value": oracle_value,
                "normalized_value": (
                    true_value / oracle_value if oracle_value > 0.0 else np.nan
                ),
                "regret": oracle_value - true_value,
                "positive_recommendations": positive,
                "deferral_rate": (
                    max(positive - served, 0) / positive
                    if positive else 0.0
                ),
                "served": served,
                "nurse_supported": nurse,
                "medicalized": medicalized,
                "nurse_share": nurse / served if served else 0.0,
                "medicalized_share": (
                    medicalized / served if served else 0.0
                ),
                "cost_used": float(stats.cost_used),
                "benefit_per_cost": (
                    true_value / float(stats.cost_used)
                    if float(stats.cost_used) > 0.0 else np.nan
                ),
                "nursing_used": float(stats.nursing_used),
                "physician_used": float(stats.physician_used),
                "budget": float(scenario.budget),
                "served_capacity": int(scenario.served_capacity),
                "nurse_capacity": int(scenario.nurse_capacity),
                "medicalized_capacity": int(
                    scenario.medicalized_capacity
                ),
                "nursing_capacity": float(scenario.nursing_capacity),
                "physician_capacity": float(scenario.physician_capacity),
                "uses_oracle": policy == "Oracle",
                "oracle_evaluation_only": policy != "Oracle",
                "oracle_access_for_policy_construction": policy == "Oracle",
                "oracle_access_for_evaluation": True,
                "_policy_order": order[(policy, optimizer)],
            })
    metrics = pd.DataFrame(rows).sort_values([
        "experiment", "scenario_order", "_policy_order"
    ]).drop(columns="_policy_order").reset_index(drop=True)

    stability_rows = []
    for experiment in ("budget_curve", "capacity_curve"):
        experiment_scenarios = frozen.scenarios.loc[
            frozen.scenarios.experiment.eq(experiment)
        ].sort_values("scenario_order")
        ids = experiment_scenarios.scenario_id.tolist()
        for left, right in zip(ids[:-1], ids[1:]):
            for policy, optimizer in SENSITIVITY_POLICY_ORDER:
                stability_rows.append({
                    "experiment": experiment,
                    "left_scenario_id": left,
                    "right_scenario_id": right,
                    "policy": policy,
                    "optimizer": optimizer,
                    "served_set_jaccard": _jaccard(
                        selected_sets[(left, policy, optimizer)],
                        selected_sets[(right, policy, optimizer)],
                    ),
                    "uses_oracle": policy == "Oracle",
                    "oracle_evaluation_only": policy != "Oracle",
                    "oracle_access_for_policy_construction": policy == "Oracle",
                    "oracle_access_for_evaluation": True,
                })
    stability = pd.DataFrame(stability_rows)

    marginal_rows = []
    budget_metrics = metrics.loc[
        metrics.experiment.eq("budget_curve")
    ].sort_values(["policy", "optimizer", "scenario_order"])
    for (policy, optimizer), group in budget_metrics.groupby([
        "policy", "optimizer"
    ], sort=False):
        previous = None
        for row in group.itertuples(index=False):
            if previous is not None:
                marginal_rows.append({
                    "policy": policy,
                    "optimizer": optimizer,
                    "left_scenario_id": previous.scenario_id,
                    "right_scenario_id": row.scenario_id,
                    "delta_budget": float(row.budget - previous.budget),
                    "delta_true_value": float(
                        row.true_value - previous.true_value
                    ),
                    "marginal_value_per_budget": (
                        float(row.true_value - previous.true_value)
                        / float(row.budget - previous.budget)
                    ),
                    "uses_oracle": policy == "Oracle",
                    "oracle_evaluation_only": policy != "Oracle",
                    "oracle_access_for_policy_construction": policy == "Oracle",
                    "oracle_access_for_evaluation": True,
                })
            previous = row
    marginal = pd.DataFrame(marginal_rows)
    audit = {
        "status": "evaluated_after_freeze",
        "oracle_scenarios_solved": int(len(oracle_summaries)),
        "scenario_count": int(frozen.audit["scenario_count"]),
        "policy_optimizer_count_including_oracle": len(
            SENSITIVITY_POLICY_ORDER
        ),
        "ranking_refit_per_scenario": False,
        "same_constraints_within_each_scenario": True,
    }
    return metrics, stability, marginal, decisions, audit


def _policy_label(policy: str, optimizer: str) -> str:
    if policy == "PROMETHEUS":
        if optimizer == "MILP":
            return "PROMETHEUS"
        if optimizer == "Greedy benefit/cost":
            return "PROMETHEUS ratio"
        return "PROMETHEUS greedy"
    if policy == "Profile-mean":
        return "Profile-mean"
    return policy


def plot_ems_constraint_sensitivity(
    metrics: pd.DataFrame,
    scenarios: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate the four prespecified sensitivity figures from actual metrics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    palette = {
        "PROMETHEUS": "#0F766E",
        "Risk-first": "#2563EB",
        "Need-first": "#D97706",
        "Outcome-first": "#DC2626",
        "Profile-mean": "#7C3AED",
        "Random": "#6B7280",
    }
    lines = (
        ("PROMETHEUS", "MILP"),
        ("Risk-first", "MILP"),
        ("Need-first", "MILP"),
        ("Outcome-first", "MILP"),
        ("Profile-mean", "MILP"),
        ("Random", "Greedy"),
    )

    def save(fig, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    budget = metrics.loc[metrics.experiment.eq("budget_curve")].merge(
        scenarios[["scenario_id", "budget_multiplier"]],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for policy, optimizer in lines:
        data = budget.loc[
            budget.policy.eq(policy) & budget.optimizer.eq(optimizer)
        ].sort_values("budget_multiplier")
        label = _policy_label(policy, optimizer)
        ax.plot(
            data.budget_multiplier,
            data.normalized_value,
            marker="o",
            linewidth=2.0,
            label=label,
            color=palette.get(policy),
        )
    ax.set_xlabel("Budget multiplier B / B₀")
    ax.set_ylabel("Normalized welfare W / W oracle")
    ax.set_title("Frozen-ranking value–budget curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "figure_value_budget_curve")

    capacity = metrics.loc[
        metrics.experiment.eq("capacity_curve")
    ].merge(
        scenarios[["scenario_id", "capacity_multiplier"]],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for policy, optimizer in lines:
        data = capacity.loc[
            capacity.policy.eq(policy) & capacity.optimizer.eq(optimizer)
        ].sort_values("capacity_multiplier")
        ax.plot(
            data.capacity_multiplier,
            data.deferral_rate,
            marker="o",
            linewidth=2.0,
            label=_policy_label(policy, optimizer),
            color=palette.get(policy),
        )
    ax.set_xlabel("Profile-capacity multiplier K / K₀")
    ax.set_ylabel("Deferral rate")
    ax.set_title("Deferral–capacity curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, "figure_deferral_capacity_curve")

    composition = metrics.loc[
        metrics.experiment.eq("combined_stress")
        & metrics.policy.eq("PROMETHEUS")
        & metrics.optimizer.eq("MILP")
    ].sort_values("scenario_order")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(composition))
    ax.bar(
        x,
        composition.nurse_share,
        label="Nurse-supported",
        color="#14B8A6",
    )
    ax.bar(
        x,
        composition.medicalized_share,
        bottom=composition.nurse_share,
        label="Medicalized",
        color="#0F3D56",
    )
    ax.set_xticks(x, composition.scenario, rotation=20, ha="right")
    ax.set_ylabel("Share of served missions")
    ax.set_title("PROMETHEUS allocation composition")
    ax.legend(frameon=False)
    save(fig, "figure_allocation_composition")

    nominal = metrics.loc[
        metrics.scenario_id.eq("combined_stress:nominal")
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for row in nominal.itertuples(index=False):
        if row.policy == "Oracle":
            continue
        label = _policy_label(row.policy, row.optimizer)
        ax.scatter(
            row.benefit_per_cost,
            row.true_value,
            s=55,
            color=palette.get(row.policy, "#111827"),
        )
        ax.annotate(
            label,
            (row.benefit_per_cost, row.true_value),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("True benefit per simulated cost unit")
    ax.set_ylabel("Total simulated true value")
    ax.set_title("Nominal welfare–efficiency trade-off")
    ax.grid(alpha=0.25)
    save(fig, "figure_welfare_efficiency")
    return written


def write_ems_constraint_sensitivity_report(
    metrics: pd.DataFrame,
    scenarios: pd.DataFrame,
    path: str | Path,
) -> dict[str, Any]:
    """Write a paper-ready narrative using only calculated sensitivity results."""

    prometheus = metrics.loc[
        metrics.policy.eq("PROMETHEUS")
        & metrics.optimizer.eq("MILP")
    ].set_index("scenario_id")
    comparators = ("Risk-first", "Need-first", "Outcome-first")
    wins = {}
    for comparator in comparators:
        other = metrics.loc[
            metrics.policy.eq(comparator)
            & metrics.optimizer.eq("MILP")
        ].set_index("scenario_id")
        wins[comparator] = int(
            (
                prometheus.normalized_value
                > other.loc[prometheus.index, "normalized_value"]
            ).sum()
        )
    nominal = metrics.loc[
        metrics.scenario_id.eq("combined_stress:nominal")
        & metrics.policy.eq("PROMETHEUS")
        & metrics.optimizer.eq("MILP")
    ].iloc[0]
    combined = metrics.loc[
        metrics.experiment.eq("combined_stress")
        & metrics.optimizer.eq("MILP")
        & metrics.policy.isin([
            "PROMETHEUS", "Risk-first", "Need-first", "Outcome-first",
        ])
    ].copy()
    pivot = combined.pivot(
        index="scenario",
        columns="policy",
        values="normalized_value",
    )
    ordered_scenarios = scenarios.loc[
        scenarios.experiment.eq("combined_stress")
    ].sort_values("scenario_order").scenario.tolist()
    lines = [
        "# Allocation constraint sensitivity",
        "",
        "## Experimental principle",
        "",
        (
            "PROMETHEUS is trained once. Direct-ranking scores, validation-only "
            "calibration, baseline allocator values and mission ordering are "
            "frozen before any constraint regime is solved. Across regimes only "
            "budget, profile capacities, simulated costs or shared-resource "
            "limits change."
        ),
        "",
        (
            f"The benchmark contains {metrics.scenario_id.nunique()} operational "
            f"regimes and {metrics[['policy', 'optimizer']].drop_duplicates().shape[0]} "
            "policy–optimizer combinations."
        ),
        "",
        "## Research questions",
        "",
        (
            "**RQ4.** How robust is the allocation value of PROMETHEUS across "
            "changes in total budget, profile-specific capacity, service costs "
            "and shared workforce constraints?"
        ),
        "",
        (
            "**RQ4b.** Does causal prioritization retain an advantage over risk-, "
            "need- and outcome-based policies across operational regimes?"
        ),
        "",
        "## Combined stress regimes",
        "",
        "| Scenario | PROMETHEUS | Risk-first | Need-first | Outcome-first |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in ordered_scenarios:
        row = pivot.loc[scenario]
        lines.append(
            f"| {scenario} | {row['PROMETHEUS']:.3f} | "
            f"{row['Risk-first']:.3f} | {row['Need-first']:.3f} | "
            f"{row['Outcome-first']:.3f} |"
        )
    lines.extend([
        "",
        (
            f"At the nominal regime PROMETHEUS retains "
            f"{float(nominal.normalized_value):.3f} of oracle welfare."
        ),
        (
            f"Across {len(prometheus)} regimes it exceeds Risk-first in "
            f"{wins['Risk-first']}, Need-first in {wins['Need-first']} and "
            f"Outcome-first in {wins['Outcome-first']} regimes."
        ),
        (
            f"Its median normalized welfare is "
            f"{float(prometheus.normalized_value.median()):.3f} and its minimum "
            f"is {float(prometheus.normalized_value.min()):.3f}."
        ),
        "",
        "## Interpretation limits",
        "",
        (
            "All welfare, regret and efficiency quantities use simulated truth "
            "after the non-oracle allocations were frozen. Resource costs are "
            "dimensionless simulation units. This is a single seeded population "
            "and does not establish stable superiority, clinical effectiveness "
            "or deployment readiness."
        ),
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return {
        "scenario_count": int(metrics.scenario_id.nunique()),
        "policy_optimizer_count": int(
            metrics[["policy", "optimizer"]].drop_duplicates().shape[0]
        ),
        "prometheus_nominal_normalized_value": float(
            nominal.normalized_value
        ),
        "prometheus_median_normalized_value": float(
            prometheus.normalized_value.median()
        ),
        "prometheus_minimum_normalized_value": float(
            prometheus.normalized_value.min()
        ),
        "prometheus_wins_vs_risk_first": wins["Risk-first"],
        "prometheus_wins_vs_need_first": wins["Need-first"],
        "prometheus_wins_vs_outcome_first": wins["Outcome-first"],
    }
