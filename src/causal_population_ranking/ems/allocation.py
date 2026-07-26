"""Capacity-constrained allocation and evaluation for the EMS case study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack
from scipy.stats import spearmanr

try:
    from causal_population_ranking.evaluation.ranking import pairwise_concordance
except ModuleNotFoundError:
    def pairwise_concordance(
        score,
        target,
        *,
        max_pairs: int = 100_000,
        seed: int = 0,
    ) -> float:
        """Standalone sampled pairwise concordance fallback."""

        score = np.asarray(score, dtype=float)
        target = np.asarray(target, dtype=float)
        if score.shape != target.shape or score.ndim != 1:
            raise ValueError("score and target must be aligned one-dimensional arrays")
        n = len(score)
        if n < 2:
            return float("nan")
        total_pairs = n * (n - 1) // 2
        if total_pairs <= int(max_pairs):
            left, right = np.triu_indices(n, k=1)
        else:
            rng = np.random.default_rng(int(seed))
            left = rng.integers(0, n, size=int(max_pairs))
            right = rng.integers(0, n - 1, size=int(max_pairs))
            right = np.where(right >= left, right + 1, right)
        target_gap = target[left] - target[right]
        supported = target_gap != 0.0
        if not supported.any():
            return float("nan")
        score_gap = score[left[supported]] - score[right[supported]]
        target_gap = target_gap[supported]
        return float(np.mean(
            (score_gap * target_gap > 0.0)
            + 0.5 * (score_gap == 0.0)
        ))


@dataclass(frozen=True)
class EMSFrozenAllocation:
    decisions: pd.DataFrame
    audit: dict[str, Any]


def _rank_with_seed(
    values: np.ndarray,
    eligible: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    selected = np.zeros(len(values), dtype=bool)
    candidates = np.flatnonzero(eligible)
    if not len(candidates) or count <= 0:
        return selected
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-1e-10, 1e-10, len(candidates))
    order = np.argsort(-(values[candidates] + jitter), kind="mergesort")
    selected[candidates[order[: min(int(count), len(candidates))]]] = True
    return selected


def freeze_ems_allocation(
    test_opportunities: pd.DataFrame,
    priority_scores: np.ndarray,
    config: Mapping[str, Any],
    *,
    method: str = "direct_causal_ranking",
) -> EMSFrozenAllocation:
    """Freeze a hierarchical learned allocation without oracle information."""

    frame = test_opportunities[
        ["mission_id", "opportunity_tier", "severity_score"]
    ].copy()
    frame["priority_score"] = np.asarray(priority_scores, dtype=float)
    wide = frame.pivot(
        index="mission_id",
        columns="opportunity_tier",
        values="priority_score",
    )
    severity = frame.groupby("mission_id", sort=False).severity_score.first()
    if set(wide.columns) != {1, 2} or wide.isna().any().any():
        raise ValueError("Every test mission requires two EMS opportunity scores")
    settings = config["allocation"]
    mission_count = len(wide)
    nurse_capacity = int(
        np.floor(mission_count * float(settings["nurse_or_higher_fraction"]))
    )
    medical_capacity = int(
        np.floor(mission_count * float(settings["medicalized_fraction"]))
    )
    if not 0 <= medical_capacity <= nurse_capacity <= mission_count:
        raise ValueError("Invalid nested EMS capacity fractions")

    medical = _rank_with_seed(
        wide[2].to_numpy(float),
        np.ones(mission_count, dtype=bool),
        medical_capacity,
        seed=int(settings["tie_seed"]),
    )
    nurse = _rank_with_seed(
        wide[1].to_numpy(float),
        ~medical,
        nurse_capacity - int(medical.sum()),
        seed=int(settings["tie_seed"]) + 1,
    )
    tier = np.zeros(mission_count, dtype=int)
    tier[nurse] = 1
    tier[medical] = 2
    decisions = pd.DataFrame({
        "mission_id": wide.index.astype(str),
        "allocated_tier": tier,
        "allocated_response": np.choose(
            tier,
            ["basic_response", "nurse_supported_response", "medicalized_response"],
        ),
        "nurse_priority_score": wide[1].to_numpy(float),
        "medicalized_priority_score": wide[2].to_numpy(float),
        "severity_score": severity.reindex(wide.index).to_numpy(int),
        "method": str(method),
    })
    return EMSFrozenAllocation(
        decisions=decisions,
        audit={
            "status": "frozen_before_oracle_evaluation",
            "method": str(method),
            "mission_count": mission_count,
            "nurse_or_higher_capacity": nurse_capacity,
            "medicalized_capacity": medical_capacity,
            "allocated_basic": int((tier == 0).sum()),
            "allocated_nurse_supported": int((tier == 1).sum()),
            "allocated_medicalized": int((tier == 2).sum()),
            "tie_seed": int(settings["tie_seed"]),
            "oracle_inputs_used": False,
        },
    )


def _baseline_allocation(
    mission_ids: np.ndarray,
    severity: np.ndarray,
    *,
    nurse_capacity: int,
    medical_capacity: int,
    medical_values: np.ndarray,
    nurse_values: np.ndarray,
    medical_seed: int,
    nurse_seed: int,
    method: str,
) -> pd.DataFrame:
    medical = _rank_with_seed(
        medical_values,
        np.ones(len(mission_ids), dtype=bool),
        medical_capacity,
        seed=medical_seed,
    )
    nurse = _rank_with_seed(
        nurse_values,
        ~medical,
        nurse_capacity - int(medical.sum()),
        seed=nurse_seed,
    )
    tier = np.zeros(len(mission_ids), dtype=int)
    tier[nurse] = 1
    tier[medical] = 2
    return pd.DataFrame({
        "mission_id": mission_ids,
        "allocated_tier": tier,
        "severity_score": severity,
        "method": method,
    })


def _oracle_allocation(
    increments: np.ndarray,
    nurse_capacity: int,
    medical_capacity: int,
    *,
    time_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(increments)
    objective = -np.concatenate([increments[:, 0], increments[:, 1]])
    row = np.arange(n)
    nesting = coo_matrix(
        (
            np.concatenate([-np.ones(n), np.ones(n)]),
            (
                np.concatenate([row, row]),
                np.concatenate([row, n + row]),
            ),
        ),
        shape=(n, 2 * n),
    ).tocsr()
    capacities = coo_matrix(
        (
            np.ones(2 * n),
            (
                np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)]),
                np.arange(2 * n),
            ),
        ),
        shape=(2, 2 * n),
    ).tocsr()
    matrix = vstack([nesting, capacities], format="csr")
    lower = np.concatenate([np.full(n, -np.inf), [-np.inf, -np.inf]])
    upper = np.concatenate([
        np.zeros(n),
        [float(nurse_capacity), float(medical_capacity)],
    ])
    result = milp(
        objective,
        integrality=np.ones(2 * n),
        bounds=Bounds(np.zeros(2 * n), np.ones(2 * n)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(time_limit)},
    )
    if result.x is None:
        raise RuntimeError(f"EMS oracle allocation failed: {result.message}")
    nurse_or_higher = result.x[:n] > 0.5
    medical = result.x[n:] > 0.5
    tier = nurse_or_higher.astype(int) + medical.astype(int)
    return tier, {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_success": bool(result.success),
        "solver_objective": float(-result.fun),
    }


def evaluate_frozen_ems_case(
    learned: EMSFrozenAllocation,
    test_opportunities: pd.DataFrame,
    priority_scores: np.ndarray,
    evaluation_only: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Open simulated truth only after learned scores and allocation are frozen."""

    opportunities = test_opportunities[
        ["mission_id", "opportunity_tier", "severity_score"]
    ].copy()
    opportunities["priority_score"] = np.asarray(priority_scores, dtype=float)
    truth = evaluation_only.set_index("mission_id")
    rows = opportunities.mission_id.astype(str)
    opportunities["true_increment"] = np.where(
        opportunities.opportunity_tier.eq(1),
        truth.loc[rows, "true_increment_nurse_supported"].to_numpy(float),
        truth.loc[rows, "true_increment_medicalized"].to_numpy(float),
    )
    settings = config["allocation"]
    ranking = config["ranking"]
    metrics = []

    def add(
        method: str,
        metric: str,
        value: float,
        *,
        oracle_policy_construction: bool = False,
    ) -> None:
        metrics.append({
            "method": method,
            "metric": metric,
            "value": float(value),
            "split": "test",
            # Backward-compatible field: True only for the oracle policy itself.
            "uses_oracle": bool(oracle_policy_construction),
            "oracle_evaluation_only": not bool(oracle_policy_construction),
            "oracle_access_for_policy_construction": bool(
                oracle_policy_construction
            ),
            "oracle_access_for_evaluation": True,
        })

    add(
        "direct_causal_ranking",
        "opportunity_pairwise_concordance",
        pairwise_concordance(
            opportunities.priority_score,
            opportunities.true_increment,
            max_pairs=int(ranking["maximum_evaluation_pairs"]),
            seed=int(ranking["evaluation_pair_seed"]),
        ),
    )
    add(
        "direct_causal_ranking",
        "opportunity_spearman",
        float(spearmanr(
            opportunities.priority_score,
            opportunities.true_increment,
        ).statistic),
    )

    mission_ids = learned.decisions.mission_id.to_numpy(str)
    severity = learned.decisions.severity_score.to_numpy(int)
    nurse_capacity = int(learned.audit["nurse_or_higher_capacity"])
    medical_capacity = int(learned.audit["medicalized_capacity"])
    random_rng = np.random.default_rng(int(settings["random_baseline_seed"]))
    severity_baseline = _baseline_allocation(
        mission_ids,
        severity,
        nurse_capacity=nurse_capacity,
        medical_capacity=medical_capacity,
        medical_values=severity.astype(float),
        nurse_values=severity.astype(float),
        medical_seed=int(settings["severity_tie_seed"]),
        nurse_seed=int(settings["severity_tie_seed"]) + 1,
        method="severity_only",
    )
    random_baseline = _baseline_allocation(
        mission_ids,
        severity,
        nurse_capacity=nurse_capacity,
        medical_capacity=medical_capacity,
        medical_values=random_rng.random(len(mission_ids)),
        nurse_values=random_rng.random(len(mission_ids)),
        medical_seed=int(settings["random_baseline_seed"]) + 1,
        nurse_seed=int(settings["random_baseline_seed"]) + 2,
        method="seeded_random",
    )
    increments = truth.loc[mission_ids, [
        "true_increment_nurse_supported",
        "true_increment_medicalized",
    ]].to_numpy(float)
    oracle_tier, oracle_audit = _oracle_allocation(
        increments,
        nurse_capacity,
        medical_capacity,
        time_limit=float(settings["oracle_solver_time_limit_seconds"]),
    )
    oracle = pd.DataFrame({
        "mission_id": mission_ids,
        "allocated_tier": oracle_tier,
        "severity_score": severity,
        "method": "evaluation_only_oracle",
    })
    policies = [
        learned.decisions[[
            "mission_id", "allocated_tier", "severity_score", "method"
        ]],
        severity_baseline,
        random_baseline,
        oracle,
    ]

    def value(policy: pd.DataFrame) -> float:
        tier = policy.allocated_tier.to_numpy(int)
        return float(
            (increments[:, 0] * (tier >= 1)
             + increments[:, 1] * (tier >= 2)).sum()
        )

    oracle_value = value(oracle)
    for policy in policies:
        method = str(policy.method.iloc[0])
        oracle_policy = method == "evaluation_only_oracle"
        policy_value = value(policy)
        add(
            method,
            "allocation_total_true_value_days",
            policy_value,
            oracle_policy_construction=oracle_policy,
        )
        add(
            method,
            "allocation_mean_true_value_days_per_mission",
            policy_value / len(policy),
            oracle_policy_construction=oracle_policy,
        )
        add(
            method,
            "normalized_allocation_value",
            policy_value / oracle_value if oracle_value > 0 else float("nan"),
            oracle_policy_construction=oracle_policy,
        )
        add(
            method,
            "allocation_regret_days",
            oracle_value - policy_value,
            oracle_policy_construction=oracle_policy,
        )
    decisions = pd.concat(policies, ignore_index=True)
    return pd.DataFrame(metrics), decisions, oracle_audit
