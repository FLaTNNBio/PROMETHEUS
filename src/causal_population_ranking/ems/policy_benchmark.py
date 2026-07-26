"""Comparable policy benchmark for the semi-synthetic EMS application.

All non-oracle allocator values and decisions are built from learner-safe data and
frozen before this module's evaluation function is allowed to see simulated truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .ranking import EMSDirectRanker
from .supervision import EMSSupervision
from .synthetic import LEARNER_FEATURE_COLUMNS


POLICY_ORDER = (
    ("Oracle", "MILP"),
    ("PROMETHEUS", "MILP"),
    ("PROMETHEUS", "Greedy ratio"),
    ("Risk-first", "MILP"),
    ("Need-first", "MILP"),
    ("Outcome-first", "MILP"),
    ("Profile-mean", "MILP"),
    ("T-learner GBDT", "MILP"),
    ("X-learner GBDT", "MILP"),
    ("DR-learner GBDT", "MILP"),
    ("FCFS", "Greedy"),
    ("Random", "Greedy"),
)


def ems_methodology_table() -> pd.DataFrame:
    """Return the prespecified interpretation of each allocator value."""

    return pd.DataFrame([
        {
            "Metodo": "PROMETHEUS",
            "Valore usato dall'allocatore": (
                "utilità allocativa calibrata su split separato dal ranking diretto"
            ),
            "Causale": "sì, metodologico",
            "Personalizzato": "sì",
            "Profile-specific": "sì",
        },
        {
            "Metodo": "Risk-first",
            "Valore usato dall'allocatore": "rischio prognostico previsto",
            "Causale": "no",
            "Personalizzato": "sì",
            "Profile-specific": "no",
        },
        {
            "Metodo": "Need-first",
            "Valore usato dall'allocatore": "severità pre-dispatch",
            "Causale": "no",
            "Personalizzato": "sì",
            "Profile-specific": "no",
        },
        {
            "Metodo": "Outcome-first",
            "Valore usato dall'allocatore": "outcome previsto sotto il profilo",
            "Causale": "no",
            "Personalizzato": "sì",
            "Profile-specific": "sì",
        },
        {
            "Metodo": "Profile-mean",
            "Valore usato dall'allocatore": "beneficio medio DR del profilo",
            "Causale": "parzialmente",
            "Personalizzato": "no",
            "Profile-specific": "sì",
        },
        {
            "Metodo": "T-learner GBDT",
            "Valore usato dall'allocatore": "differenza tra outcome-model separati",
            "Causale": "sì, meta-learner pubblico",
            "Personalizzato": "sì",
            "Profile-specific": "sì",
        },
        {
            "Metodo": "X-learner GBDT",
            "Valore usato dall'allocatore": "effetto X-learner per livello di risposta",
            "Causale": "sì, meta-learner pubblico",
            "Personalizzato": "sì",
            "Profile-specific": "sì",
        },
        {
            "Metodo": "DR-learner GBDT",
            "Valore usato dall'allocatore": "regressione GBDT dei segnali doubly robust",
            "Causale": "sì, baseline DR",
            "Personalizzato": "sì",
            "Profile-specific": "sì",
        },
        {
            "Metodo": "FCFS",
            "Valore usato dall'allocatore": "ordine di arrivo simulato",
            "Causale": "no",
            "Personalizzato": "no",
            "Profile-specific": "no",
        },
        {
            "Metodo": "Random",
            "Valore usato dall'allocatore": "ordine casuale con seed",
            "Causale": "no",
            "Personalizzato": "no",
            "Profile-specific": "no",
        },
        {
            "Metodo": "Oracle",
            "Valore usato dall'allocatore": "vero beneficio sintetico",
            "Causale": "sì, evaluation-only",
            "Personalizzato": "sì",
            "Profile-specific": "sì",
        },
    ])


@dataclass(frozen=True)
class EMSPolicyInputs:
    allocator_values: pd.DataFrame
    mission_metadata: pd.DataFrame
    methodology: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class EMSFrozenPolicyBenchmark:
    decisions: pd.DataFrame
    audit: dict[str, Any]


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        column
        for column in LEARNER_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric = [
        column for column in LEARNER_FEATURE_COLUMNS
        if column not in categorical
    ]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("categorical", encoder, categorical),
        ("numeric", StandardScaler(), numeric),
    ])


def build_ems_policy_inputs(
    learner_data: pd.DataFrame,
    supervision: EMSSupervision,
    ranker: EMSDirectRanker,
    config: Mapping[str, Any],
) -> EMSPolicyInputs:
    """Build all non-oracle allocator values using only prespecified splits."""

    if "simulated_arrival_order" not in learner_data:
        raise ValueError("EMS FCFS benchmark requires simulated arrival order")
    nuisance = learner_data.loc[
        learner_data.split.eq("nuisance_train")
    ].reset_index(drop=True)
    test_missions = learner_data.loc[
        learner_data.split.eq("test")
    ].sort_values("mission_id").reset_index(drop=True)
    calibration = supervision.opportunities.loc[
        supervision.opportunities.split.eq("calibration")
    ].reset_index(drop=True)
    test_opportunities = supervision.opportunities.loc[
        supervision.opportunities.split.eq("test")
    ].reset_index(drop=True)
    if nuisance.empty or calibration.empty or test_missions.empty:
        raise ValueError("EMS policy benchmark requires nuisance, calibration and test")

    processor = _preprocessor(nuisance)
    nuisance_x = np.asarray(
        processor.fit_transform(nuisance[list(LEARNER_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    test_x = np.asarray(
        processor.transform(test_missions[list(LEARNER_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    settings = config["policy_benchmark"]
    max_iter = int(settings["model_max_iter"])
    treatment = nuisance.assigned_tier.to_numpy(int)
    observed_outcome = nuisance.observed_outcome.to_numpy(float)

    basic = np.flatnonzero(treatment == 0)
    if len(basic) < 20:
        raise ValueError("Risk-first requires sufficient basic-response missions")
    risk_model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_leaf_nodes=15,
        learning_rate=0.06,
        random_state=int(settings["risk_model_seed"]),
    )
    risk_model.fit(nuisance_x[basic], 30.0 - observed_outcome[basic])
    predicted_risk = np.clip(risk_model.predict(test_x), 0.0, 30.0)

    predicted_outcome = {}
    for tier in (1, 2):
        arm = np.flatnonzero(treatment == tier)
        if len(arm) < 20:
            raise ValueError(
                f"Outcome-first requires sufficient tier-{tier} missions"
            )
        outcome_model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_leaf_nodes=15,
            learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + tier,
        )
        outcome_model.fit(nuisance_x[arm], observed_outcome[arm])
        predicted_outcome[tier] = np.clip(
            outcome_model.predict(test_x),
            0.0,
            30.0,
        )

    # Public causal meta-learner baselines, trained only on learner-safe rows.
    basic_outcome_model = HistGradientBoostingRegressor(
        max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
        random_state=int(settings["outcome_model_seed"]) + 90,
    ).fit(nuisance_x[basic], observed_outcome[basic])
    predicted_basic_outcome = np.clip(
        basic_outcome_model.predict(test_x), 0.0, 30.0
    )
    t_learner_nurse = predicted_outcome[1] - predicted_basic_outcome
    t_learner_medicalized = predicted_outcome[2] - predicted_basic_outcome

    def x_learner_value(tier: int) -> np.ndarray:
        keep = np.isin(treatment, [0, tier])
        x = nuisance_x[keep]
        d = (treatment[keep] == tier).astype(int)
        y = observed_outcome[keep]
        mu0 = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + 100 + tier,
        ).fit(x[d == 0], y[d == 0])
        mu1 = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + 110 + tier,
        ).fit(x[d == 1], y[d == 1])
        d1 = y[d == 1] - mu0.predict(x[d == 1])
        d0 = mu1.predict(x[d == 0]) - y[d == 0]
        tau1 = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + 120 + tier,
        ).fit(x[d == 1], d1)
        tau0 = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + 130 + tier,
        ).fit(x[d == 0], d0)
        propensity = HistGradientBoostingClassifier(
            max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + 140 + tier,
        ).fit(x, d)
        e = np.clip(propensity.predict_proba(test_x)[:, 1], 0.05, 0.95)
        return (1.0 - e) * tau1.predict(test_x) + e * tau0.predict(test_x)

    x_learner_nurse = x_learner_value(1)
    x_learner_medicalized = x_learner_value(2)

    calibration_score = ranker.score(calibration)
    test_score = ranker.score(test_opportunities)
    calibration_signal = calibration[
        list(supervision.repeat_signal_columns)
    ].mean(axis=1).to_numpy(float)
    prometheus_increment = {}
    calibrator_audit = {}
    for tier in (1, 2):
        calibration_mask = calibration.opportunity_tier.eq(tier).to_numpy()
        test_mask = test_opportunities.opportunity_tier.eq(tier).to_numpy()
        calibrator = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )
        calibrator.fit(
            calibration_score[calibration_mask],
            calibration_signal[calibration_mask],
        )
        tier_values = pd.Series(
            calibrator.predict(test_score[test_mask]),
            index=test_opportunities.loc[test_mask, "mission_id"].astype(str),
        )
        prometheus_increment[tier] = tier_values.reindex(
            test_missions.mission_id.astype(str)
        ).to_numpy(float)
        calibrator_audit[str(tier)] = {
            "calibration_rows": int(calibration_mask.sum()),
            "score_min": float(calibration_score[calibration_mask].min()),
            "score_max": float(calibration_score[calibration_mask].max()),
            "fitted_thresholds": int(len(calibrator.X_thresholds_)),
        }

    rank_train = supervision.opportunities.loc[
        supervision.opportunities.split.eq("rank_train")
    ].copy()
    rank_train["_mean_dr"] = rank_train[
        list(supervision.repeat_signal_columns)
    ].mean(axis=1)
    profile_mean_increment = rank_train.groupby(
        "opportunity_tier"
    )["_mean_dr"].mean()
    if set(profile_mean_increment.index) != {1, 2}:
        raise ValueError("Profile-mean requires both EMS opportunities")

    rank_processor = _preprocessor(rank_train)
    rank_x = np.asarray(
        rank_processor.fit_transform(rank_train[list(LEARNER_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    rank_test_x = np.asarray(
        rank_processor.transform(test_missions[list(LEARNER_FEATURE_COLUMNS)]),
        dtype=np.float32,
    )
    dr_increment = {}
    for tier in (1, 2):
        mask = rank_train.opportunity_tier.eq(tier).to_numpy()
        dr_model = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=15, learning_rate=0.06,
            random_state=int(settings["outcome_model_seed"]) + 200 + tier,
        ).fit(rank_x[mask], rank_train.loc[mask, "_mean_dr"])
        dr_increment[tier] = dr_model.predict(rank_test_x)

    severity = test_missions.severity_score.to_numpy(float)
    values = []

    def add(policy: str, nurse_value, medicalized_value) -> None:
        values.append(pd.DataFrame({
            "policy": policy,
            "mission_id": test_missions.mission_id.astype(str),
            "nurse_value": np.asarray(nurse_value, dtype=float),
            "medicalized_value": np.asarray(medicalized_value, dtype=float),
        }))

    add(
        "PROMETHEUS",
        prometheus_increment[1],
        prometheus_increment[1] + prometheus_increment[2],
    )
    add("Risk-first", predicted_risk, predicted_risk)
    add("Need-first", severity, severity)
    add(
        "Outcome-first",
        predicted_outcome[1],
        predicted_outcome[2],
    )
    add(
        "Profile-mean",
        np.full(len(test_missions), profile_mean_increment.loc[1]),
        np.full(
            len(test_missions),
            profile_mean_increment.loc[1] + profile_mean_increment.loc[2],
        ),
    )
    add("T-learner GBDT", t_learner_nurse, t_learner_medicalized)
    add("X-learner GBDT", x_learner_nurse, x_learner_medicalized)
    add(
        "DR-learner GBDT",
        dr_increment[1],
        dr_increment[1] + dr_increment[2],
    )
    allocator_values = pd.concat(values, ignore_index=True)
    numeric = allocator_values[["nurse_value", "medicalized_value"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("EMS allocator values must be finite")

    metadata = test_missions[[
        "mission_id",
        "severity_score",
        "simulated_arrival_order",
    ]].copy()
    return EMSPolicyInputs(
        allocator_values=allocator_values,
        mission_metadata=metadata,
        methodology=ems_methodology_table(),
        audit={
            "status": "learner_safe_allocator_values_built",
            "raw_priority_score_semantics": (
                "ordinal_not_calibrated_treatment_effect"
            ),
            "prometheus_allocator_value": (
                "validation_calibrated_utility_from_direct_ranking_score"
            ),
            "calibration_split_used": "calibration",
            "calibration_split_used_for_ranker_selection": False,
            "oracle_inputs_used": False,
            "risk_first_training_arm": "simulated_basic_response_only",
            "outcome_first_training": "separate_observed_outcome_model_per_tier",
            "profile_mean_training_split": "rank_train",
            "calibrators": calibrator_audit,
            "risk_model_seed": int(settings["risk_model_seed"]),
            "outcome_model_seed": int(settings["outcome_model_seed"]),
            "encoded_feature_count": int(nuisance_x.shape[1]),
        },
    )


def _stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}|{token}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _capacity_settings(
    mission_count: int,
    config: Mapping[str, Any],
) -> dict[str, float | int]:
    allocation = config["allocation"]
    benchmark = config["policy_benchmark"]
    served_capacity = int(np.floor(
        mission_count * float(allocation["nurse_or_higher_fraction"])
    ))
    medicalized_capacity = int(np.floor(
        mission_count * float(allocation["medicalized_fraction"])
    ))
    nurse_cost = float(benchmark["nurse_response_cost"])
    medicalized_cost = float(benchmark["medicalized_response_cost"])
    budget = float(benchmark["budget_per_test_mission"]) * mission_count
    full_capacity_cost = (
        (served_capacity - medicalized_capacity) * nurse_cost
        + medicalized_capacity * medicalized_cost
    )
    if not (
        0.0 < nurse_cost < medicalized_cost
        and 0 <= medicalized_capacity <= served_capacity <= mission_count
        and budget > 0.0
    ):
        raise ValueError("Invalid EMS policy benchmark capacity settings")
    return {
        "served_capacity": served_capacity,
        "medicalized_capacity": medicalized_capacity,
        "nurse_cost": nurse_cost,
        "medicalized_cost": medicalized_cost,
        "budget": budget,
        "full_capacity_cost": full_capacity_cost,
    }


def _solve_milp(
    mission_ids: np.ndarray,
    nurse_value: np.ndarray,
    medicalized_value: np.ndarray,
    config: Mapping[str, Any],
    *,
    policy: str,
    oracle: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(mission_ids)
    capacity = _capacity_settings(n, config)
    settings = config["policy_benchmark"]
    costs = np.concatenate([
        np.full(n, capacity["nurse_cost"]),
        np.full(n, capacity["medicalized_cost"]),
    ])
    values = np.concatenate([nurse_value, medicalized_value]).astype(float)
    tie_seed = _stable_seed(int(settings["optimizer_tie_seed"]), policy)
    rng = np.random.default_rng(tie_seed)
    tie = rng.uniform(0.0, 1e-10, 2 * n)
    objective = -(values + tie) + costs * 1e-8

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
    served = coo_matrix(
        (np.ones(2 * n), (np.zeros(2 * n, dtype=int), np.arange(2 * n))),
        shape=(1, 2 * n),
    ).tocsr()
    medicalized = coo_matrix(
        (
            np.ones(n),
            (np.zeros(n, dtype=int), n + np.arange(n)),
        ),
        shape=(1, 2 * n),
    ).tocsr()
    budget = coo_matrix(
        (costs, (np.zeros(2 * n, dtype=int), np.arange(2 * n))),
        shape=(1, 2 * n),
    ).tocsr()
    matrix = vstack([one_profile, served, medicalized, budget], format="csr")
    lower = np.concatenate([
        np.full(n, -np.inf),
        [
            float(capacity["served_capacity"]),
            float(capacity["medicalized_capacity"]),
            -np.inf,
        ],
    ])
    upper = np.concatenate([
        np.ones(n),
        [
            float(capacity["served_capacity"]),
            float(capacity["medicalized_capacity"]),
            float(capacity["budget"]),
        ],
    ])
    result = milp(
        objective,
        integrality=np.ones(2 * n),
        bounds=Bounds(np.zeros(2 * n), np.ones(2 * n)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={
            "time_limit": float(
                config["allocation"]["oracle_solver_time_limit_seconds"]
            )
        },
    )
    if result.x is None:
        raise RuntimeError(f"EMS policy MILP failed for {policy}: {result.message}")
    nurse = result.x[:n] > 0.5
    medical = result.x[n:] > 0.5
    tier = nurse.astype(int) + 2 * medical.astype(int)
    selected_value = np.where(nurse, nurse_value, 0.0)
    selected_value += np.where(medical, medicalized_value, 0.0)
    selected_cost = np.where(nurse, capacity["nurse_cost"], 0.0)
    selected_cost += np.where(
        medical, capacity["medicalized_cost"], 0.0
    )
    decisions = pd.DataFrame({
        "mission_id": mission_ids.astype(str),
        "policy": policy,
        "optimizer": "MILP",
        "allocated_tier": tier,
        "allocator_value": selected_value,
        "allocated_cost": selected_cost,
        "uses_oracle_allocator_value": bool(oracle),
    })
    return decisions, {
        "policy": policy,
        "optimizer": "MILP",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_success": bool(result.success),
        "allocator_objective": float(selected_value.sum()),
        "cost_used": float(selected_cost.sum()),
        "served": int((tier > 0).sum()),
        "medicalized": int((tier == 2).sum()),
        "tie_seed": tie_seed,
        **capacity,
    }


def _greedy_ratio(
    mission_ids: np.ndarray,
    nurse_value: np.ndarray,
    medicalized_value: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(mission_ids)
    capacity = _capacity_settings(n, config)
    costs = np.concatenate([
        np.full(n, capacity["nurse_cost"]),
        np.full(n, capacity["medicalized_cost"]),
    ])
    values = np.concatenate([nurse_value, medicalized_value])
    tiers = np.concatenate([np.ones(n, dtype=int), np.full(n, 2, dtype=int)])
    missions = np.concatenate([np.arange(n), np.arange(n)])
    seed = int(config["policy_benchmark"]["greedy_tie_seed"])
    rng = np.random.default_rng(seed)
    ratio = values / costs + rng.uniform(0.0, 1e-12, 2 * n)
    order = np.argsort(-ratio, kind="mergesort")
    allocated = np.zeros(n, dtype=int)
    allocator_value = np.zeros(n, dtype=float)
    allocated_cost = np.zeros(n, dtype=float)
    used_budget = 0.0
    served = 0
    medicalized = 0
    for candidate in order:
        mission = int(missions[candidate])
        tier = int(tiers[candidate])
        cost = float(costs[candidate])
        value = float(values[candidate])
        if (
            allocated[mission] != 0
            or value <= 0.0
            or served >= int(capacity["served_capacity"])
            or used_budget + cost > float(capacity["budget"]) + 1e-9
            or (
                tier == 2
                and medicalized >= int(capacity["medicalized_capacity"])
            )
        ):
            continue
        allocated[mission] = tier
        allocator_value[mission] = value
        allocated_cost[mission] = cost
        used_budget += cost
        served += 1
        medicalized += int(tier == 2)
    decisions = pd.DataFrame({
        "mission_id": mission_ids.astype(str),
        "policy": "PROMETHEUS",
        "optimizer": "Greedy ratio",
        "allocated_tier": allocated,
        "allocator_value": allocator_value,
        "allocated_cost": allocated_cost,
        "uses_oracle_allocator_value": False,
    })
    return decisions, {
        "policy": "PROMETHEUS",
        "optimizer": "Greedy ratio",
        "allocator_objective": float(allocator_value.sum()),
        "cost_used": float(used_budget),
        "served": int(served),
        "medicalized": int(medicalized),
        "tie_seed": seed,
        **capacity,
    }


def _ordered_greedy(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(metadata)
    capacity = _capacity_settings(n, config)
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
        raise ValueError(f"Unsupported ordered EMS policy: {policy}")
    tier = np.zeros(n, dtype=int)
    cost = np.zeros(n, dtype=float)
    budget = 0.0
    served = 0
    medicalized = 0
    for mission in order:
        if served >= int(capacity["served_capacity"]):
            break
        if (
            medicalized < int(capacity["medicalized_capacity"])
            and budget + float(capacity["medicalized_cost"])
            <= float(capacity["budget"]) + 1e-9
        ):
            tier[mission] = 2
            cost[mission] = float(capacity["medicalized_cost"])
            medicalized += 1
        elif (
            budget + float(capacity["nurse_cost"])
            <= float(capacity["budget"]) + 1e-9
        ):
            tier[mission] = 1
            cost[mission] = float(capacity["nurse_cost"])
        else:
            break
        budget += cost[mission]
        served += 1
    decisions = pd.DataFrame({
        "mission_id": metadata.mission_id.astype(str),
        "policy": policy,
        "optimizer": "Greedy",
        "allocated_tier": tier,
        "allocator_value": np.zeros(n),
        "allocated_cost": cost,
        "uses_oracle_allocator_value": False,
    })
    return decisions, {
        "policy": policy,
        "optimizer": "Greedy",
        "cost_used": float(budget),
        "served": int(served),
        "medicalized": int(medicalized),
        "random_seed": seed,
        **capacity,
    }


def freeze_ems_policy_benchmark(
    inputs: EMSPolicyInputs,
    config: Mapping[str, Any],
) -> EMSFrozenPolicyBenchmark:
    """Solve and freeze every non-oracle policy before truth is opened."""

    metadata = inputs.mission_metadata.sort_values(
        "mission_id"
    ).reset_index(drop=True)
    mission_ids = metadata.mission_id.to_numpy(str)
    decisions = []
    audits = []
    for policy in (
        "PROMETHEUS",
        "Risk-first",
        "Need-first",
        "Outcome-first",
        "Profile-mean",
        "T-learner GBDT",
        "X-learner GBDT",
        "DR-learner GBDT",
    ):
        values = inputs.allocator_values.loc[
            inputs.allocator_values.policy.eq(policy)
        ].set_index("mission_id").loc[mission_ids]
        solved, audit = _solve_milp(
            mission_ids,
            values.nurse_value.to_numpy(float),
            values.medicalized_value.to_numpy(float),
            config,
            policy=policy,
            oracle=False,
        )
        decisions.append(solved)
        audits.append(audit)
        if policy == "PROMETHEUS":
            greedy, greedy_audit = _greedy_ratio(
                mission_ids,
                values.nurse_value.to_numpy(float),
                values.medicalized_value.to_numpy(float),
                config,
            )
            decisions.append(greedy)
            audits.append(greedy_audit)
    for policy in ("FCFS", "Random"):
        solved, audit = _ordered_greedy(metadata, config, policy=policy)
        decisions.append(solved)
        audits.append(audit)
    result = pd.concat(decisions, ignore_index=True)
    if result.uses_oracle_allocator_value.astype(bool).any():
        raise AssertionError("Oracle value entered frozen EMS policy benchmark")
    return EMSFrozenPolicyBenchmark(
        decisions=result,
        audit={
            "status": "frozen_before_oracle_evaluation",
            "oracle_inputs_used": False,
            "same_budget_and_capacities_for_all_policies": True,
            "milp_resource_use_fixed_at_common_capacities": True,
            "solvers": audits,
        },
    )


def evaluate_ems_policy_benchmark(
    frozen: EMSFrozenPolicyBenchmark,
    evaluation_only: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate frozen policies and solve the evaluation-only oracle."""

    first = frozen.decisions.loc[
        frozen.decisions.policy.eq("PROMETHEUS")
        & frozen.decisions.optimizer.eq("MILP")
    ].sort_values("mission_id")
    mission_ids = first.mission_id.to_numpy(str)
    truth = evaluation_only.set_index("mission_id").loc[mission_ids]
    nurse_true = truth.true_increment_nurse_supported.to_numpy(float)
    medical_true = (
        nurse_true + truth.true_increment_medicalized.to_numpy(float)
    )
    oracle, oracle_audit = _solve_milp(
        mission_ids,
        nurse_true,
        medical_true,
        config,
        policy="Oracle",
        oracle=True,
    )
    decisions = pd.concat([frozen.decisions, oracle], ignore_index=True)

    def true_value(group: pd.DataFrame) -> float:
        aligned = group.set_index("mission_id").loc[mission_ids]
        tier = aligned.allocated_tier.to_numpy(int)
        return float(
            np.where(
                tier == 1,
                nurse_true,
                np.where(tier == 2, medical_true, 0.0),
            ).sum()
        )

    oracle_value = true_value(oracle)
    rows = []
    for policy, optimizer in POLICY_ORDER:
        group = decisions.loc[
            decisions.policy.eq(policy)
            & decisions.optimizer.eq(optimizer)
        ]
        if len(group) != len(mission_ids):
            raise ValueError(
                f"Missing EMS policy decisions for {policy}/{optimizer}"
            )
        value = true_value(group)
        cost = float(group.allocated_cost.sum())
        served = int(group.allocated_tier.gt(0).sum())
        rows.append({
            "Policy": policy,
            "Optimizer": optimizer,
            "True value": value,
            "Normalized value": (
                value / oracle_value if oracle_value > 0.0 else float("nan")
            ),
            "Regret": oracle_value - value,
            "Benefit/cost": value / cost if cost > 0.0 else float("nan"),
            "Served": served,
            "Medicalized": int(group.allocated_tier.eq(2).sum()),
            "Cost used": cost,
            "Budget": float(
                _capacity_settings(len(mission_ids), config)["budget"]
            ),
            # Oracle truth is used to construct only the Oracle policy.
            "uses_oracle": policy == "Oracle",
            "oracle_evaluation_only": policy != "Oracle",
            "oracle_access_for_policy_construction": policy == "Oracle",
            "oracle_access_for_evaluation": True,
        })
    return pd.DataFrame(rows), decisions, oracle_audit
