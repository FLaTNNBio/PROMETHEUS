"""Fully synthetic observational DGP for explicit care actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.validation import assert_no_oracle_columns
from ..dm77.action_models import CareActionCatalog


NO_ACTION = "no_new_action"
ACTION_DGP_SCENARIOS = (
    "baseline_identifiable",
    "strong_observed_confounding",
    "poor_overlap",
    "risk_benefit_misalignment",
    "targeted_selection_observed",
    "hidden_confounding",
    "combined_stress",
    "null_treatment_effect",
    "placebo_outcome",
)


@dataclass(frozen=True)
class ActionSimulationResult:
    learner: pd.DataFrame
    ground_truth: pd.DataFrame
    action_learners: dict[str, pd.DataFrame]
    feature_columns: tuple[str, ...]
    metadata: dict


def _z(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-8)


def _patient_splits(n: int, seed: int) -> np.ndarray:
    if n < 100:
        raise ValueError("Action DGP requires at least 100 patients")
    order = np.random.default_rng(seed).permutation(n)
    cuts = (int(0.35 * n), int(0.65 * n), int(0.80 * n))
    split = np.empty(n, dtype=object)
    split[order[:cuts[0]]] = "nuisance_train"
    split[order[cuts[0]:cuts[1]]] = "rank_train"
    split[order[cuts[1]:cuts[2]]] = "validation"
    split[order[cuts[2]:]] = "test"
    return split


def _feature_columns(context: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "current_care_state_index", "dm77_need_level", "treatment_level",
        "observed_outcome", "observed_action_id",
    }
    features = tuple(
        column for column in context.columns
        if column not in excluded
        and column != "patient_id"
        and pd.api.types.is_numeric_dtype(context[column])
        and not column.startswith(("true_", "oracle_", "potential_", "dm77_"))
    )
    if not features or not np.isfinite(context.loc[:, features].to_numpy(float)).all():
        raise ValueError("Action DGP requires finite numeric pre-index features")
    return features


def simulate_observational_care_actions(
    patient_context: pd.DataFrame,
    authoritative_opportunities: pd.DataFrame,
    catalog: CareActionCatalog,
    seed: int,
    outcome_noise_sd: float = 6.0,
    minimum_action_assignments_per_split: int = 6,
    scenario: str = "baseline_identifiable",
) -> ActionSimulationResult:
    """Generate one observed action and common outcome per patient.

    Candidate actions are supplied only by the authoritative DM77 catalogue adapter.
    Potential outcomes and true benefits are returned in a separate table.
    """

    scenario = {
        "baseline": "baseline_identifiable",
        "targeted_selection": "targeted_selection_observed",
    }.get(str(scenario), str(scenario))
    if scenario not in ACTION_DGP_SCENARIOS:
        raise ValueError(f"Unknown action DGP scenario {scenario!r}")
    if outcome_noise_sd < 0 or minimum_action_assignments_per_split < 1:
        raise ValueError("Invalid action-DGP noise or assignment coverage")
    context = patient_context.reset_index(drop=True).copy()
    if context.patient_id.astype(str).duplicated().any():
        raise ValueError("Action DGP requires one context row per patient")
    context["patient_id"] = context.patient_id.astype(str)
    features = _feature_columns(context)
    n = len(context)
    rng = np.random.default_rng(int(seed))
    split = _patient_splits(n, seed + 17)
    index_by_patient = {patient: index for index, patient in enumerate(context.patient_id)}

    age = _z(context.age)
    burden = _z(np.log1p(context.condition_distinct))
    inpatient = _z(np.log1p(context.prior_inpatient))
    emergency = _z(np.log1p(context.prior_emergency))
    medications = _z(np.log1p(context.medication_distinct))
    frailty = _z(context.frailty_index)
    functional = _z(context.functional_limitation_score)
    social = _z(context.social_fragility_score)
    risk = _z(
        0.25 * age + 0.60 * burden + 0.50 * inpatient + 0.35 * emergency
        + 0.22 * medications + 0.22 * frailty + 0.15 * functional + 0.10 * social
    )
    hidden = scenario in {"hidden_confounding", "combined_stress"}
    poor_overlap = scenario in {"poor_overlap", "combined_stress"}
    strong_observed = scenario in {"strong_observed_confounding", "combined_stress"}
    misaligned = scenario in {"risk_benefit_misalignment", "combined_stress"}
    targeted = scenario in {"targeted_selection_observed", "combined_stress"}
    null_effect = scenario in {"null_treatment_effect", "placebo_outcome"}
    placebo = scenario == "placebo_outcome"
    latent_u = rng.normal(0.0, 1.0, n) if hidden else np.zeros(n, dtype=float)
    mu0 = (
        276.0 - 30.0 * np.tanh(risk / 1.7) + 3.0 * np.sin(age)
        + (6.0 * latent_u if hidden else 0.0)
    )
    if placebo:
        # Negative-control endpoint: prognostic variation remains realistic,
        # but every ranked action has exactly zero causal effect on it.
        mu0 = 210.0 + 18.0 * np.tanh(
            (0.45 * age - 0.35 * medications + 0.25 * social) / 1.6
        )
    shared_response = _z(
        0.42 * burden - 0.32 * inpatient + 0.25 * medications
        + 0.30 * np.sin(age) + 0.20 * functional * social
    )

    ranked_actions = catalog.ranked_actions
    action_ids = tuple(action.action_id for action in ranked_actions)
    true_cate = np.empty((n, len(ranked_actions)), dtype=float)
    latent_individual_effect = np.zeros_like(true_cate)
    for local_index, action in enumerate(ranked_actions):
        # Fully observed, deterministic effect modification. No individual
        # random term is permitted in the identifiable baseline target.
        specific = _z(
            np.sin((local_index + 1) * age / 2.0)
            + (0.18 + 0.05 * local_index) * burden
            - (0.12 + 0.03 * local_index) * emergency
            + 0.18 * functional * ((local_index % 2) * 2 - 1)
            + 0.10 * social * burden
        )
        response = _z(
            (-0.58 if misaligned else 0.52) * shared_response
            + 0.48 * specific
        )
        true_cate[:, local_index] = (
            action.synthetic_effect_mean_days
            + action.synthetic_effect_scale_days * np.tanh(response / 1.4)
        )
        if hidden:
            latent_individual_effect[:, local_index] = (
                1.4 + 0.25 * local_index
            ) * latent_u
    if null_effect:
        true_cate.fill(0.0)
        latent_individual_effect.fill(0.0)
    individual_effect = true_cate + latent_individual_effect
    potential = mu0[:, None] + individual_effect
    if potential.min() < 0.0 or potential.max() > 365.0:
        raise RuntimeError("Action DGP potential outcomes left [0,365]; adjust effect scales")

    eligible_by_patient: dict[str, list[str]] = {
        patient: [] for patient in context.patient_id
    }
    for row in authoritative_opportunities.itertuples(index=False):
        if bool(row.discretionary_rank_candidate):
            eligible_by_patient[str(row.patient_id)].append(str(row.action_id))
    action_lookup = {action.action_id: local for local, action in enumerate(ranked_actions)}
    observed_action = np.full(n, NO_ACTION, dtype=object)
    for row, patient in enumerate(context.patient_id):
        candidates = [action for action in eligible_by_patient[patient] if action in action_lookup]
        if not candidates:
            continue
        logits = [0.35]
        for action_id in candidates:
            local = action_lookup[action_id]
            # Observed targeting score is constructed independently from the
            # oracle target, although it can be correlated with f_a(X).
            targeting_score = (
                0.35 * shared_response[row]
                + 0.25 * np.sin((local + 1) * age[row] / 2.0)
                + 0.12 * burden[row]
            )
            logit = (
                -0.20
                + (0.72 if strong_observed else (0.55 if poor_overlap else 0.24))
                * risk[row]
                + 0.08 * context.current_care_state_index.iat[row]
                + 0.06 * local
            )
            if targeted or strong_observed:
                logit += (0.70 if strong_observed else 0.40) * targeting_score
            if hidden:
                logit += 0.65 * latent_u[row]
            logits.append(logit)
        values = np.asarray(logits, dtype=float)
        probability = np.exp(values - values.max())
        probability /= probability.sum()
        choice = int(rng.choice(len(probability), p=probability))
        if choice:
            observed_action[row] = candidates[choice - 1]

    # Ensure bounded per-split action-arm coverage for small CPU smoke runs.
    # A shared, deterministic no-action reserve supplies controls for every
    # eligible action.  Coverage treatments are then assigned rare-action first
    # and never reuse a reserved/control patient or another coverage treatment.
    # This keeps the observed treatment multi-valued (one action per patient)
    # while preventing an overlapping common action from consuming every row of
    # a rarer action-specific comparison.
    for split_name in ("nuisance_train", "rank_train", "validation", "test"):
        eligible_rows_by_action = {}
        for action_id in action_ids:
            eligible_rows_by_action[action_id] = np.asarray([
                index_by_patient[patient]
                for patient, candidates in eligible_by_patient.items()
                if action_id in candidates and split[index_by_patient[patient]] == split_name
            ], dtype=int)
        priority = rng.random(n)
        reserved_controls: set[int] = set()
        targets: dict[str, int] = {}
        for action_id, eligible_rows in eligible_rows_by_action.items():
            target = min(
                int(minimum_action_assignments_per_split),
                max(0, len(eligible_rows) // 3),
            )
            targets[action_id] = target
            if target:
                ordered = eligible_rows[np.argsort(priority[eligible_rows], kind="mergesort")]
                reserved_controls.update(map(int, ordered[:target]))
        if reserved_controls:
            observed_action[np.fromiter(reserved_controls, dtype=int)] = NO_ACTION

        coverage_treatments: set[int] = set()
        for action_id in sorted(action_ids, key=lambda value: len(eligible_rows_by_action[value])):
            target = targets[action_id]
            eligible_rows = eligible_rows_by_action[action_id]
            available = np.asarray([
                int(row) for row in eligible_rows
                if int(row) not in reserved_controls and int(row) not in coverage_treatments
            ], dtype=int)
            if target and len(available):
                ordered = available[np.argsort(priority[available], kind="mergesort")]
                chosen = ordered[-min(target, len(ordered)):]
                observed_action[chosen] = action_id
                coverage_treatments.update(map(int, chosen))

        # Make failures explicit: an action with too little catalogue support is
        # a data-adequacy problem, not a reason to silently drop that action.
        for action_id, eligible_rows in eligible_rows_by_action.items():
            target = targets[action_id]
            controls = int(np.sum(observed_action[eligible_rows] == NO_ACTION))
            treated = int(np.sum(observed_action[eligible_rows] == action_id))
            if target < 2 or controls < target or treated < target:
                raise ValueError(
                    "Insufficient catalogue-supported observational coverage for "
                    f"{action_id} in {split_name}: eligible={len(eligible_rows)}, "
                    f"target={target}, controls={controls}, treated={treated}. "
                    "Increase the synthetic sample or broaden the action rule."
                )

    selected_mean = mu0.copy()
    for action_id, local_index in action_lookup.items():
        mask = observed_action == action_id
        selected_mean[mask] = potential[mask, local_index]
    raw_noise = rng.normal(0.0, float(outcome_noise_sd), n)
    observed_outcome = np.clip(selected_mean + raw_noise, 0.0, 365.0)

    learner = context.loc[:, ["patient_id", *features, "current_care_state"]].copy()
    learner["observed_action_id"] = observed_action
    learner["observed_outcome"] = observed_outcome
    learner["split"] = split
    assert_no_oracle_columns(learner)

    truth = pd.DataFrame({
        "patient_id": context.patient_id,
        "prognostic_risk_truth": risk,
        "potential_outcome_no_new_action": mu0,
        "observed_outcome_noise": observed_outcome - selected_mean,
    })
    for action_id, local_index in action_lookup.items():
        truth[f"true_cate__{action_id}"] = true_cate[:, local_index]
        truth[f"latent_individual_effect__{action_id}"] = latent_individual_effect[:, local_index]
        truth[f"individual_effect__{action_id}"] = individual_effect[:, local_index]
        # Compatibility alias for prior synthetic evaluation readers. New
        # integrated metrics use true_cate explicitly.
        truth[f"true_benefit__{action_id}"] = true_cate[:, local_index]
        truth[f"potential_outcome__{action_id}"] = potential[:, local_index]
    if hidden:
        truth["latent_confounder_u"] = latent_u

    action_learners: dict[str, pd.DataFrame] = {}
    populations = {}
    for action in ranked_actions:
        candidate_ids = set(
            authoritative_opportunities.loc[
                (authoritative_opportunities.action_id == action.action_id)
                & authoritative_opportunities.discretionary_rank_candidate.to_numpy(bool),
                "patient_id",
            ].astype(str)
        )
        mask = (
            learner.patient_id.isin(candidate_ids)
            & learner.observed_action_id.isin((NO_ACTION, action.action_id))
        )
        frame = learner.loc[mask].copy()
        frame["action_id"] = action.action_id
        frame["transition"] = action.action_id
        frame["transition_index"] = action.action_index
        frame["transition_treatment"] = (
            frame.observed_action_id == action.action_id
        ).astype(int)
        frame.reset_index(drop=True, inplace=True)
        assert_no_oracle_columns(frame)
        action_learners[action.action_id] = frame
        populations[action.action_id] = {
            "candidate_patients": int(len(candidate_ids)),
            "observational_rows": int(len(frame)),
            "treated_rows": int(frame.transition_treatment.sum()),
            "split_counts": frame.split.value_counts().to_dict(),
        }
    metadata = {
        "dgp": "single_observed_care_action_common_outcome_v2",
        "scenario": scenario,
        "primary_oracle_target": "true_cate_f_of_observed_x",
        "secondary_oracle_target": (
            "individual_effect_true_cate_plus_latent_component" if hidden else None
        ),
        "conditional_exchangeability_by_construction": not hidden,
        "observed_confounding_strength": (
            "strong" if strong_observed else ("poor_overlap" if poor_overlap else "standard")
        ),
        "hidden_confounding_failure_scenario": hidden,
        "negative_control": (
            "placebo_outcome" if placebo else
            "sharp_null_treatment_effect" if null_effect else None
        ),
        "sharp_null_treatment_effect": bool(null_effect),
        "latent_individual_effect_zero": not hidden,
        "treatment_assignment_uses_true_cate": False,
        "treatment_assignment_uses_oracle": False,
        "treatment_assignment_observed_inputs": (
            "preindex_covariates_and_current_care_state"
        ),
        "historical_treatment_semantics": "care_action_id_not_dm77_level",
        "no_action_id": NO_ACTION,
        "outcome_name": "days_alive_outside_acute_hospital_365d",
        "followup_horizon": 365,
        "outcome_unit": "days",
        "action_ids": list(action_ids),
        "action_populations": populations,
        "observed_action_counts": pd.Series(observed_action).value_counts().to_dict(),
        "oracle_physically_separate": True,
        "dm77_need_level_in_learner": False,
        "minimum_action_assignments_per_split": int(minimum_action_assignments_per_split),
    }
    return ActionSimulationResult(learner, truth, action_learners, features, metadata)
