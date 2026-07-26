"""Prespecified discovery, confirmation and fixed-dataset stability campaign."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .allocation import allocate_recommended_profiles
from .evaluation import (
    evaluate_baseline_need,
    evaluate_profile_allocation,
    evaluate_profile_ranking_scores,
    evaluate_profile_recommendations,
)
from .nuisance import build_profile_causal_supervision
from .ranking import train_profile_rankers
from .recommendation import calibrate_and_recommend_profiles
from .synthetic import generate_synthetic_population, generate_synthetic_profile_dgp


DATA_SEED_COMPONENTS = (
    "population_seed",
    "reference_seed",
    "need_model_seed",
    "need_split_seed",
    "current_care_seed",
    "effect_seed",
    "assignment_seed",
    "outcome_seed",
)
RUN_SEED_COMPONENTS = (
    "split_seed",
    "nuisance_seed",
    "ranker_model_seed",
    "ranker_pair_seed",
    "ranker_validation_pair_seed",
    "ranker_contrastive_seed",
    "ranker_gbdt_seed",
    "ranker_random_seed",
    "allocator_tie_seed",
    "evaluation_seed",
)
METRIC_COLUMNS = (
    "record_type",
    "phase",
    "run_seed",
    "dataset_seed",
    "layer",
    "variant",
    "metric",
    "value",
    "uses_oracle",
    "oracle_unblinded_after_freeze",
    "n_runs",
    "ci_lower",
    "ci_upper",
    "paired_comparator",
    "notes",
)


@dataclass(frozen=True)
class DiscoveryCampaignResult:
    records: pd.DataFrame
    selected_variant: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class ConfirmationCampaignResult:
    metrics: pd.DataFrame
    freezes: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class StabilizationDiagnosticResult:
    records: pd.DataFrame
    audit: dict[str, Any]


def _metric_row(**values: Any) -> dict[str, Any]:
    row = {column: np.nan for column in METRIC_COLUMNS}
    row.update(
        {
            "uses_oracle": False,
            "oracle_unblinded_after_freeze": False,
            "paired_comparator": "",
            "notes": "",
        }
    )
    row.update(values)
    return row


def _seed_values(seed: int, count: int, namespace: int) -> list[int]:
    state = np.random.SeedSequence(
        int(seed), spawn_key=(int(namespace),)
    ).generate_state(int(count), dtype=np.uint32)
    values = [1 + int(value) % 2_000_000_000 for value in state]
    if len(set(values)) != len(values):
        raise RuntimeError("Experiment component-seed derivation produced a collision")
    return values


def derive_experiment_seeds(
    run_seed: int, dataset_seed: int | None = None
) -> dict[str, int]:
    """Derive explicit data and learner seeds without crossing registry partitions."""

    data_authority = int(run_seed if dataset_seed is None else dataset_seed)
    data = dict(zip(
        DATA_SEED_COMPONENTS,
        _seed_values(data_authority, len(DATA_SEED_COMPONENTS), 17),
    ))
    learner = dict(zip(
        RUN_SEED_COMPONENTS,
        _seed_values(int(run_seed), len(RUN_SEED_COMPONENTS), 29),
    ))
    if set(data.values()).intersection(learner.values()):
        raise RuntimeError("Experiment data and learner component seeds collided")
    return {**data, **learner}


def _canonical_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    selected = selected.sort_values(list(columns), kind="stable")
    return hashlib.sha256(selected.to_csv(index=False).encode("utf-8")).hexdigest()


def _canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _profile_ids(catalog) -> list[str]:
    return [profile.care_profile_id for profile in catalog.automatic_rank_profiles]


def _learning_stack(
    config: Mapping[str, Any],
    catalog,
    *,
    run_seed: int,
    dataset_seed: int | None,
    primary_variant: str,
    full_decision_stack: bool,
    component_seeds: Mapping[str, int] | None = None,
    ranker_settings_override: Mapping[str, Any] | None = None,
    causal_supervision_override: Mapping[str, Any] | None = None,
    candidate_variants: Sequence[str] | None = None,
    dgp_stress_override: Mapping[str, Any] | None = None,
    prepared_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    experiments = config["experiments"]
    seeds = (
        derive_experiment_seeds(run_seed, dataset_seed)
        if component_seeds is None
        else {key: int(value) for key, value in component_seeds.items()}
    )
    required_seeds = set(DATA_SEED_COMPONENTS) | set(RUN_SEED_COMPONENTS)
    if set(seeds) != required_seeds or len(set(seeds.values())) != len(seeds):
        raise ValueError("Experiment component seeds must be complete and distinct")
    if prepared_state is None:
        synthetic = config["synthetic_population"]
        population = generate_synthetic_population(
            population_size=int(experiments["population_size"]),
            seed=int(seeds["population_seed"]),
            history_months=int(config["run"]["history_months"]),
            reference_date=str(synthetic["reference_date"]),
            profile=str(synthetic["profile"]),
            scenario=str(synthetic["scenario"]),
        )
        dgp = generate_synthetic_profile_dgp(
            population.patients,
            catalog,
            scenario=str(experiments["scenario"]),
            need_mode=str(config["baseline_need"]["mode"]),
            baseline_need_contract=config["baseline_need"]["contract"],
            index_date=str(synthetic["reference_date"]),
            stress_settings=dgp_stress_override,
            **{
                name: int(seeds[name])
                for name in DATA_SEED_COMPONENTS
                if name != "population_seed"
            },
        )
        supervision_settings = copy.deepcopy(dict(config["causal_supervision"]))
        if causal_supervision_override:
            supervision_settings.update(
                copy.deepcopy(dict(causal_supervision_override))
            )
        supervision = build_profile_causal_supervision(
            dgp.learner,
            dgp.patient_profile_opportunities,
            supervision_settings,
            split_seed=int(seeds["split_seed"]),
            nuisance_seed=int(seeds["nuisance_seed"]),
            profile_candidates=[
                (profile.care_profile_id, profile.care_profile_index)
                for profile in catalog.automatic_rank_profiles
            ],
        )
    else:
        if int(prepared_state.get("run_seed", -1)) != int(run_seed):
            raise ValueError("Prepared experiment data must use the same run seed")
        prepared_seeds = {
            key: int(value) for key, value in prepared_state.get("seeds", {}).items()
        }
        if prepared_seeds != seeds:
            raise ValueError("Prepared experiment data must use identical component seeds")
        for key in ("population", "dgp", "supervision"):
            if key not in prepared_state:
                raise ValueError(f"Prepared experiment data is missing {key}")
        population = prepared_state["population"]
        dgp = prepared_state["dgp"]
        supervision = prepared_state["supervision"]
    ranker_settings = copy.deepcopy(dict(config["ranker"]))
    if ranker_settings_override:
        ranker_settings.update(copy.deepcopy(dict(ranker_settings_override)))
    ranker_settings["variants"] = list(
        experiments["candidate_variants"]
        if candidate_variants is None
        else map(str, candidate_variants)
    )
    ranker_settings["primary_variant"] = str(primary_variant)
    ranker = train_profile_rankers(
        dgp.learner,
        supervision.supervision,
        supervision.supported_opportunities,
        supervision.patient_splits,
        ranker_settings,
        numeric_features=config["causal_supervision"]["numeric_features"],
        categorical_features=config["causal_supervision"]["categorical_features"],
        profile_ids=_profile_ids(catalog),
        model_seed=int(seeds["ranker_model_seed"]),
        pair_seed=int(seeds["ranker_pair_seed"]),
        validation_pair_seed=int(seeds["ranker_validation_pair_seed"]),
        contrastive_seed=int(seeds["ranker_contrastive_seed"]),
        gbdt_seed=int(seeds["ranker_gbdt_seed"]),
        random_seed=int(seeds["ranker_random_seed"]),
    )
    if ranker.audit["status"] != "trained":
        raise RuntimeError(f"Experiment ranker did not train for seed {run_seed}")
    state = {
        "run_seed": int(run_seed),
        "dataset_seed": None if dataset_seed is None else int(dataset_seed),
        "seeds": seeds,
        "dgp": dgp,
        "supervision": supervision,
        "ranker": ranker,
        "population": population,
    }
    if not full_decision_stack:
        return state
    recommendation = calibrate_and_recommend_profiles(
        ranker.scores,
        supervision.supervision,
        supervision.supported_opportunities,
        dgp.baseline_need_assessments,
        dgp.current_care_profiles,
        supervision.patient_splits,
        supervision.support_diagnostics,
        primary_variant=str(primary_variant),
        calibration_contract=config["scientific_contract"]["calibration_contract"],
        recommendation_contract=config["scientific_contract"]["recommendation_contract"],
        implementation_settings=config["recommendation"],
    )
    allocation = allocate_recommended_profiles(
        recommendation.recommendations,
        supervision.supported_opportunities,
        dgp.resource_capacities,
        dgp.profile_resources,
        allocation_contract=config["scientific_contract"]["allocation_contract"],
        settings=config["allocation"],
        tie_seed=int(seeds["allocator_tie_seed"]),
    )
    state.update({"recommendation": recommendation, "allocation": allocation})
    return state


def run_discovery_campaign(
    config: Mapping[str, Any], catalog
) -> DiscoveryCampaignResult:
    """Select one candidate using validation losses only; never open truth."""

    experiments = config["experiments"]
    seeds = tuple(map(int, experiments["discovery"]["run_seeds"]))
    variants = tuple(map(str, experiments["candidate_variants"]))
    records: list[dict[str, Any]] = []
    derived: dict[str, dict[str, int]] = {}
    for run_seed in seeds:
        state = _learning_stack(
            config,
            catalog,
            run_seed=run_seed,
            dataset_seed=None,
            primary_variant=variants[0],
            full_decision_stack=False,
        )
        derived[str(run_seed)] = state["seeds"]
        ranker = state["ranker"]
        for variant in variants:
            audit = ranker.audit["variants"][variant]
            records.append({
                "record_type": "discovery_run",
                "run_seed": int(run_seed),
                "candidate_variant": variant,
                "selection_partition": "validation",
                "selection_metric": "best_validation_pairwise_loss",
                "selection_value": float(audit["best_validation_pairwise_loss"]),
                "fixed_validation_pairs": int(audit["fixed_validation_pairs"]),
                "validation_within_profile_pairs": int(
                    audit["validation_within_profile_pairs"]
                ),
                "validation_cross_profile_pairs": int(
                    audit["validation_cross_profile_pairs"]
                ),
                "oracle_used": False,
                "test_partition_used_for_selection": False,
                "confirmation_seed_used": False,
            })
    frame = pd.DataFrame(records)
    aggregate = (
        frame.groupby("candidate_variant", sort=True).selection_value
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_selection_value",
            "std": "std_selection_value",
            "count": "discovery_runs",
        })
    )
    selected = aggregate.sort_values(
        ["mean_selection_value", "std_selection_value", "candidate_variant"],
        ascending=[True, True, True],
        kind="stable",
    ).iloc[0]
    selected_variant = str(selected.candidate_variant)
    summary_records = []
    for row in aggregate.itertuples(index=False):
        summary_records.append({
            "record_type": "discovery_summary",
            "run_seed": np.nan,
            "candidate_variant": str(row.candidate_variant),
            "selection_partition": "validation",
            "selection_metric": "mean_best_validation_pairwise_loss",
            "selection_value": float(row.mean_selection_value),
            "selection_std": float(row.std_selection_value),
            "discovery_runs": int(row.discovery_runs),
            "selected": str(row.candidate_variant) == selected_variant,
            "oracle_used": False,
            "test_partition_used_for_selection": False,
            "confirmation_seed_used": False,
        })
    frame = pd.concat([frame, pd.DataFrame(summary_records)], ignore_index=True)
    return DiscoveryCampaignResult(
        records=frame,
        selected_variant=selected_variant,
        audit={
            "status": "candidate_selected",
            "run_seeds": list(seeds),
            "candidate_variants": list(variants),
            "selected_variant": selected_variant,
            "selector": "minimum_mean_validation_pairwise_loss",
            "tie_breakers": [
                "minimum_standard_deviation", "lexicographic_variant_name"
            ],
            "selection_partition": "validation",
            "test_partition_used": False,
            "oracle_used": False,
            "confirmation_seeds_used": False,
            "derived_component_seeds": derived,
        },
    )


def build_candidate_declaration(
    config: Mapping[str, Any],
    discovery: DiscoveryCampaignResult,
    *,
    discovery_artifact_sha256: str,
    stabilization_diagnostic_sha256: str,
) -> dict[str, Any]:
    """Create the immutable configuration payload written before confirmation."""

    experiment = config["experiments"]
    candidate_configuration = {
        "primary_variant": discovery.selected_variant,
        "ranker": {
            key: config["ranker"][key]
            for key in (
                "hidden_dim", "profile_embedding_dim", "projection_dim",
                "model_restarts", "restart_aggregation", "epochs",
                "patience", "minimum_improvement", "learning_rate", "weight_decay",
                "training_pairs", "validation_pairs", "within_profile_fraction",
                "minimum_signal_gap", "minimum_direction_agreement",
                "maximum_pair_weight", "allow_cross_profile_training_pairs",
                "allow_same_patient_cross_profile_pairs", "contrastive_pairs",
                "contrastive_response_bins", "contrastive_weight",
                "contrastive_margin",
            )
        },
        "causal_supervision": dict(config["causal_supervision"]),
        "calibration": dict(config["scientific_contract"]["calibration_contract"]),
        "recommendation": dict(
            config["scientific_contract"]["recommendation_contract"]
        ),
        "recommendation_implementation": dict(config["recommendation"]),
        "allocation": {
            "contract": dict(config["scientific_contract"]["allocation_contract"]),
            "settings": dict(config["allocation"]),
        },
    }
    commitments = {
        "confirmation_seed_commitment_sha256": _canonical_payload_hash({
            "run_seeds": list(map(int, experiment["confirmation"]["run_seeds"]))
        }),
        "fixed_dataset_seed_commitment_sha256": _canonical_payload_hash({
            "dataset_seed": int(experiment["fixed_dataset_stability"]["dataset_seed"]),
            "run_seeds": list(map(
                int, experiment["fixed_dataset_stability"]["run_seeds"]
            )),
        }),
    }
    return {
        "declaration_version": "prometheus_stabilized_locked_candidate_v9",
        "status": "locked_before_confirmation",
        "scenario": str(experiment["scenario"]),
        "population_size": int(experiment["population_size"]),
        "selected_primary_variant": discovery.selected_variant,
        "selection_rule": discovery.audit["selector"],
        "selection_partition": "validation",
        "discovery_run_seeds": discovery.audit["run_seeds"],
        "discovery_artifact_sha256": str(discovery_artifact_sha256),
        "stabilization_diagnostic_sha256": str(
            stabilization_diagnostic_sha256
        ),
        "stabilization_gates_passed_before_discovery": True,
        "candidate_configuration": candidate_configuration,
        "candidate_configuration_sha256": _canonical_payload_hash(
            candidate_configuration
        ),
        **commitments,
        "oracle_used_for_selection": False,
        "test_partition_used_for_selection": False,
        "confirmation_results_seen_before_lock": False,
    }


def _freeze_row(
    state: Mapping[str, Any],
    *,
    phase: str,
    candidate_declaration_sha256: str,
) -> dict[str, Any]:
    ranker = state["ranker"]
    recommendation = state["recommendation"]
    allocation = state["allocation"]
    dgp = state["dgp"]
    supervision = state["supervision"]
    return {
        "phase": str(phase),
        "run_seed": int(state["run_seed"]),
        "dataset_seed": (
            np.nan if state["dataset_seed"] is None else int(state["dataset_seed"])
        ),
        "candidate_declaration_sha256": str(candidate_declaration_sha256),
        "learner_dataset_sha256": _canonical_frame_hash(
            dgp.learner,
            ["patient_id", "observed_treatment_profile", "observed_outcome"],
        ),
        "supervision_sha256": _canonical_frame_hash(
            supervision.supervision,
            ["patient_id", "care_profile_id", "split", "dr_pseudo_outcome"],
        ),
        "priority_scores_sha256": _canonical_frame_hash(
            ranker.scores,
            ["patient_id", "care_profile_id", "method", "method_score"],
        ),
        "recommendations_sha256": _canonical_frame_hash(
            recommendation.recommendations,
            [
                "patient_id", "recommended_profile_id", "recommendation_abstained",
                "recommended_actionable_level",
            ],
        ),
        "allocations_sha256": _canonical_frame_hash(
            allocation.decisions,
            ["patient_id", "allocated_profile_id", "allocation_status"],
        ),
        "primary_variant": str(ranker.audit["primary_variant"]),
        "recommendation_policy_frozen": True,
        "allocation_policy_frozen": True,
        "constraint_violations": int(allocation.audit["constraint_violations_total"]),
        "oracle_used_before_freeze": False,
        "truth_unblinded_after_this_record": True,
        "component_seeds_json": json.dumps(state["seeds"], sort_keys=True),
    }


def _append_freeze(path: Path, row: Mapping[str, Any]) -> pd.DataFrame:
    existing = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([dict(row)])], ignore_index=True)
    updated.to_csv(path, index=False)
    return updated


def _baseline_metric_rows(state: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    dgp = state["dgp"]
    evaluation = evaluate_baseline_need(
        dgp.need_ground_truth,
        dgp.baseline_need_assessments,
        subgroup_frame=dgp.learner,
        subgroup_columns=("female", "rurality", "caregiver_available"),
        minimum_subgroup_size=30,
    )
    common = {
        "record_type": "run_metric",
        "phase": phase,
        "run_seed": int(state["run_seed"]),
        "dataset_seed": (
            np.nan if state["dataset_seed"] is None else int(state["dataset_seed"])
        ),
        "layer": "baseline_need",
        "variant": str(state["dgp"].baseline_need_assessments.baseline_need_mode.iloc[0]),
        "uses_oracle": True,
        "oracle_unblinded_after_freeze": True,
    }
    rows = []
    for metric, value in evaluation.summary.items():
        if isinstance(value, (bool, str)):
            continue
        rows.append(_metric_row(**common, metric=str(metric), value=float(value)))
    for matrix_row in evaluation.confusion_matrix.itertuples(index=False):
        reference = int(matrix_row.reference_level)
        for level in range(1, 7):
            rows.append(_metric_row(
                **common,
                metric=f"confusion_reference_{reference}_predicted_{level}",
                value=float(getattr(matrix_row, f"predicted_level_{level}")),
            ))
    for subgroup in evaluation.subgroup_metrics.itertuples(index=False):
        if str(subgroup.status) != "reported":
            continue
        for metric in (
            "quadratic_weighted_kappa", "macro_f1", "balanced_accuracy",
            "ordinal_mae", "within_one_level_accuracy", "spearman",
        ):
            rows.append(_metric_row(
                **common,
                metric=(
                    f"subgroup:{subgroup.subgroup_column}={subgroup.subgroup_value}:"
                    f"{metric}"
                ),
                value=float(getattr(subgroup, metric)),
            ))
    return rows


def _evaluation_rows(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    phase: str,
    selected_variant: str,
) -> list[dict[str, Any]]:
    rows = _baseline_metric_rows(state, phase)
    common = {
        "record_type": "run_metric",
        "phase": phase,
        "run_seed": int(state["run_seed"]),
        "dataset_seed": (
            np.nan if state["dataset_seed"] is None else int(state["dataset_seed"])
        ),
        "oracle_unblinded_after_freeze": True,
    }
    ranking = evaluate_profile_ranking_scores(
        state["ranker"].scores,
        state["supervision"].supervision,
        state["dgp"].profile_ground_truth,
        split="test",
        fractions=config["experiments"]["reporting"]["capacity_fractions"],
        seed=int(state["seeds"]["evaluation_seed"]),
    )
    for item in ranking.itertuples(index=False):
        rows.append(_metric_row(
            **common,
            layer="causal_ranking",
            variant=str(item.method),
            metric=str(item.metric),
            value=float(item.value),
            uses_oracle=bool(item.uses_oracle),
            notes=(
                "locked_candidate" if str(item.method) == selected_variant
                else "prespecified_comparator"
            ),
        ))
    recommendation = evaluate_profile_recommendations(
        state["recommendation"].recommendations,
        state["dgp"].profile_ground_truth,
        state["supervision"].supported_opportunities,
        benefit_threshold_days=float(
            config["scientific_contract"]["recommendation_contract"][
                "primary_minimum_benefit_threshold_days"
            ]
        ),
        split="test",
    )
    for item in recommendation.itertuples(index=False):
        rows.append(_metric_row(
            **common,
            layer="recommendation",
            variant=selected_variant,
            metric=str(item.metric),
            value=float(item.value),
            uses_oracle=bool(item.uses_oracle),
        ))
    oracle_solver_settings = dict(state["allocation"].audit["solver_settings"])
    oracle_solver_settings["time_limit_seconds"] = float(
        config["experiments"]["reporting"]["oracle_solver_time_limit_seconds"]
    )
    allocation = evaluate_profile_allocation(
        state["allocation"].decisions,
        state["allocation"].allocation_candidates,
        state["allocation"].diagnostic_selections,
        state["dgp"].profile_ground_truth,
        shared_budget_limit=float(state["allocation"].audit["shared_budget_limit"]),
        pool_capacities=state["allocation"].audit["pool_capacities"],
        profile_capacities=state["allocation"].audit["profile_capacities"],
        tie_seed=int(state["allocation"].audit["tie_seed"]),
        tie_break_epsilon=float(state["allocation"].audit["tie_break_epsilon"]),
        solver_settings=oracle_solver_settings,
    )
    for item in allocation.itertuples(index=False):
        rows.append(_metric_row(
            **common,
            layer="allocation",
            variant=selected_variant,
            metric=str(item.metric),
            value=float(item.value),
            uses_oracle=bool(item.uses_oracle),
        ))
    return rows


def _stability_state(state: Mapping[str, Any], selected_variant: str) -> dict[str, Any]:
    scores = state["ranker"].scores.loc[
        state["ranker"].scores.method.astype(str).eq(selected_variant),
        ["patient_id", "care_profile_id", "method_score"],
    ].copy()
    scores["opportunity_id"] = (
        scores.patient_id.astype(str) + "|" + scores.care_profile_id.astype(str)
    )
    return {
        "run_seed": int(state["run_seed"]),
        "scores": scores[["opportunity_id", "method_score"]],
        "recommendations": state["recommendation"].recommendations[[
            "patient_id", "recommended_profile_id", "recommendation_abstained"
        ]].copy(),
        "allocations": state["allocation"].decisions[[
            "patient_id", "allocated_profile_id", "allocation_status"
        ]].copy(),
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def run_stabilization_source_diagnostics(
    config: Mapping[str, Any], catalog
) -> StabilizationDiagnosticResult:
    """Isolate non-oracle stability sources before opening new discovery seeds."""

    experiment = config["experiments"]
    settings = experiment["stability_diagnostics"]
    dataset_seed = int(settings["dataset_seed"])
    reference_seed = int(settings["reference_run_seed"])
    perturbation_seeds = tuple(map(int, settings["perturbation_run_seeds"]))
    sources = {
        str(name): tuple(map(str, components))
        for name, components in settings["sources"].items()
    }
    score_variant = str(settings["score_variant"])
    diagnostic_primary = (
        score_variant
        if score_variant in {"global_rank_only", "global_rank_plus_contrastive"}
        else "global_rank_only"
    )
    diagnostic_variants = tuple(dict.fromkeys((diagnostic_primary, score_variant)))
    configurations = {
        "short_training_baseline": (
            dict(settings["baseline_ranker_settings"]),
            dict(settings["baseline_causal_supervision_settings"]),
        ),
        "stabilized": (
            dict(settings["stabilized_ranker_settings"]),
            dict(settings["stabilized_causal_supervision_settings"]),
        ),
    }
    reference_components = derive_experiment_seeds(reference_seed, dataset_seed)
    records: list[dict[str, Any]] = []
    seed_audit: dict[str, Any] = {}
    for configuration, overrides in configurations.items():
        ranker_override, supervision_override = overrides
        full_decision_stack = configuration == "stabilized"
        reference = _learning_stack(
            config,
            catalog,
            run_seed=reference_seed,
            dataset_seed=dataset_seed,
            primary_variant=diagnostic_primary,
            full_decision_stack=full_decision_stack,
            component_seeds=reference_components,
            ranker_settings_override=ranker_override,
            causal_supervision_override=supervision_override,
            candidate_variants=diagnostic_variants,
        )
        reference_scores = reference["ranker"].scores.loc[
            reference["ranker"].scores.method.astype(str).eq(score_variant),
            ["patient_id", "care_profile_id", "method_score"],
        ].copy()
        reference_scores["opportunity_id"] = (
            reference_scores.patient_id.astype(str)
            + "|"
            + reference_scores.care_profile_id.astype(str)
        )
        reference_scores = reference_scores[["opportunity_id", "method_score"]]
        if full_decision_stack:
            reference_recommended = set(
                reference["recommendation"].recommendations.loc[
                    ~reference["recommendation"].recommendations[
                        "recommendation_abstained"
                    ].astype(bool),
                    "patient_id",
                ].astype(str)
            )
            reference_allocated = set(
                reference["allocation"].decisions.loc[
                    reference["allocation"].decisions.allocated_profile_id.notna(),
                    "patient_id",
                ].astype(str)
            )
        for source, varied_components in sources.items():
            for perturbation_seed in perturbation_seeds:
                perturbation = derive_experiment_seeds(
                    perturbation_seed, dataset_seed
                )
                component_seeds = dict(reference_components)
                for component in varied_components:
                    if component not in RUN_SEED_COMPONENTS:
                        raise ValueError(
                            f"Invalid stabilization component {component!r}"
                        )
                    component_seeds[component] = int(perturbation[component])
                if len(set(component_seeds.values())) != len(component_seeds):
                    raise RuntimeError("Stabilization diagnostic seed collision")
                state = _learning_stack(
                    config,
                    catalog,
                    run_seed=perturbation_seed,
                    dataset_seed=dataset_seed,
                    primary_variant=diagnostic_primary,
                    full_decision_stack=full_decision_stack,
                    component_seeds=component_seeds,
                    ranker_settings_override=ranker_override,
                    causal_supervision_override=supervision_override,
                    candidate_variants=diagnostic_variants,
                )
                seed_audit[
                    f"{configuration}:{source}:{perturbation_seed}"
                ] = component_seeds
                scores = state["ranker"].scores.loc[
                    state["ranker"].scores.method.astype(str).eq(score_variant),
                    ["patient_id", "care_profile_id", "method_score"],
                ].copy()
                scores["opportunity_id"] = (
                    scores.patient_id.astype(str)
                    + "|"
                    + scores.care_profile_id.astype(str)
                )
                common = reference_scores.merge(
                    scores[["opportunity_id", "method_score"]],
                    on="opportunity_id",
                    suffixes=("_reference", "_perturbed"),
                    validate="one_to_one",
                )
                if len(common) < 2:
                    raise RuntimeError(
                        "Stabilization diagnostics require common supported opportunities"
                    )
                metrics = {
                    "ranking_score_spearman": float(spearmanr(
                        common.method_score_reference,
                        common.method_score_perturbed,
                    ).statistic),
                }
                for fraction in map(float, settings["fractions"]):
                    count = max(1, int(np.ceil(fraction * len(common))))
                    reference_top = set(common.sort_values(
                        ["method_score_reference", "opportunity_id"],
                        ascending=[False, True],
                        kind="stable",
                    ).head(count).opportunity_id)
                    perturbed_top = set(common.sort_values(
                        ["method_score_perturbed", "opportunity_id"],
                        ascending=[False, True],
                        kind="stable",
                    ).head(count).opportunity_id)
                    metrics[
                        f"ranking_top_{int(round(100 * fraction))}pct_jaccard"
                    ] = _jaccard(reference_top, perturbed_top)
                if full_decision_stack:
                    perturbed_recommended = set(
                        state["recommendation"].recommendations.loc[
                            ~state["recommendation"].recommendations[
                                "recommendation_abstained"
                            ].astype(bool),
                            "patient_id",
                        ].astype(str)
                    )
                    perturbed_allocated = set(
                        state["allocation"].decisions.loc[
                            state["allocation"].decisions.allocated_profile_id.notna(),
                            "patient_id",
                        ].astype(str)
                    )
                    metrics["recommended_patient_jaccard"] = _jaccard(
                        reference_recommended, perturbed_recommended
                    )
                    metrics["allocated_patient_jaccard"] = _jaccard(
                        reference_allocated, perturbed_allocated
                    )
                for metric, value in metrics.items():
                    records.append({
                        "record_type": "source_replicate",
                        "configuration": configuration,
                        "source": source,
                        "reference_run_seed": reference_seed,
                        "perturbation_run_seed": int(perturbation_seed),
                        "dataset_seed": dataset_seed,
                        "varied_components": ",".join(varied_components),
                        "metric": metric,
                        "value": float(value),
                        "common_opportunities": int(len(common)),
                        "uses_outcome_in_stability_metric": False,
                        "uses_oracle": False,
                    })
    frame = pd.DataFrame(records)
    summaries = []
    for (configuration, source, metric), group in frame.groupby(
        ["configuration", "source", "metric"], sort=True
    ):
        summaries.append({
            "record_type": "source_summary",
            "configuration": configuration,
            "source": source,
            "reference_run_seed": reference_seed,
            "perturbation_run_seed": np.nan,
            "dataset_seed": dataset_seed,
            "varied_components": "",
            "metric": metric,
            "value": float(group.value.mean()),
            "common_opportunities": int(group.common_opportunities.min()),
            "uses_outcome_in_stability_metric": False,
            "uses_oracle": False,
        })
    for (configuration, metric), group in frame.groupby(
        ["configuration", "metric"], sort=True
    ):
        summaries.append({
            "record_type": "source_summary",
            "configuration": configuration,
            "source": "all",
            "reference_run_seed": reference_seed,
            "perturbation_run_seed": np.nan,
            "dataset_seed": dataset_seed,
            "varied_components": "",
            "metric": metric,
            "value": float(group.value.mean()),
            "common_opportunities": int(group.common_opportunities.min()),
            "uses_outcome_in_stability_metric": False,
            "uses_oracle": False,
        })
    frame = pd.concat([frame, pd.DataFrame(summaries)], ignore_index=True)

    def summary_value(
        configuration: str, metric: str, source: str = "all"
    ) -> float:
        selected = frame.loc[
            frame.record_type.eq("source_summary")
            & frame.configuration.eq(configuration)
            & frame.source.eq(source)
            & frame.metric.eq(metric),
            "value",
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Missing stabilization summary for {metric}")
        return float(selected.iloc[0])

    acceptance = settings["acceptance"]
    stabilized_spearman = summary_value("stabilized", "ranking_score_spearman")
    baseline_spearman = summary_value(
        "short_training_baseline", "ranking_score_spearman"
    )
    stabilized_top10 = summary_value(
        "stabilized", "ranking_top_10pct_jaccard"
    )
    baseline_top10 = summary_value(
        "short_training_baseline", "ranking_top_10pct_jaccard"
    )
    stabilized_top20 = summary_value(
        "stabilized", "ranking_top_20pct_jaccard"
    )
    stabilized_recommended = summary_value(
        "stabilized", "recommended_patient_jaccard"
    )
    stabilized_allocated = summary_value(
        "stabilized", "allocated_patient_jaccard"
    )
    gates = {
        "stabilized_score_spearman": stabilized_spearman >= float(
            acceptance["minimum_stabilized_mean_score_spearman"]
        ),
        "stabilized_top_10pct_jaccard": stabilized_top10 >= float(
            acceptance["minimum_stabilized_mean_top_10pct_jaccard"]
        ),
        "stabilized_top_20pct_jaccard": stabilized_top20 >= float(
            acceptance["minimum_stabilized_mean_top_20pct_jaccard"]
        ),
        "stabilized_recommended_patient_jaccard": stabilized_recommended >= float(
            acceptance["minimum_stabilized_mean_recommended_patient_jaccard"]
        ),
        "stabilized_allocated_patient_jaccard": stabilized_allocated >= float(
            acceptance["minimum_stabilized_mean_allocated_patient_jaccard"]
        ),
        "oracle_and_outcome_metric_isolation": bool(
            not frame.uses_oracle.astype(bool).any()
            and not frame.uses_outcome_in_stability_metric.astype(bool).any()
        ),
    }
    source_metric_thresholds = {
        "ranking_score_spearman": float(
            acceptance["minimum_each_source_score_spearman"]
        ),
        "ranking_top_10pct_jaccard": float(
            acceptance["minimum_each_source_top_10pct_jaccard"]
        ),
        "ranking_top_20pct_jaccard": float(
            acceptance["minimum_each_source_top_20pct_jaccard"]
        ),
        "recommended_patient_jaccard": float(
            acceptance["minimum_each_source_recommended_patient_jaccard"]
        ),
        "allocated_patient_jaccard": float(
            acceptance["minimum_each_source_allocated_patient_jaccard"]
        ),
    }
    for source in sources:
        for metric, threshold in source_metric_thresholds.items():
            gates[f"{source}:{metric}"] = bool(
                summary_value("stabilized", metric, source) >= threshold
            )
    return StabilizationDiagnosticResult(
        records=frame,
        audit={
            "status": "passed" if all(gates.values()) else "failed",
            "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "dataset_seed": dataset_seed,
            "reference_run_seed": reference_seed,
            "perturbation_run_seeds": list(perturbation_seeds),
            "sources": {key: list(value) for key, value in sources.items()},
            "score_variant": score_variant,
            "baseline_overall": {
                "ranking_score_spearman": baseline_spearman,
                "ranking_top_10pct_jaccard": baseline_top10,
            },
            "baseline_comparison_role": str(
                settings["baseline_comparison_role"]
            ),
            "stabilized_overall": {
                "ranking_score_spearman": stabilized_spearman,
                "ranking_top_10pct_jaccard": stabilized_top10,
                "ranking_top_20pct_jaccard": stabilized_top20,
                "recommended_patient_jaccard": stabilized_recommended,
                "allocated_patient_jaccard": stabilized_allocated,
            },
            "oracle_used": False,
            "observed_outcome_used_for_causal_supervision": True,
            "outcome_used_in_stability_metric": False,
            "component_seed_audit": seed_audit,
        },
    )


def _pairwise_stability_rows(
    states: Sequence[Mapping[str, Any]],
    *,
    dataset_seed: int,
    fractions: Sequence[float],
    selected_variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(states, 2):
        pair = f"{left['run_seed']}__{right['run_seed']}"
        scores = left["scores"].merge(
            right["scores"], on="opportunity_id", suffixes=("_left", "_right")
        )
        correlation = float(spearmanr(
            scores.method_score_left, scores.method_score_right
        ).statistic)
        rows.append(_metric_row(
            record_type="stability_pair",
            phase="fixed_dataset_stability",
            run_seed=pair,
            dataset_seed=int(dataset_seed),
            layer="causal_ranking",
            variant=selected_variant,
            metric="ranking_score_spearman",
            value=correlation,
            uses_oracle=False,
            notes=f"common_opportunities={len(scores)}",
        ))
        for fraction in map(float, fractions):
            count = max(1, int(np.ceil(fraction * len(scores))))
            left_top = set(scores.sort_values(
                ["method_score_left", "opportunity_id"],
                ascending=[False, True], kind="stable"
            ).head(count).opportunity_id)
            right_top = set(scores.sort_values(
                ["method_score_right", "opportunity_id"],
                ascending=[False, True], kind="stable"
            ).head(count).opportunity_id)
            rows.append(_metric_row(
                record_type="stability_pair",
                phase="fixed_dataset_stability",
                run_seed=pair,
                dataset_seed=int(dataset_seed),
                layer="causal_ranking",
                variant=selected_variant,
                metric=f"ranking_top_{int(round(100 * fraction))}pct_jaccard",
                value=_jaccard(left_top, right_top),
                uses_oracle=False,
            ))
        recommendation = left["recommendations"].merge(
            right["recommendations"], on="patient_id", suffixes=("_left", "_right")
        )
        left_recommended = set(
            recommendation.loc[
                ~recommendation.recommendation_abstained_left.astype(bool), "patient_id"
            ].astype(str)
        )
        right_recommended = set(
            recommendation.loc[
                ~recommendation.recommendation_abstained_right.astype(bool), "patient_id"
            ].astype(str)
        )
        exact_profile = (
            recommendation.recommended_profile_id_left.fillna("").astype(str)
            == recommendation.recommended_profile_id_right.fillna("").astype(str)
        ).mean()
        rows.extend([
            _metric_row(
                record_type="stability_pair",
                phase="fixed_dataset_stability",
                run_seed=pair,
                dataset_seed=int(dataset_seed),
                layer="recommendation",
                variant=selected_variant,
                metric="recommended_patient_jaccard",
                value=_jaccard(left_recommended, right_recommended),
                uses_oracle=False,
            ),
            _metric_row(
                record_type="stability_pair",
                phase="fixed_dataset_stability",
                run_seed=pair,
                dataset_seed=int(dataset_seed),
                layer="recommendation",
                variant=selected_variant,
                metric="recommended_profile_exact_agreement",
                value=float(exact_profile),
                uses_oracle=False,
            ),
        ])
        allocation = left["allocations"].merge(
            right["allocations"], on="patient_id", suffixes=("_left", "_right")
        )
        left_allocated = set(
            allocation.loc[
                allocation.allocated_profile_id_left.notna(), "patient_id"
            ].astype(str)
        )
        right_allocated = set(
            allocation.loc[
                allocation.allocated_profile_id_right.notna(), "patient_id"
            ].astype(str)
        )
        rows.append(_metric_row(
            record_type="stability_pair",
            phase="fixed_dataset_stability",
            run_seed=pair,
            dataset_seed=int(dataset_seed),
            layer="allocation",
            variant=selected_variant,
            metric="allocated_patient_jaccard",
            value=_jaccard(left_allocated, right_allocated),
            uses_oracle=False,
        ))
    return rows


def _bootstrap_mean(
    values: np.ndarray, *, samples: int, seed: int
) -> tuple[float, float, float]:
    if not len(values) or not np.isfinite(values).all():
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, len(values), size=(int(samples), len(values)))].mean(
        axis=1
    )
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def summarize_experiment_metrics(
    metrics: pd.DataFrame,
    *,
    selected_variant: str,
    comparator_variant: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Add across-run uncertainty and selected-vs-comparator paired intervals."""

    rows: list[dict[str, Any]] = []
    confirmation = metrics.loc[
        metrics.record_type.eq("run_metric") & metrics.phase.eq("confirmation")
    ].copy()
    group_columns = ["layer", "variant", "metric", "uses_oracle"]
    for key, group in confirmation.groupby(group_columns, dropna=False, sort=True):
        values = pd.to_numeric(group.value, errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        token = "|".join(map(str, key))
        seed = int.from_bytes(
            hashlib.sha256(f"{bootstrap_seed}|{token}".encode()).digest()[:4], "big"
        )
        mean, lower, upper = _bootstrap_mean(
            values, samples=bootstrap_samples, seed=seed
        )
        rows.append(_metric_row(
            record_type="uncertainty_summary",
            phase="confirmation_summary",
            layer=str(key[0]),
            variant=str(key[1]),
            metric=str(key[2]),
            value=mean,
            uses_oracle=bool(key[3]),
            oracle_unblinded_after_freeze=True,
            n_runs=int(len(values)),
            ci_lower=lower,
            ci_upper=upper,
            notes="run-level percentile bootstrap of the mean",
        ))

    ranking = confirmation.loc[confirmation.layer.eq("causal_ranking")]
    selected = ranking.loc[ranking.variant.eq(selected_variant)]
    comparator = ranking.loc[ranking.variant.eq(comparator_variant)]
    paired = selected.merge(
        comparator,
        on=["run_seed", "metric", "uses_oracle"],
        suffixes=("_selected", "_comparator"),
    )
    for (metric, uses_oracle), group in paired.groupby(
        ["metric", "uses_oracle"], sort=True
    ):
        delta = (
            pd.to_numeric(group.value_selected, errors="coerce")
            - pd.to_numeric(group.value_comparator, errors="coerce")
        ).dropna().to_numpy(float)
        if not len(delta):
            continue
        token = f"paired|{metric}|{uses_oracle}"
        seed = int.from_bytes(
            hashlib.sha256(f"{bootstrap_seed}|{token}".encode()).digest()[:4], "big"
        )
        mean, lower, upper = _bootstrap_mean(
            delta, samples=bootstrap_samples, seed=seed
        )
        rows.append(_metric_row(
            record_type="paired_uncertainty_summary",
            phase="confirmation_paired_summary",
            layer="causal_ranking",
            variant=selected_variant,
            metric=f"selected_minus_comparator:{metric}",
            value=mean,
            uses_oracle=bool(uses_oracle),
            oracle_unblinded_after_freeze=True,
            n_runs=int(len(delta)),
            ci_lower=lower,
            ci_upper=upper,
            paired_comparator=comparator_variant,
            notes="paired by untouched confirmation run seed",
        ))

    stability = metrics.loc[metrics.record_type.eq("stability_pair")]
    for key, group in stability.groupby(["layer", "metric"], sort=True):
        values = pd.to_numeric(group.value, errors="coerce").dropna().to_numpy(float)
        if not len(values):
            continue
        token = f"stability|{key[0]}|{key[1]}"
        seed = int.from_bytes(
            hashlib.sha256(f"{bootstrap_seed}|{token}".encode()).digest()[:4], "big"
        )
        mean, lower, upper = _bootstrap_mean(
            values, samples=bootstrap_samples, seed=seed
        )
        rows.append(_metric_row(
            record_type="stability_summary",
            phase="fixed_dataset_stability_summary",
            layer=str(key[0]),
            variant=selected_variant,
            metric=str(key[1]),
            value=mean,
            uses_oracle=False,
            n_runs=int(len(values)),
            ci_lower=lower,
            ci_upper=upper,
            notes="pairwise fixed-dataset stability interval",
        ))
    return pd.concat(
        [metrics, pd.DataFrame(rows, columns=METRIC_COLUMNS)], ignore_index=True
    )


def run_confirmation_and_stability_campaign(
    config: Mapping[str, Any],
    catalog,
    *,
    selected_variant: str,
    candidate_declaration_sha256: str,
    freeze_path: str | Path,
) -> ConfirmationCampaignResult:
    """Run untouched confirmations, fixed-dataset stability, then summarize."""

    experiments = config["experiments"]
    confirmation_seeds = tuple(map(int, experiments["confirmation"]["run_seeds"]))
    stability = experiments["fixed_dataset_stability"]
    stability_seeds = tuple(map(int, stability["run_seeds"]))
    dataset_seed = int(stability["dataset_seed"])
    fixed_component_names = tuple(map(
        str, stability.get("fixed_component_names", ())
    ))
    invalid_fixed_components = sorted(
        set(fixed_component_names).difference(RUN_SEED_COMPONENTS)
    )
    if invalid_fixed_components:
        raise ValueError(
            f"Invalid fixed stability components: {invalid_fixed_components}"
        )
    stability_reference_components = derive_experiment_seeds(
        stability_seeds[0], dataset_seed
    )
    freeze_path = Path(freeze_path)
    if freeze_path.exists():
        raise FileExistsError(f"Experiment freeze log already exists: {freeze_path}")
    metrics: list[dict[str, Any]] = []
    stability_states: list[dict[str, Any]] = []
    derived: dict[str, dict[str, int]] = {}

    for phase, run_seeds, fixed_seed in (
        ("confirmation", confirmation_seeds, None),
        ("fixed_dataset_stability", stability_seeds, dataset_seed),
    ):
        for run_seed in run_seeds:
            component_seeds = None
            if phase == "fixed_dataset_stability":
                component_seeds = derive_experiment_seeds(run_seed, dataset_seed)
                for component in fixed_component_names:
                    component_seeds[component] = int(
                        stability_reference_components[component]
                    )
                if len(set(component_seeds.values())) != len(component_seeds):
                    raise RuntimeError("Fixed stability component seeds collided")
            state = _learning_stack(
                config,
                catalog,
                run_seed=run_seed,
                dataset_seed=fixed_seed,
                primary_variant=selected_variant,
                full_decision_stack=True,
                component_seeds=component_seeds,
            )
            derived[f"{phase}:{run_seed}"] = state["seeds"]
            freeze = _freeze_row(
                state,
                phase=phase,
                candidate_declaration_sha256=candidate_declaration_sha256,
            )
            _append_freeze(freeze_path, freeze)
            metrics.extend(_evaluation_rows(
                state,
                config,
                phase=phase,
                selected_variant=selected_variant,
            ))
            if phase == "fixed_dataset_stability":
                stability_states.append(_stability_state(state, selected_variant))

    metrics.extend(_pairwise_stability_rows(
        stability_states,
        dataset_seed=dataset_seed,
        fractions=experiments["reporting"]["stability_fractions"],
        selected_variant=selected_variant,
    ))
    raw_metrics = pd.DataFrame(metrics, columns=METRIC_COLUMNS)
    comparator = next(
        variant for variant in experiments["candidate_variants"]
        if str(variant) != selected_variant
    )
    summarized = summarize_experiment_metrics(
        raw_metrics,
        selected_variant=selected_variant,
        comparator_variant=str(comparator),
        bootstrap_samples=int(experiments["reporting"]["bootstrap_samples"]),
        bootstrap_seed=int(experiments["reporting"]["paired_bootstrap_seed"]),
    )
    freezes = pd.read_csv(freeze_path)
    expected_freezes = len(confirmation_seeds) + len(stability_seeds)
    confirmation_present = set(
        freezes.loc[freezes.phase.eq("confirmation"), "run_seed"].astype(int)
    )
    stability_present = set(
        freezes.loc[freezes.phase.eq("fixed_dataset_stability"), "run_seed"].astype(int)
    )
    fixed_hashes = freezes.loc[
        freezes.phase.eq("fixed_dataset_stability"), "learner_dataset_sha256"
    ].nunique()
    allocation_violations = int(pd.to_numeric(
        freezes.constraint_violations, errors="coerce"
    ).sum())
    ci_rows = summarized.loc[
        summarized.record_type.isin(
            ("uncertainty_summary", "paired_uncertainty_summary", "stability_summary")
        )
    ]
    gates = {
        "candidate_declaration_referenced_by_every_run": bool(
            len(freezes) == expected_freezes
            and freezes.candidate_declaration_sha256.astype(str).eq(
                candidate_declaration_sha256
            ).all()
        ),
        "confirmation_seed_set_exact": confirmation_present == set(confirmation_seeds),
        "fixed_stability_seed_set_exact": stability_present == set(stability_seeds),
        "fixed_dataset_identical": int(fixed_hashes) == 1,
        "all_decisions_frozen_before_unblinding": bool(
            freezes.truth_unblinded_after_this_record.astype(bool).all()
            and ~freezes.oracle_used_before_freeze.astype(bool).any()
        ),
        "zero_allocation_constraint_violations": allocation_violations == 0,
        "paired_uncertainty_present": bool(
            summarized.record_type.eq("paired_uncertainty_summary").any()
        ),
        "all_four_reporting_layers_present": set(
            summarized.loc[summarized.record_type.eq("run_metric"), "layer"]
        ) == {"baseline_need", "causal_ranking", "recommendation", "allocation"},
        "confidence_intervals_finite": bool(
            len(ci_rows)
            and pd.to_numeric(ci_rows.ci_lower, errors="coerce").notna().all()
            and pd.to_numeric(ci_rows.ci_upper, errors="coerce").notna().all()
        ),
    }
    stability_acceptance = stability.get("acceptance", {})
    if stability_acceptance:
        stability_summaries = summarized.loc[
            summarized.record_type.eq("stability_summary")
        ].set_index("metric").value.astype(float)
        acceptance_metrics = {
            "fixed_score_spearman": (
                "ranking_score_spearman",
                "minimum_mean_score_spearman",
            ),
            "fixed_top_10pct_jaccard": (
                "ranking_top_10pct_jaccard",
                "minimum_mean_top_10pct_jaccard",
            ),
            "fixed_top_20pct_jaccard": (
                "ranking_top_20pct_jaccard",
                "minimum_mean_top_20pct_jaccard",
            ),
            "fixed_recommended_patient_jaccard": (
                "recommended_patient_jaccard",
                "minimum_mean_recommended_patient_jaccard",
            ),
            "fixed_allocated_patient_jaccard": (
                "allocated_patient_jaccard",
                "minimum_mean_allocated_patient_jaccard",
            ),
        }
        for gate, (metric, threshold) in acceptance_metrics.items():
            gates[gate] = bool(
                metric in stability_summaries
                and float(stability_summaries.loc[metric])
                >= float(stability_acceptance[threshold])
            )
    return ConfirmationCampaignResult(
        metrics=summarized,
        freezes=freezes,
        audit={
            "status": "passed" if all(gates.values()) else "failed",
            "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "selected_variant": selected_variant,
            "comparator_variant": str(comparator),
            "confirmation_run_seeds": list(confirmation_seeds),
            "fixed_dataset_seed": dataset_seed,
            "fixed_dataset_run_seeds": list(stability_seeds),
            "fixed_component_names": list(fixed_component_names),
            "freeze_records": int(len(freezes)),
            "allocation_constraint_violations": allocation_violations,
            "oracle_used_for_candidate_selection": False,
            "confirmation_results_seen_before_candidate_lock": False,
            "derived_component_seeds": derived,
        },
    )


__all__ = [
    "ConfirmationCampaignResult",
    "DiscoveryCampaignResult",
    "StabilizationDiagnosticResult",
    "build_candidate_declaration",
    "derive_experiment_seeds",
    "run_confirmation_and_stability_campaign",
    "run_discovery_campaign",
    "run_stabilization_source_diagnostics",
    "summarize_experiment_metrics",
]
