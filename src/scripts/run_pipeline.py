"""Run the current PROMETHEUS care-profile pipeline for one experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from causal_population_ranking.allocation import allocate_recommended_profiles
from causal_population_ranking.dm77 import (
    CARE_PROFILE,
    cross_fit_baseline_need,
    load_care_profile_catalog,
    resolve_decision_unit,
    validate_ranked_profile_comparability,
)
from causal_population_ranking.diagnostics import (
    derive_diagnostic_seeds,
    finalize_diagnostic_campaign,
    hidden_confounding_sensitivity_row,
    need_stratification_ablation_rows,
    recommendation_ablation_rows,
    run_nonoracle_ranker_diagnostics,
)
from causal_population_ranking.evaluation import (
    evaluate_baseline_need,
    evaluate_profile_allocation,
    evaluate_profile_ranking_scores,
    evaluate_profile_recommendations,
)
from causal_population_ranking.experiments import (
    build_candidate_declaration,
    run_confirmation_and_stability_campaign,
    run_discovery_campaign,
    run_stabilization_source_diagnostics,
)
from causal_population_ranking.nuisance import build_profile_causal_supervision
from causal_population_ranking.recommendation import (
    CALIBRATION_METHODS,
    calibrate_and_recommend_profiles,
)
from causal_population_ranking.ranking import train_profile_rankers
from causal_population_ranking.synthetic import (
    PROFILE_DGP_SCENARIOS,
    generate_synthetic_need_reference,
    generate_synthetic_population,
    generate_synthetic_profile_dgp,
    validate_synthetic_profile_dgp,
)


DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def _load_config(source: str | Path | Mapping) -> dict:
    if isinstance(source, Mapping):
        return dict(source)
    return yaml.safe_load(Path(source).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_config(config: Mapping) -> None:
    required = {
        "config_version", "scientific_contract", "seed_registry",
        "decision_unit", "opportunity_source", "run", "synthetic_population",
        "baseline_need", "care", "causal_supervision", "ranker", "recommendation",
        "allocation", "diagnostics", "experiments", "scenarios", "gates",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"PROMETHEUS pipeline config is missing: {missing}")
    if resolve_decision_unit(config).decision_unit != CARE_PROFILE:
        raise ValueError("The current pipeline requires decision_unit: care_profile")
    scenarios = tuple(map(str, config["scenarios"]))
    expected_order = tuple(name for name in PROFILE_DGP_SCENARIOS if name in scenarios)
    if not scenarios or scenarios != expected_order:
        raise ValueError(
            "Pipeline scenarios must be a non-empty ordered subset of the frozen catalogue"
        )
    seed_keys = (
        "population_seed", "reference_seed", "need_model_seed", "need_split_seed",
        "current_care_seed", "effect_seed", "assignment_seed", "outcome_seed",
        "split_seed", "nuisance_seed", "ranker_model_seed", "ranker_pair_seed",
        "ranker_validation_pair_seed", "ranker_contrastive_seed",
        "ranker_gbdt_seed", "ranker_random_seed", "allocator_tie_seed",
    )
    seeds = [int(config["run"][key]) for key in seed_keys]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Pipeline components require distinct explicit seeds")
    registry = config["seed_registry"]
    if not isinstance(config["scientific_contract"], Mapping):
        raise ValueError("scientific_contract must be embedded in the pipeline config")
    if not isinstance(registry, Mapping):
        raise ValueError("seed_registry must be embedded in the pipeline config")
    if not isinstance(config["baseline_need"].get("contract"), Mapping):
        raise ValueError("baseline_need.contract must be embedded in the pipeline config")
    consumed = {
        int(seed)
        for group in registry.get("consumed", {}).values()
        for seed in group.get("run_seeds", [])
    }
    if not set(seeds).issubset(consumed):
        raise ValueError("The pipeline may use only consumed implementation seeds")
    supervision = config["causal_supervision"]
    repeat_seeds = {
        int(config["run"]["nuisance_seed"])
        + repeat * int(supervision["repeat_seed_stride"])
        for repeat in range(int(supervision["repeats"]))
    }
    if not repeat_seeds.issubset(consumed):
        raise ValueError("Every repeated nuisance seed must be consumed in the registry")
    ranker_seeds = {
        int(config["run"][key])
        for key in (
            "ranker_model_seed", "ranker_pair_seed", "ranker_validation_pair_seed",
            "ranker_contrastive_seed", "ranker_gbdt_seed", "ranker_random_seed",
        )
    }
    if len(ranker_seeds) != 6 or not ranker_seeds.issubset(consumed):
        raise ValueError("Every ranker model and sampler seed must be distinct and consumed")
    if config["ranker"].get("allow_cross_profile_training_pairs") is not True:
        raise ValueError("The primary pipeline requires comparable cross-profile pairs")
    if int(config["ranker"].get("model_restarts", 1)) < 1:
        raise ValueError("Ranker model_restarts must be positive")
    if config["ranker"].get("restart_aggregation") != "mean_percentile_rank":
        raise ValueError("The stabilized ranker requires mean percentile-rank aggregation")
    if config["gates"]["truth_tables_written_after_learner_boundary"] is not True:
        raise ValueError("The pipeline requires the physical learner/truth boundary")
    if config["gates"]["oracle_used_for_model_selection"] is not False:
        raise ValueError("The pipeline cannot use oracle data for selection")
    scientific = config["scientific_contract"]
    calibration_contract = scientific["calibration_contract"]
    recommendation_contract = scientific["recommendation_contract"]
    if tuple(calibration_contract["candidate_methods"]) != CALIBRATION_METHODS:
        raise ValueError("The pipeline requires every frozen Phase-6 calibrator")
    if calibration_contract["fit_partition"] != "validation":
        raise ValueError("Calibration must fit on validation only")
    if calibration_contract["oracle_target_allowed"] is not False:
        raise ValueError("Calibration cannot use an oracle target")
    if recommendation_contract["minimum_benefit_operator"] != "strictly_greater_than":
        raise ValueError("Recommendation requires the frozen strict benefit threshold")
    support = recommendation_contract["support_requirements"]
    if list(map(float, config["causal_supervision"]["overlap_support_bounds"])) != [
        float(support["propensity_lower"]), float(support["propensity_upper"])
    ]:
        raise ValueError("Recommendation and causal-supervision overlap bounds must match")
    if float(config["causal_supervision"]["minimum_profile_effective_sample_size"]) != float(
        support["minimum_profile_effective_sample_size"]
    ):
        raise ValueError("Recommendation and causal-supervision profile ESS gates must match")
    if support["require_profile_validation_lower_bound_above_null"] is not True:
        raise ValueError("Recommendation requires positive profile validation evidence")
    if support["profile_validation_evidence_is_individual_interval"] is not False:
        raise ValueError("Profile evidence cannot be represented as an individual interval")
    implementation = config["recommendation"]
    if int(implementation["minimum_validation_rows"]) < 2:
        raise ValueError("Recommendation calibration requires at least two validation rows")
    if int(implementation["monotonic_bins"]) < 2:
        raise ValueError("Monotonic calibration requires at least two bins")
    if float(implementation["profile_mean_shrinkage_strength"]) <= 0.0:
        raise ValueError("Profile-mean shrinkage strength must be positive")
    if not 0.0 < float(implementation["uncertainty_residual_quantile"]) < 1.0:
        raise ValueError("Recommendation residual quantile must be in (0, 1)")
    if int(implementation["minimum_profile_validation_rows"]) < 2:
        raise ValueError("Profile validation evidence requires at least two rows")
    if float(implementation["profile_evidence_one_sided_z"]) <= 0.0:
        raise ValueError("Profile validation evidence z value must be positive")
    if float(implementation["profile_evidence_null_days"]) != 0.0:
        raise ValueError("The frozen profile evidence null is zero days")
    allocation_contract = scientific["allocation_contract"]
    if allocation_contract["stage"] != "after_recommendation_freeze":
        raise ValueError("Allocation must occur after recommendation freeze")
    if allocation_contract["candidate_profiles"] != "recommended_profile_only":
        raise ValueError("Allocation may consider only recommended profiles")
    if allocation_contract["alternative_profile_substitution_allowed"] is not False:
        raise ValueError("Allocation cannot substitute alternative profiles")
    if allocation_contract["recommendation_may_be_overwritten"] is not False:
        raise ValueError("Allocation cannot overwrite recommendations")
    allocation = config["allocation"]
    if float(allocation["shared_budget_units_per_patient"]) <= 0.0:
        raise ValueError("Allocation shared budget per patient must be positive")
    if not 0.0 < float(allocation["tie_break_epsilon"]) < 1e-3:
        raise ValueError("Allocation tie-break epsilon must be positive and negligible")
    budget_multipliers = tuple(map(float, allocation["budget_curve_multipliers"]))
    if (
        not budget_multipliers
        or tuple(sorted(set(budget_multipliers))) != budget_multipliers
        or 1.0 not in budget_multipliers
        or any(value <= 0.0 for value in budget_multipliers)
    ):
        raise ValueError("Budget-curve multipliers must be unique, ordered and include 1")
    fixed_fractions = tuple(map(float, allocation["fixed_capacity_fractions"]))
    if (
        not fixed_fractions
        or tuple(sorted(set(fixed_fractions))) != fixed_fractions
        or any(not 0.0 < value < 1.0 for value in fixed_fractions)
    ):
        raise ValueError("Fixed-capacity fractions must be unique ordered values in (0, 1)")
    solver = allocation["solver"]
    if float(solver["mip_relative_gap"]) < 0.0 or float(
        solver["time_limit_seconds"]
    ) <= 0.0:
        raise ValueError("Allocation solver gap/time settings are invalid")
    diagnostics = config["diagnostics"]
    diagnostic_contract = scientific["diagnostic_contract"]
    if str(diagnostics["evaluation_partition"]) != "test":
        raise ValueError("Phase-8 diagnostics require the untouched test partition")
    if diagnostics["evaluation_partition_used_for_fitting_or_early_stopping"] is not False:
        raise ValueError("Diagnostic test rows cannot enter fitting or early stopping")
    if diagnostics["eligible_for_model_selection"] is not False:
        raise ValueError("Phase-8 diagnostics cannot select the primary model")
    diagnostic_seeds = tuple(map(int, diagnostics["base_seeds"]))
    consumed_diagnostics = tuple(map(
        int,
        registry["consumed"]["negative_control_diagnostics"]["run_seeds"],
    ))
    if (
        len(diagnostic_seeds) < 3
        or len(set(diagnostic_seeds)) != len(diagnostic_seeds)
        or diagnostic_seeds != consumed_diagnostics
    ):
        raise ValueError("Diagnostic seeds must be the frozen consumed campaign seeds")
    required_conditions = {
        "baseline_identifiable": (
            "standard",
            "within_profile_training_only",
            "permuted_training_pair_labels",
        ),
        "no_shared_response": ("standard", "within_profile_training_only"),
        "sharp_null": ("standard",),
        "placebo_outcome": ("standard",),
    }
    configured_conditions = {
        str(name): tuple(map(str, values))
        for name, values in diagnostics["scenario_conditions"].items()
    }
    if configured_conditions != required_conditions:
        raise ValueError("The Phase-8 scenario/condition campaign is frozen")
    if tuple(map(float, diagnostics["threshold_sensitivity_days"])) != (0.0, 2.0, 5.0):
        raise ValueError("Threshold ablations must remain 0, 2 and 5 days")
    need_ablation = diagnostics["supervised_need_ablation"]
    if (
        int(need_ablation["base_seed"]) not in diagnostic_seeds
        or int(need_ablation["population_size"]) < 5_000
        or str(need_ablation["reference"])
        != "synthetic_noisy_need_panel_v1_nonoracle"
    ):
        raise ValueError("The rules/supervised need ablation contract is invalid")
    controls = diagnostics["controls"]
    if not 0.0 < float(
        controls["null_max_median_concordance_distance_from_chance"]
    ) < 0.5:
        raise ValueError("Null concordance tolerance must be in (0, 0.5)")
    if not 0.0 <= float(controls["null_max_recommendation_rate"]) <= 0.25:
        raise ValueError("Null recommendation-rate tolerance must be in [0, 0.25]")
    if not 0.5 <= float(controls["permuted_max_median_concordance"]) < 0.75:
        raise ValueError("Permuted-label concordance maximum must be in [0.5, 0.75)")
    if diagnostic_contract["heldout_diagnostic_partition"] != "test":
        raise ValueError("Scientific diagnostic contract requires held-out test rows")
    if diagnostic_contract["oracle_targets_allowed"] is not False:
        raise ValueError("Oracle targets cannot enter Phase-8 diagnostics")
    if diagnostic_contract["diagnostic_results_select_primary_model"] is not False:
        raise ValueError("Diagnostic results cannot select the primary model")
    if config["gates"]["require_phase8_diagnostic_exit_gates"] is not True:
        raise ValueError("The unified pipeline must enforce Phase-8 exit gates")
    experiments = config["experiments"]
    experiment_contract = scientific["experiment_contract"]
    candidate_variants = tuple(map(str, experiments["candidate_variants"]))
    if candidate_variants != (
        "global_rank_only", "global_rank_plus_contrastive"
    ):
        raise ValueError("Phase-9 candidate variants are frozen")
    discovery = experiments["discovery"]
    confirmation = experiments["confirmation"]
    stability = experiments["fixed_dataset_stability"]
    discovery_seeds = tuple(map(int, discovery["run_seeds"]))
    confirmation_seeds = tuple(map(int, confirmation["run_seeds"]))
    stability_seeds = tuple(map(int, stability["run_seeds"]))
    expected_seed_groups = {
        "discovery": discovery_seeds,
        "confirmation": confirmation_seeds,
        "fixed_dataset_stability": stability_seeds,
    }
    for name, expected in expected_seed_groups.items():
        section = experiments[
            "fixed_dataset_stability" if name == "fixed_dataset_stability" else name
        ]
        registry_group = str(section["registry_group"])
        registered = tuple(map(
            int, registry["consumed"][registry_group]["run_seeds"]
        ))
        if expected != registered or len(set(expected)) != len(expected):
            raise ValueError(f"Phase-9 {name} seeds do not match the registry")
    if set(discovery_seeds) & set(confirmation_seeds) or (
        set(discovery_seeds) | set(confirmation_seeds)
    ) & set(stability_seeds):
        raise ValueError("Discovery, confirmation and stability seeds must be disjoint")
    if len(discovery_seeds) != 5 or len(confirmation_seeds) != 10 or len(stability_seeds) != 5:
        raise ValueError("Phase-9 run counts are frozen at 5/10/5")
    registered_dataset_seeds = tuple(map(
        int,
        registry["consumed"][str(stability["registry_group"])]["dataset_seeds"],
    ))
    if (int(stability["dataset_seed"]),) != registered_dataset_seeds:
        raise ValueError("Fixed-dataset stability seed does not match the registry")
    reporting = experiments["reporting"]
    if int(reporting["paired_bootstrap_seed"]) not in set(map(
        int,
        registry["consumed"][str(reporting["registry_group"])]["run_seeds"],
    )):
        raise ValueError("Phase-9 reporting seed must be consumed in the registry")
    if int(reporting["bootstrap_samples"]) < 500:
        raise ValueError("Phase-9 paired uncertainty requires at least 500 bootstraps")
    if float(reporting["oracle_solver_time_limit_seconds"]) < float(
        config["allocation"]["solver"]["time_limit_seconds"]
    ):
        raise ValueError(
            "Oracle evaluation solver time cannot be shorter than operational allocation"
        )
    stabilization = experiments["stability_diagnostics"]
    if int(experiments["population_size"]) < 5000:
        raise ValueError("Phase-9B stabilization requires at least 5000 patients")
    stabilization_registry = registry["consumed"][str(
        stabilization["registry_group"]
    )]
    stabilization_seeds = (
        int(stabilization["reference_run_seed"]),
        *map(int, stabilization["perturbation_run_seeds"]),
    )
    if tuple(map(int, stabilization_registry["run_seeds"])) != stabilization_seeds:
        raise ValueError("Phase-9B diagnostic seeds do not match the registry")
    if tuple(map(int, stabilization_registry["dataset_seeds"])) != (
        int(stabilization["dataset_seed"]),
    ):
        raise ValueError("Phase-9B diagnostic dataset seed does not match the registry")
    all_experiment_seeds = (
        set(stabilization_seeds)
        | set(discovery_seeds)
        | set(confirmation_seeds)
        | set(stability_seeds)
        | {int(reporting["paired_bootstrap_seed"])}
    )
    completed_phase9_seeds = {
        int(seed)
        for group in (
            "discovery", "confirmation", "fixed_dataset_stability", "phase9_reporting"
        )
        for seed in registry["consumed"][group].get("run_seeds", [])
    }
    if all_experiment_seeds & completed_phase9_seeds:
        raise ValueError("Phase-9B cannot reuse completed Phase-9 seeds")
    stabilized_settings = stabilization["stabilized_ranker_settings"]
    for key, expected in stabilized_settings.items():
        if config["ranker"][key] != expected:
            raise ValueError(
                f"Active ranker setting {key!r} differs from the frozen stabilization"
            )
    for key, expected in stabilization[
        "stabilized_causal_supervision_settings"
    ].items():
        if config["causal_supervision"][key] != expected:
            raise ValueError(
                f"Active causal-supervision setting {key!r} differs from stabilization"
            )
    if (
        stabilization["oracle_allowed"] is not False
        or stabilization["outcome_allowed_in_stability_metric"] is not False
        or stabilization["observed_outcome_allowed_for_causal_supervision"] is not True
    ):
        raise ValueError(
            "Stability diagnostics may use observed outcomes only inside causal supervision"
        )
    if stabilization.get("baseline_comparison_role") != (
        "descriptive_only_not_an_acceptance_gate"
    ):
        raise ValueError("The short-training baseline must remain descriptive only")
    if any("improvement" in str(key) for key in stabilization["acceptance"]):
        raise ValueError("Phase-9B acceptance must use absolute stability gates")
    if tuple(map(str, stability["fixed_component_names"])) != (
        "ranker_model_seed",
    ):
        raise ValueError("Phase-9B fixed-dataset stability must freeze only model seeds")
    if (
        str(discovery["selection_partition"]) != "validation"
        or discovery["oracle_allowed"] is not False
        or discovery["test_partition_allowed"] is not False
    ):
        raise ValueError("Discovery must use validation records without oracle/test data")
    expected_candidate = discovery.get("expected_locked_candidate")
    if expected_candidate is not None and str(expected_candidate) not in candidate_variants:
        raise ValueError("Expected locked candidate is not a prespecified variant")
    if experiment_contract["discovery_oracle_allowed"] is not False:
        raise ValueError("The Phase-9 scientific contract forbids discovery oracle use")
    if experiment_contract["candidate_may_change_after_confirmation_unblinding"] is not False:
        raise ValueError("Confirmation cannot change the locked candidate")
    if config["gates"]["require_phase9_protocol_exit_gates"] is not True:
        raise ValueError("The unified pipeline must enforce Phase-9 protocol gates")


def _direction_checks(
    results: dict[str, object],
) -> pd.DataFrame:
    diagnostics = {
        name: result.metadata["scenario_diagnostics_evaluation_only"]
        for name, result in results.items()
    }
    checks: list[tuple[str, bool]] = []

    def check(name: str, required: tuple[str, ...], predicate) -> None:
        if set(required) <= set(results):
            checks.append((name, bool(predicate())))

    check(
        "need_current_alignment_increases_spearman",
        ("baseline_identifiable", "need_current_alignment"),
        lambda: diagnostics["need_current_alignment"]["need_current_spearman"]
        > diagnostics["baseline_identifiable"]["need_current_spearman"],
    )
    check(
        "unmet_need_shifts_current_level_down",
        ("baseline_identifiable", "unmet_need"),
        lambda: diagnostics["unmet_need"][
            "mean_current_minus_true_need_clipped_to_five"
        ] < diagnostics["baseline_identifiable"][
            "mean_current_minus_true_need_clipped_to_five"
        ] - 0.50,
    )
    check(
        "over_intensive_shifts_current_level_up",
        ("baseline_identifiable", "over_intensive_care"),
        lambda: diagnostics["over_intensive_care"][
            "mean_current_minus_true_need_clipped_to_five"
        ] > diagnostics["baseline_identifiable"][
            "mean_current_minus_true_need_clipped_to_five"
        ] + 0.50,
    )
    check(
        "risk_benefit_alignment_positive",
        ("risk_benefit_aligned",),
        lambda: diagnostics["risk_benefit_aligned"]["risk_benefit_spearman"] > 0.45,
    )
    check(
        "risk_benefit_misalignment_negative",
        ("risk_benefit_misaligned",),
        lambda: diagnostics["risk_benefit_misaligned"]["risk_benefit_spearman"] < -0.25,
    )
    check(
        "no_shared_response_zero",
        ("no_shared_response",),
        lambda: np.allclose(
            results["no_shared_response"].profile_ground_truth[
                "latent_shared_causal_response"
            ],
            0.0,
        ),
    )
    check(
        "sharp_null_zero",
        ("sharp_null",),
        lambda: np.allclose(
            results["sharp_null"].profile_ground_truth["true_profile_benefit"], 0.0
        ),
    )
    check(
        "placebo_analysis_null_primary_nonzero",
        ("placebo_outcome",),
        lambda: np.allclose(
            results["placebo_outcome"].profile_ground_truth["true_profile_benefit"],
            0.0,
        ) and float(np.abs(
            results["placebo_outcome"].profile_ground_truth[
                "true_primary_profile_benefit"
            ]
        ).mean()) > 0.1,
    )
    check(
        "poor_overlap_more_extreme",
        ("baseline_identifiable", "poor_overlap"),
        lambda: diagnostics["poor_overlap"]["maximum_profile_assignment_probability"]
        > diagnostics["baseline_identifiable"][
            "maximum_profile_assignment_probability"
        ],
    )
    check(
        "hidden_confounding_declared_failure",
        ("hidden_confounding",),
        lambda: results["hidden_confounding"].metadata["identification"][
            "conditional_exchangeability_given_learner_covariates"
        ] is False,
    )
    check(
        "capacity_scarcity_reduces_capacity",
        ("baseline_identifiable", "capacity_scarcity"),
        lambda: results["capacity_scarcity"].resource_capacities[
            "capacity_fraction_of_population"
        ].max() < results["baseline_identifiable"].resource_capacities[
            "capacity_fraction_of_population"
        ].min(),
    )
    check(
        "heterogeneous_profile_costs_nonconstant",
        ("heterogeneous_profile_costs",),
        lambda: results["heterogeneous_profile_costs"].profile_resources[
            "scenario_cost_multiplier"
        ].nunique() > 3,
    )
    return pd.DataFrame([
        {"check": name, "status": "pass" if passed else "fail", "passed": bool(passed)}
        for name, passed in checks
    ], columns=("check", "status", "passed"))


def run_pipeline(
    config_source: str | Path | Mapping = DEFAULT_CONFIG,
) -> Path:
    config = _load_config(config_source)
    _validate_config(config)
    run = config["run"]
    synthetic = config["synthetic_population"]
    run_id = str(run.get("run_id", "auto"))
    if run_id == "auto":
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(run["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    catalog = load_care_profile_catalog(config["care"]["profile_catalog"])
    profile_comparability = validate_ranked_profile_comparability(catalog)
    population = generate_synthetic_population(
        population_size=int(run["population_size"]),
        seed=int(run["population_seed"]),
        history_months=int(run["history_months"]),
        reference_date=str(synthetic["reference_date"]),
        profile=str(synthetic["profile"]),
        scenario=str(synthetic["scenario"]),
    )
    dgp_seed_keys = (
        "reference_seed", "need_model_seed", "need_split_seed",
        "current_care_seed", "effect_seed", "assignment_seed", "outcome_seed",
    )
    results = {}
    summary_rows = []
    supervision_summary_rows = []
    ranker_summary_rows = []
    recommendation_summary_rows = []
    allocation_summary_rows = []
    diagnostic_frames = []
    diagnostic_audits = {}
    for scenario in config["scenarios"]:
        scenario = str(scenario)
        scenario_dir = output / scenario
        scenario_dir.mkdir()
        result = generate_synthetic_profile_dgp(
            population.patients,
            catalog,
            scenario=scenario,
            need_mode=str(config["baseline_need"]["mode"]),
            baseline_need_contract=config["baseline_need"]["contract"],
            index_date=str(synthetic["reference_date"]),
            **{key: int(run[key]) for key in dgp_seed_keys},
        )
        results[scenario] = result
        safe_artifacts = {
            "learner_profile_dataset.csv": result.learner,
            "baseline_need_assessment.csv": result.baseline_need_assessments,
            "synthetic_need_reference_labels.csv": result.need_reference_labels,
            "patient_current_care_profile.csv": result.current_care_profiles,
            "care_profile_eligibility.csv": result.profile_eligibility,
            "patient_profile_opportunities.csv": result.patient_profile_opportunities,
            "profile_resources.csv": result.profile_resources,
            "resource_capacities.csv": result.resource_capacities,
        }
        safe_paths = {}
        for filename, frame in safe_artifacts.items():
            path = scenario_dir / filename
            frame.to_csv(path, index=False)
            safe_paths[filename] = path
        boundary = {
            "stage": "learner_artifacts_frozen_before_truth_materialization",
            "scenario": scenario,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "safe_artifact_sha256": {
                name: _checksum(path) for name, path in safe_paths.items()
            },
            "oracle_columns_in_learner": result.audit["learner_oracle_columns"],
            "oracle_columns_in_eligibility": result.audit[
                "eligibility_oracle_columns"
            ],
            "oracle_used_for_model_selection": False,
        }
        _write_json(scenario_dir / "learner_truth_boundary.json", boundary)

        causal_supervision = build_profile_causal_supervision(
            result.learner,
            result.patient_profile_opportunities,
            config["causal_supervision"],
            split_seed=int(run["split_seed"]),
            nuisance_seed=int(run["nuisance_seed"]),
            profile_candidates=[
                (profile.care_profile_id, profile.care_profile_index)
                for profile in catalog.automatic_rank_profiles
            ],
        )
        supervision_artifacts = {
            "patient_splits.csv": causal_supervision.patient_splits,
            "profile_causal_supervision.csv": causal_supervision.supervision,
            "profile_nuisance_predictions.csv": causal_supervision.nuisance_predictions,
            "profile_causal_support.csv": causal_supervision.support_diagnostics,
            "supported_profile_opportunities.csv": (
                causal_supervision.supported_opportunities
            ),
        }
        supervision_paths = {}
        for filename, frame in supervision_artifacts.items():
            path = scenario_dir / filename
            frame.to_csv(path, index=False)
            supervision_paths[filename] = path
        audit_path = scenario_dir / "profile_causal_supervision_audit.json"
        _write_json(audit_path, causal_supervision.audit)
        supervision_paths[audit_path.name] = audit_path
        supervision_freeze = {
            "stage": "causal_supervision_frozen_before_evaluation_truth_materialization",
            "scenario": scenario,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": {
                name: _checksum(path) for name, path in supervision_paths.items()
            },
            "oracle_columns_detected": causal_supervision.audit[
                "oracle_columns_detected"
            ],
            "oracle_used": False,
        }
        _write_json(scenario_dir / "causal_supervision_freeze.json", supervision_freeze)
        supervision_summary_rows.append({
            "scenario": scenario,
            "profiles_considered": causal_supervision.audit["profiles_considered"],
            "profiles_fitted": causal_supervision.audit["profiles_fitted"],
            "profiles_supported": causal_supervision.audit["profiles_supported"],
            "profiles_unsupported": len(
                causal_supervision.audit["unsupported_profiles"]
            ),
            "supervision_rows": len(causal_supervision.supervision),
            "oracle_used": causal_supervision.audit["oracle_used"],
        })

        ranker = train_profile_rankers(
            result.learner,
            causal_supervision.supervision,
            causal_supervision.supported_opportunities,
            causal_supervision.patient_splits,
            config["ranker"],
            numeric_features=config["causal_supervision"]["numeric_features"],
            categorical_features=config["causal_supervision"]["categorical_features"],
            profile_ids=[
                profile.care_profile_id for profile in catalog.automatic_rank_profiles
            ],
            model_seed=int(run["ranker_model_seed"]),
            pair_seed=int(run["ranker_pair_seed"]),
            validation_pair_seed=int(run["ranker_validation_pair_seed"]),
            contrastive_seed=int(run["ranker_contrastive_seed"]),
            gbdt_seed=int(run["ranker_gbdt_seed"]),
            random_seed=int(run["ranker_random_seed"]),
        )
        ranker_artifacts = {
            "profile_priority_scores.csv": ranker.scores,
            "profile_ranker_training_history.csv": ranker.training_history,
            "profile_ranker_validation_pairs.csv": ranker.validation_pairs,
        }
        ranker_paths = {}
        for filename, frame in ranker_artifacts.items():
            path = scenario_dir / filename
            frame.to_csv(path, index=False)
            ranker_paths[filename] = path
        ranker_audit_path = scenario_dir / "profile_ranker_audit.json"
        _write_json(ranker_audit_path, ranker.audit)
        ranker_paths[ranker_audit_path.name] = ranker_audit_path
        if ranker.primary_model_bundle is not None:
            model_path = scenario_dir / "primary_profile_ranker.pt"
            torch.save(ranker.primary_model_bundle, model_path)
            ranker_paths[model_path.name] = model_path
        ranker_freeze = {
            "stage": "profile_ranker_frozen_before_evaluation_truth_materialization",
            "scenario": scenario,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": {
                name: _checksum(path) for name, path in ranker_paths.items()
            },
            "primary_variant": config["ranker"]["primary_variant"],
            "score_semantics": "globally_comparable_ordinal_priority_no_causal_zero",
            "individual_cate_estimated_then_sorted": False,
            "oracle_used": False,
        }
        _write_json(scenario_dir / "profile_ranker_freeze.json", ranker_freeze)
        ranker_summary_rows.append({
            "scenario": scenario,
            "status": ranker.audit["status"],
            "primary_variant": config["ranker"]["primary_variant"],
            "supported_opportunities_scored": ranker.audit.get(
                "scored_supported_opportunities", 0
            ),
            "fixed_validation_pairs": ranker.audit.get("fixed_validation_pairs", 0),
            "validation_within_profile_pairs": ranker.audit.get(
                "validation_within_profile_pairs", 0
            ),
            "validation_cross_profile_pairs": ranker.audit.get(
                "validation_cross_profile_pairs", 0
            ),
            "primary_model_saved": ranker.primary_model_bundle is not None,
            "oracle_used": ranker.audit["oracle_used_for_training"]
            if ranker.audit["status"] == "trained" else ranker.audit["oracle_used"],
        })

        recommendation = calibrate_and_recommend_profiles(
            ranker.scores,
            causal_supervision.supervision,
            causal_supervision.supported_opportunities,
            result.baseline_need_assessments,
            result.current_care_profiles,
            causal_supervision.patient_splits,
            causal_supervision.support_diagnostics,
            primary_variant=str(config["ranker"]["primary_variant"]),
            calibration_contract=config["scientific_contract"][
                "calibration_contract"
            ],
            recommendation_contract=config["scientific_contract"][
                "recommendation_contract"
            ],
            implementation_settings=config["recommendation"],
        )
        recommendation_artifacts = {
            "profile_calibration_diagnostics.csv": (
                recommendation.calibration_diagnostics
            ),
            "actionable_profile_recommendations.csv": recommendation.recommendations,
        }
        recommendation_paths = {}
        for filename, frame in recommendation_artifacts.items():
            path = scenario_dir / filename
            frame.to_csv(path, index=False)
            recommendation_paths[filename] = path
        decision_contract = {
            **recommendation.decision_contract,
            "scenario": scenario,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": (
                "recommendation_frozen_before_allocation_and_evaluation_truth_materialization"
            ),
            "artifact_sha256": {
                name: _checksum(path) for name, path in recommendation_paths.items()
            },
        }
        decision_contract_path = scenario_dir / "profile_decision_contract.json"
        _write_json(decision_contract_path, decision_contract)
        recommendation_summary_rows.append({
            "scenario": scenario,
            "status": recommendation.audit["status"],
            "selected_calibration_method": recommendation.audit[
                "selected_calibration_method"
            ],
            "calibration_validation_rows": recommendation.audit[
                "calibration_validation_rows"
            ],
            "patients": recommendation.audit["patients"],
            "patients_recommended": recommendation.audit["patients_recommended"],
            "patients_abstained": recommendation.audit["patients_abstained"],
            "raw_to_calibrated_ordering_violations": recommendation.audit[
                "raw_to_calibrated_ordering_violations"
            ],
            "deintensification_recommendations": recommendation.audit[
                "deintensification_recommendations"
            ],
            "capacity_inputs_used": recommendation.audit["capacity_inputs_used"],
            "oracle_used": recommendation.audit["oracle_used"],
        })

        allocation = allocate_recommended_profiles(
            recommendation.recommendations,
            causal_supervision.supported_opportunities,
            result.resource_capacities,
            result.profile_resources,
            allocation_contract=config["scientific_contract"]["allocation_contract"],
            settings=config["allocation"],
            tie_seed=int(run["allocator_tie_seed"]),
        )
        allocation_artifacts = {
            "profile_allocation_decisions.csv": allocation.decisions,
            "profile_allocation_diagnostics.csv": allocation.diagnostics,
        }
        allocation_paths = {}
        for filename, frame in allocation_artifacts.items():
            path = scenario_dir / filename
            frame.to_csv(path, index=False)
            allocation_paths[filename] = path
        allocation_contract = {
            **allocation.allocation_contract,
            "scenario": scenario,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "allocation_frozen_before_evaluation_truth_materialization",
            "frozen_recommendation_contract_sha256": _checksum(
                decision_contract_path
            ),
            "artifact_sha256": {
                name: _checksum(path) for name, path in allocation_paths.items()
            },
        }
        _write_json(
            scenario_dir / "profile_allocation_contract.json", allocation_contract
        )
        allocation_summary_rows.append({
            "scenario": scenario,
            "status": allocation.audit["status"],
            "recommended_candidates": allocation.audit["recommended_candidates"],
            "allocated_recommendations": allocation.audit[
                "allocated_recommendations"
            ],
            "deferred_recommendations": allocation.audit[
                "deferred_recommendations"
            ],
            "conditional_deferral_rate": (
                allocation.audit["deferred_recommendations"]
                / max(allocation.audit["recommended_candidates"], 1)
            ),
            "shared_budget_limit": allocation.audit["shared_budget_limit"],
            "shared_budget_used": allocation.audit["shared_budget_used"],
            "calibrated_objective_value": allocation.audit[
                "calibrated_objective_value"
            ],
            "constraint_violations": allocation.audit[
                "constraint_violations_total"
            ],
            "recommendation_records_modified": allocation.audit[
                "recommendation_records_modified"
            ],
            "alternative_profile_substitution": allocation.audit[
                "alternative_profile_substitution"
            ],
            "oracle_used": allocation.audit["oracle_used"],
        })

        diagnostic_conditions = config["diagnostics"][
            "scenario_conditions"
        ].get(scenario)
        if diagnostic_conditions:
            diagnostic = run_nonoracle_ranker_diagnostics(
                result.learner,
                causal_supervision.supervision,
                causal_supervision.supported_opportunities,
                causal_supervision.patient_splits,
                config["ranker"],
                scenario=scenario,
                conditions=diagnostic_conditions,
                base_seeds=config["diagnostics"]["base_seeds"],
                campaign_settings=config["diagnostics"],
                numeric_features=config["causal_supervision"]["numeric_features"],
                categorical_features=config["causal_supervision"][
                    "categorical_features"
                ],
                profile_ids=[
                    profile.care_profile_id
                    for profile in catalog.automatic_rank_profiles
                ],
            )
            diagnostic_frames.append(diagnostic.rows)
            diagnostic_audits[scenario] = diagnostic.audit

        if scenario == "baseline_identifiable":
            threshold_results = {2.0: recommendation}
            for threshold in config["diagnostics"]["threshold_sensitivity_days"]:
                threshold = float(threshold)
                if threshold == 2.0:
                    continue
                threshold_contract = copy.deepcopy(
                    config["scientific_contract"]["recommendation_contract"]
                )
                threshold_contract["primary_minimum_benefit_threshold_days"] = threshold
                threshold_results[threshold] = calibrate_and_recommend_profiles(
                    ranker.scores,
                    causal_supervision.supervision,
                    causal_supervision.supported_opportunities,
                    result.baseline_need_assessments,
                    result.current_care_profiles,
                    causal_supervision.patient_splits,
                    causal_supervision.support_diagnostics,
                    primary_variant=str(config["ranker"]["primary_variant"]),
                    calibration_contract=config["scientific_contract"][
                        "calibration_contract"
                    ],
                    recommendation_contract=threshold_contract,
                    implementation_settings=config["recommendation"],
                )
            diagnostic_frames.append(
                recommendation_ablation_rows(
                    scenario=scenario,
                    calibration_diagnostics=recommendation.calibration_diagnostics,
                    threshold_results=threshold_results,
                )
            )

            supervised_settings = config["diagnostics"][
                "supervised_need_ablation"
            ]
            supervised_base_seed = int(supervised_settings["base_seed"])
            supervised_seeds = derive_diagnostic_seeds(supervised_base_seed)
            need_population = generate_synthetic_population(
                population_size=int(supervised_settings["population_size"]),
                seed=int(supervised_seeds["need_population_seed"]),
                history_months=int(run["history_months"]),
                reference_date=str(synthetic["reference_date"]),
                profile=str(synthetic["profile"]),
                scenario=str(synthetic["scenario"]),
            )
            need_reference = generate_synthetic_need_reference(
                need_population.patients,
                seed=int(supervised_seeds["need_reference_seed"]),
            )
            need_frame = need_population.patients.merge(
                need_reference.learner_labels,
                on="patient_id",
                how="inner",
                validate="one_to_one",
            )
            rules_need = cross_fit_baseline_need(
                need_population.patients,
                mode="rules",
                model_seed=int(supervised_seeds["need_model_seed"]),
                split_seed=int(supervised_seeds["need_split_seed"]),
                contract=config["baseline_need"]["contract"],
            )
            supervised_need = cross_fit_baseline_need(
                need_frame,
                mode="supervised",
                model_seed=int(supervised_seeds["need_model_seed"]),
                split_seed=int(supervised_seeds["need_split_seed"]),
                contract=config["baseline_need"]["contract"],
            )
            diagnostic_frames.append(
                need_stratification_ablation_rows(
                    rules_assessments=rules_need.assessments,
                    supervised_assessments=supervised_need.assessments,
                    reference_labels=need_reference.learner_labels,
                )
            )
            diagnostic_audits["rules_vs_supervised_need"] = {
                "population_size": int(len(need_population.patients)),
                "reference": str(supervised_settings["reference"]),
                "need_population_seed": int(
                    supervised_seeds["need_population_seed"]
                ),
                "need_reference_seed": int(
                    supervised_seeds["need_reference_seed"]
                ),
                "need_model_seed": int(supervised_seeds["need_model_seed"]),
                "need_split_seed": int(supervised_seeds["need_split_seed"]),
                "supervised_predictions_out_of_fold": True,
                "synthetic_true_need_used": False,
                "oracle_used": False,
            }

        if scenario == "hidden_confounding":
            diagnostic_frames.append(
                hidden_confounding_sensitivity_row(
                    scenario=scenario,
                    conditional_exchangeability=result.metadata["identification"][
                        "conditional_exchangeability_given_learner_covariates"
                    ],
                )
            )

        evaluation_dir = scenario_dir / "evaluation_only"
        evaluation_dir.mkdir()
        result.need_ground_truth.to_csv(
            evaluation_dir / "baseline_need_ground_truth.csv", index=False
        )
        result.profile_ground_truth.to_csv(
            evaluation_dir / "profile_causal_ground_truth.csv", index=False
        )
        need_evaluation = evaluate_baseline_need(
            result.need_ground_truth,
            result.baseline_need_assessments,
            subgroup_frame=result.learner,
            subgroup_columns=tuple(
                map(str, config["experiments"]["reporting"]["subgroup_columns"])
            ),
            minimum_subgroup_size=30,
        )
        need_metric_rows = [
            {
                "record_type": "summary",
                "metric": str(metric),
                "value": float(value),
                "subgroup": "all",
                "uses_oracle": True,
            }
            for metric, value in need_evaluation.summary.items()
            if not isinstance(value, (bool, str))
        ]
        for matrix_row in need_evaluation.confusion_matrix.itertuples(index=False):
            for predicted_level in range(1, 7):
                need_metric_rows.append({
                    "record_type": "confusion_matrix",
                    "metric": (
                        f"reference_{int(matrix_row.reference_level)}_"
                        f"predicted_{predicted_level}"
                    ),
                    "value": float(getattr(
                        matrix_row, f"predicted_level_{predicted_level}"
                    )),
                    "subgroup": "all",
                    "uses_oracle": True,
                })
        for subgroup in need_evaluation.subgroup_metrics.itertuples(index=False):
            if str(subgroup.status) != "reported":
                continue
            for metric in (
                "quadratic_weighted_kappa", "macro_f1", "balanced_accuracy",
                "ordinal_mae", "within_one_level_accuracy", "spearman",
            ):
                need_metric_rows.append({
                    "record_type": "subgroup",
                    "metric": metric,
                    "value": float(getattr(subgroup, metric)),
                    "subgroup": (
                        f"{subgroup.subgroup_column}={subgroup.subgroup_value}"
                    ),
                    "uses_oracle": True,
                })
        pd.DataFrame(need_metric_rows).to_csv(
            evaluation_dir / "baseline_need_metrics.csv", index=False
        )
        if ranker.scores.empty:
            ranking_metrics = pd.DataFrame([{
                "method": "none",
                "score_role": "no_supported_opportunities",
                "metric": "evaluated_opportunities",
                "value": 0.0,
                "split": "test",
                "uses_oracle": False,
                "oracle_evaluation_only": True,
                "globally_comparable": False,
            }])
        else:
            ranking_metrics = evaluate_profile_ranking_scores(
                ranker.scores,
                causal_supervision.supervision,
                result.profile_ground_truth,
                split="test",
                fractions=config["experiments"]["reporting"]["capacity_fractions"],
                seed=int(run["ranker_random_seed"]),
            )
        ranking_metrics.to_csv(
            evaluation_dir / "profile_ranking_metrics.csv", index=False
        )
        checks, validation = validate_synthetic_profile_dgp(result, catalog)
        checks.to_csv(evaluation_dir / "dgp_validation_checks.csv", index=False)
        _write_json(evaluation_dir / "dgp_validation_report.json", validation)
        recommendation_metrics = evaluate_profile_recommendations(
            recommendation.recommendations,
            result.profile_ground_truth,
            causal_supervision.supported_opportunities,
            benefit_threshold_days=float(
                config["scientific_contract"]["recommendation_contract"][
                    "primary_minimum_benefit_threshold_days"
                ]
            ),
            split="test",
        )
        recommendation_metrics.to_csv(
            evaluation_dir / "recommendation_metrics.csv", index=False
        )
        allocation_metrics = evaluate_profile_allocation(
            allocation.decisions,
            allocation.allocation_candidates,
            allocation.diagnostic_selections,
            result.profile_ground_truth,
            shared_budget_limit=float(allocation.audit["shared_budget_limit"]),
            pool_capacities=allocation.audit["pool_capacities"],
            profile_capacities=allocation.audit["profile_capacities"],
            tie_seed=int(allocation.audit["tie_seed"]),
            tie_break_epsilon=float(allocation.audit["tie_break_epsilon"]),
            solver_settings=allocation.audit["solver_settings"],
        )
        allocation_metrics.to_csv(
            evaluation_dir / "profile_allocation_metrics.csv", index=False
        )
        _write_json(scenario_dir / "dgp_metadata.json", result.metadata)
        _write_json(scenario_dir / "dgp_audit.json", result.audit)
        diagnostics = result.metadata["scenario_diagnostics_evaluation_only"]
        summary_rows.append({
            "scenario": scenario,
            "critical_failures": validation["critical_failures"],
            "patients": len(result.learner),
            "structural_opportunities": len(result.patient_profile_opportunities),
            **diagnostics,
            "conditional_exchangeability": result.metadata["identification"][
                "conditional_exchangeability_given_learner_covariates"
            ],
            "analysis_outcome_is_placebo": result.metadata["identification"][
                "analysis_outcome_is_placebo"
            ],
            "capacity_fraction": result.resource_capacities[
                "capacity_fraction_of_population"
            ].max(),
            "profile_cost_multiplier_count": result.profile_resources[
                "scenario_cost_multiplier"
            ].nunique(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "profile_dgp_scenario_summary.csv", index=False)
    supervision_summary = pd.DataFrame(supervision_summary_rows)
    supervision_summary.to_csv(
        output / "profile_causal_supervision_summary.csv", index=False
    )
    ranker_summary = pd.DataFrame(ranker_summary_rows)
    ranker_summary.to_csv(output / "profile_ranker_summary.csv", index=False)
    recommendation_summary = pd.DataFrame(recommendation_summary_rows)
    recommendation_summary.to_csv(
        output / "profile_recommendation_summary.csv", index=False
    )
    allocation_summary = pd.DataFrame(allocation_summary_rows)
    allocation_summary.to_csv(
        output / "profile_allocation_summary.csv", index=False
    )
    raw_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostic_campaign = finalize_diagnostic_campaign(
        raw_diagnostics,
        settings=config["diagnostics"],
        recommendation_summary=recommendation_summary,
        allocation_summary=allocation_summary,
    )
    diagnostic_path = output / "profile_diagnostic_campaign.csv"
    diagnostic_campaign.rows.to_csv(diagnostic_path, index=False)
    diagnostic_contract = {
        "campaign_version": config["diagnostics"]["campaign_version"],
        "stage": "after_complete_decision_stack_before_large_experiments",
        "per_scenario_computation_stage": (
            "after_allocation_freeze_before_scenario_evaluation_only_writes"
        ),
        "root_aggregation_serialization_stage": "after_scenario_loop",
        "evaluation_partition": "test_nonoracle",
        "evaluation_partition_used_for_fitting_or_early_stopping": False,
        "base_seeds": list(map(int, config["diagnostics"]["base_seeds"])),
        "component_seed_derivation": config["diagnostics"][
            "component_seed_derivation"
        ],
        "scenario_conditions": config["diagnostics"]["scenario_conditions"],
        "thresholds": config["diagnostics"]["controls"],
        "threshold_sensitivity_days": config["diagnostics"][
            "threshold_sensitivity_days"
        ],
        "scenario_audits": diagnostic_audits,
        "campaign_audit": diagnostic_campaign.audit,
        "artifact_sha256": {diagnostic_path.name: _checksum(diagnostic_path)},
        "eligible_for_model_selection": False,
        "oracle_used": False,
        "failure_policy": config["diagnostics"]["failure_policy"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "profile_diagnostic_contract.json", diagnostic_contract)
    direction_checks = _direction_checks(results)
    direction_checks.to_csv(output / "profile_dgp_direction_checks.csv", index=False)
    if config["gates"]["require_all_structural_checks"] and summary.critical_failures.sum():
        raise AssertionError("A pipeline scenario failed structural validation")
    if config["gates"]["require_scenario_direction_checks"] and not direction_checks.passed.all():
        failed = direction_checks.loc[~direction_checks.passed, "check"].tolist()
        raise AssertionError(f"Pipeline direction checks failed: {failed}")
    if (
        config["gates"]["require_phase8_diagnostic_exit_gates"]
        and diagnostic_campaign.audit["status"] != "passed"
    ):
        raise AssertionError(
            "Phase-8 diagnostics failed: "
            f"{diagnostic_campaign.audit['failed_gates']}"
        )
    primary_scenario = str(config["gates"]["primary_support_scenario"])
    primary = supervision_summary.loc[supervision_summary.scenario.eq(primary_scenario)]
    if (
        config["gates"]["require_all_primary_scenario_profiles_supported"]
        and len(primary)
        and int(primary.iloc[0].profiles_supported)
        != int(primary.iloc[0].profiles_considered)
    ):
        raise AssertionError(
            f"Primary scenario {primary_scenario!r} has unsupported profile comparisons"
        )
    primary_ranker = ranker_summary.loc[ranker_summary.scenario.eq(primary_scenario)]
    if (
        config["gates"]["require_primary_ranker_trained"]
        and (
            primary_ranker.empty
            or primary_ranker.iloc[0].status != "trained"
            or not bool(primary_ranker.iloc[0].primary_model_saved)
        )
    ):
        raise AssertionError(f"Primary scenario {primary_scenario!r} did not train the ranker")
    if (
        config["gates"]["require_primary_validation_within_and_cross_pairs"]
        and (
            primary_ranker.empty
            or int(primary_ranker.iloc[0].validation_within_profile_pairs) < 1
            or int(primary_ranker.iloc[0].validation_cross_profile_pairs) < 1
        )
    ):
        raise AssertionError(
            f"Primary scenario {primary_scenario!r} lacks fixed within/cross validation pairs"
        )
    primary_recommendation = recommendation_summary.loc[
        recommendation_summary.scenario.eq(primary_scenario)
    ]
    if (
        config["gates"]["require_primary_recommendation_calibrated"]
        and (
            primary_recommendation.empty
            or primary_recommendation.iloc[0].status != "calibrated_and_recommended"
            or pd.isna(primary_recommendation.iloc[0].selected_calibration_method)
        )
    ):
        raise AssertionError(
            f"Primary scenario {primary_scenario!r} did not fit a validation calibrator"
        )
    if (
        config["gates"]["require_recommendations_for_all_patients"]
        and not recommendation_summary.patients.eq(len(population.patients)).all()
    ):
        raise AssertionError("Every scenario must write one recommendation per patient")
    if config["gates"][
        "require_zero_recommendation_ordering_and_deintensification_violations"
    ] and (
        recommendation_summary.raw_to_calibrated_ordering_violations.sum() > 0
        or recommendation_summary.deintensification_recommendations.sum() > 0
        or recommendation_summary.capacity_inputs_used.astype(bool).any()
        or recommendation_summary.oracle_used.astype(bool).any()
    ):
        raise AssertionError(
            "Recommendation ordering, deintensification, capacity or oracle gate failed"
        )
    if config["gates"]["require_zero_allocation_constraint_violations"] and (
        allocation_summary.constraint_violations.sum() > 0
        or allocation_summary.oracle_used.astype(bool).any()
    ):
        raise AssertionError("Allocation constraint or oracle gate failed")
    if config["gates"]["require_allocation_preserves_recommendations"] and (
        allocation_summary.recommendation_records_modified.astype(bool).any()
        or allocation_summary.alternative_profile_substitution.astype(bool).any()
    ):
        raise AssertionError("Allocation changed or substituted a frozen recommendation")
    if config["gates"]["require_capacity_scarcity_defers_more_recommendations"]:
        baseline_allocation = allocation_summary.loc[
            allocation_summary.scenario.eq("baseline_identifiable")
        ]
        scarcity_allocation = allocation_summary.loc[
            allocation_summary.scenario.eq("capacity_scarcity")
        ]
        if (
            baseline_allocation.empty
            or scarcity_allocation.empty
            or float(scarcity_allocation.iloc[0].conditional_deferral_rate)
            <= float(baseline_allocation.iloc[0].conditional_deferral_rate)
        ):
            raise AssertionError(
                "Capacity-scarcity scenario did not increase conditional deferral"
            )

    experiment_config = config["experiments"]
    stability_diagnostic_path = output / "profile_stability_diagnostics.csv"
    stability_diagnostic = run_stabilization_source_diagnostics(config, catalog)
    stability_diagnostic.records.to_csv(stability_diagnostic_path, index=False)
    stability_contract_path = output / "profile_stability_contract.json"
    _write_json(stability_contract_path, {
        "contract_version": "prometheus_ranker_stability_diagnostic_v7",
        "protocol_version": experiment_config["protocol_version"],
        **stability_diagnostic.audit,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    stability_diagnostic_sha256 = _checksum(stability_diagnostic_path)
    if stability_diagnostic.audit["status"] != "passed":
        raise AssertionError(
            "Phase-9B non-oracle stabilization gates failed before discovery: "
            + ", ".join(stability_diagnostic.audit["failed_gates"])
        )
    discovery_path = output / "profile_experiment_discovery.csv"
    discovery = run_discovery_campaign(config, catalog)
    discovery.records.to_csv(discovery_path, index=False)
    discovery_sha256 = _checksum(discovery_path)

    candidate_path = output / "profile_candidate_declaration.json"
    candidate_declaration = build_candidate_declaration(
        config,
        discovery,
        discovery_artifact_sha256=discovery_sha256,
        stabilization_diagnostic_sha256=stability_diagnostic_sha256,
    )
    candidate_declaration.update({
        "protocol_version": experiment_config["protocol_version"],
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(candidate_path, candidate_declaration)
    candidate_sha256 = _checksum(candidate_path)

    expected_candidate = experiment_config["discovery"][
        "expected_locked_candidate"
    ]
    if expected_candidate is None:
        raise RuntimeError(
            "Discovery completed and the candidate declaration was written before "
            "opening confirmation. Set experiments.discovery.expected_locked_candidate "
            f"to {discovery.selected_variant!r}, then rerun the pipeline."
        )
    if str(expected_candidate) != discovery.selected_variant:
        raise AssertionError(
            "Prespecified discovery did not reproduce the locked candidate: "
            f"expected {expected_candidate!r}, observed {discovery.selected_variant!r}. "
            "Confirmation remains unopened."
        )

    freeze_path = output / "profile_experiment_freezes.csv"
    confirmation = run_confirmation_and_stability_campaign(
        config,
        catalog,
        selected_variant=discovery.selected_variant,
        candidate_declaration_sha256=candidate_sha256,
        freeze_path=freeze_path,
    )
    experiment_metrics_path = output / "profile_experiment_metrics.csv"
    confirmation.metrics.to_csv(experiment_metrics_path, index=False)
    experiment_contract_path = output / "profile_experiment_contract.json"
    experiment_contract = {
        "contract_version": "prometheus_phase9b_stabilized_experiment_contract_v7",
        "protocol_version": experiment_config["protocol_version"],
        "status": confirmation.audit["status"],
        "candidate_declaration_sha256": candidate_sha256,
        "selected_candidate": discovery.selected_variant,
        "discovery": discovery.audit,
        "stabilization_diagnostics": stability_diagnostic.audit,
        "confirmation_and_stability": confirmation.audit,
        "artifacts": {
            discovery_path.name: discovery_sha256,
            stability_diagnostic_path.name: stability_diagnostic_sha256,
            stability_contract_path.name: _checksum(stability_contract_path),
            candidate_path.name: candidate_sha256,
            freeze_path.name: _checksum(freeze_path),
            experiment_metrics_path.name: _checksum(experiment_metrics_path),
        },
        "oracle_used_for_candidate_selection": False,
        "confirmation_opened_only_after_candidate_declaration": True,
        "priority_score_interpretation": "ordinal_not_calibrated_treatment_effect",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(experiment_contract_path, experiment_contract)
    if (
        config["gates"]["require_phase9_protocol_exit_gates"]
        and confirmation.audit["status"] != "passed"
    ):
        raise AssertionError(
            "Phase-9 experiment gates failed: "
            + ", ".join(confirmation.audit["failed_gates"])
        )

    manifest = {
        "manifest_version": "prometheus_profile_stratification_manifest_v1",
        "config_version": config["config_version"],
        "run_id": run_id,
        "pipeline": "prometheus_care_profile",
        "completed_stage": "stabilized_prespecified_synthetic_evaluation",
        "status": "complete",
        "decision_unit": "care_profile",
        "treatment": "exact_observed_treatment_profile",
        "comparator": "no_new_profile",
        "scenarios": list(config["scenarios"]),
        "scenario_count": len(config["scenarios"]),
        "population_size": len(population.patients),
        "structural_validation_failures": int(summary.critical_failures.sum()),
        "direction_checks_failed": int((~direction_checks.passed).sum()),
        "truth_storage": "scenario/evaluation_only_after_profile_allocation_freeze",
        "oracle_used_for_model_selection": False,
        "model_training_performed": True,
        "nuisance_model_training_performed": True,
        "causal_ranker_training_performed": bool(
            ranker_summary.status.eq("trained").any()
        ),
        "primary_ranker_variant": config["ranker"]["primary_variant"],
        "ranker_variants": list(config["ranker"]["variants"]),
        "oracle_evaluation_ranker_deferred": True,
        "ranker_scenarios_trained": int(ranker_summary.status.eq("trained").sum()),
        "ranker_supported_opportunities_scored": int(
            ranker_summary.supported_opportunities_scored.sum()
        ),
        "validation_calibration_performed": bool(
            recommendation_summary.selected_calibration_method.notna().any()
        ),
        "calibration_scenarios_fitted": int(
            recommendation_summary.selected_calibration_method.notna().sum()
        ),
        "actionable_profile_recommendation_performed": True,
        "recommendation_patients": int(recommendation_summary.patients.sum()),
        "recommendation_patients_recommended": int(
            recommendation_summary.patients_recommended.sum()
        ),
        "recommendation_patients_abstained": int(
            recommendation_summary.patients_abstained.sum()
        ),
        "recommendation_capacity_inputs_used": False,
        "recommendation_oracle_used": False,
        "recommendation_oracle_metrics_evaluation_only": True,
        "profile_allocation_performed": True,
        "allocation_recommended_candidates": int(
            allocation_summary.recommended_candidates.sum()
        ),
        "allocation_profiles_activated": int(
            allocation_summary.allocated_recommendations.sum()
        ),
        "allocation_recommendations_deferred": int(
            allocation_summary.deferred_recommendations.sum()
        ),
        "allocation_constraint_violations": int(
            allocation_summary.constraint_violations.sum()
        ),
        "allocation_recommendations_modified": False,
        "allocation_alternative_profile_substitution": False,
        "allocation_oracle_used": False,
        "allocation_oracle_metrics_evaluation_only": True,
        "phase8_diagnostics_performed": True,
        "phase8_diagnostic_campaign_version": config["diagnostics"][
            "campaign_version"
        ],
        "phase8_diagnostic_base_seeds": list(map(
            int, config["diagnostics"]["base_seeds"]
        )),
        "phase8_required_gates": int(
            diagnostic_campaign.audit["required_gate_count"]
        ),
        "phase8_required_gates_passed": int(
            diagnostic_campaign.audit["required_gates_passed"]
        ),
        "phase8_failed_gates": diagnostic_campaign.audit["failed_gates"],
        "phase8_diagnostics_oracle_used": False,
        "phase8_diagnostics_eligible_for_model_selection": False,
        "phase9b_stabilization_performed": True,
        "phase9b_protocol_version": experiment_config["protocol_version"],
        "phase9b_stability_diagnostic_gates": stability_diagnostic.audit["gates"],
        "phase9b_selected_candidate": discovery.selected_variant,
        "phase9b_discovery_run_seeds": list(map(
            int, experiment_config["discovery"]["run_seeds"]
        )),
        "phase9b_confirmation_run_seeds": list(map(
            int, experiment_config["confirmation"]["run_seeds"]
        )),
        "phase9b_fixed_dataset_seed": int(
            experiment_config["fixed_dataset_stability"]["dataset_seed"]
        ),
        "phase9b_fixed_dataset_run_seeds": list(map(
            int, experiment_config["fixed_dataset_stability"]["run_seeds"]
        )),
        "phase9b_bootstrap_samples": int(
            experiment_config["reporting"]["bootstrap_samples"]
        ),
        "phase9b_required_gates": int(len(confirmation.audit["gates"])),
        "phase9b_required_gates_passed": int(sum(
            bool(value) for value in confirmation.audit["gates"].values()
        )),
        "phase9b_failed_gates": confirmation.audit["failed_gates"],
        "phase9b_freeze_records": int(confirmation.audit["freeze_records"]),
        "phase9b_confirmation_opened_after_candidate_lock": True,
        "phase9b_oracle_used_for_candidate_selection": False,
        "profile_comparability": profile_comparability,
        "ranker_architecture_used_to_design_dgp": False,
        "implicit_action_to_profile_conversion": False,
        "source_population_fully_synthetic": True,
        "causal_supervision": {
            "profile_comparison": "exact_profile_vs_no_new_profile",
            "profiles_fitted": int(supervision_summary.profiles_fitted.sum()),
            "profiles_supported": int(supervision_summary.profiles_supported.sum()),
            "profiles_unsupported": int(supervision_summary.profiles_unsupported.sum()),
            "supervision_rows": int(supervision_summary.supervision_rows.sum()),
            "oracle_used": False,
        },
        "seeds": {
            key: int(run[key]) for key in (
                "population_seed", *dgp_seed_keys, "split_seed", "nuisance_seed",
                "ranker_model_seed", "ranker_pair_seed",
                "ranker_validation_pair_seed", "ranker_contrastive_seed",
                "ranker_gbdt_seed", "ranker_random_seed",
                "allocator_tie_seed",
            )
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "run_manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    print(run_pipeline(args.config))


if __name__ == "__main__":
    main()
