from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..allocation.global_allocator import allocate_global_opportunities
from ..data.validation import assert_no_oracle_columns
from ..dm77 import (
    assess_dm77_population,
    derive_intervention_eligibility,
    load_dm77_settings,
    summarize_dm77_population,
)
from ..evaluation.global_metrics import (
    allocation_value,
    compare_allocation_to_oracle,
    global_pairwise_concordance,
    pooled_opportunity_autoc,
    transition_ranking_metrics,
)
from ..nuisance.cross_fitting import fit_repeated_partitioned_nuisance
from ..ranking.calibration import MonotoneScoreCalibrator, TransitionwiseCalibrator
from ..ranking.causal_signals import aggregate_repeated_signals, repeated_doubly_robust_signals
from ..ranking.global_ranker import GLOBAL_TRAINING_MODES, GlobalPrometheusRanker
from ..ranking.opportunities import OpportunityArrays, concatenate_opportunities
from ..reproducibility import seed_everything, write_json
from ..synthetic import (
    SYNTHETIC_POPULATION_SCENARIOS,
    generate_synthetic_population,
    validate_synthetic_population,
)
from ..version import PROMETHEUS_SYNTHETIC_LEGACY_GLOBAL_METHOD_VERSION
from .global_simulation import TRANSITION_NAMES, simulate_multivalued_care
from .global_support import TransitionSupportOutcomePredictor
from .helpers import _load_cohort


METHOD_VERSION = PROMETHEUS_SYNTHETIC_LEGACY_GLOBAL_METHOD_VERSION
LOCAL_MODES = {"independent_local_rankers", "unified_local_ranker"}


def _validate_config(config: dict) -> dict:
    sections = {"run", "data", "simulation", "nuisance", "prometheus", "allocation", "dm77"}
    missing = sorted(sections.difference(config))
    if missing:
        raise ValueError(f"Missing global PROMETHEUS config sections: {missing}")
    run = config["run"]
    simulation = config["simulation"]
    nuisance = config["nuisance"]
    model = config["prometheus"]
    allocation = config["allocation"]
    dm77 = config["dm77"]
    data = config["data"]
    if int(run["seed"]) < 0 or int(run["sample_size"]) < 100:
        raise ValueError("run.seed must be non-negative and run.sample_size at least 100")
    if model.get("training_mode") not in GLOBAL_TRAINING_MODES:
        raise ValueError(f"Unknown global training mode {model.get('training_mode')!r}")
    required_model = {
        "within_pairs_per_epoch", "cross_pairs_per_epoch", "beta_cross",
        "min_signal_gap_days", "min_pair_direction_agreement", "lambda_con",
        "lambda_reg", "contrastive_margin", "positive_max_gap_days",
        "negative_min_gap_days", "propensity_clip_epsilon", "dr_signal_aggregation",
        "calibration",
    }
    missing_model = sorted(required_model.difference(model))
    if missing_model:
        raise ValueError(f"Missing global PROMETHEUS settings: {missing_model}")
    if int(nuisance.get("folds", 0)) < 2 or int(nuisance.get("repeats", 0)) < 1:
        raise ValueError("Nuisance folds must be at least 2 and repeats positive")
    if not 0.0 < float(model["propensity_clip_epsilon"]) < 0.5:
        raise ValueError("prometheus.propensity_clip_epsilon must be in (0, 0.5)")
    if model["dr_signal_aggregation"] not in {"mean", "median"}:
        raise ValueError("prometheus.dr_signal_aggregation must be mean or median")
    calibration = model["calibration"]
    if calibration.get("method") != "isotonic":
        raise ValueError("prometheus.calibration.method must be isotonic")
    if model["training_mode"] not in LOCAL_MODES and not bool(
        calibration.get("pooled_for_global_models", False)
    ):
        raise ValueError("Global modes require pooled_for_global_models=true")
    eligibility = simulation.get("eligibility", {})
    if eligibility.get("mode") not in {"all", "synthetic_clinical"}:
        raise ValueError("simulation.eligibility.mode must be all or synthetic_clinical")
    costs = allocation.get("transition_costs", {})
    if set(costs) != set(TRANSITION_NAMES) or any(float(value) <= 0 for value in costs.values()):
        raise ValueError("allocation.transition_costs must define five positive transition costs")
    support = allocation.get("propensity_support", {})
    lower, upper = float(support.get("lower", -1)), float(support.get("upper", -1))
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("allocation.propensity_support must satisfy 0 <= lower < upper <= 1")
    if float(allocation.get("shared_budget", -1)) < 0:
        raise ValueError("allocation.shared_budget must be non-negative")
    if model["training_mode"] in LOCAL_MODES and float(model["lambda_con"]) != 0.0:
        raise ValueError("Local ranking modes require prometheus.lambda_con = 0")
    if not bool(dm77.get("enabled", False)):
        raise ValueError("The global pipeline requires dm77.enabled=true")
    if bool(dm77.get("use_level_as_ranker_feature", False)):
        raise ValueError("dm77_need_level must never be used as a ranker feature")
    if bool(dm77.get("use_level_as_treatment", False)):
        raise ValueError("dm77_need_level must never be used as historical treatment")
    load_dm77_settings(dm77)
    source = data.get("source", "cohort_cache")
    if source not in {"fully_synthetic", "cohort_cache"}:
        raise ValueError("data.source must be fully_synthetic or cohort_cache")
    if source == "fully_synthetic":
        synthetic = data.get("synthetic", {})
        if synthetic.get("generator") != "structured_longitudinal_population_v1":
            raise ValueError(
                "data.synthetic.generator must be structured_longitudinal_population_v1"
            )
        if int(synthetic.get("history_months", 0)) < 12:
            raise ValueError("Fully synthetic data require at least 12 history months")
        if synthetic.get("scenario", "baseline") not in SYNTHETIC_POPULATION_SCENARIOS:
            raise ValueError("Unknown data.synthetic.scenario")
    elif not data.get("cohort_cache"):
        raise ValueError("data.cohort_cache is required when data.source=cohort_cache")
    return config


def _load_global_population(data: dict, sample_size: int, seed: int):
    source = data.get("source", "cohort_cache")
    if source == "cohort_cache":
        cohort = _load_cohort(data["cohort_cache"], sample_size, seed)
        return cohort, None, None, {
            "data_source": "cohort_cache",
            "population_is_fully_synthetic": False,
            "real_person_source_records": "unknown",
            "formal_privacy_guarantee_claimed": False,
        }
    settings = data["synthetic"]
    generated = generate_synthetic_population(
        population_size=sample_size,
        seed=seed + int(settings.get("seed_offset", 0)),
        history_months=int(settings["history_months"]),
        reference_date=str(settings.get("reference_date", "2025-01-01")),
        minimum_age=int(settings.get("minimum_age", 18)),
        maximum_age=int(settings.get("maximum_age", 100)),
        profile=str(settings.get("population_profile", "methodological_adult_population_v1")),
        scenario=str(settings.get("scenario", "baseline")),
    )
    checks, report = validate_synthetic_population(
        generated.patients, generated.monthly_history, int(settings["history_months"])
    )
    if int(report["critical_failures"]) > 0:
        failed = checks.loc[checks.status == "fail", "check"].tolist()
        raise RuntimeError(f"Synthetic population failed critical validation: {failed}")
    return generated.patients, generated, (checks, report), generated.privacy_audit


def _repeated_nuisance_tensor(repetitions: list[pd.DataFrame], patient_ids) -> np.ndarray:
    identifiers = pd.Index(np.asarray(patient_ids).astype(str), name="patient_id")
    arrays = []
    for repeat, frame in enumerate(repetitions):
        indexed = frame.assign(patient_id=frame.patient_id.astype(str)).set_index("patient_id")
        if not identifiers.isin(indexed.index).all():
            raise RuntimeError(f"Nuisance repetition {repeat} is missing patient predictions")
        arrays.append(indexed.loc[identifiers, ["e_hat", "mu0_hat", "mu1_hat"]].to_numpy(float))
    return np.stack(arrays, axis=1)


def _build_observed_opportunities(
    prepared: dict,
    features: tuple[str, ...],
    split: str,
) -> OpportunityArrays:
    blocks = []
    for transition_index, name in enumerate(TRANSITION_NAMES):
        frame = prepared[name]["frame"]
        positions = np.flatnonzero(frame.split.astype(str).to_numpy() == split)
        blocks.append(OpportunityArrays(
            frame.loc[positions, features].to_numpy(float),
            np.full(len(positions), transition_index, dtype=np.int64),
            frame.loc[positions, "patient_id"].astype(str).to_numpy(),
            prepared[name]["signal"][positions],
            prepared[name]["repeated_signal"][positions],
            positions,
        ))
    return concatenate_opportunities(blocks)


def _ranker_from_config(config: dict, transition_names, seed: int, mode: str | None = None):
    model = config["prometheus"]
    selected_mode = mode or model["training_mode"]
    return GlobalPrometheusRanker(
        transition_names=transition_names,
        training_mode=selected_mode,
        within_pairs_per_epoch=int(model["within_pairs_per_epoch"]),
        cross_pairs_per_epoch=int(model["cross_pairs_per_epoch"]),
        beta_cross=float(model["beta_cross"]),
        min_signal_gap_days=float(model["min_signal_gap_days"]),
        min_pair_direction_agreement=float(model["min_pair_direction_agreement"]),
        balance_transition_pairs=bool(model.get("balance_transition_pairs", True)),
        allow_same_patient_cross_pairs=bool(model.get("allow_same_patient_cross_pairs", False)),
        contrastive_pairs_per_epoch=int(model.get("contrastive_pairs_per_epoch", model["within_pairs_per_epoch"])),
        positive_max_gap_days=float(model["positive_max_gap_days"]),
        negative_min_gap_days=float(model["negative_min_gap_days"]),
        lambda_con=float(model["lambda_con"]),
        lambda_reg=float(model["lambda_reg"]),
        contrastive_margin=float(model["contrastive_margin"]),
        reliability_weighting=bool(model.get("pair_reliability_weighting", {}).get("enabled", False)),
        reliability_max_weight=float(model.get("pair_reliability_weighting", {}).get("max_weight", 25.0)),
        validation_beta_cross=float(model.get("validation_beta_cross", model["beta_cross"])),
        checkpoint_metric=str(config.get("checkpoint", {}).get("metric", "global_concordance")),
        validation_policy_fraction=float(
            config.get("checkpoint", {}).get("validation_policy_fraction", 0.20)
        ),
        contrastive_pair_scope=str(model.get("contrastive_pair_scope", "pooled")),
        contrastive_require_stable_direction=bool(
            model.get("contrastive_require_stable_direction", False)
        ),
        contrastive_reliability_weighting=bool(
            model.get("contrastive_reliability_weighting", False)
        ),
        epochs=int(model.get("epochs", 20)),
        batch_size=int(model.get("batch_size", 512)),
        hidden_dim=int(model.get("hidden_dim", 128)),
        latent_dim=int(model.get("latent_dim", 32)),
        learning_rate=float(model.get("learning_rate", 3e-4)),
        patience=int(model.get("patience", 4)),
        permute_training_pair_labels=bool(
            config.get("negative_control", {}).get(
                "permute_training_pair_labels", False
            )
        ),
        seed=seed,
        device=config["run"].get("device", "cpu"),
    )


def _train_rankers(
    config: dict,
    training: OpportunityArrays,
    validation: OpportunityArrays,
    seed: int,
    transition_names=TRANSITION_NAMES,
):
    mode = config["prometheus"]["training_mode"]
    if mode != "independent_local_rankers":
        ranker = _ranker_from_config(config, transition_names, seed).fit(training, validation)
        return {name: ranker for name in transition_names}, pd.DataFrame(ranker.history), ranker.training_diagnostics
    models, histories, diagnostics = {}, [], {}
    for transition_index, name in enumerate(transition_names):
        train_mask = training.transition_index == transition_index
        validation_mask = validation.transition_index == transition_index
        local_training = OpportunityArrays(
            training.x[train_mask], np.zeros(train_mask.sum(), dtype=np.int64), training.patient_ids[train_mask],
            training.signal[train_mask], training.repeated_signal[train_mask], training.source_row[train_mask],
        )
        local_validation = OpportunityArrays(
            validation.x[validation_mask], np.zeros(validation_mask.sum(), dtype=np.int64),
            validation.patient_ids[validation_mask], validation.signal[validation_mask],
            validation.repeated_signal[validation_mask], validation.source_row[validation_mask],
        )
        model = _ranker_from_config(
            config, (name,), seed + 1009 * transition_index,
            mode="independent_local_rankers",
        ).fit(local_training, local_validation)
        models[name] = model
        histories.append(pd.DataFrame(model.history).assign(transition=name))
        diagnostics[name] = model.training_diagnostics
    return models, pd.concat(histories, ignore_index=True), {
        "model": "independent_local_rankers",
        "training_mode": mode,
        "cross_transition_pairs": False,
        "score_semantics": "ordinal_within_transition",
        "per_transition": diagnostics,
    }


def _score_observed(
    models: dict,
    opportunities: OpportunityArrays,
    transition_names=TRANSITION_NAMES,
) -> np.ndarray:
    scores = np.empty(len(opportunities.x), dtype=float)
    for transition_index, name in enumerate(transition_names):
        mask = opportunities.transition_index == transition_index
        model = models[name]
        model_index = np.zeros(mask.sum(), dtype=np.int64) if len(model.transition_names) == 1 else np.full(mask.sum(), transition_index)
        scores[mask] = model.predict_opportunity_scores(opportunities.x[mask], model_index)
    return scores


def _fit_calibrator(mode: str, score, validation: OpportunityArrays):
    if mode in LOCAL_MODES:
        return TransitionwiseCalibrator(TRANSITION_NAMES).fit(
            score, validation.signal, validation.transition_index
        )
    return MonotoneScoreCalibrator().fit(score, validation.signal)


def _calibrate(calibrator, score, transition_index) -> np.ndarray:
    if isinstance(calibrator, TransitionwiseCalibrator):
        return calibrator.predict(score, transition_index)
    return calibrator.predict(score)


def _build_test_opportunities(
    learner: pd.DataFrame,
    features: tuple[str, ...],
    predictors: dict,
    models: dict,
    calibrator,
    config: dict,
) -> pd.DataFrame:
    test = learner.loc[learner.split.astype(str) == "test"].reset_index(drop=True)
    parts = []
    costs = config["allocation"]["transition_costs"]
    support_bounds = config["allocation"]["propensity_support"]
    for transition_index, name in enumerate(TRANSITION_NAMES):
        x = test.loc[:, features].to_numpy(float)
        propensity, mu0, mu1 = predictors[name].predict(x)
        model = models[name]
        model_index = np.zeros(len(test), dtype=np.int64) if len(model.transition_names) == 1 else np.full(len(test), transition_index)
        score = model.predict_opportunity_scores(x, model_index)
        parts.append(pd.DataFrame({
            "patient_id": test.patient_id.astype(str),
            "transition": name,
            "transition_index": transition_index,
            "raw_score": score,
            "cost": float(costs[name]),
            "eligibility": test[f"eligible_{name}"].to_numpy(bool),
            "eligible": test[f"eligible_{name}"].to_numpy(bool),
            "support_propensity": propensity,
            "support": (propensity >= float(support_bounds["lower"])) & (propensity <= float(support_bounds["upper"])),
            "supported": (propensity >= float(support_bounds["lower"])) & (propensity <= float(support_bounds["upper"])),
            "transition_cate_baseline": mu1 - mu0,
        }))
    frame = pd.concat(parts, ignore_index=True).sort_values(
        ["patient_id", "transition_index"], kind="mergesort"
    ).reset_index(drop=True)
    frame["calibrated_benefit"] = _calibrate(
        calibrator, frame.raw_score.to_numpy(float), frame.transition_index.to_numpy(int)
    )
    return frame


def _allocation(frame: pd.DataFrame, config: dict):
    allocation = config["allocation"]
    return allocate_global_opportunities(
        frame,
        float(allocation["shared_budget"]),
        allocation.get("transition_capacities"),
        bool(allocation.get("require_empirical_support", True)),
        bool(allocation.get("allocate_negative_predicted_benefit", False)),
        bool(allocation.get("exact_solver", True)),
        float(allocation.get("solver_time_limit_seconds", 120.0)),
    )


def _apply_dm77_standard_pathway_exclusions(simulation, assessments: pd.DataFrame, settings: dict) -> dict:
    """Exclude protected/abstained patients from the standard causal allocation."""

    protected = assessments.dm77_protected_pathway.to_numpy(bool)
    manual_review = assessments.dm77_assessment_status.eq("manual_review").to_numpy(bool)
    blocked = np.zeros(len(assessments), dtype=bool)
    if bool(settings.get("exclude_level_vi_from_standard_allocation", True)):
        blocked |= protected
    if bool(settings.get("exclude_manual_review_from_standard_allocation", True)):
        blocked |= manual_review
    blocked_ids = set(assessments.loc[blocked, "patient_id"].astype(str))
    for name in TRANSITION_NAMES:
        eligibility_column = f"eligible_{name}"
        simulation.learner.loc[
            simulation.learner.patient_id.astype(str).isin(blocked_ids), eligibility_column
        ] = False
        transition = simulation.transition_learners[name]
        simulation.transition_learners[name] = transition.loc[
            ~transition.patient_id.astype(str).isin(blocked_ids)
        ].reset_index(drop=True)
    return {
        "standard_pathway_excluded_patients": int(len(blocked_ids)),
        "level_vi_excluded_patients": int(protected.sum()) if bool(
            settings.get("exclude_level_vi_from_standard_allocation", True)
        ) else 0,
        "manual_review_excluded_patients": int(manual_review.sum()) if bool(
            settings.get("exclude_manual_review_from_standard_allocation", True)
        ) else 0,
        "exclusion_is_pre_treatment_governed_eligibility": True,
    }


def _priority_baseline(frame: pd.DataFrame, values) -> pd.DataFrame:
    out = frame.copy()
    values = np.asarray(values, dtype=float)
    order = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    out["calibrated_benefit"] = order
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def _write_report(
    out: Path,
    manifest: dict,
    metrics: dict,
    policies: pd.DataFrame,
    allocation: dict,
    dm77_summary: pd.DataFrame,
    dm77_audit: dict,
) -> None:
    source_description = (
        "The configured primary source is fully synthetic and generated from an explicit "
        "seed, with no real-person records. The longitudinal validation report checks "
        "identities, ranges, dependency directions, and broad methodological plausibility."
        if bool(manifest["source_population_fully_synthetic"])
        else "The configured source is a cohort cache; its provenance must be governed "
        "externally and no fully-synthetic lineage claim is made by this run."
    )
    text = f"""# PROMETHEUS global multi-transition causal ranking

Method version: `{manifest['method_version']}`
Training mode: `{manifest['model_variant']}`
Source population: `{manifest['data_source']}`

{source_description} These checks do not establish Italian-population
representativeness or external clinical validity.

Historical treatment `T_i` is the single multivalued care package observed during the
365-day study period. The learned `s_theta(X_i, k)` is a globally comparable ordinal
priority over patient-transition opportunities, not a calibrated treatment-effect
estimate. Validation-only isotonic calibration produces an approximate benefit in days.
The joint allocation variable `Z_(i,k)` respects eligibility, empirical support,
precedence, transition capacities when configured, and one shared budget. The resulting
`T_i*` is the allocated care package and is not a severity class.

No oracle column, potential outcome, true effect, propensity truth, or latent rank enters
nuisance fitting, pair construction, model fitting, early stopping, calibration, support
prediction, or learned allocation. Oracle effects are joined only after the operational
allocation is fixed for semi-synthetic evaluation.

## DM 77 need stratification

DM 77 need level, prognostic risk, causal priority, and allocation decision are separate
outputs. The level is never the historical treatment, a causal target, or a ranker
feature. The six labels follow the DM 77 population-stratification model; the executable
thresholds used here are an auditable research operationalization, not official national
cut-offs. Level VI follows a protected palliative pathway. Incomplete multidimensional
data trigger abstention and professional review rather than an automatic level.

{_markdown_table(dm77_summary)}

Standard-pathway exclusions: `{dm77_audit['standard_pathway_excluded_patients']}`.

## Global metrics

{_markdown_table(pd.DataFrame([metrics]))}

## Feasibility

{_markdown_table(pd.DataFrame([allocation]))}

## Shared-budget policy evaluation

{_markdown_table(policies)}

The intervention catalogue expresses candidate eligibility for professional review; it
does not establish causal effectiveness. This methodological synthetic run is not
evidence of clinical effectiveness, Italian-population validity, superiority over ACG,
or deployment readiness.
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def _run_synthetic_legacy_prometheus(config: dict) -> Path:
    """Execute the global multivalued PROMETHEUS pipeline and write real run artifacts."""

    config = _validate_config(config)
    if config["run"]["scenario"] == "heterogeneous_transition_costs":
        base_costs = config["allocation"]["transition_costs"]
        config["allocation"]["transition_costs"] = {
            name: float(base_costs[name]) * (1.0 + 0.35 * index)
            for index, name in enumerate(TRANSITION_NAMES)
        }
    started = time.time()
    run, simulation_config, nuisance_config, model_config = (
        config["run"], config["simulation"], config["nuisance"], config["prometheus"]
    )
    dm77_settings = load_dm77_settings(config["dm77"])
    seed = int(run["seed"])
    seed_everything(seed)
    population_scenario = (
        str(config["data"].get("synthetic", {}).get("scenario", "not_applicable"))
        if config["data"].get("source") == "fully_synthetic" else "cohort_cache"
    )
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_prometheus_global_{model_config['training_mode']}_{run['scenario']}"
        + f"_population_{population_scenario}_s{seed}"
    )
    out = Path(run["output_root"]) / run_id
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "resolved_config.json", config)

    cohort, synthetic_result, synthetic_validation, source_privacy_audit = _load_global_population(
        config["data"], int(run["sample_size"]), seed
    )
    if synthetic_result is not None:
        synthetic_result.patients.to_csv(out / "synthetic_patient_features.csv", index=False)
        if bool(config["data"]["synthetic"].get("write_monthly_history", True)):
            synthetic_result.monthly_history.to_csv(
                out / "synthetic_monthly_history.csv", index=False
            )
        write_json(
            out / "synthetic_generation_metadata.json", synthetic_result.metadata
        )
        synthetic_checks, synthetic_validation_report = synthetic_validation
        synthetic_checks.to_csv(out / "synthetic_validation_checks.csv", index=False)
        write_json(out / "synthetic_validation_report.json", synthetic_validation_report)
        write_json(out / "synthetic_privacy_audit.json", source_privacy_audit)
    else:
        synthetic_validation_report = {
            "validation_profile": "not_applicable_cohort_cache",
            "critical_failures": None,
            "warnings": None,
            "italian_population_representativeness_established": False,
        }
    simulation = simulate_multivalued_care(
        cohort, seed=seed, scenario=run["scenario"],
        outcome_horizon_days=int(simulation_config.get("outcome_horizon_days", 365)),
        treatment_levels=int(simulation_config.get("treatment_levels", 6)),
        shared_effect_correlation=float(simulation_config["shared_effect_correlation"]),
        overlap_strength=float(simulation_config["overlap_strength"]),
        risk_benefit_alignment=float(simulation_config["risk_benefit_alignment"]),
        transition_sample_imbalance=float(simulation_config["transition_sample_imbalance"]),
        transition_effect_scales=simulation_config.get("transition_effect_scales"),
        eligibility_mode=simulation_config["eligibility"]["mode"],
        outcome_noise_sd=float(simulation_config.get("outcome_noise_sd", 6.0)),
    )
    assert_no_oracle_columns(simulation.learner)
    dm77_assessments, dm77_assessment_audit = assess_dm77_population(
        simulation.learner, dm77_settings
    )
    dm77_summary = summarize_dm77_population(dm77_assessments)
    dm77_intervention_eligibility = derive_intervention_eligibility(
        dm77_assessments, dm77_settings["intervention_catalog"]
    )
    dm77_routing_audit = _apply_dm77_standard_pathway_exclusions(
        simulation, dm77_assessments, dm77_settings
    )
    dm77_audit = {**dm77_assessment_audit, **dm77_routing_audit}
    dm77_assessments.to_csv(out / "dm77_patient_assessments.csv", index=False)
    dm77_summary.to_csv(out / "dm77_population_summary.csv", index=False)
    dm77_assessments.loc[
        dm77_assessments.dm77_assessment_status == "manual_review"
    ].to_csv(out / "dm77_manual_review_queue.csv", index=False)
    dm77_intervention_eligibility.to_csv(out / "dm77_intervention_eligibility.csv", index=False)
    write_json(out / "dm77_audit_report.json", dm77_audit)
    simulation.learner.to_csv(out / "learner_multivalued_dataset.csv", index=False)
    simulation.ground_truth.to_csv(out / "evaluation_multilevel_ground_truth.csv", index=False)
    features = simulation.feature_columns
    if "dm77_need_level" in features or any(column.startswith("dm77_") for column in features):
        raise AssertionError("A derived DM 77 assessment entered the ranker feature set")

    nuisance_dir = out / "transition_nuisance_predictions"
    nuisance_dir.mkdir()
    prepared, predictors = {}, {}
    epsilon = float(model_config["propensity_clip_epsilon"])
    for transition_index, name in enumerate(TRANSITION_NAMES):
        learner = simulation.transition_learners[name].copy()
        assert_no_oracle_columns(learner)
        nuisance, repetitions, diagnostics = fit_repeated_partitioned_nuisance(
            learner, list(features), int(nuisance_config["folds"]), nuisance_config["model"],
            seed + 1009 * transition_index, int(nuisance_config["repeats"]),
            int(nuisance_config.get("repeat_seed_stride", 1009)), "nuisance_train",
            (epsilon, 1.0 - epsilon),
        )
        nuisance.to_csv(nuisance_dir / f"{name}.csv", index=False)
        pd.concat(repetitions, ignore_index=True).to_csv(nuisance_dir / f"{name}_repeated.csv", index=False)
        write_json(nuisance_dir / f"{name}_diagnostics.json", diagnostics)
        repeated_nuisance = _repeated_nuisance_tensor(repetitions, learner.patient_id)
        repeated_signal = repeated_doubly_robust_signals(
            learner.transition_treatment.to_numpy(float), learner.observed_outcome.to_numpy(float),
            repeated_nuisance, epsilon,
        )
        signal = aggregate_repeated_signals(repeated_signal, model_config["dr_signal_aggregation"])
        prepared[name] = {
            "frame": learner, "signal": signal, "repeated_signal": repeated_signal,
            "diagnostics": diagnostics,
        }
        predictors[name] = TransitionSupportOutcomePredictor(
            nuisance_config["model"], seed + 20_011 + transition_index
        ).fit(learner, features)

    training = _build_observed_opportunities(prepared, features, "rank_train")
    validation = _build_observed_opportunities(prepared, features, "validation")
    training.to_frame(TRANSITION_NAMES).to_csv(out / "rank_training_opportunities.csv", index=False)
    models, history, training_diagnostics = _train_rankers(config, training, validation, seed + 50_021)
    history.to_csv(out / "prometheus_training_history.csv", index=False)
    training_diagnostics["feature_columns"] = list(features)
    training_diagnostics["dm77_need_level_in_ranker_features"] = False
    training_diagnostics["dm77_need_level_used_for_pair_construction"] = False
    write_json(out / "prometheus_training_diagnostics.json", training_diagnostics)
    validation_score = _score_observed(models, validation)
    calibrator = _fit_calibrator(model_config["training_mode"], validation_score, validation)
    validation_cate = np.empty(len(validation.x), dtype=float)
    for transition_index, name in enumerate(TRANSITION_NAMES):
        mask = validation.transition_index == transition_index
        _, mu0, mu1 = predictors[name].predict(validation.x[mask])
        validation_cate[mask] = mu1 - mu0
    cate_calibrator = TransitionwiseCalibrator(TRANSITION_NAMES).fit(
        validation_cate, validation.signal, validation.transition_index
    )
    validation_frame = validation.to_frame(TRANSITION_NAMES)
    validation_frame["raw_score"] = validation_score
    validation_frame["calibrated_benefit"] = _calibrate(
        calibrator, validation_score, validation.transition_index
    )
    validation_frame.to_csv(out / "validation_opportunities.csv", index=False)
    calibration_diagnostics = {
        "prometheus_score_calibration": calibrator.diagnostics,
        "baseline_transition_cate_calibration": cate_calibrator.diagnostics,
    }
    write_json(out / "calibration_diagnostics.json", calibration_diagnostics)

    test_opportunities = _build_test_opportunities(
        simulation.learner, features, predictors, models, calibrator, config
    )
    test_opportunities["transition_cate_calibrated_baseline"] = cate_calibrator.predict(
        test_opportunities.transition_cate_baseline.to_numpy(float),
        test_opportunities.transition_index.to_numpy(int),
    )
    learned = _allocation(test_opportunities, config)
    test_opportunities["selected"] = learned.selected
    test_opportunities = test_opportunities.merge(learned.final_packages, on="patient_id", validate="many_to_one")
    dm77_operational_columns = [
        "patient_id", "dm77_need_level", "dm77_need_label", "dm77_assessment_status",
        "dm77_needs_multidimensional_assessment", "dm77_protected_pathway",
    ]
    test_opportunities = test_opportunities.merge(
        dm77_assessments.loc[:, dm77_operational_columns],
        on="patient_id", validate="many_to_one",
    )
    routed_outside_standard = (
        test_opportunities.dm77_protected_pathway.to_numpy(bool)
        | test_opportunities.dm77_assessment_status.eq("manual_review").to_numpy(bool)
    )
    test_opportunities["final_package"] = test_opportunities.final_package.astype("Int64")
    test_opportunities.loc[routed_outside_standard, "final_package"] = pd.NA
    test_opportunities["recommended_pathway"] = np.select(
        [
            test_opportunities.dm77_protected_pathway.to_numpy(bool),
            test_opportunities.dm77_assessment_status.eq("manual_review").to_numpy(bool),
        ],
        ["protected_palliative_pathway", "manual_multidimensional_review"],
        default="standard_causal_allocation",
    )
    operational_columns = [
        "patient_id", "transition", "transition_index", "raw_score", "calibrated_benefit",
        "cost", "eligibility", "support", "support_propensity", "selected", "final_package",
        "dm77_need_level", "dm77_need_label", "dm77_assessment_status",
        "dm77_needs_multidimensional_assessment", "dm77_protected_pathway",
        "recommended_pathway",
    ]
    test_opportunities.loc[:, operational_columns].to_csv(out / "test_global_opportunities.csv", index=False)
    test_opportunities.loc[:, operational_columns].to_csv(out / "test_global_allocation.csv", index=False)
    test_opportunities.loc[:, [
        "patient_id", "final_package", "dm77_need_level", "dm77_need_label",
        "dm77_assessment_status", "dm77_protected_pathway", "recommended_pathway",
    ]].drop_duplicates("patient_id").to_csv(out / "patient_final_packages.csv", index=False)

    # Evaluation starts here. Oracle truth is joined only after model, calibration,
    # support flags, and learned allocation have been fixed.
    truth = simulation.ground_truth.set_index("patient_id")
    evaluation = test_opportunities.copy()
    evaluation["true_benefit"] = [
        float(truth.loc[patient, f"true_benefit_{transition}"])
        for patient, transition in zip(evaluation.patient_id, evaluation.transition)
    ]
    evaluation.to_csv(out / "evaluation_test_global_opportunities.csv", index=False)
    oracle_frame = evaluation.copy()
    oracle_frame["calibrated_benefit"] = oracle_frame.true_benefit
    oracle = _allocation(oracle_frame, config)

    test_patient_features = simulation.learner.set_index("patient_id").loc[
        evaluation.patient_id.astype(str)
    ]
    risk_priority = (
        0.25 * test_patient_features.age.to_numpy(float)
        + 6.0 * test_patient_features.condition_distinct.to_numpy(float)
        + 10.0 * test_patient_features.prior_inpatient.to_numpy(float)
        + 6.0 * test_patient_features.prior_emergency.to_numpy(float)
        + 2.0 * test_patient_features.medication_distinct.to_numpy(float)
    )
    morbidity_priority = (
        test_patient_features.condition_distinct.to_numpy(float)
        + 0.5 * test_patient_features.medication_distinct.to_numpy(float)
    )
    random_priority = np.random.default_rng(seed + 90_001).uniform(size=len(evaluation))
    baselines = {
        "random_feasible_allocation": _allocation(_priority_baseline(evaluation, random_priority), config),
        "prognostic_risk_allocation": _allocation(_priority_baseline(evaluation, risk_priority), config),
        "morbidity_burden_allocation": _allocation(_priority_baseline(evaluation, morbidity_priority), config),
        "transition_cate_calibrated_global_optimization": _allocation(
            evaluation.assign(
                calibrated_benefit=evaluation.transition_cate_calibrated_baseline.to_numpy(float)
            ),
            config,
        ),
    }
    concordance_frame = evaluation.loc[evaluation.eligible.to_numpy(bool)].reset_index(drop=True)
    global_metrics = global_pairwise_concordance(
        concordance_frame.raw_score, concordance_frame.true_benefit,
        concordance_frame.transition_index, seed=seed,
    )
    global_metrics["global_pooled_autoc"] = pooled_opportunity_autoc(
        concordance_frame.raw_score, concordance_frame.true_benefit
    )
    global_metrics.update(allocation_value(evaluation, learned.selected))
    global_metrics.update(compare_allocation_to_oracle(
        evaluation, learned.selected, oracle.selected,
        baselines["random_feasible_allocation"].selected,
    ))
    transition_metrics = transition_ranking_metrics(concordance_frame)
    write_json(out / "global_metrics.json", global_metrics)
    write_json(out / "transition_metrics.json", transition_metrics)
    write_json(out / "allocation_diagnostics.json", {
        "learned": learned.diagnostics,
        "oracle_evaluation_only": oracle.diagnostics,
        "baselines": {name: value.diagnostics for name, value in baselines.items()},
    })

    policy_rows = []
    for name, result in {
        model_config["training_mode"]: learned,
        **baselines,
        "oracle_allocation_evaluation_only": oracle,
    }.items():
        values = allocation_value(evaluation, result.selected)
        policy_rows.append({"policy": name, **values, "shared_budget": float(config["allocation"]["shared_budget"])})
    policy_evaluation = pd.DataFrame(policy_rows)
    policy_evaluation.to_csv(out / "policy_evaluation.csv", index=False)

    if model_config["training_mode"] == "independent_local_rankers":
        torch.save({
            "method_version": METHOD_VERSION,
            "training_mode": model_config["training_mode"],
            "per_transition": {
                name: {
                    "state_dict": model.model.state_dict(),
                    "feature_mean": model.scaler.mean_,
                    "feature_scale": model.scaler.scale_,
                    "hidden_dim": model.hidden_dim,
                    "latent_dim": model.latent_dim,
                    "training_diagnostics": model.training_diagnostics,
                }
                for name, model in models.items()
            },
        }, out / "prometheus_global_ranker.pt")
    else:
        models[TRANSITION_NAMES[0]].save(out / "prometheus_global_ranker.pt")
    manifest = {
        "run_id": run_id,
        "method_version": METHOD_VERSION,
        "model_name": "PROMETHEUS",
        "model_variant": model_config["training_mode"],
        "ranking_unit": "patient_transition_opportunity",
        "cross_transition_pairs": model_config["training_mode"] not in LOCAL_MODES,
        "score_semantics": "globally_comparable_ordinal_priority" if model_config["training_mode"] not in LOCAL_MODES else "ordinal_within_transition",
        "historical_treatment": "single_multivalued_care_package",
        "outcome": "365_day_hospital_free_days",
        "allocation": "joint_shared_budget_constrained_optimization",
        "data_source": (
            synthetic_result.metadata["data_source"]
            if synthetic_result is not None else "cohort_cache"
        ),
        "source_population_fully_synthetic": bool(synthetic_result is not None),
        "real_patient_records_used": False if synthetic_result is not None else "unknown",
        "synthetic_generator_version": (
            synthetic_result.metadata["generator_version"]
            if synthetic_result is not None else None
        ),
        "synthetic_history_months": (
            synthetic_result.metadata["history_months"]
            if synthetic_result is not None else None
        ),
        "synthetic_population_scenario": population_scenario,
        "synthetic_validation_critical_failures": synthetic_validation_report["critical_failures"],
        "synthetic_validation_warnings": synthetic_validation_report["warnings"],
        "synthetic_italian_population_representativeness_established": False,
        "formal_privacy_guarantee_claimed": False,
        "dm77_assessment_enabled": True,
        "dm77_assessment_version": dm77_audit["assessment_version"],
        "dm77_operationalization_status": dm77_audit["operationalization_status"],
        "dm77_level_used_as_historical_treatment": False,
        "dm77_level_used_as_ranker_feature": False,
        "dm77_level_used_for_pair_construction": False,
        "dm77_level_vi_pathway": "protected_palliative_pathway",
        "dm77_manual_review_policy": "abstain_from_level_and_standard_allocation",
        "dm77_protected_level_vi_patients": dm77_audit["protected_level_vi_patients"],
        "dm77_manual_review_patients": dm77_audit["manual_review_patients"],
        "oracle_used_for_training": False,
        "oracle_used_for_pair_construction": False,
        "oracle_used_for_calibration": False,
        "oracle_used_for_allocation": False,
        "oracle_table_join_stage": "evaluation_after_learned_allocation_fixed",
        "nuisance_protocol": "repeated_strict_partition_patient_grouped_cross_fitting",
        "patient_splits": ["nuisance_train", "rank_train", "validation", "test"],
        "common_signal_scale": "days_alive_and_outside_acute_hospitalization",
        "transitionwise_signal_standardization": False,
        "sample_size": int(len(simulation.learner)),
        "seed": seed,
        "scenario": run["scenario"],
        "duration_seconds": time.time() - started,
        "simulation_only": True,
    }
    write_json(out / "manifest.json", manifest)
    _write_report(
        out, manifest, global_metrics, policy_evaluation, learned.diagnostics,
        dm77_summary, dm77_audit,
    )
    return out


def run_global_prometheus(config: dict) -> Path:
    source = str(config.get("opportunity_source", "dm77_catalog"))
    if source == "dm77_catalog":
        from .dm77_integrated_runner import run_dm77_integrated_prometheus

        return run_dm77_integrated_prometheus(config)
    if source != "synthetic_legacy":
        raise ValueError("opportunity_source must be dm77_catalog or synthetic_legacy")
    return _run_synthetic_legacy_prometheus(config)


run_prometheus = run_global_prometheus
