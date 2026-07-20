"""DM 77 integrated patient-action causal-ranking runner."""

from __future__ import annotations

import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..allocation.action_allocator import (
    allocate_care_actions,
    allocate_care_actions_greedy,
    apply_action,
)
from ..data.validation import assert_no_oracle_columns
from ..dm77 import (
    assess_dm77_population,
    attach_current_care_state_features,
    build_patient_action_opportunities,
    derive_current_care_states,
    derive_dm77_action_eligibility,
    load_care_action_catalog,
    load_dm77_settings,
    summarize_dm77_population,
    validate_ranked_action_comparability,
)
from ..evaluation.global_metrics import (
    global_pairwise_concordance,
    hard_cross_action_concordance,
)
from ..evaluation.action_policy_metrics import (
    actionwise_off_policy_metrics,
    allocation_budget_metrics,
    budget_curve_area,
    linear_calibration_metrics,
    normalized_oracle_efficiency,
    rank_weighted_effect_metrics,
    score_group_diagnostics,
    synthetic_top_q_metrics,
)
from ..nuisance.cross_fitting import fit_repeated_partitioned_nuisance
from ..ranking.action_calibration import fit_action_calibrators
from ..ranking.causal_signals import (
    dr_signal_diagnostics,
    repeated_causal_signals,
    robustify_repeated_signals,
)
from ..ranking.tree_baselines import (
    DirectPairwiseGBDTRanker,
    DRGBDTPriority,
    DRPolicyTreePriority,
    DRRandomForestPriority,
)
from ..ranking.opportunities import OpportunityArrays, concatenate_opportunities
from ..reproducibility import seed_everything, write_json
from ..version import PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION
from .action_simulation import (
    ACTION_DGP_SCENARIOS,
    NO_ACTION,
    simulate_observational_care_actions,
)
from .global_prometheus_runner import (
    _load_global_population,
    _repeated_nuisance_tensor,
    _score_observed,
    _train_rankers,
)
from .global_support import TransitionSupportOutcomePredictor


METHOD_VERSION = PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION
DGP_RUN_SLUGS = {
    "baseline_identifiable": "bi",
    "strong_observed_confounding": "soc",
    "poor_overlap": "po",
    "risk_benefit_misalignment": "rbm",
    "targeted_selection_observed": "tso",
    "hidden_confounding": "hc",
    "combined_stress": "cs",
    "null_treatment_effect": "nte",
    "placebo_outcome": "plc",
}
REQUIRED_CALIBRATION_METHODS = {
    "pooled_isotonic",
    "reliability_weighted_pooled_isotonic",
    "monotonic_binned",
    "action_mean_shrinkage",
}
REQUIRED_BASELINES = {
    "random_priority",
    "risk_priority",
    "action_mean_priority",
    "independent_action_rankers",
    "unified_local_ranker",
    "unified_global_ranker",
    "oracle_priority",
    "dm77_need_fixed_action_rule",
    "historical_action_propensity",
    "cost_normalized_risk_priority",
    "direct_pairwise_gbdt_ranker",
    "pooled_dr_gbdt",
    "independent_dr_gbdt",
    "dr_random_forest_priority",
    "dr_policy_tree",
}
PRIMARY_RANKING_METHODS = {
    "unified_global_ranker",
    "prometheus_global_causal_contrastive",
    "prometheus_global_causal_contrastive_v2",
}


def _validate_config(config: dict) -> dict:
    required = {
        "run", "data", "dm77", "care", "simulation", "nuisance",
        "prometheus", "allocation", "causal_signal", "calibration",
        "checkpoint", "baselines",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing integrated DM 77 config sections: {missing}")
    if config.get("opportunity_source") != "dm77_catalog":
        raise ValueError("Integrated runner requires opportunity_source=dm77_catalog")
    if config["data"].get("source") != "fully_synthetic":
        raise ValueError("Integrated v1 currently requires data.source=fully_synthetic")
    if int(config["run"].get("sample_size", 0)) < 500:
        raise ValueError("Integrated action smoke requires at least 500 synthetic patients")
    if int(config["nuisance"].get("folds", 0)) < 2 or int(config["nuisance"].get("repeats", 0)) < 1:
        raise ValueError("Integrated nuisance folds/repeats are invalid")
    scenario = str(config["simulation"].get("scenario", "baseline_identifiable"))
    if scenario not in ACTION_DGP_SCENARIOS:
        raise ValueError(f"Unknown integrated action-DGP scenario {scenario!r}")
    causal_signal = config["causal_signal"]
    if causal_signal.get("estimator", "dr") not in {
        "dr", "outcome_regression", "ipw", "naive_outcome",
    }:
        raise ValueError("Unknown causal_signal.estimator")
    if causal_signal.get("aggregation") not in {"mean", "median"}:
        raise ValueError("causal_signal.aggregation must be mean or median")
    quantiles = causal_signal.get("winsorize_quantiles", ())
    if len(quantiles) != 2 or not 0 <= float(quantiles[0]) < float(quantiles[1]) <= 1:
        raise ValueError("Invalid causal_signal.winsorize_quantiles")
    if not 0 <= float(causal_signal.get("min_direction_agreement", -1)) <= 1:
        raise ValueError("Invalid causal_signal.min_direction_agreement")
    model = config["prometheus"]
    model["dr_signal_aggregation"] = causal_signal["aggregation"]
    model["min_pair_direction_agreement"] = causal_signal["min_direction_agreement"]
    model["pair_reliability_weighting"] = {
        **model.get("pair_reliability_weighting", {}),
        "enabled": bool(causal_signal.get("reliability_weighting", True)),
    }
    contrastive = config.get("contrastive", {})
    enabled = bool(contrastive.get("enabled", False))
    if not enabled:
        if model.get("training_mode") != "unified_global_ranker" or float(model.get("lambda_con", 0)) != 0:
            raise ValueError("Integrated baseline requires unified_global_ranker and lambda_con=0")
    else:
        training_mode = str(model.get("training_mode"))
        if training_mode not in {
            "prometheus_global_causal_contrastive",
            "prometheus_global_causal_contrastive_v2",
        }:
            raise ValueError(
                "contrastive.enabled requires a supported causal contrastive mode"
            )
        if float(model.get("lambda_con", 0)) <= 0 or int(
            model.get("contrastive_pairs_per_epoch", 0)
        ) < 1:
            raise ValueError(
                "Enabled causal contrastive training requires positive lambda_con and pair budget"
            )
        required_scope = (
            "balanced_mixed"
            if training_mode == "prometheus_global_causal_contrastive_v2"
            else "within_action"
        )
        if contrastive.get("pair_scope") != required_scope:
            raise ValueError(
                f"Integrated {training_mode} requires {required_scope} scope"
            )
        if not bool(contrastive.get("require_stable_direction", True)):
            raise ValueError("Integrated causal contrastive mode requires stable directions")
        if not bool(contrastive.get("require_overlap_support", True)):
            raise ValueError("Integrated causal contrastive mode requires overlap support")
        if (
            training_mode == "prometheus_global_causal_contrastive_v2"
            and not bool(contrastive.get("reliability_weighting", False))
        ):
            raise ValueError(
                "Integrated causal contrastive v2 requires reliability weighting"
            )
        model["contrastive_pair_scope"] = required_scope
        model["contrastive_require_stable_direction"] = True
        model["contrastive_reliability_weighting"] = bool(
            contrastive.get("reliability_weighting", False)
        )
    if int(config["allocation"].get("max_primary_actions_per_patient", 0)) < 1:
        raise ValueError("allocation.max_primary_actions_per_patient must be positive")
    if config["allocation"].get("value_mode", "calibrated_benefit") not in {
        "calibrated_benefit", "equal_cost_rank_only",
    }:
        raise ValueError("Unknown allocation.value_mode")
    calibration = config["calibration"]
    configured_calibrations = set(map(str, calibration.get("methods", ())))
    if calibration.get("selected_method") not in configured_calibrations:
        raise ValueError("Selected calibration method must be included in methods")
    if not REQUIRED_CALIBRATION_METHODS.issubset(configured_calibrations):
        missing_methods = sorted(REQUIRED_CALIBRATION_METHODS - configured_calibrations)
        raise ValueError(
            f"Integrated validation requires all calibration methods: {missing_methods}"
        )
    configured_baselines = set(map(str, config["baselines"].get("enabled", ())))
    if not REQUIRED_BASELINES.issubset(configured_baselines):
        missing_baselines = sorted(REQUIRED_BASELINES - configured_baselines)
        raise ValueError(
            f"Integrated validation requires all baseline methods: {missing_baselines}"
        )
    if config["checkpoint"].get("metric") not in {
        "global_concordance", "dr_policy_value",
    }:
        raise ValueError("checkpoint.metric must be global_concordance or dr_policy_value")
    evaluation = config.get("evaluation", {})
    top_quantiles = tuple(evaluation.get("top_quantiles", (0.05, 0.10, 0.20)))
    if not top_quantiles or any(not 0.0 < float(value) < 1.0 for value in top_quantiles):
        raise ValueError("evaluation.top_quantiles must contain fractions in (0,1)")
    if int(evaluation.get("calibration_score_groups", 10)) < 2:
        raise ValueError("evaluation.calibration_score_groups must be at least 2")
    return config


def _primary_ranking_method(config: dict) -> str:
    method = str(config["prometheus"]["training_mode"])
    if method not in PRIMARY_RANKING_METHODS:
        raise ValueError(f"Unsupported integrated primary ranking method {method!r}")
    return method


def _build_observed_action_opportunities(
    prepared: dict,
    features: tuple[str, ...],
    split: str,
    ranked_actions,
    require_overlap_support: bool,
    support_bounds: tuple[float, float],
) -> OpportunityArrays:
    blocks = []
    for action in ranked_actions:
        item = prepared[action.action_id]
        frame = item["frame"]
        mask = frame.split.astype(str).to_numpy() == split
        if require_overlap_support:
            propensity = item["nuisance"].e_hat.to_numpy(float)
            mask &= (propensity >= support_bounds[0]) & (propensity <= support_bounds[1])
        positions = np.flatnonzero(mask)
        if not len(positions):
            raise RuntimeError(f"No {split} observations for action {action.action_id}")
        blocks.append(OpportunityArrays(
            frame.loc[positions, features].to_numpy(float),
            np.full(len(positions), action.action_index, dtype=np.int64),
            frame.loc[positions, "patient_id"].astype(str).to_numpy(),
            item["signal"][positions],
            item["repeated_signal"][positions],
            positions,
        ))
    return concatenate_opportunities(blocks)


def _build_observed_action_attribute(
    prepared: dict,
    split: str,
    ranked_actions,
    key: str,
    require_overlap_support: bool,
    support_bounds: tuple[float, float],
) -> np.ndarray:
    values = []
    for action in ranked_actions:
        item = prepared[action.action_id]
        frame = item["frame"]
        mask = frame.split.astype(str).to_numpy() == split
        if require_overlap_support:
            propensity = item["nuisance"].e_hat.to_numpy(float)
            mask &= (propensity >= support_bounds[0]) & (propensity <= support_bounds[1])
        values.append(np.asarray(item[key], dtype=float)[mask])
    return np.concatenate(values)


def _calibration_csv(diagnostics: dict) -> pd.DataFrame:
    keys = (
        "method", "fit_partition", "target", "oracle_target", "pooled",
        "out_of_bounds", "mae", "mse", "monotonicity_check", "unique_score_points",
    )
    return pd.DataFrame([{key: diagnostics.get(key) for key in keys}])


def _risk_priority_model(training: OpportunityArrays, features: tuple[str, ...]):
    weights = {
        "age": 0.20,
        "condition_distinct": 0.70,
        "prior_inpatient": 0.60,
        "prior_emergency": 0.45,
        "medication_distinct": 0.25,
        "frailty_index": 0.30,
        "functional_limitation_score": 0.25,
    }
    selected = [(features.index(name), weight) for name, weight in weights.items() if name in features]
    mean = training.x.mean(axis=0)
    scale = training.x.std(axis=0) + 1e-8

    def score(x) -> np.ndarray:
        standardized = (np.asarray(x, dtype=float) - mean) / scale
        return sum(weight * standardized[:, index] for index, weight in selected)

    return score


def _dm77_need_fixed_action_score(need_level, action_index, action_count: int) -> np.ndarray:
    """Transparent non-causal baseline using need plus a fixed action rule."""

    need = np.asarray(need_level, dtype=float)
    action = np.asarray(action_index, dtype=int)
    if need.shape != action.shape or not np.isfinite(need).all():
        raise ValueError("DM77 fixed-rule baseline inputs must align and be finite")
    preferred = np.clip(np.rint(need).astype(int) - 1, 0, int(action_count) - 1)
    return 0.01 * need - np.abs(action - preferred).astype(float)


def _allocate_action_policy(
    frame: pd.DataFrame,
    score,
    cardinal_value,
    config: dict,
):
    candidate = frame.copy()
    score = np.asarray(score, dtype=float)
    cardinal = np.asarray(cardinal_value, dtype=float)
    value_mode = str(config["allocation"].get("value_mode", "calibrated_benefit"))
    if value_mode == "equal_cost_rank_only":
        candidate["calibrated_incremental_benefit"] = pd.Series(score).rank(
            ascending=True, method="first"
        ).to_numpy(float)
        candidate["cost"] = 1.0
    else:
        candidate["calibrated_incremental_benefit"] = cardinal
    result = allocate_care_actions(
        candidate,
        shared_budget=float(config["allocation"]["shared_budget"]),
        capacity_pools=config["allocation"].get("capacity_pools"),
        max_primary_actions_per_patient=int(
            config["allocation"]["max_primary_actions_per_patient"]
        ),
        require_empirical_support=bool(
            config["allocation"].get("require_empirical_support", True)
        ),
        allocate_negative_predicted_benefit=bool(
            config["allocation"].get("allocate_negative_predicted_benefit", False)
        ),
        solver_time_limit_seconds=float(
            config["allocation"].get("solver_time_limit_seconds", 120.0)
        ),
    )
    result.diagnostics["value_mode"] = value_mode
    return result


def _write_report(out: Path, manifest: dict, dm77_summary: pd.DataFrame, metrics: dict) -> None:
    level_rows = ", ".join(
        f"{row.dm77_need_level}:{row.patient_count}"
        for row in dm77_summary.dropna(subset=["dm77_need_level"]).itertuples()
    )
    text = f"""# PROMETHEUS DM 77 integrated global action ranking

Method version: `{manifest['method_version']}`

The need level describes multidimensional need. `current_care_state` describes services
active before the index date. `care_action_id` is the causal treatment. The network
ranks globally comparable patient-action opportunities; it does not learn or redefine
DM 77 levels and its raw score is ordinal, not an individual CATE.

DM 77 level counts: {level_rows}.

- Candidate actions: {manifest['candidate_actions']}
- Ranked test opportunities: {manifest['ranked_opportunities']}
- Valid cross-action training pairs: {manifest['cross_action_pairs']}
- Allocated discretionary actions: {manifest['allocated_actions']}
- Protected cases: {manifest['protected_cases']}
- Manual-review cases: {manifest['manual_review_cases']}
- Primary ranking method: {manifest['primary_ranking_method']}
- Contrastive configuration: {manifest['contrastive_configuration']}
- Action-DGP scenario: {manifest['action_dgp_scenario']}
- Primary synthetic oracle target: {manifest['primary_oracle_target']}
- Robust DR supervision: {manifest['ranking_signal']}
- Selected validation-only calibration: {manifest['selected_calibration_method']}
- Allocation value mode: {manifest['allocation_value_mode']}

Synthetic evaluation diagnostics: `{metrics}`.

This fully synthetic methodological run does not establish clinical effectiveness,
Italian-population validity, superiority over ACG, or deployment readiness. Protected,
mandatory, and manual-review pathways do not compete in discretionary causal ranking.
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def run_dm77_integrated_prometheus(config: dict) -> Path:
    config = _validate_config(config)
    started = time.time()
    seed = int(config["run"]["seed"])
    dataset_seed = int(config["run"].get("dataset_seed", seed))
    primary_ranking_method = _primary_ranking_method(config)
    seed_everything(seed)
    catalog = load_care_action_catalog(config["care"]["catalog"])
    comparability = validate_ranked_action_comparability(catalog)
    ranked_actions = catalog.ranked_actions
    if tuple(action.action_index for action in ranked_actions) != tuple(range(len(ranked_actions))):
        raise ValueError("Integrated v1 requires ranked actions before protected/mandatory actions")
    action_names = tuple(action.action_id for action in ranked_actions)
    population_scenario = str(config["data"]["synthetic"].get("scenario", "baseline"))
    action_dgp_scenario = str(config["simulation"].get("scenario", "baseline_identifiable"))
    action_dgp_slug = DGP_RUN_SLUGS[action_dgp_scenario]
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_dm77_{action_dgp_slug}_s{seed}"
        + (f"_d{dataset_seed}" if dataset_seed != seed else "")
    )
    out = Path(config["run"]["output_root"]) / run_id
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "resolved_config.json", config)

    patients, synthetic_result, synthetic_validation, privacy_audit = _load_global_population(
        config["data"], int(config["run"]["sample_size"]), dataset_seed
    )
    if synthetic_result is None:
        raise AssertionError("Integrated v1 expected a fully synthetic source")
    synthetic_result.patients.to_csv(out / "synthetic_patient_features.csv", index=False)
    if bool(config["data"]["synthetic"].get("write_monthly_history", True)):
        synthetic_result.monthly_history.to_csv(out / "synthetic_monthly_history.csv", index=False)
    checks, validation_report = synthetic_validation
    checks.to_csv(out / "synthetic_validation_checks.csv", index=False)
    write_json(out / "synthetic_validation_report.json", validation_report)
    write_json(out / "synthetic_generation_metadata.json", synthetic_result.metadata)
    write_json(out / "synthetic_privacy_audit.json", privacy_audit)

    dm77_settings = load_dm77_settings(config["dm77"])
    assessments, dm77_audit = assess_dm77_population(patients, dm77_settings)
    dm77_summary = summarize_dm77_population(assessments)
    current_states = derive_current_care_states(
        patients, catalog,
        dataset_seed + int(config["care"].get("current_state_seed_offset", 31_337))
    )
    patient_context, care_state_features = attach_current_care_state_features(
        patients, current_states, catalog
    )
    action_eligibility = derive_dm77_action_eligibility(
        patients, assessments, current_states, catalog
    )
    authoritative_candidates = build_patient_action_opportunities(action_eligibility)
    assessments.to_csv(out / "dm77_need_assessment.csv", index=False)
    dm77_summary.to_csv(out / "dm77_population_summary.csv", index=False)
    write_json(out / "dm77_assessment_audit.json", dm77_audit)
    current_states.to_csv(out / "patient_current_care_state.csv", index=False)
    action_eligibility.to_csv(out / "dm77_action_eligibility.csv", index=False)
    authoritative_candidates.to_csv(out / "patient_action_opportunities.csv", index=False)
    protected_manual = assessments.loc[
        assessments.dm77_protected_pathway.to_numpy(bool)
        | assessments.dm77_assessment_status.eq("manual_review").to_numpy(bool)
    ].merge(current_states, on="patient_id", validate="one_to_one")
    protected_manual.to_csv(out / "protected_and_manual_review_cases.csv", index=False)

    simulation = simulate_observational_care_actions(
        patient_context,
        action_eligibility,
        catalog,
        seed=dataset_seed + int(config["simulation"].get("action_dgp_seed_offset", 70_001)),
        outcome_noise_sd=float(config["simulation"].get("outcome_noise_sd", 6.0)),
        minimum_action_assignments_per_split=int(
            config["simulation"].get("minimum_action_assignments_per_split", 6)
        ),
        scenario=str(config["simulation"].get("scenario", "baseline_identifiable")),
    )
    assert_no_oracle_columns(simulation.learner)
    if "dm77_need_level" in simulation.feature_columns:
        raise AssertionError("DM 77 need level entered action-ranker features")
    simulation.learner.to_csv(out / "action_observational_learner.csv", index=False)
    simulation.ground_truth.to_csv(out / "evaluation_action_ground_truth.csv", index=False)
    write_json(out / "action_simulation_metadata.json", simulation.metadata)

    exclude_current_care_features = bool(
        config.get("ablation", {}).get("exclude_current_care_state_features", False)
    )
    features = tuple(
        feature for feature in simulation.feature_columns
        if not (exclude_current_care_features and feature in set(care_state_features))
    )
    if not features:
        raise ValueError("Current-care ablation removed every learner feature")
    nuisance_dir = out / "action_nuisance_predictions"
    nuisance_dir.mkdir()
    prepared, predictors, signal_tables, signal_diagnostics = {}, {}, [], []
    epsilon = float(config["prometheus"]["propensity_clip_epsilon"])
    for local_index, action in enumerate(ranked_actions):
        learner = simulation.action_learners[action.action_id].copy()
        assert_no_oracle_columns(learner)
        nuisance, repetitions, diagnostics = fit_repeated_partitioned_nuisance(
            learner,
            list(features),
            int(config["nuisance"]["folds"]),
            config["nuisance"]["model"],
            seed + 1009 * local_index,
            int(config["nuisance"]["repeats"]),
            int(config["nuisance"].get("repeat_seed_stride", 1009)),
            "nuisance_train",
            (epsilon, 1.0 - epsilon),
        )
        nuisance.to_csv(nuisance_dir / f"{action.action_id}.csv", index=False)
        pd.concat(repetitions, ignore_index=True).to_csv(
            nuisance_dir / f"{action.action_id}_repeated.csv", index=False
        )
        write_json(nuisance_dir / f"{action.action_id}_diagnostics.json", diagnostics)
        tensor = _repeated_nuisance_tensor(repetitions, learner.patient_id)
        raw_repeated_signal = repeated_causal_signals(
            learner.transition_treatment.to_numpy(float),
            learner.observed_outcome.to_numpy(float),
            tensor,
            epsilon,
            estimator=str(config["causal_signal"].get("estimator", "dr")),
        )
        robust_signal = robustify_repeated_signals(
            raw_repeated_signal,
            learner.split.astype(str).to_numpy(),
            aggregation=str(config["causal_signal"]["aggregation"]),
            winsorize=bool(config["causal_signal"].get("winsorize", True)),
            winsorize_quantiles=config["causal_signal"].get(
                "winsorize_quantiles", (0.01, 0.99)
            ),
        )
        prepared[action.action_id] = {
            "frame": learner,
            "nuisance": nuisance,
            "signal": robust_signal.robust_aggregate,
            "repeated_signal": robust_signal.robust_repeated,
            "raw_signal": robust_signal.raw_aggregate,
            "raw_repeated_signal": raw_repeated_signal,
            "reliability_weight": robust_signal.reliability_weight,
            "propensity": nuisance.e_hat.to_numpy(float),
            "winsor_bounds": robust_signal.winsor_bounds,
        }
        predictors[action.action_id] = TransitionSupportOutcomePredictor(
            config["nuisance"]["model"], seed + 20_011 + local_index
        ).fit(learner, features)
        signal_frame = pd.DataFrame({
            "patient_id": learner.patient_id.astype(str),
            "action_id": action.action_id,
            "transition_index": action.action_index,
            "split": learner.split.astype(str),
            "action_treatment": learner.transition_treatment.to_numpy(int),
            "dr_signal_raw_days": robust_signal.raw_aggregate,
            "dr_signal_robust_days": robust_signal.robust_aggregate,
            "dr_reliability_weight": robust_signal.reliability_weight,
            "e_hat": nuisance.e_hat.to_numpy(float),
            "causal_signal_estimator": str(
                config["causal_signal"].get("estimator", "dr")
            ),
        })
        for repeat in range(raw_repeated_signal.shape[1]):
            signal_frame[f"dr_signal_raw_repeat_{repeat}"] = raw_repeated_signal[:, repeat]
            signal_frame[f"dr_signal_robust_repeat_{repeat}"] = (
                robust_signal.robust_repeated[:, repeat]
            )
        for split_name, group in signal_frame.groupby("split", sort=True):
            positions = group.index.to_numpy(int)
            signal_diagnostics.append(dr_signal_diagnostics(
                robust_signal.robust_aggregate[positions],
                nuisance.e_hat.to_numpy(float)[positions],
                learner.transition_treatment.to_numpy(int)[positions],
                action.action_id,
                str(split_name),
            ))
        signal_tables.append(signal_frame)
    pd.concat(signal_tables, ignore_index=True).to_csv(
        out / "action_specific_dr_signals.csv", index=False
    )
    pd.DataFrame(signal_diagnostics).to_csv(
        out / "action_specific_dr_diagnostics.csv", index=False
    )

    support_cfg = config["allocation"]["propensity_support"]
    support_bounds = (float(support_cfg["lower"]), float(support_cfg["upper"]))
    contrastive = config.get("contrastive", {})
    require_training_support = bool(
        contrastive.get("enabled", False)
        and contrastive.get("require_overlap_support", True)
    )
    training = _build_observed_action_opportunities(
        prepared, features, "rank_train", ranked_actions,
        require_training_support, support_bounds,
    )
    validation = _build_observed_action_opportunities(
        prepared, features, "validation", ranked_actions,
        require_training_support, support_bounds,
    )
    training.to_frame(action_names).rename(columns={"transition": "action_id"}).to_csv(
        out / "rank_training_action_opportunities.csv", index=False
    )
    models, history, training_diagnostics = _train_rankers(
        config, training, validation, seed + 50_021, transition_names=action_names
    )
    enabled_baselines = tuple(map(str, config["baselines"].get("enabled", ())))
    neural_modes = {
        "independent_action_rankers": "independent_local_rankers",
        "unified_local_ranker": "unified_local_ranker",
        "unified_global_ranker": "unified_global_ranker",
    }
    baseline_models = {primary_ranking_method: models}
    baseline_histories, baseline_training_diagnostics = [], {}
    for baseline_name, mode in neural_modes.items():
        if baseline_name not in enabled_baselines or (
            mode == config["prometheus"]["training_mode"]
        ):
            continue
        baseline_config = copy.deepcopy(config)
        baseline_config["prometheus"]["training_mode"] = mode
        baseline_config["prometheus"]["lambda_con"] = 0.0
        baseline_seed = (
            seed + 50_021
            if mode == "unified_global_ranker"
            else seed + 60_031 + 10_007 * len(baseline_models)
        )
        fitted, fitted_history, fitted_diagnostics = _train_rankers(
            baseline_config,
            training,
            validation,
            baseline_seed,
            transition_names=action_names,
        )
        baseline_models[baseline_name] = fitted
        baseline_histories.append(fitted_history.assign(baseline=baseline_name))
        baseline_training_diagnostics[baseline_name] = fitted_diagnostics
    tree_settings = config["baselines"].get("settings", {})
    action_count = len(ranked_actions)
    tree_baseline_models = {}
    if "direct_pairwise_gbdt_ranker" in enabled_baselines:
        settings = tree_settings.get("direct_pairwise_gbdt_ranker", {})
        tree_baseline_models["direct_pairwise_gbdt_ranker"] = DirectPairwiseGBDTRanker(
            action_count=action_count,
            within_pairs=int(settings.get("within_pairs", 4_000)),
            cross_pairs=int(settings.get("cross_pairs", 4_000)),
            min_signal_gap_days=float(settings.get(
                "min_signal_gap_days", config["prometheus"]["min_signal_gap_days"]
            )),
            min_direction_agreement=float(settings.get(
                "min_direction_agreement", config["causal_signal"]["min_direction_agreement"]
            )),
            anchor_count=int(settings.get("anchor_count", 128)),
            max_iter=int(settings.get("max_iter", 80)),
            max_leaf_nodes=int(settings.get("max_leaf_nodes", 31)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            seed=seed + 120_011,
        ).fit(training)
    for name, pooled in (("pooled_dr_gbdt", True), ("independent_dr_gbdt", False)):
        if name in enabled_baselines:
            settings = tree_settings.get(name, {})
            tree_baseline_models[name] = DRGBDTPriority(
                action_count=action_count,
                pooled=pooled,
                max_iter=int(settings.get("max_iter", 80)),
                max_leaf_nodes=int(settings.get("max_leaf_nodes", 31)),
                learning_rate=float(settings.get("learning_rate", 0.05)),
                seed=seed + (130_013 if pooled else 140_009),
            ).fit(training)
    if "dr_random_forest_priority" in enabled_baselines:
        settings = tree_settings.get("dr_random_forest_priority", {})
        tree_baseline_models["dr_random_forest_priority"] = DRRandomForestPriority(
            action_count=action_count,
            estimators=int(settings.get("estimators", 120)),
            min_samples_leaf=int(settings.get("min_samples_leaf", 20)),
            seed=seed + 150_001,
        ).fit(training)
    if "dr_policy_tree" in enabled_baselines:
        settings = tree_settings.get("dr_policy_tree", {})
        tree_baseline_models["dr_policy_tree"] = DRPolicyTreePriority(
            action_count=action_count,
            max_depth=int(settings.get("max_depth", 3)),
            min_samples_leaf=int(settings.get("min_samples_leaf", 30)),
            thresholds_per_feature=int(settings.get("thresholds_per_feature", 7)),
            seed=seed + 160_001,
        ).fit(training)
    baseline_training_diagnostics.update({
        name: model.diagnostics for name, model in tree_baseline_models.items()
    })
    training_diagnostics.update({
        "primary_ranking_method": primary_ranking_method,
        "ranking_unit": "patient_action_opportunity",
        "cross_action_pairs": training_diagnostics["cross_transition_pairs"],
        "dm77_need_level_in_ranker_features": False,
        "current_care_state_one_hot_features": [
            feature for feature in care_state_features if feature in features
        ],
        "current_care_state_features_excluded_ablation": exclude_current_care_features,
        "feature_columns": list(features),
        "action_id_to_transition_index": {
            action.action_id: action.action_index for action in ranked_actions
        },
        "outcome_comparability": comparability,
    })
    history.rename(columns={
        "valid_cross_pairs": "valid_cross_action_pairs",
        "valid_within_pairs": "valid_within_action_pairs",
    }).to_csv(out / "prometheus_training_history.csv", index=False)
    if baseline_histories:
        pd.concat(baseline_histories, ignore_index=True).to_csv(
            out / "baseline_training_history.csv", index=False
        )
    write_json(
        out / "baseline_training_diagnostics.json", baseline_training_diagnostics
    )
    write_json(out / "prometheus_training_diagnostics.json", training_diagnostics)
    validation_score = _score_observed(
        models, validation, transition_names=action_names
    )
    validation_reliability = _build_observed_action_attribute(
        prepared, "validation", ranked_actions, "reliability_weight",
        require_training_support, support_bounds,
    )
    calibration_config = config["calibration"]
    evaluation_config = config.get("evaluation", {})
    top_quantiles = tuple(map(float, evaluation_config.get(
        "top_quantiles", (0.05, 0.10, 0.20)
    )))
    calibration_score_groups = int(
        evaluation_config.get("calibration_score_groups", 10)
    )
    calibrators = fit_action_calibrators(
        validation_score,
        validation.signal,
        validation.transition_index,
        validation_reliability,
        calibration_config["methods"],
        int(calibration_config.get("minimum_bin_size", 25)),
        float(calibration_config.get("shrinkage_pooled_weight", 0.70)),
    )
    selected_calibration_method = str(calibration_config["selected_method"])
    calibrator = calibrators[selected_calibration_method]
    calibration_group_rows = []
    for method, fitted_calibrator in calibrators.items():
        validation_prediction = fitted_calibrator.predict(
            validation_score, validation.transition_index
        )
        calibration_group_rows.append(score_group_diagnostics(
            validation_score,
            validation_prediction,
            validation.signal,
            method=method,
            partition="validation",
            target_name="heldout_robust_doubly_robust_signal_days",
            oracle_target=False,
            groups=calibration_score_groups,
        ))
    validation_scores = {primary_ranking_method: validation_score}
    for baseline_name, fitted in baseline_models.items():
        if baseline_name == primary_ranking_method:
            continue
        validation_scores[baseline_name] = _score_observed(
            fitted, validation, transition_names=action_names
        )
    for baseline_name, fitted in tree_baseline_models.items():
        validation_scores[baseline_name] = fitted.predict_opportunity_scores(
            validation.x, validation.transition_index
        )
    risk_score_model = _risk_priority_model(training, features)
    action_mean_by_index = {
        int(index): float(np.median(training.signal[training.transition_index == index]))
        for index in np.unique(training.transition_index)
    }
    validation_scores["risk_priority"] = risk_score_model(validation.x)
    validation_scores["action_mean_priority"] = np.asarray([
        action_mean_by_index[int(index)] for index in validation.transition_index
    ])
    need_by_patient = assessments.assign(
        patient_id=assessments.patient_id.astype(str)
    ).set_index("patient_id").dm77_need_level
    validation_need = need_by_patient.loc[
        pd.Index(validation.patient_ids.astype(str))
    ].to_numpy(float)
    action_cost_by_index = {
        int(action.action_index): float(action.cost) for action in ranked_actions
    }
    validation_cost = np.asarray([
        action_cost_by_index[int(index)] for index in validation.transition_index
    ])
    validation_propensity = _build_observed_action_attribute(
        prepared, "validation", ranked_actions, "propensity",
        require_training_support, support_bounds,
    )
    validation_scores["dm77_need_fixed_action_rule"] = _dm77_need_fixed_action_score(
        validation_need, validation.transition_index, len(ranked_actions)
    )
    validation_scores["historical_action_propensity"] = validation_propensity
    validation_scores["cost_normalized_risk_priority"] = (
        validation_scores["risk_priority"] / validation_cost
    )
    validation_scores["random_priority"] = np.random.default_rng(
        seed + 81_019
    ).uniform(size=len(validation.x))
    baseline_calibrators = {primary_ranking_method: calibrator}
    for baseline_name, score in validation_scores.items():
        if baseline_name == primary_ranking_method:
            continue
        baseline_calibrators[baseline_name] = fit_action_calibrators(
            score,
            validation.signal,
            validation.transition_index,
            validation_reliability,
            [selected_calibration_method],
            int(calibration_config.get("minimum_bin_size", 25)),
            float(calibration_config.get("shrinkage_pooled_weight", 0.70)),
        )[selected_calibration_method]

    split_by_patient = simulation.learner.set_index("patient_id").split.astype(str)
    test_candidates = authoritative_candidates.loc[
        authoritative_candidates.patient_id.astype(str).map(split_by_patient).eq("test")
    ].copy()
    test_context = simulation.learner.loc[
        simulation.learner.split.astype(str) == "test", ["patient_id", *features]
    ]
    test_candidates = test_candidates.merge(
        test_context, on="patient_id", validate="many_to_one"
    )
    ranked_parts = []
    for action in ranked_actions:
        group = test_candidates.loc[test_candidates.action_id == action.action_id].copy()
        if group.empty:
            continue
        x = group.loc[:, features].to_numpy(float)
        propensity, _, _ = predictors[action.action_id].predict(x)
        indices = np.full(len(group), action.action_index, dtype=np.int64)
        for baseline_name, fitted in baseline_models.items():
            model = fitted[action.action_id]
            model_indices = (
                np.zeros(len(group), dtype=np.int64)
                if len(model.transition_names) == 1 else indices
            )
            group[f"score__{baseline_name}"] = model.predict_opportunity_scores(
                x, model_indices
            )
        group["support_propensity"] = propensity
        group["support_flag"] = (
            (propensity >= support_bounds[0]) & (propensity <= support_bounds[1])
        )
        ranked_parts.append(group)
    if not ranked_parts:
        raise RuntimeError("No test patient-action opportunities were generated")
    ranking = pd.concat(ranked_parts, ignore_index=True)
    for baseline_name, fitted in tree_baseline_models.items():
        ranking[f"score__{baseline_name}"] = fitted.predict_opportunity_scores(
            ranking.loc[:, features].to_numpy(float),
            ranking.transition_index.to_numpy(np.int64),
        )
    ranking["score__risk_priority"] = risk_score_model(
        ranking.loc[:, features].to_numpy(float)
    )
    ranking["score__action_mean_priority"] = np.asarray([
        action_mean_by_index[int(index)] for index in ranking.transition_index
    ])
    ranking["score__random_priority"] = np.random.default_rng(
        seed + 91_021
    ).uniform(size=len(ranking))
    ranking["score__dm77_need_fixed_action_rule"] = _dm77_need_fixed_action_score(
        ranking.dm77_need_level.to_numpy(float),
        ranking.transition_index.to_numpy(int),
        len(ranked_actions),
    )
    ranking["score__historical_action_propensity"] = ranking.support_propensity.to_numpy(float)
    ranking["score__cost_normalized_risk_priority"] = (
        ranking["score__risk_priority"].to_numpy(float) / ranking.cost.to_numpy(float)
    )
    for baseline_name, fitted_calibrator in baseline_calibrators.items():
        score_column = f"score__{baseline_name}"
        if score_column not in ranking:
            continue
        ranking[f"calibrated__{baseline_name}"] = fitted_calibrator.predict(
            ranking[score_column].to_numpy(float),
            ranking.transition_index.to_numpy(int),
        )
    ranking["raw_priority_score"] = ranking[f"score__{primary_ranking_method}"]
    ranking["causal_priority_score"] = ranking["raw_priority_score"]
    for method, fitted_calibrator in calibrators.items():
        ranking[f"calibrated_method__{method}"] = fitted_calibrator.predict(
            ranking.raw_priority_score.to_numpy(float),
            ranking.transition_index.to_numpy(int),
        )
    ranking["calibrated_incremental_benefit"] = ranking[
        f"calibrated_method__{selected_calibration_method}"
    ]
    ranking["global_rank"] = ranking.raw_priority_score.rank(
        ascending=False, method="first"
    ).astype(int)
    ranking["within_action_rank"] = ranking.groupby("action_id").raw_priority_score.rank(
        ascending=False, method="first"
    ).astype(int)
    ranking["prerequisites_satisfied"] = ~ranking.eligibility_reasons.str.contains(
        "PREREQUISITE_NOT_ACTIVE", regex=False
    )
    value_mode = str(config["allocation"].get("value_mode", "calibrated_benefit"))
    if value_mode == "equal_cost_rank_only":
        ranking["allocation_objective_value"] = (
            len(ranking) - ranking["global_rank"] + 1
        ).astype(float)
        ranking["allocation_cost"] = 1.0
    else:
        ranking["allocation_objective_value"] = ranking["calibrated_incremental_benefit"]
        ranking["allocation_cost"] = ranking["cost"].astype(float)
    global_ranking_columns = [
        "patient_id", "dm77_need_level", "current_care_state", "action_id",
        "transition_index", "raw_priority_score", "causal_priority_score",
        "global_rank", "within_action_rank",
        "support_flag", "support_propensity", "eligibility_reasons",
        "contraindication_reasons", "calibrated_incremental_benefit",
        "allocation_objective_value", "allocation_cost", "cost",
        "capacity_pool", "from_states", "to_state", "prerequisites",
        "mutually_exclusive_with", "mandatory", "protected_action",
        "prerequisites_satisfied",
    ]
    ranking.loc[:, global_ranking_columns].to_csv(out / "global_ranking.csv", index=False)

    allocation_input = ranking.copy()
    allocation_input["calibrated_incremental_benefit"] = (
        allocation_input["allocation_objective_value"]
    )
    allocation_input["cost"] = allocation_input["allocation_cost"]
    allocation = allocate_care_actions(
        allocation_input,
        shared_budget=float(config["allocation"]["shared_budget"]),
        capacity_pools=config["allocation"].get("capacity_pools"),
        max_primary_actions_per_patient=int(
            config["allocation"]["max_primary_actions_per_patient"]
        ),
        require_empirical_support=bool(
            config["allocation"].get("require_empirical_support", True)
        ),
        allocate_negative_predicted_benefit=bool(
            config["allocation"].get("allocate_negative_predicted_benefit", False)
        ),
        solver_time_limit_seconds=float(
            config["allocation"].get("solver_time_limit_seconds", 120.0)
        ),
    )
    allocation.diagnostics["value_mode"] = value_mode
    allocation.diagnostics["uses_cardinal_calibration"] = value_mode == "calibrated_benefit"
    write_json(out / "allocation_diagnostics.json", allocation.diagnostics)
    baseline_allocations = {primary_ranking_method: allocation}
    for baseline_name in enabled_baselines:
        if baseline_name in {primary_ranking_method, "oracle_priority"}:
            continue
        score_column = f"score__{baseline_name}"
        calibrated_column = f"calibrated__{baseline_name}"
        if score_column not in ranking or calibrated_column not in ranking:
            continue
        baseline_allocations[baseline_name] = _allocate_action_policy(
            ranking,
            ranking[score_column].to_numpy(float),
            ranking[calibrated_column].to_numpy(float),
            config,
        )
    write_json(out / "baseline_allocation_diagnostics.json", {
        name: result.diagnostics for name, result in baseline_allocations.items()
    })
    baseline_assignments = ranking.loc[:, [
        "patient_id", "action_id", "transition_index",
    ]].copy()
    for name, result in baseline_allocations.items():
        baseline_assignments[f"selected__{name}"] = result.selected
    baseline_assignments.to_csv(out / "baseline_policy_assignments.csv", index=False)
    greedy_allocations = {
        "greedy_calibrated_score": allocate_care_actions_greedy(
            allocation_input,
            shared_budget=float(config["allocation"]["shared_budget"]),
            capacity_pools=config["allocation"].get("capacity_pools"),
            strategy="score",
            max_primary_actions_per_patient=int(
                config["allocation"]["max_primary_actions_per_patient"]
            ),
            require_empirical_support=bool(
                config["allocation"].get("require_empirical_support", True)
            ),
            allocate_negative_predicted_benefit=bool(
                config["allocation"].get("allocate_negative_predicted_benefit", False)
            ),
        ),
        "greedy_calibrated_score_per_cost": allocate_care_actions_greedy(
            allocation_input,
            shared_budget=float(config["allocation"]["shared_budget"]),
            capacity_pools=config["allocation"].get("capacity_pools"),
            strategy="score_per_cost",
            max_primary_actions_per_patient=int(
                config["allocation"]["max_primary_actions_per_patient"]
            ),
            require_empirical_support=bool(
                config["allocation"].get("require_empirical_support", True)
            ),
            allocate_negative_predicted_benefit=bool(
                config["allocation"].get("allocate_negative_predicted_benefit", False)
            ),
        ),
    }
    raw_rank_input = allocation_input.copy()
    raw_rank_input["calibrated_incremental_benefit"] = (
        raw_rank_input.raw_priority_score.rank(ascending=True, method="first")
    ).to_numpy(float)
    greedy_allocations["greedy_raw_ordinal_score"] = allocate_care_actions_greedy(
        raw_rank_input,
        shared_budget=float(config["allocation"]["shared_budget"]),
        capacity_pools=config["allocation"].get("capacity_pools"),
        strategy="score",
        max_primary_actions_per_patient=int(
            config["allocation"]["max_primary_actions_per_patient"]
        ),
        require_empirical_support=bool(
            config["allocation"].get("require_empirical_support", True)
        ),
        allocate_negative_predicted_benefit=False,
    )
    write_json(out / "allocator_ablation_diagnostics.json", {
        "milp_calibrated_score": allocation.diagnostics,
        **{name: result.diagnostics for name, result in greedy_allocations.items()},
    })

    # Held-out observational evaluation. Test outcomes are opened only here,
    # after ranking, calibration, and allocation have all been fixed.
    observed_test = _build_observed_action_opportunities(
        prepared, features, "test", ranked_actions,
        require_training_support, support_bounds,
    )
    observed_scores = {
        primary_ranking_method: _score_observed(
            models, observed_test, transition_names=action_names
        ),
        "risk_priority": risk_score_model(observed_test.x),
    }
    for name in ("unified_global_ranker",):
        if name in baseline_models and name not in observed_scores:
            observed_scores[name] = _score_observed(
                baseline_models[name], observed_test, transition_names=action_names
            )
    for name in (
        "direct_pairwise_gbdt_ranker", "pooled_dr_gbdt",
        "dr_random_forest_priority", "dr_policy_tree",
    ):
        if name in tree_baseline_models:
            observed_scores[name] = tree_baseline_models[name].predict_opportunity_scores(
                observed_test.x, observed_test.transition_index
            )
    heldout_dr_parts = []
    for action in ranked_actions:
        item = prepared[action.action_id]
        frame = item["frame"]
        mask = frame.split.astype(str).to_numpy() == "test"
        if require_training_support:
            e_all = item["nuisance"].e_hat.to_numpy(float)
            mask &= (e_all >= support_bounds[0]) & (e_all <= support_bounds[1])
        e = np.clip(item["nuisance"].e_hat.to_numpy(float)[mask], epsilon, 1.0 - epsilon)
        mu0 = item["nuisance"].mu0_hat.to_numpy(float)[mask]
        mu1 = item["nuisance"].mu1_hat.to_numpy(float)[mask]
        d = frame.transition_treatment.to_numpy(float)[mask]
        y = frame.observed_outcome.to_numpy(float)[mask]
        heldout_dr_parts.append(
            mu1 - mu0 + d / e * (y - mu1) - (1.0 - d) / (1.0 - e) * (y - mu0)
        )
    heldout_dr = np.concatenate(heldout_dr_parts)
    rate_rows = []
    for method, score in observed_scores.items():
        rate_rows.append({
            "method": method,
            "partition": "heldout_observational_test",
            "target": "cross_fitted_doubly_robust_signal",
            "oracle_target": False,
            **rank_weighted_effect_metrics(
                score, heldout_dr, observed_test.patient_ids,
                bootstrap_samples=int(evaluation_config.get(
                    "rate_bootstrap_samples", 0
                )),
                seed=seed + 170_003 + len(rate_rows),
            ),
        })
    pd.DataFrame(rate_rows).to_csv(out / "observed_test_rate_metrics.csv", index=False)

    ope_rows = []
    ope_fraction = float(evaluation_config.get("ope_policy_fraction", 0.20))
    for action in ranked_actions:
        item = prepared[action.action_id]
        frame = item["frame"]
        mask = frame.split.astype(str).to_numpy() == "test"
        positions = np.flatnonzero(mask)
        x_action = frame.loc[positions, features].to_numpy(float)
        action_index = np.full(len(positions), action.action_index, dtype=np.int64)
        nuisance = item["nuisance"]
        e = nuisance.e_hat.to_numpy(float)[positions]
        mu0 = nuisance.mu0_hat.to_numpy(float)[positions]
        mu1 = nuisance.mu1_hat.to_numpy(float)[positions]
        d = frame.transition_treatment.to_numpy(float)[positions]
        y = frame.observed_outcome.to_numpy(float)[positions]
        clipped = np.clip(e, epsilon, 1.0 - epsilon)
        dr = mu1 - mu0 + d / clipped * (y - mu1) - (
            1.0 - d
        ) / (1.0 - clipped) * (y - mu0)
        action_scores = {
            primary_ranking_method: models[action.action_id].predict_opportunity_scores(
                x_action,
                np.zeros(len(x_action), dtype=np.int64)
                if len(models[action.action_id].transition_names) == 1 else action_index,
            ),
            "risk_priority": risk_score_model(x_action),
        }
        for name, fitted in tree_baseline_models.items():
            action_scores[name] = fitted.predict_opportunity_scores(x_action, action_index)
        for method, score in action_scores.items():
            count = max(1, int(np.ceil(ope_fraction * len(score))))
            selected_action = np.zeros(len(score), dtype=bool)
            selected_action[np.argsort(-score, kind="mergesort")[:count]] = True
            ope_rows.append({
                "method": method,
                "action_id": action.action_id,
                "partition": "heldout_observational_test",
                "oracle_target": False,
                **actionwise_off_policy_metrics(
                    d, y, e, mu0, mu1, dr, selected_action
                ),
            })
    pd.DataFrame(ope_rows).to_csv(out / "actionwise_ope_metrics.csv", index=False)
    calibration_allocations = {selected_calibration_method: allocation}
    for method in calibrators:
        if method == selected_calibration_method:
            continue
        calibration_allocations[method] = _allocate_action_policy(
            ranking,
            ranking.raw_priority_score.to_numpy(float),
            ranking[f"calibrated_method__{method}"].to_numpy(float),
            config,
        )

    test_people = (
        simulation.learner.loc[simulation.learner.split.astype(str) == "test", [
            "patient_id", "current_care_state",
        ]]
        .merge(assessments, on="patient_id", validate="one_to_one")
    )
    allocation_by_patient = allocation.decisions.set_index("patient_id")
    mandatory_test = action_eligibility.loc[
        action_eligibility.eligible.to_numpy(bool)
        & action_eligibility.mandatory.to_numpy(bool)
        & ~action_eligibility.protected_action.to_numpy(bool)
        & action_eligibility.patient_id.astype(str).map(split_by_patient).eq("test")
    ]
    protected_test = action_eligibility.loc[
        action_eligibility.eligible.to_numpy(bool)
        & action_eligibility.protected_action.to_numpy(bool)
        & action_eligibility.patient_id.astype(str).map(split_by_patient).eq("test")
    ]
    mandatory_map = mandatory_test.drop_duplicates("patient_id").set_index("patient_id")
    protected_map = protected_test.drop_duplicates("patient_id").set_index("patient_id")
    decision_rows = []
    action_to_state = {action.action_id: action.to_state for action in catalog.actions}
    score_lookup = ranking.set_index(["patient_id", "action_id"]).raw_priority_score
    calibrated_lookup = ranking.set_index(
        ["patient_id", "action_id"]
    ).calibrated_incremental_benefit
    objective_lookup = ranking.set_index(
        ["patient_id", "action_id"]
    ).allocation_objective_value
    for person in test_people.itertuples(index=False):
        patient = str(person.patient_id)
        current_state = str(person.current_care_state)
        common = {
            "patient_id": patient,
            "dm77_need_level": person.dm77_need_level,
            "current_care_state": current_state,
            "reason_codes": person.dm77_reason_codes,
            "manual_review": person.dm77_assessment_status == "manual_review",
            "protected_pathway": bool(person.dm77_protected_pathway),
        }
        if person.dm77_assessment_status == "manual_review":
            row = {
                **common, "recommended_action": None, "allocated_action": None,
                "calibrated_incremental_benefit": np.nan, "cost": 0.0,
                "capacity_pool": None, "allocation_status": "manual_review",
                "resulting_care_state": current_state,
                "allocation_reason": "AUTOMATIC_ALLOCATION_WITHHELD_MANUAL_REVIEW",
            }
        elif patient in protected_map.index:
            action_id = str(protected_map.loc[patient].action_id)
            row = {
                **common, "recommended_action": action_id, "allocated_action": action_id,
                "calibrated_incremental_benefit": np.nan,
                "cost": float(protected_map.loc[patient].cost),
                "capacity_pool": str(protected_map.loc[patient].capacity_pool),
                "allocation_status": "protected_pathway",
                "resulting_care_state": apply_action(current_state, action_id, action_to_state),
                "allocation_reason": "PROTECTED_PATHWAY_BYPASSED_DISCRETIONARY_RANKING",
            }
        elif patient in mandatory_map.index:
            action_id = str(mandatory_map.loc[patient].action_id)
            row = {
                **common, "recommended_action": action_id, "allocated_action": action_id,
                "calibrated_incremental_benefit": np.nan,
                "cost": float(mandatory_map.loc[patient].cost),
                "capacity_pool": str(mandatory_map.loc[patient].capacity_pool),
                "allocation_status": "mandatory_allocated_before_discretionary_budget",
                "resulting_care_state": apply_action(current_state, action_id, action_to_state),
                "allocation_reason": "MANDATORY_ACTION_BYPASSED_CAUSAL_RANKING",
            }
        elif patient in allocation_by_patient.index:
            allocated = allocation_by_patient.loc[patient]
            row = {
                **common,
                **allocated[[
                    "recommended_action", "allocated_action",
                    "calibrated_incremental_benefit", "cost", "capacity_pool",
                    "allocation_status", "resulting_care_state", "allocation_reason",
                ]].to_dict(),
            }
        else:
            row = {
                **common, "recommended_action": None, "allocated_action": None,
                "calibrated_incremental_benefit": np.nan, "cost": 0.0,
                "capacity_pool": None, "allocation_status": "no_admissible_action",
                "resulting_care_state": current_state,
                "allocation_reason": "NO_DM77_CATALOG_ACTION_FROM_CURRENT_STATE",
            }
        allocated_action = (
            str(row["allocated_action"])
            if pd.notna(row["allocated_action"]) and str(row["allocated_action"]) else None
        )
        recommended_action = (
            str(row["recommended_action"])
            if pd.notna(row["recommended_action"]) and str(row["recommended_action"]) else None
        )
        row["allocated_action"] = allocated_action
        row["recommended_action"] = recommended_action
        chosen_action = allocated_action or recommended_action
        row["recommended_or_allocated_action"] = chosen_action
        score_key = (patient, str(chosen_action)) if chosen_action is not None else None
        ranked_candidate = score_key is not None and score_key in score_lookup.index
        row["candidate_action_id"] = chosen_action if ranked_candidate else None
        row["causal_priority_score"] = float(
            score_lookup.loc[score_key]
        ) if ranked_candidate else np.nan
        row["calibrated_incremental_benefit"] = float(
            calibrated_lookup.loc[score_key]
        ) if ranked_candidate else row["calibrated_incremental_benefit"]
        row["allocation_objective_value"] = float(
            objective_lookup.loc[score_key]
        ) if ranked_candidate else np.nan
        row["allocation_value_mode"] = value_mode
        decision_rows.append(row)
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(out / "allocation_decisions.csv", index=False)

    # Evaluation starts only after observational recommendations and allocations are fixed.
    truth = simulation.ground_truth.set_index("patient_id")
    evaluation = ranking.copy()
    evaluation["true_cate"] = [
        float(truth.loc[str(patient), f"true_cate__{action_id}"])
        for patient, action_id in zip(evaluation.patient_id, evaluation.action_id)
    ]
    evaluation["latent_individual_effect"] = [
        float(truth.loc[str(patient), f"latent_individual_effect__{action_id}"])
        for patient, action_id in zip(evaluation.patient_id, evaluation.action_id)
    ]
    evaluation["individual_effect"] = [
        float(truth.loc[str(patient), f"individual_effect__{action_id}"])
        for patient, action_id in zip(evaluation.patient_id, evaluation.action_id)
    ]
    evaluation["true_benefit"] = evaluation["true_cate"]
    evaluation.to_csv(out / "evaluation_ranked_actions.csv", index=False)
    learned_selected = allocation.selected
    oracle_allocation = _allocate_action_policy(
        evaluation,
        evaluation.true_cate.to_numpy(float),
        evaluation.true_cate.to_numpy(float),
        config,
    )
    metrics = global_pairwise_concordance(
        evaluation.raw_priority_score,
        evaluation.true_cate,
        evaluation.transition_index,
        seed=seed + 503,
    )
    metrics.update(hard_cross_action_concordance(
        evaluation.raw_priority_score,
        evaluation.true_cate,
        evaluation.transition_index,
        evaluation.support_flag,
        seed=seed + 509,
    ))
    learned_value = float(evaluation.loc[learned_selected, "true_cate"].sum())
    oracle_value = float(evaluation.loc[oracle_allocation.selected, "true_cate"].sum())

    baseline_allocations["oracle_priority"] = oracle_allocation
    baseline_rows = []
    for method, result in baseline_allocations.items():
        score = (
            evaluation.true_cate.to_numpy(float)
            if method == "oracle_priority"
            else evaluation[f"score__{method}"].to_numpy(float)
        )
        concordance = global_pairwise_concordance(
            score, evaluation.true_cate, evaluation.transition_index,
            seed=seed + 503,
        )
        hard = hard_cross_action_concordance(
            score, evaluation.true_cate, evaluation.transition_index,
            evaluation.support_flag, seed=seed + 509,
        )
        value = float(evaluation.loc[result.selected, "true_cate"].sum())
        top_q = synthetic_top_q_metrics(
            score, evaluation.true_cate, quantiles=top_quantiles
        )
        budget = allocation_budget_metrics(
            evaluation.true_cate,
            result.selected,
            evaluation.allocation_cost,
            oracle_allocation.selected,
        )
        baseline_rows.append({
            "method": method,
            **{key: value_ for key, value_ in concordance.items() if key != "oracle_evaluation_only"},
            **hard,
            **top_q,
            **budget,
            "global_allocation_value": value,
            "global_regret": oracle_value - value,
            "oracle_priority_evaluation_only": method == "oracle_priority",
        })
    baseline_comparison = pd.DataFrame(baseline_rows)
    random_value = float(
        baseline_comparison.set_index("method").loc[
            "random_priority", "global_allocation_value"
        ]
    )
    baseline_comparison["normalized_oracle_efficiency"] = [
        normalized_oracle_efficiency(value, random_value, oracle_value)
        for value in baseline_comparison.global_allocation_value
    ]
    baseline_comparison.to_csv(out / "baseline_comparison.csv", index=False)
    baseline_index = baseline_comparison.set_index("method")
    action_mean_cross = float(
        baseline_index.loc["action_mean_priority", "cross_action_concordance"]
    )
    action_mean_value = float(
        baseline_index.loc["action_mean_priority", "global_allocation_value"]
    )
    risk_value = float(baseline_index.loc["risk_priority", "global_allocation_value"])
    primary_efficiency = float(
        baseline_index.loc[primary_ranking_method, "normalized_oracle_efficiency"]
    )
    metrics.update({
        "primary_oracle_target": "true_cate_f_of_observed_x",
        "action_mean_cross_action_concordance": action_mean_cross,
        "incremental_cross_action_concordance_over_action_mean": (
            metrics["cross_action_concordance"] - action_mean_cross
        ),
        "global_allocation_value": learned_value,
        "learned_allocation_value": learned_value,
        "oracle_allocation_value": oracle_value,
        "global_regret": oracle_value - learned_value,
        "value_over_action_mean": learned_value - action_mean_value,
        "value_over_risk": learned_value - risk_value,
        "normalized_oracle_efficiency": primary_efficiency,
        "oracle_evaluation_only": True,
    })
    metrics.update(synthetic_top_q_metrics(
        evaluation.raw_priority_score,
        evaluation.true_cate,
        quantiles=top_quantiles,
    ))
    metrics.update(allocation_budget_metrics(
        evaluation.true_cate,
        learned_selected,
        evaluation.allocation_cost,
        oracle_allocation.selected,
    ))
    write_json(out / "global_metrics.json", metrics)

    allocator_rows = []
    for name, result in {
        "milp_calibrated_score": allocation,
        **greedy_allocations,
    }.items():
        value = float(evaluation.loc[result.selected, "true_cate"].sum())
        allocator_rows.append({
            "allocator": name,
            "ranking_method": primary_ranking_method,
            "global_allocation_value": value,
            "global_regret": oracle_value - value,
            "normalized_oracle_efficiency": normalized_oracle_efficiency(
                value, random_value, oracle_value
            ),
            "selected_actions": int(result.selected.sum()),
            "budget_used": float(result.diagnostics["budget_used"]),
            "feasible": True,
            "oracle_evaluation_only": True,
        })
    pd.DataFrame(allocator_rows).to_csv(out / "allocator_ablation.csv", index=False)

    curve_methods = [
        primary_ranking_method, "risk_priority", "direct_pairwise_gbdt_ranker",
        "pooled_dr_gbdt", "dr_policy_tree", "oracle_priority",
    ]
    budget_rows = []
    for fraction in map(float, evaluation_config.get("budget_fractions", (0.5, 1.0))):
        curve_config = copy.deepcopy(config)
        curve_config["allocation"]["shared_budget"] = (
            float(config["allocation"]["shared_budget"]) * fraction
        )
        for method in curve_methods:
            if method == "oracle_priority":
                score = evaluation.true_cate.to_numpy(float)
                cardinal = score
            elif f"score__{method}" in evaluation and method in baseline_calibrators:
                score = evaluation[f"score__{method}"].to_numpy(float)
                cardinal = evaluation[f"calibrated__{method}"].to_numpy(float)
            else:
                continue
            try:
                result = _allocate_action_policy(
                    evaluation, score, cardinal, curve_config
                )
                value = float(evaluation.loc[result.selected, "true_cate"].sum())
                budget_rows.append({
                    "method": method,
                    "budget_fraction": fraction,
                    "shared_budget": float(
                        curve_config["allocation"]["shared_budget"]
                    ),
                    "global_allocation_value": value,
                    "selected_actions": int(result.selected.sum()),
                    "budget_used": float(result.diagnostics["budget_used"]),
                    "feasible": True,
                    "oracle_evaluation_only": True,
                })
            except (RuntimeError, ValueError) as error:
                budget_rows.append({
                    "method": method,
                    "budget_fraction": fraction,
                    "shared_budget": float(
                        curve_config["allocation"]["shared_budget"]
                    ),
                    "global_allocation_value": np.nan,
                    "selected_actions": 0,
                    "budget_used": np.nan,
                    "feasible": False,
                    "failure_reason": str(error),
                    "oracle_evaluation_only": True,
                })
    budget_curve = pd.DataFrame(budget_rows)
    budget_curve.to_csv(out / "budget_value_curve.csv", index=False)
    budget_summary_rows = []
    for method, group in budget_curve.loc[budget_curve.feasible].groupby("method"):
        if len(group) >= 2:
            budget_summary_rows.append({
                "method": method,
                "budget_value_auc": budget_curve_area(
                    group.budget_fraction, group.global_allocation_value
                ),
                "budget_points": int(len(group)),
                "oracle_evaluation_only": True,
            })
    pd.DataFrame(budget_summary_rows).to_csv(
        out / "budget_curve_summary.csv", index=False
    )

    subgroup_definitions = {
        "dm77_need_level": evaluation.dm77_need_level.astype(str),
    }
    if "age" in evaluation:
        subgroup_definitions["age_band"] = pd.cut(
            evaluation.age, bins=[17, 44, 64, 74, 84, np.inf],
            labels=["18-44", "45-64", "65-74", "75-84", "85+"],
        ).astype(str)
    if "deprivation_index" in evaluation:
        subgroup_definitions["deprivation_quartile"] = pd.qcut(
            evaluation.deprivation_index, 4, labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        ).astype(str)
    if "rurality_index" in evaluation:
        subgroup_definitions["rurality"] = pd.cut(
            evaluation.rurality_index, bins=[-np.inf, 0.33, 0.66, np.inf],
            labels=["urban", "mixed", "rural"],
        ).astype(str)
    if "sex_female" in evaluation:
        subgroup_definitions["sex"] = np.where(
            evaluation.sex_female.to_numpy(float) >= 0.5, "female", "male"
        )
    subgroup_methods = [
        value for value in (
            primary_ranking_method, "risk_priority", "random_priority",
            "direct_pairwise_gbdt_ranker", "pooled_dr_gbdt", "dr_policy_tree",
            "oracle_priority",
        ) if value in baseline_allocations
    ]
    subgroup_rows = []
    random_selected = baseline_allocations["random_priority"].selected
    for attribute, labels in subgroup_definitions.items():
        labels = np.asarray(labels).astype(str)
        for group_name in sorted(set(labels)):
            mask = labels == group_name
            random_group_value = float(
                evaluation.loc[mask & random_selected, "true_cate"].sum()
            )
            oracle_group_value = float(
                evaluation.loc[mask & oracle_allocation.selected, "true_cate"].sum()
            )
            for method in subgroup_methods:
                selected = baseline_allocations[method].selected
                value = float(evaluation.loc[mask & selected, "true_cate"].sum())
                subgroup_rows.append({
                    "attribute": attribute,
                    "group": group_name,
                    "method": method,
                    "opportunities": int(mask.sum()),
                    "patients": int(evaluation.loc[mask, "patient_id"].nunique()),
                    "supported_fraction": float(evaluation.loc[mask, "support_flag"].mean()),
                    "selected_actions": int((mask & selected).sum()),
                    "global_allocation_value": value,
                    "normalized_oracle_efficiency": normalized_oracle_efficiency(
                        value, random_group_value, oracle_group_value
                    ),
                    "oracle_evaluation_only": True,
                })
    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(out / "subgroup_policy_metrics.csv", index=False)
    primary_subgroups = subgroup_frame.loc[
        subgroup_frame.method == primary_ranking_method
    ]
    worst_rows = []
    for attribute, group in primary_subgroups.groupby("attribute"):
        finite = group.loc[np.isfinite(group.normalized_oracle_efficiency)]
        if len(finite):
            worst = finite.sort_values("normalized_oracle_efficiency").iloc[0]
            worst_rows.append(worst.to_dict())
    pd.DataFrame(worst_rows).to_csv(out / "worst_group_summary.csv", index=False)

    risk_selected = baseline_allocations["risk_priority"].selected
    discordance_rows = []
    for label, mask in {
        "selected_by_both": learned_selected & risk_selected,
        "primary_only": learned_selected & ~risk_selected,
        "risk_only": ~learned_selected & risk_selected,
        "selected_by_neither": ~learned_selected & ~risk_selected,
    }.items():
        discordance_rows.append({
            "discordance_group": label,
            "opportunities": int(mask.sum()),
            "true_cate_sum": float(evaluation.loc[mask, "true_cate"].sum()),
            "true_cate_mean": float(evaluation.loc[mask, "true_cate"].mean())
            if mask.any() else np.nan,
            "oracle_evaluation_only": True,
        })
    pd.DataFrame(discordance_rows).to_csv(
        out / "risk_causal_policy_discordance.csv", index=False
    )

    calibration_rows = []
    for method, fitted_calibrator in calibrators.items():
        prediction = evaluation[f"calibrated_method__{method}"].to_numpy(float)
        selected = calibration_allocations[method].selected
        row = dict(fitted_calibrator.diagnostics)
        row.update({
            "selected_for_observational_allocation": method == selected_calibration_method,
            "synthetic_oracle_mae": float(np.mean(np.abs(prediction - evaluation.true_cate))),
            "synthetic_oracle_bias": float(np.mean(prediction - evaluation.true_cate)),
            "selected_benefit_bias": float(np.mean(
                prediction[selected] - evaluation.true_cate.to_numpy(float)[selected]
            )) if selected.any() else float("nan"),
            "oracle_metrics_computed_after_allocation_fixed": True,
            **linear_calibration_metrics(
                prediction, evaluation.true_cate, "synthetic_oracle"
            ),
        })
        calibration_rows.append(row)
        calibration_group_rows.append(score_group_diagnostics(
            evaluation.raw_priority_score,
            prediction,
            evaluation.true_cate,
            method=method,
            partition="test_synthetic_evaluation_after_allocation_fixed",
            target_name="true_cate_f_of_observed_x",
            oracle_target=True,
            groups=calibration_score_groups,
        ))
    calibration_diagnostics = pd.DataFrame(calibration_rows)
    calibration_diagnostics.to_csv(out / "calibration_diagnostics.csv", index=False)
    write_json(out / "calibration_diagnostics.json", {
        row["method"]: row for row in calibration_rows
    })
    pd.concat(calibration_group_rows, ignore_index=True).to_csv(
        out / "calibration_score_group_diagnostics.csv", index=False
    )

    if not simulation.metadata["latent_individual_effect_zero"]:
        latent_metrics = global_pairwise_concordance(
            evaluation.raw_priority_score,
            evaluation.individual_effect,
            evaluation.transition_index,
            seed=seed + 607,
        )
        latent_metrics.update({
            "oracle_target": "individual_effect_true_cate_plus_latent_component",
            "secondary_stress_diagnostic_only": True,
        })
        write_json(out / "latent_individual_effect_diagnostics.json", latent_metrics)

    model = models[action_names[0]]
    model.save(
        out / "prometheus_dm77_action_ranker.pt",
        method_version=METHOD_VERSION,
    )
    pair_diag = training_diagnostics.get("best_pair_diagnostics", {})
    manifest = {
        "run_id": run_id,
        "method_version": METHOD_VERSION,
        "opportunity_source": "dm77_catalog",
        "ranking_unit": "patient_action_opportunity",
        "primary_ranking_method": primary_ranking_method,
        "score_semantics": "globally_comparable_ordinal_priority_not_cate",
        "need_level_semantics": "dm77_multidimensional_need_not_treatment",
        "historical_treatment_semantics": "care_action_id",
        "action_dgp_scenario": simulation.metadata["scenario"],
        "observed_confounding_strength": simulation.metadata[
            "observed_confounding_strength"
        ],
        "hidden_confounding_failure_scenario": simulation.metadata[
            "hidden_confounding_failure_scenario"
        ],
        "dgp_negative_control": simulation.metadata.get("negative_control"),
        "conditional_exchangeability_by_construction": simulation.metadata[
            "conditional_exchangeability_by_construction"
        ],
        "primary_oracle_target": "true_cate_f_of_observed_x",
        "secondary_oracle_target": simulation.metadata["secondary_oracle_target"],
        "primary_ranking_metrics_use": "true_cate",
        "individual_effect_metrics": (
            "secondary_stress_diagnostic_only"
            if not simulation.metadata["latent_individual_effect_zero"] else "not_applicable_zero"
        ),
        "latent_individual_effect_zero": simulation.metadata[
            "latent_individual_effect_zero"
        ],
        "treatment_assignment_uses_oracle": False,
        "current_care_state_distinct_from_dm77_need": True,
        "current_care_state_features_excluded_ablation": exclude_current_care_features,
        "action_id_to_transition_index": {
            action.action_id: action.action_index for action in ranked_actions
        },
        "cross_action_pairing_enabled": True,
        "all_ranked_actions_outcome_comparable": True,
        "contrastive_enabled": bool(contrastive.get("enabled", False)),
        "contrastive_configuration": {
            "enabled": bool(contrastive.get("enabled", False)),
            "training_mode": str(config["prometheus"]["training_mode"]),
            "pair_scope": contrastive.get("pair_scope"),
            "pairs_per_epoch": int(
                config["prometheus"].get("contrastive_pairs_per_epoch", 0)
            ),
            "lambda_con": float(config["prometheus"].get("lambda_con", 0.0)),
            "margin": float(config["prometheus"].get("contrastive_margin", 1.0)),
            "positive_max_gap_days": float(
                config["prometheus"].get("positive_max_gap_days", 0.0)
            ),
            "negative_min_gap_days": float(
                config["prometheus"].get("negative_min_gap_days", 0.0)
            ),
            "require_stable_direction": bool(
                contrastive.get("require_stable_direction", False)
            ),
            "require_overlap_support": bool(
                contrastive.get("require_overlap_support", False)
            ),
            "reliability_weighting": bool(
                contrastive.get("reliability_weighting", False)
            ),
            "separate_contrastive_head": bool(
                training_diagnostics.get("separate_contrastive_head", False)
            ),
        },
        "ranking_signal": (
            "split_local_robust_repeated_"
            + str(config["causal_signal"].get("estimator", "dr"))
            + "_aggregate"
        ),
        "causal_signal_estimator": str(
            config["causal_signal"].get("estimator", "dr")
        ),
        "raw_dr_signals_preserved_for_audit": True,
        "causal_signal_configuration": config["causal_signal"],
        "ranking_beta_cross": float(config["prometheus"]["beta_cross"]),
        "selected_calibration_method": selected_calibration_method,
        "calibration_methods_compared": list(calibrators),
        "allocation_value_mode": value_mode,
        "checkpoint_metric": config["checkpoint"]["metric"],
        "fixed_validation_pair_set": True,
        "training_pair_labels_permuted_negative_control": bool(
            config.get("negative_control", {}).get(
                "permute_training_pair_labels", False
            )
        ),
        "initial_validation_metric": training_diagnostics.get(
            "initial_validation_metric"
        ),
        "best_epoch": training_diagnostics.get("best_epoch"),
        "best_validation_within": training_diagnostics.get("best_validation_within"),
        "best_validation_cross": training_diagnostics.get("best_validation_cross"),
        "best_validation_global": training_diagnostics.get("best_validation_global"),
        "validation_policy_value_dr": training_diagnostics.get(
            "validation_policy_value_dr"
        ),
        "baselines_evaluated": baseline_comparison.method.astype(str).tolist(),
        "top_q_fractions": list(top_quantiles),
        "calibration_score_groups": calibration_score_groups,
        "interval_targets_prespecified": evaluation_config.get(
            "interval_targets", []
        ),
        "aggregate_interval_coverage_established": False,
        "individual_effect_interval_coverage_claimed": False,
        "oracle_used_for_nuisance": False,
        "oracle_used_for_ranking": False,
        "oracle_used_for_calibration": False,
        "oracle_used_for_observational_allocation": False,
        "oracle_join_stage": "evaluation_after_allocation_decisions_fixed",
        "need_assessments": int(len(assessments)),
        "candidate_actions": int(len(authoritative_candidates)),
        "ranked_opportunities": int(len(ranking)),
        "within_action_pairs": int(pair_diag.get("valid_within_pairs", 0)),
        "cross_action_pairs": int(pair_diag.get("valid_cross_pairs", 0)),
        "allocated_actions": int(decisions.allocated_action.isin(action_names).sum()),
        "protected_cases": int(assessments.dm77_protected_pathway.sum()),
        "manual_review_cases": int(
            assessments.dm77_assessment_status.eq("manual_review").sum()
        ),
        "protected_cases_in_test_decisions": int(decisions.protected_pathway.sum()),
        "manual_review_cases_in_test_decisions": int(decisions.manual_review.sum()),
        "mandatory_cases": int(
            decisions.allocation_status.eq(
                "mandatory_allocated_before_discretionary_budget"
            ).sum()
        ),
        "allocation_violations": {
            key: value for key, value in allocation.diagnostics.items()
            if key.endswith("violations") or key == "budget_violation"
        },
        "resulting_state_uses_action_application": True,
        "resulting_state_uses_1_plus_sum": False,
        "sample_size": int(len(patients)),
        "seed": seed,
        "model_and_nuisance_seed": seed,
        "dataset_seed": dataset_seed,
        "population_scenario": population_scenario,
        "duration_seconds": time.time() - started,
        "fully_synthetic": True,
        "italian_population_validity_established": False,
        "deployment_readiness_established": False,
    }
    # Keep both names: run_manifest is the integrated contract; manifest supports
    # existing discovery tooling.
    write_json(out / "run_manifest.json", manifest)
    write_json(out / "manifest.json", manifest)
    _write_report(out, manifest, dm77_summary, metrics)
    return out
