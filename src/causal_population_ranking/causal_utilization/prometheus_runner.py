from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.validation import assert_no_oracle_columns, assert_patient_level_split_integrity
from ..version import PROMETHEUS_LEGACY_LOCAL_METHOD_VERSION
from ..evaluation.metrics import (
    evaluate_observational_ranking,
    evaluate_ranker,
    macro_average_transition_metrics,
    operational_capacity_metrics,
    tail_metrics,
)
from ..nuisance.cross_fitting import fit_repeated_partitioned_nuisance
from ..ranking.causal_signals import aggregate_repeated_signals, repeated_doubly_robust_signals
from ..ranking.losses import causal_contrastive_loss, sample_contrastive_pairs
from ..ranking.prometheus_ranker import PrometheusRanker, TRAINING_MODES, TransitionArrays
from ..reproducibility import seed_everything, write_json
from .capacity_thresholds import fit_capacity_threshold, select_at_capacity
from .policy_benchmarks import (
    burden_only_score,
    deterministic_score_equal_resource_policy,
    global_oracle_equal_resource_bound,
    random_equal_resource_policy,
    transition_score_equal_resource_policy,
)
from .helpers import _align, _load_cohort, _percentile, _policy_value, _risk_proxy
from .simulation import TRANSITIONS, simulate_care_intensity
from .stratifier import HierarchicalCausalStratifier
from .support import support_label, supported


METHOD_VERSION = PROMETHEUS_LEGACY_LOCAL_METHOD_VERSION


def _repeated_nuisance_tensor(repetitions: list[pd.DataFrame], patient_ids) -> np.ndarray:
    identifiers = pd.Index(np.asarray(patient_ids).astype(str), name="patient_id")
    arrays = []
    for repeat, frame in enumerate(repetitions):
        indexed = frame.assign(patient_id=frame.patient_id.astype(str)).set_index("patient_id")
        if not identifiers.isin(indexed.index).all():
            raise RuntimeError(f"Nuisance repetition {repeat} is missing patient predictions")
        arrays.append(indexed.loc[identifiers, ["e_hat", "mu0_hat", "mu1_hat"]].to_numpy(float))
    return np.stack(arrays, axis=1)


def _aggregated_dr_signal(
    frame: pd.DataFrame,
    repetitions: list[pd.DataFrame],
    epsilon: float,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    repeated = repeated_doubly_robust_signals(
        frame.transition_treatment.to_numpy(float),
        frame.observed_outcome.to_numpy(float),
        _repeated_nuisance_tensor(repetitions, frame.patient_id),
        epsilon,
    )
    return aggregate_repeated_signals(repeated, aggregation), repeated


def _contrastive_evaluation_diagnostics(
    ranker: PrometheusRanker,
    x: np.ndarray,
    signal: np.ndarray,
    transition_name: str,
    config: dict,
    seed: int,
) -> dict:
    bins = ranker.response_bins(signal, transition_name)
    budget = min(int(config["contrastive_pairs_per_transition"]), max(2, len(signal) * 2))
    pairs = sample_contrastive_pairs(
        bins,
        budget,
        seed,
        int(config["negative_bin_separation"]),
        float(config["positive_negative_pair_ratio"]),
        True,
    )
    if not len(pairs.left):
        return {
            "contrastive_positive_pair_count": 0,
            "contrastive_negative_pair_count": 0,
            "average_contrastive_positive_distance": float("nan"),
            "average_contrastive_negative_distance": float("nan"),
            "active_contrastive_margin_fraction": 0.0,
        }
    representation = torch.as_tensor(
        ranker.predict_representation(x, transition_name), dtype=torch.float32
    )
    similar = torch.as_tensor(pairs.target, dtype=torch.float32)
    _, diagnostics = causal_contrastive_loss(
        representation[pairs.left],
        representation[pairs.right],
        similar,
        float(config["contrastive_margin"]),
        bool(config["normalize_contrastive_embeddings"]),
        return_diagnostics=True,
    )
    return {
        "contrastive_positive_pair_count": diagnostics["positive_pair_count"],
        "contrastive_negative_pair_count": diagnostics["negative_pair_count"],
        "average_contrastive_positive_distance": diagnostics["mean_positive_distance"],
        "average_contrastive_negative_distance": diagnostics["mean_negative_distance"],
        "active_contrastive_margin_fraction": diagnostics["active_negative_margin_fraction"],
    }


def _validate_config(config: dict) -> dict:
    required_sections = {"run", "data", "simulation", "nuisance", "prometheus"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing PROMETHEUS config sections: {sorted(missing)}")
    required = {
        "training_mode", "lambda_con", "lambda_reg", "contrastive_margin",
        "num_response_bins", "negative_bin_separation", "ranking_min_signal_gap",
        "propensity_clip_epsilon", "pairs_per_transition", "positive_negative_pair_ratio",
        "normalize_contrastive_embeddings", "pair_reliability_weighting",
        "dr_signal_aggregation", "min_pair_direction_agreement",
    }
    missing = required.difference(config["prometheus"])
    if missing:
        raise ValueError(f"Missing PROMETHEUS settings: {sorted(missing)}")
    if config["prometheus"]["training_mode"] not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode {config['prometheus']['training_mode']!r}")
    if int(config["nuisance"].get("repeats", 0)) < 1:
        raise ValueError("nuisance.repeats must be positive")
    simulation_required = {
        "shared_effect_correlation", "overlap_strength", "risk_benefit_alignment",
        "transition_imbalance_rho", "capacities",
    }
    simulation_missing = simulation_required.difference(config["simulation"])
    if simulation_missing:
        raise ValueError(f"Missing simulation settings: {sorted(simulation_missing)}")
    if not 0.0 <= float(config["simulation"]["shared_effect_correlation"]) <= 1.0:
        raise ValueError("shared_effect_correlation must be in [0, 1]")
    if not -1.0 <= float(config["simulation"]["risk_benefit_alignment"]) <= 1.0:
        raise ValueError("risk_benefit_alignment must be in [-1, 1]")
    return config


def run_prometheus(config: dict):
    """Run one fully resolved PROMETHEUS experiment."""
    config = _validate_config(config)
    run = config["run"]
    data = config["data"]
    simulation_config = config["simulation"]
    nuisance_config = config["nuisance"]
    model_config = config["prometheus"]
    seed = int(run["seed"])
    started = time.time()
    seed_everything(seed)
    mode = str(model_config["training_mode"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_prometheus_{mode}_{run['scenario']}_s{seed}"
    out = Path(run["output_root"]) / run_id
    out.mkdir(parents=True)
    write_json(out / "resolved_config.json", config)

    cohort = _load_cohort(data["cohort_cache"], int(run["sample_size"]), seed)
    learners, truths, truth_all, clinical, metadata, features, split = simulate_care_intensity(
        cohort=cohort,
        seed=seed,
        scenario=run["scenario"],
        shared_effect_correlation=float(simulation_config["shared_effect_correlation"]),
        overlap_strength=float(simulation_config["overlap_strength"]),
        risk_benefit_alignment=float(simulation_config["risk_benefit_alignment"]),
        transition_imbalance_rho=float(simulation_config["transition_imbalance_rho"]),
        capacities=simulation_config["capacities"],
    )
    assert_patient_level_split_integrity(learners)
    clinical = clinical.copy()
    clinical["split"] = split
    clinical.to_csv(out / "clinical_state.csv", index=False)
    truth_all.to_csv(out / "multilevel_ground_truth.csv", index=False)

    prepared = {}
    epsilon = float(model_config["propensity_clip_epsilon"])
    for transition_index, name in enumerate(TRANSITIONS):
        transition_dir = out / name
        transition_dir.mkdir()
        learner = learners[name]
        assert_no_oracle_columns(learner)
        learner.to_csv(transition_dir / "learner_dataset.csv", index=False)
        truths[name].to_csv(transition_dir / "evaluation_ground_truth.csv", index=False)
        nuisance, nuisance_repetitions, diagnostics = fit_repeated_partitioned_nuisance(
            learner,
            features,
            int(nuisance_config["folds"]),
            nuisance_config["model"],
            seed + transition_index,
            int(nuisance_config["repeats"]),
            int(nuisance_config.get("repeat_seed_stride", 1009)),
            "nuisance_train",
            (epsilon, 1.0 - epsilon),
        )
        nuisance.to_csv(transition_dir / "nuisance_predictions.csv", index=False)
        pd.concat(nuisance_repetitions, ignore_index=True).to_csv(
            transition_dir / "nuisance_predictions_repeated.csv", index=False
        )
        joined = learner.merge(nuisance, on="patient_id", validate="one_to_one")
        rank = joined[joined.split == "rank_train"].reset_index(drop=True)
        validation = joined[joined.split == "validation"].reset_index(drop=True)
        test = joined[joined.split == "test"].reset_index(drop=True)
        if min(len(rank), len(validation), len(test)) < int(run.get("minimum_split_n", 10)):
            raise RuntimeError(f"Insufficient eligible sample for {name}")
        prepared[name] = {
            "rank": rank, "validation": validation, "test": test,
            "nuisance_repetitions": nuisance_repetitions,
            "nuisance_diagnostics": diagnostics,
        }
        write_json(transition_dir / "nuisance_diagnostics.json", diagnostics)

    rho = float(simulation_config["transition_imbalance_rho"])
    imbalance_sizes = {name: len(prepared[name]["rank"]) for name in TRANSITIONS}

    training_data = {}
    validation_data = {}
    for name in TRANSITIONS:
        rank = prepared[name]["rank"]
        validation = prepared[name]["validation"]
        training_data[name] = TransitionArrays(
            rank[features].to_numpy(float),
            rank.transition_treatment.to_numpy(float),
            rank.observed_outcome.to_numpy(float),
            _repeated_nuisance_tensor(
                prepared[name]["nuisance_repetitions"], rank.patient_id
            ),
            rank.patient_id.astype(str).to_numpy(),
        )
        validation_data[name] = TransitionArrays(
            validation[features].to_numpy(float),
            validation.transition_treatment.to_numpy(float),
            validation.observed_outcome.to_numpy(float),
            _repeated_nuisance_tensor(
                prepared[name]["nuisance_repetitions"], validation.patient_id
            ),
            validation.patient_id.astype(str).to_numpy(),
        )
    train_patients = set().union(*(set(value.patient_ids) for value in training_data.values()))
    test_patients = set().union(*(set(prepared[name]["test"].patient_id.astype(str)) for name in TRANSITIONS))
    if train_patients.intersection(test_patients):
        raise AssertionError("Test patients overlap rank-training patients")

    def make_ranker(names, model_seed, selected_mode):
        return PrometheusRanker(
            transition_names=names,
            training_mode=selected_mode,
            pairs_per_transition=int(model_config["pairs_per_transition"]),
            contrastive_pairs_per_transition=int(model_config.get(
                "contrastive_pairs_per_transition", max(1, int(model_config["pairs_per_transition"]) // 2)
            )),
            epochs=int(model_config.get("epochs", 8)),
            batch_size=int(model_config.get("batch_size", 1024)),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            latent_dim=int(model_config.get("latent_dim", 32)),
            lr=float(model_config.get("learning_rate", 3e-4)),
            patience=int(model_config.get("patience", 4)),
            seed=model_seed,
            device=run.get("device", "cpu"),
            lambda_con=0.0 if selected_mode == "independent_rankers" else float(model_config["lambda_con"]),
            lambda_reg=float(model_config["lambda_reg"]),
            contrastive_margin=float(model_config["contrastive_margin"]),
            num_response_bins=int(model_config["num_response_bins"]),
            negative_bin_separation=int(model_config["negative_bin_separation"]),
            ranking_min_signal_gap=model_config["ranking_min_signal_gap"],
            propensity_clip_epsilon=epsilon,
            positive_negative_pair_ratio=float(model_config["positive_negative_pair_ratio"]),
            normalize_contrastive_embeddings=bool(model_config["normalize_contrastive_embeddings"]),
            pair_reliability_weighting=model_config["pair_reliability_weighting"],
            dr_signal_aggregation=model_config["dr_signal_aggregation"],
            min_pair_direction_agreement=float(model_config["min_pair_direction_agreement"]),
        )

    if mode == "independent_rankers":
        models = {}
        history_parts = []
        diagnostics_by_transition = {}
        for transition_index, name in enumerate(TRANSITIONS):
            ranker = make_ranker((name,), seed + 5000 + 100 * transition_index, mode).fit(
                {name: training_data[name]}, {name: validation_data[name]}
            )
            models[name] = ranker
            history_parts.append(pd.DataFrame(ranker.history).assign(transition=name))
            diagnostics_by_transition[name] = ranker.training_diagnostics
        history = pd.concat(history_parts, ignore_index=True)
        training_diagnostics = {
            "model": "independent_transition_rankers",
            "training_mode": mode,
            "per_transition": diagnostics_by_transition,
            "transition_imbalance_rho": rho,
            "rank_training_sizes": imbalance_sizes,
        }
    else:
        ranker = make_ranker(tuple(TRANSITIONS), seed + 5000, mode).fit(
            training_data, validation_data
        )
        models = {name: ranker for name in TRANSITIONS}
        history = pd.DataFrame(ranker.history)
        training_diagnostics = dict(ranker.training_diagnostics)
        training_diagnostics.update({
            "transition_imbalance_rho": rho,
            "rank_training_sizes": imbalance_sizes,
        })
    history.to_csv(out / "prometheus_training_history.csv", index=False)
    write_json(out / "prometheus_training_diagnostics.json", training_diagnostics)

    predictions = {}
    thresholds = {}
    transition_metrics = {}
    comparator_metric_rows = []
    cate_score_frames = {}
    selection = {}
    for transition_index, name in enumerate(TRANSITIONS):
        capacity = float(metadata[name]["capacity"])
        transition_dir = out / name
        validation = prepared[name]["validation"]
        test = prepared[name]["test"]
        ranker = models[name]
        validation_score = ranker.predict_score(validation[features].to_numpy(float), name)
        test_x = test[features].to_numpy(float)
        test_score = ranker.predict_score(test_x, name)
        threshold = fit_capacity_threshold(validation_score, capacity)
        thresholds[name] = threshold
        support = supported(test.e_hat.to_numpy())
        selected = select_at_capacity(test_score, capacity, support)
        budget = int(np.floor(capacity * len(test_score)))
        if int(selected.sum()) > budget:
            raise AssertionError(f"Capacity allocation exceeds B_k for {name}")
        labels = support_label(test.e_hat.to_numpy())
        prediction = pd.DataFrame({
            "patient_id": test.patient_id.astype(str),
            "score": test_score,
            "score_percentile": _percentile(validation_score, test_score),
            "e_hat": test.e_hat.to_numpy(),
            "support_label": labels,
            "supported": support,
            "selected": selected,
        })
        predictions[name] = prediction
        prediction.to_csv(transition_dir / "test_transition_priorities.csv", index=False)

        heldout_signal, heldout_signal_repeated = _aggregated_dr_signal(
            test,
            prepared[name]["nuisance_repetitions"],
            epsilon,
            model_config["dr_signal_aggregation"],
        )
        test_truth = _align(test, truths[name])
        metrics = evaluate_ranker(
            test_score, test_truth.true_latent_rank, test_truth.true_benefit,
            capacities=(0.05, 0.10, 0.20),
        )
        metrics.update(tail_metrics(
            test_score, test_truth.true_benefit, capacities=(0.05, 0.10, 0.20), seed=seed
        ))
        metrics.update(operational_capacity_metrics(
            test_score,
            test_truth.true_benefit.to_numpy(float),
            test_truth.potential_outcome_lower.to_numpy(float),
            capacity,
            selected,
        ))
        gap = ranker.ranking_gaps[name]
        metrics.update(evaluate_observational_ranking(
            test_score, heldout_signal, test.e_hat.to_numpy(float),
            capacities=(0.05, 0.10, 0.20), min_signal_gap=gap,
            propensity_clip_epsilon=epsilon, seed=seed + transition_index,
        ))
        metrics.update({
            "dr_signal_repetitions": int(heldout_signal_repeated.shape[1]),
            "mean_dr_signal_between_repeat_sd": float(
                heldout_signal_repeated.std(axis=1).mean()
            ),
        })
        metrics.update(_contrastive_evaluation_diagnostics(
            ranker, test_x, heldout_signal, name, model_config, seed + 7000 + transition_index
        ))
        best_history = ranker.history[ranker.best_epoch]
        metrics["valid_training_pair_fraction"] = float(
            best_history[f"valid_pair_fraction_{name}"]
        )
        transition_metrics[name] = metrics
        write_json(transition_dir / "metrics.json", metrics)
        selection[name] = {
            "ranker": mode,
            "capacity": capacity,
            "capacity_budget": budget,
            "selected_count": int(selected.sum()),
            "validation_threshold_for_reporting": threshold,
            "eligible_counts": {
                "rank": len(prepared[name]["rank"]),
                "validation": len(validation),
                "test": len(test),
            },
            "support_rate_test": float(prediction.supported.mean()),
            "ranking_min_signal_gap": gap,
            "validation_observed_autoc": ranker.best_validation_by_transition[name],
        }

        transition_base = cohort.set_index("patient_id").loc[test.patient_id.astype(str)].reset_index()
        risk_score = _risk_proxy(transition_base)
        cate_score = test.mu1_hat.to_numpy(float) - test.mu0_hat.to_numpy(float)
        cate_score_frames[name] = pd.DataFrame({
            "patient_id": test.patient_id.astype(str),
            "score": cate_score,
        })
        comparator_scores = {
            "learned_direct_ranker": test_score,
            "random_ranking": np.random.default_rng(seed + 9000 + transition_index).normal(size=len(test)),
            "prognostic_risk_ranking": risk_score,
            "cate_estimator_then_sort": cate_score,
            "oracle_ranking": test_truth.true_benefit.to_numpy(float),
        }
        for comparator_name, comparator_score in comparator_scores.items():
            comparator_selected = (
                selected if comparator_name == "learned_direct_ranker"
                else select_at_capacity(comparator_score, capacity, support)
            )
            comparator_metrics = evaluate_ranker(
                comparator_score,
                test_truth.true_latent_rank,
                test_truth.true_benefit,
                capacities=(capacity,),
            )
            comparator_metrics.update(operational_capacity_metrics(
                comparator_score,
                test_truth.true_benefit.to_numpy(float),
                test_truth.potential_outcome_lower.to_numpy(float),
                capacity,
                comparator_selected,
            ))
            comparator_metric_rows.append({
                "method": comparator_name,
                "transition": name,
                "seed": seed,
                "autoc": comparator_metrics["autoc"],
                "pairwise_concordance": comparator_metrics["pairwise_concordance"],
                "benefit_at_operational_capacity": comparator_metrics["benefit_at_operational_capacity"],
                "policy_value_at_operational_capacity": comparator_metrics["policy_value_at_operational_capacity"],
                "policy_regret_at_operational_capacity": comparator_metrics["policy_regret_at_operational_capacity"],
                "top_capacity_overlap_with_oracle": comparator_metrics["top_capacity_overlap_with_oracle"],
                "overlap_coverage": metrics["overlap_coverage"],
                "valid_training_pair_fraction": (
                    metrics["valid_training_pair_fraction"]
                    if comparator_name == "learned_direct_ranker" else float("nan")
                ),
            })

    macro_metrics = macro_average_transition_metrics(transition_metrics)
    write_json(out / "transition_metrics.json", transition_metrics)
    write_json(out / "macro_metrics.json", macro_metrics)
    comparator_metrics_frame = pd.DataFrame(comparator_metric_rows)
    comparator_metrics_frame.to_csv(out / "comparator_transition_metrics.csv", index=False)
    comparator_macro = comparator_metrics_frame.groupby("method", as_index=False).agg(
        **{
            column: (column, "mean")
            for column in (
                "autoc", "pairwise_concordance", "benefit_at_operational_capacity",
                "policy_value_at_operational_capacity", "policy_regret_at_operational_capacity",
                "top_capacity_overlap_with_oracle", "overlap_coverage",
                "valid_training_pair_fraction",
            )
        }
    )
    comparator_macro.to_csv(out / "comparator_macro_metrics.csv", index=False)

    checkpoint = {
        "features": features,
        "transition_names": list(TRANSITIONS),
        "thresholds": thresholds,
        "resolved_configuration": config,
        "training_mode": mode,
    }
    if mode == "independent_rankers":
        checkpoint.update({
            "state_dict_by_transition": {name: models[name].model.state_dict() for name in TRANSITIONS},
            "scaler_mean_by_transition": {name: models[name].scaler.mean_ for name in TRANSITIONS},
            "scaler_scale_by_transition": {name: models[name].scaler.scale_ for name in TRANSITIONS},
        })
    else:
        checkpoint.update({
            "state_dict": ranker.model.state_dict(),
            "scaler_mean": ranker.scaler.mean_,
            "scaler_scale": ranker.scaler.scale_,
        })
    torch.save(checkpoint, out / "prometheus_ranker.pt")

    test_clinical = clinical[clinical.split == "test"].drop(columns="split").reset_index(drop=True)
    assigned = HierarchicalCausalStratifier(thresholds).assign(test_clinical, predictions)
    base = cohort.set_index("patient_id").loc[assigned.patient_id].reset_index()
    assigned["prognostic_risk_score"] = _risk_proxy(base)
    assigned.to_csv(out / "test_causal_utilization_levels.csv", index=False)
    random_policy, random_diagnostics = random_equal_resource_policy(test_clinical, assigned, seed + 999)
    risk_policy, risk_diagnostics = deterministic_score_equal_resource_policy(
        test_clinical, assigned, assigned.prognostic_risk_score, "pre-index prognostic risk only"
    )
    burden_policy, burden_diagnostics = deterministic_score_equal_resource_policy(
        test_clinical, assigned, burden_only_score(base), "pre-index morbidity burden only"
    )
    cate_policy, cate_diagnostics = transition_score_equal_resource_policy(
        test_clinical, assigned, cate_score_frames, "cross-fitted CATE estimator then sort"
    )
    learned_value, current_value, actionable_n = _policy_value(assigned, truth_all)
    random_value, _, _ = _policy_value(random_policy, truth_all)
    risk_value, _, _ = _policy_value(risk_policy, truth_all)
    burden_value, _, _ = _policy_value(burden_policy, truth_all)
    cate_value, _, _ = _policy_value(cate_policy, truth_all)
    oracle_increment_total, oracle_diagnostics = global_oracle_equal_resource_bound(
        test_clinical, assigned, truth_all
    )
    oracle_value = current_value + oracle_increment_total / actionable_n
    policy = {
        "actionable_test_n": actionable_n,
        "current_care_policy_value": current_value,
        "learned_policy_value": learned_value,
        "risk_only_policy_value": risk_value,
        "burden_only_policy_value": burden_value,
        "cate_sort_policy_value": cate_value,
        "random_capacity_policy_value": random_value,
        "oracle_capacity_policy_value": oracle_value,
        "learned_increment_vs_current_care": learned_value - current_value,
        "learned_minus_risk_only": learned_value - risk_value,
        "learned_minus_burden_only": learned_value - burden_value,
        "learned_minus_cate_sort": learned_value - cate_value,
        "learned_minus_random": learned_value - random_value,
        "fraction_of_oracle_increment": (
            (learned_value - current_value) / (oracle_value - current_value + 1e-12)
        ),
        "benchmark_definition": (
            "current care is the no-escalation baseline; risk, burden, and random use exact "
            "learned resources; oracle is evaluation-only"
        ),
    }
    pd.DataFrame({
        "policy": [
            "current_care_no_escalation", "learned_direct_causal_prometheus",
            "risk_only_equal_resource", "burden_only_equal_resource",
            "cate_estimator_then_sort_equal_resource",
            "random_equal_resource", "componentwise_topk_oracle_upper_bound",
        ],
        "value": [
            current_value, learned_value, risk_value, burden_value,
            cate_value, random_value, oracle_value,
        ],
    }).to_csv(out / "policy_evaluation.csv", index=False)
    write_json(out / "policy_benchmark_diagnostics.json", {
        "risk_only": risk_diagnostics,
        "burden_only": burden_diagnostics,
        "cate_estimator_then_sort": cate_diagnostics,
        "random": random_diagnostics,
        "componentwise_topk_oracle_bound": oracle_diagnostics,
    })
    level_summary = assigned.groupby(["assigned_causal_level", "recommended_package"]).agg(
        n=("patient_id", "size"), mean_risk=("prognostic_risk_score", "mean")
    ).reset_index()
    level_summary.to_csv(out / "level_summary.csv", index=False)
    write_json(out / "transition_metadata.json", metadata)
    write_json(out / "transition_selection.json", selection)
    write_json(out / "policy_summary.json", policy)
    manifest = {
        "run_id": run_id,
        "method_version": METHOD_VERSION,
        "model_name": "PROMETHEUS",
        "model_variant": mode,
        "nuisance_protocol": "repeated_strict_partition_grouped_cross_fitting",
        "nuisance_repeats": int(nuisance_config["repeats"]),
        "nuisance_repeat_seed_stride": int(nuisance_config.get("repeat_seed_stride", 1009)),
        "supervision": "median_aggregated_repetition_specific_doubly_robust_signals",
        "min_pair_direction_agreement": float(model_config["min_pair_direction_agreement"]),
        "ranking_objective": "pairwise_causal_ranking_loss",
        "contrastive_objective": "causal_contrastive_regularization_on_latent_representation",
        "supervision_updates_from_ranker": False,
        "cross_transition_pairs": False,
        "seed": seed,
        "device": run.get("device", "cpu"),
        "sample_size": len(cohort),
        "source_cache": data["cohort_cache"],
        "duration_seconds": time.time() - started,
        "scenario": run["scenario"],
        "shared_effect_correlation": float(simulation_config["shared_effect_correlation"]),
        "overlap_strength": float(simulation_config["overlap_strength"]),
        "risk_benefit_alignment": float(simulation_config["risk_benefit_alignment"]),
        "transition_imbalance_rho": float(simulation_config["transition_imbalance_rho"]),
        "transition_sample_sizes": {
            name: int(metadata[name]["realized_sample_n"]) for name in TRANSITIONS
        },
        "transition_capacities": {
            name: float(metadata[name]["capacity"]) for name in TRANSITIONS
        },
        "conditional_exchangeability_by_construction": all(
            metadata[name]["conditional_exchangeability_by_construction"] for name in TRANSITIONS
        ),
        "direct_ranking_declaration": (
            "The method learns transition-specific causal order directly and never estimates "
            "individual CATEs for sorting."
        ),
        "priority_score_semantics": "ordinal_within_transition_not_calibrated_treatment_effect",
        "oracle_training_columns": [],
        "simulation_only": True,
    }
    write_json(out / "manifest.json", manifest)
    _write_report(out, level_summary, selection, transition_metrics, macro_metrics, policy, manifest)
    return out


def _write_report(out, levels, selection, metrics, macro_metrics, policy, manifest):
    transition_rows = []
    for name, metric in metrics.items():
        transition_rows.append({
            "transition": name,
            "capacity": selection[name]["capacity"],
            "selected": selection[name]["selected_count"],
            "AUTOC": metric["autoc"],
            "heldout DR AUTOC": metric["observed_autoc"],
            "concordance": metric["pairwise_concordance"],
            "overlap coverage": metric["overlap_coverage"],
        })
    def markdown_table(frame: pd.DataFrame) -> str:
        frame = frame.copy()
        columns = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header, divider, *body])

    text = f"""# PROMETHEUS revised causal-ranking objective

Training mode: `{manifest['model_variant']}`

The implementation uses repeated patient-grouped nuisance cross-fitting, median-aggregated
transition-specific doubly robust signals, and ranking pairs whose direction is stable
across repetitions. A unified transition-conditioned ranker combines standard pairwise
causal ranking loss with optional causal contrastive regularization on the latent
representation. Scores are ordinal and are comparable only within a transition; they are
not calibrated treatment-effect estimates.

No oracle columns, potential outcomes, or latent ranks enter fitting, pair construction,
response-bin fitting, early stopping, or capacity allocation. This is a semi-synthetic
methodological experiment, not evidence of clinical effectiveness or deployment readiness.

## Transition performance

{markdown_table(pd.DataFrame(transition_rows))}

## Macro average

{markdown_table(pd.DataFrame([macro_metrics]))}

## Assigned levels

{markdown_table(levels)}

## Evaluation-only policy comparison

{markdown_table(pd.DataFrame([policy]))}
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "prometheus_protocol_run.md").write_text(text, encoding="utf-8")
