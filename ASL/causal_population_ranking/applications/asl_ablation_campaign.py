"""Fast paired ablations for PROMETHEUS on real ASL covariates.

For each scenario/run pair the ASL sample, semi-synthetic treatment/outcome DGP,
splits and repeated doubly robust supervision are built once.  Ranker variants
then reuse exactly those objects, making the comparison paired and avoiding the
large cost of refitting nuisance models for every ablation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .asl_semisynthetic_campaign import ASL_RESPONSE_SCENARIOS
from .asl_semisynthetic_case import (
    ASL_FEATURE_COLUMNS,
    _asl_patient_profile_confusion,
    _asl_profile_diagnostics,
    build_asl_supervision,
    generate_asl_semisynthetic_cohort,
)
from .contrastive_ranker import fit_contrastive_causal_ranker
from .dm77_case import (
    PROFILE_COST,
    _concordance,
    _cross_profile_concordance,
    _milp_allocate,
    _ndcg_fraction,
    _within_patient_recommendation_accuracy,
    sample_pairs,
)


ASL_ABLATION_VARIANTS = (
    # Contrastive-pretraining protocol.
    "full",
    "contrastive_pretrained_retained",
    "contrastive_pretrained_finetuned",
    "contrastive_pretrained_frozen",
    "random_init",
    "joint_uact",
    # Component and backward-compatible ablations.
    "no_contrastive",
    "legacy_triplet",
    "uact_no_uncertainty",
    "no_calibration",
    "no_within_patient_auxiliary",
    "rank_only",
)



def _interval(values: np.ndarray, seed: int, repetitions: int = 3000) -> tuple[float, float]:
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _variant_config(base: Mapping[str, Any], variant: str) -> dict[str, Any]:
    config = copy.deepcopy(base)
    settings = config["asl_semisynthetic"]
    ranking = settings["ranking"]
    pretraining = ranking.setdefault("contrastive_pretraining", {})
    finetuning = ranking.setdefault("ranking_finetuning", {})
    causal = ranking.setdefault("causal_contrastive", {})
    legacy = ranking.setdefault("legacy_triplet", ranking.setdefault("triplet", {}))

    def disable_pretraining() -> None:
        pretraining["enabled"] = False
        pretraining["epochs"] = 0

    def disable_retention() -> None:
        causal["enabled"] = False
        causal["weight"] = 0.0
        causal["maximum_anchors"] = 0

    def disable_legacy() -> None:
        legacy["enabled"] = False
        legacy["weight"] = 0.0
        legacy["maximum_triplets"] = 0

    # Full is the contrastively pretrained encoder, fine-tuned with a small
    # UACT geometry-retention term.
    if variant in {"full", "contrastive_pretrained_retained"}:
        pretraining["enabled"] = True
        finetuning["freeze_encoder_entire_finetuning"] = False
        finetuning["freeze_encoder_epochs"] = int(
            finetuning.get("freeze_encoder_epochs", 3)
        )
        causal["enabled"] = True
        causal["weight"] = float(causal.get("weight", 0.0005) or 0.0005)
        disable_legacy()
        return config

    if variant in {"random_init", "no_contrastive"}:
        disable_pretraining()
        disable_retention()
        disable_legacy()
        finetuning["freeze_encoder_epochs"] = 0
        finetuning["freeze_encoder_entire_finetuning"] = False
        return config

    if variant == "contrastive_pretrained_frozen":
        pretraining["enabled"] = True
        disable_retention()
        disable_legacy()
        finetuning["freeze_encoder_epochs"] = 0
        finetuning["freeze_encoder_entire_finetuning"] = True
        return config

    if variant == "contrastive_pretrained_finetuned":
        pretraining["enabled"] = True
        disable_retention()
        disable_legacy()
        finetuning["freeze_encoder_entire_finetuning"] = False
        finetuning["freeze_encoder_epochs"] = int(
            finetuning.get("freeze_encoder_epochs", 3)
        )
        return config

    if variant == "joint_uact":
        disable_pretraining()
        disable_legacy()
        causal["enabled"] = True
        causal["mode"] = "uncertainty_aware_ordinal_triplet"
        causal["weight"] = 0.0025
        finetuning["freeze_encoder_epochs"] = 0
        finetuning["freeze_encoder_entire_finetuning"] = False
        return config

    if variant == "legacy_triplet":
        disable_pretraining()
        disable_retention()
        legacy["enabled"] = True
        legacy["weight"] = float(legacy.get("weight", 0.005) or 0.005)
        legacy["maximum_triplets"] = max(
            int(legacy.get("maximum_triplets", 6000)), 1
        )
        finetuning["freeze_encoder_epochs"] = 0
        finetuning["freeze_encoder_entire_finetuning"] = False
        return config

    if variant == "uact_no_uncertainty":
        pretraining["enabled"] = True
        pretraining["uncertainty_filtering"] = False
        pretraining["uncertainty_weighting"] = False
        pretraining["uncertainty_max_quantile"] = 1.0
        causal["enabled"] = True
        causal["uncertainty_filtering"] = False
        causal["uncertainty_weighting"] = False
        causal["uncertainty_max_quantile"] = 1.0
        disable_legacy()
        return config

    if variant in {"no_calibration", "no_within_patient_auxiliary", "rank_only"}:
        # These component ablations keep the same pretrained initialization.
        pretraining["enabled"] = True
        disable_legacy()
        if variant in {"no_calibration", "rank_only"}:
            calibration = ranking.setdefault("calibration", {})
            calibration["enabled"] = False
            calibration["weight"] = 0.0
        if variant in {"no_within_patient_auxiliary", "rank_only"}:
            within = ranking.setdefault("within_unit_auxiliary", {})
            within["enabled"] = False
            within["weight"] = 0.0
            within["maximum_pairs"] = 0
        if variant == "rank_only":
            disable_retention()
        return config

    raise ValueError(f"Unknown ASL ablation variant: {variant}")



def _ranker_for_variant(supervision, settings: Mapping[str, Any], smoke: bool):
    ranking = copy.deepcopy(settings["ranking"])
    if smoke:
        ranking.update({
            "epochs": 20,
            "patience": 5,
            "minimum_training_epochs": 12,
            "maximum_contrastive_pairs": 1600,
        })
        legacy = ranking.setdefault("legacy_triplet", ranking.setdefault("triplet", {}))
        legacy["maximum_triplets"] = min(
            int(legacy.get("maximum_triplets", 2000)),
            2000,
        )
        causal = ranking.setdefault("causal_contrastive", {})
        causal["maximum_anchors"] = min(
            int(causal.get("maximum_anchors", 2000)),
            2000,
        )
        causal["warmup_epochs"] = min(
            int(causal.get("warmup_epochs", 2)),
            2,
        )
        causal["ramp_epochs"] = min(
            int(causal.get("ramp_epochs", 4)),
            4,
        )
        causal["minimum_post_ramp_epochs"] = min(
            int(causal.get("minimum_post_ramp_epochs", 2)),
            2,
        )
        pretraining = ranking.setdefault("contrastive_pretraining", {})
        if bool(pretraining.get("enabled", False)):
            pretraining["epochs"] = min(int(pretraining.get("epochs", 6)), 6)
            pretraining["patience"] = min(int(pretraining.get("patience", 3)), 3)
            pretraining["maximum_anchors"] = min(
                int(pretraining.get("maximum_anchors", 1500)), 1500
            )
            pretraining["maximum_validation_anchors"] = min(
                int(pretraining.get("maximum_validation_anchors", 800)), 800
            )
            pretraining["maximum_validation_triplets"] = min(
                int(pretraining.get("maximum_validation_triplets", 1000)), 1000
            )
        finetuning = ranking.setdefault("ranking_finetuning", {})
        finetuning["freeze_encoder_epochs"] = min(
            int(finetuning.get("freeze_encoder_epochs", 2)), 2
        )
        ranking.setdefault("within_unit_auxiliary", {})["maximum_pairs"] = min(
            int(ranking.setdefault("within_unit_auxiliary", {}).get("maximum_pairs", 4000)),
            4000,
        )
    pairs = settings["pairs"]
    pair_common = {
        "minimum_repeat_agreement": float(pairs.get("minimum_repeat_agreement", 0.70)),
        "pair_type_fractions": pairs.get("pair_type_fractions", {
            "within_unit": 0.50,
            "within_treatment": 0.20,
            "global_cross_treatment": 0.30,
        }),
        "top_region_fraction": float(pairs.get("top_region_fraction", 0.25)),
        "top_pair_multiplier": float(pairs.get("top_pair_multiplier", 2.0)),
        "gap_clip_quantile": float(pairs.get("gap_clip_quantile", 0.95)),
    }
    train_pairs = sample_pairs(
        supervision.opportunities,
        supervision.signal_columns,
        split="rank_train",
        maximum_pairs=int(6000 if smoke else pairs["maximum_train_pairs"]),
        minimum_gap=float(pairs["minimum_gap"]),
        seed=int(pairs["train_seed"]),
        **pair_common,
    )
    validation_pairs = sample_pairs(
        supervision.opportunities,
        supervision.signal_columns,
        split="validation",
        maximum_pairs=int(1800 if smoke else pairs["maximum_validation_pairs"]),
        minimum_gap=float(pairs["minimum_gap"]),
        seed=int(pairs["validation_seed"]),
        **pair_common,
    )
    return fit_contrastive_causal_ranker(
        supervision.opportunities,
        train_pairs,
        validation_pairs,
        feature_columns=ASL_FEATURE_COLUMNS,
        treatment_column="profile_name",
        signal_columns=supervision.signal_columns,
        unit_id_column="patient_id",
        config=ranking,
    )


def _evaluate_ranker(cohort, supervision, ranker, settings: Mapping[str, Any]):
    test = supervision.opportunities.loc[
        supervision.opportunities.split.eq("test")
    ].reset_index(drop=True)
    score = np.asarray(ranker.score(test), dtype=float)
    profile = test.profile_id.to_numpy(int)
    patient = test.patient_id.astype(str).to_numpy()
    oracle = cohort.oracle.set_index("patient_id")
    truth = np.asarray([
        oracle.loc[patient_id, f"tau_{profile_id}"]
        for patient_id, profile_id in zip(test.patient_id, profile)
    ], dtype=float)
    test_count = test.patient_id.nunique()
    allocation = settings["allocation"]
    capacities = {
        1: max(1, int(round(test_count * float(allocation["capacity_fraction_monitoring"])))),
        2: max(1, int(round(test_count * float(allocation["capacity_fraction_home_care"])))),
        3: max(1, int(round(test_count * float(allocation["capacity_fraction_case_management"])))),
    }
    budget = float(allocation["budget_multiplier"]) * sum(
        capacities[p] * PROFILE_COST[p] for p in capacities
    )
    oracle_selected = _milp_allocate(test, truth, budget=budget, capacities=capacities)
    selected = _milp_allocate(test, score, budget=budget, capacities=capacities)
    oracle_value = float(truth[oracle_selected].sum())
    true_value = float(truth[selected].sum())
    policy = {
        "normalized_value": true_value / oracle_value if oracle_value else np.nan,
        "regret": oracle_value - true_value,
        "ranking_concordance": _concordance(score, truth),
        "cross_profile_concordance": _cross_profile_concordance(
            score, truth, profile, patient
        ),
        "within_patient_accuracy": _within_patient_recommendation_accuracy(
            score, truth, patient
        ),
        "ndcg_at_25pct": _ndcg_fraction(score, truth, 0.25),
        "harm_at_25pct": float(np.mean(
            truth[np.argsort(-score)[:max(1, int(round(0.25 * len(score))))]] < 0.0
        )),
        "selected_harm_rate": float(np.mean(truth[selected] < 0.0))
        if selected.any() else 0.0,
        "served": int(selected.sum()),
    }
    score_rows = pd.DataFrame({
        "patient_id": np.concatenate([patient, patient]),
        "profile_id": np.concatenate([profile, profile]),
        "method": ["PROMETHEUS-Contrastive"] * len(test) + ["Oracle"] * len(test),
        "allocator_value": np.concatenate([score, truth]),
        "true_effect_evaluation_only": np.concatenate([truth, truth]),
        "selected": np.concatenate([selected, oracle_selected]),
    })
    return policy, score_rows


def run_asl_ablation_campaign(
    base_config: Mapping[str, Any],
    input_path: str | Path,
    output_root: str | Path,
    *,
    variants: Iterable[str] = ASL_ABLATION_VARIANTS,
    scenarios: Iterable[str] = ("partially_aligned", "risk_benefit_misaligned", "mixed_response"),
    runs_per_scenario: int = 3,
    smoke: bool = False,
) -> Path:
    if runs_per_scenario < 2:
        raise ValueError("At least two runs are required for paired ablations")
    variants = tuple(variants)
    scenarios = tuple(scenarios)
    unknown_variants = set(variants) - set(ASL_ABLATION_VARIANTS)
    unknown_scenarios = set(scenarios) - set(ASL_RESPONSE_SCENARIOS)
    if unknown_variants:
        raise ValueError(f"Unknown variants: {sorted(unknown_variants)}")
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {sorted(unknown_scenarios)}")
    if "full" not in variants:
        raise ValueError("The full variant is required as paired reference")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios):
        for run_id in range(runs_per_scenario):
            offset = 10_000 * (scenario_index * runs_per_scenario + run_id + 1)
            shared_config = copy.deepcopy(base_config)
            shared_settings = shared_config["asl_semisynthetic"]
            shared_settings["response_scenario"] = scenario
            shared_settings["seed"] += offset
            shared_settings["causal_supervision"]["nuisance_seed"] += offset
            shared_settings["ranking"]["model_seed"] += offset
            shared_settings["pairs"]["train_seed"] += offset
            shared_settings["pairs"]["validation_seed"] += offset
            if smoke:
                shared_settings["causal_supervision"].update({
                    "nuisance_repeats": 3,
                    "propensity_model_max_iter": 45,
                    "outcome_model_max_iter": 50,
                })
            patient_count = int(
                shared_settings["patients_smoke" if smoke else "patients"]
            )
            if smoke:
                patient_count = min(patient_count, 3000)
            cohort = generate_asl_semisynthetic_cohort(
                input_path,
                patient_count,
                seed=int(shared_settings["seed"]),
                response_scenario=scenario,
                minimum_age=int(shared_settings.get("minimum_age", 18)),
            )
            supervision = build_asl_supervision(cohort.learner, shared_settings)

            for variant in variants:
                variant_config = _variant_config(shared_config, variant)
                settings = variant_config["asl_semisynthetic"]
                run_dir = root / "runs" / f"{scenario}__run_{run_id:03d}__{variant}"
                run_dir.mkdir(parents=True, exist_ok=False)
                ranker = _ranker_for_variant(supervision, settings, smoke)
                policy, score_rows = _evaluate_ranker(
                    cohort,
                    supervision,
                    ranker,
                    settings,
                )
                profile_diagnostics = _asl_profile_diagnostics(score_rows)
                profile_confusion = _asl_patient_profile_confusion(score_rows)
                pd.DataFrame([{**policy, "variant": variant}]).to_csv(
                    run_dir / "asl_ablation_policy_result.csv",
                    index=False,
                )
                profile_diagnostics.to_csv(
                    run_dir / "asl_profile_diagnostics.csv",
                    index=False,
                )
                profile_confusion.to_csv(
                    run_dir / "asl_patient_profile_confusion.csv",
                    index=False,
                )
                # Preserve opportunity-level decisions and the exact cohort/supervision
                # metadata used by this paired run.  These artifacts are required for
                # cohort-flow reporting, recommendation/allocation audits, profile-level
                # safety analyses, and reproducible paper tables.
                score_rows.to_csv(
                    run_dir / "asl_opportunity_scores.csv",
                    index=False,
                )
                cohort.baseline_summary.to_csv(
                    run_dir / "asl_baseline_summary.csv",
                    index=False,
                )
                supervision_quality = _asl_supervision_quality(cohort, supervision)
                supervision_quality.to_csv(
                    run_dir / "asl_supervision_quality.csv",
                    index=False,
                )
                (run_dir / "asl_cohort_audit.json").write_text(
                    json.dumps(cohort.audit, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "asl_supervision_audit.json").write_text(
                    json.dumps(supervision.audit, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "asl_ranking_audit.json").write_text(
                    json.dumps(ranker.audit, indent=2),
                    encoding="utf-8",
                )
                rows.append({
                    "scenario": scenario,
                    "run_id": run_id,
                    "variant": variant,
                    **policy,
                    "contrastive_pretraining_enabled": bool(
                        ranker.audit.get("contrastive_pretraining_enabled", False)
                    ),
                    "contrastive_pretraining_best_epoch": int(
                        ranker.audit.get("contrastive_pretraining", {}).get("best_epoch", -1)
                    ),
                    "contrastive_pretraining_validation_metric": float(
                        ranker.audit.get("contrastive_pretraining", {}).get(
                            "best_validation_selection_metric", float("nan")
                        )
                    ),
                    "contrastive_pretraining_triplets": int(
                        ranker.audit.get("contrastive_pretraining", {}).get(
                            "total_sampled_triplets", 0
                        )
                    ),
                    "finetuning_freeze_encoder_epochs": int(
                        ranker.audit.get("ranking_finetuning_freeze_encoder_epochs", 0)
                    ),
                    "finetuning_encoder_frozen_entire": bool(
                        ranker.audit.get("ranking_finetuning_freeze_encoder_entire", False)
                    ),
                    "triplet_enabled": bool(ranker.audit.get("triplet_enabled", False)),
                    "causal_contrastive_enabled": bool(
                        ranker.audit.get("causal_contrastive_enabled", False)
                    ),
                    "causal_contrastive_mode": str(
                        ranker.audit.get("causal_contrastive_mode", "disabled")
                    ),
                    "causal_contrastive_weight": float(
                        ranker.audit.get("causal_contrastive_weight", 0.0)
                    ),
                    "causal_contrastive_active_epochs": int(
                        ranker.audit.get("causal_contrastive_active_epochs", 0)
                    ),
                    "sampled_uact_triplets": int(
                        ranker.audit.get(
                            "causal_contrastive_total_sampled_triplets",
                            0,
                        )
                    ),
                    "same_patient_negative_count": int(
                        ranker.audit.get(
                            "causal_contrastive_total_same_patient_negatives",
                            0,
                        )
                    ),
                    "contrastive_gradient_norm": float(
                        ranker.audit.get(
                            "causal_contrastive_max_gradient_norm",
                            0.0,
                        )
                    ),
                    "calibration_enabled": bool(ranker.audit.get("calibration_enabled", False)),
                    "within_unit_auxiliary_enabled": bool(
                        ranker.audit.get("within_unit_auxiliary_enabled", False)
                    ),
                    "best_epoch": int(ranker.audit.get("best_epoch", -1)),
                    "validation_selection_metric": float(
                        ranker.audit.get(
                            "best_validation_selection_metric",
                            float("nan"),
                        )
                    ),
                    "validation_global_concordance": float(
                        ranker.audit.get("best_validation_metrics", {}).get(
                            "global_concordance",
                            float("nan"),
                        )
                    ),
                    "validation_cross_profile_concordance": float(
                        ranker.audit.get("best_validation_metrics", {}).get(
                            "cross_treatment_concordance",
                            float("nan"),
                        )
                    ),
                    "validation_within_patient_concordance": float(
                        ranker.audit.get("best_validation_metrics", {}).get(
                            "within_unit_concordance",
                            float("nan"),
                        )
                    ),
                    "validation_ndcg": float(
                        ranker.audit.get("best_validation_metrics", {}).get(
                            "ndcg",
                            float("nan"),
                        )
                    ),
                    "validation_topk_dr_value": float(
                        ranker.audit.get("best_validation_metrics", {}).get(
                            "topk_dr_value",
                            float("nan"),
                        )
                    ),
                })

    results = pd.DataFrame(rows)
    results.to_csv(root / "asl_ablation_results.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for (scenario, variant), group in results.groupby(["scenario", "variant"], sort=True):
        values = group.normalized_value.to_numpy(float)
        low, high = _interval(values, seed=203_000 + len(summary_rows))
        summary_rows.append({
            "scenario": scenario,
            "variant": variant,
            "runs": int(len(group)),
            "mean_normalized_value": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "mean_ranking_concordance": float(group.ranking_concordance.mean()),
            "mean_cross_profile_concordance": float(group.cross_profile_concordance.mean()),
            "mean_within_patient_accuracy": float(group.within_patient_accuracy.mean()),
            "mean_ndcg_at_25pct": float(group.ndcg_at_25pct.mean()),
            "mean_harm_at_25pct": float(group.harm_at_25pct.mean()),
            "mean_selected_harm_rate": float(group.selected_harm_rate.mean()),
            "mean_contrastive_active_epochs": float(
                group.causal_contrastive_active_epochs.mean()
            ),
            "mean_sampled_uact_triplets": float(
                group.sampled_uact_triplets.mean()
            ),
            "mean_same_patient_negative_count": float(
                group.same_patient_negative_count.mean()
            ),
            "mean_contrastive_gradient_norm": float(
                group.contrastive_gradient_norm.mean()
            ),
            "mean_validation_selection_metric": float(
                group.validation_selection_metric.mean()
            ),
            "mean_validation_global_concordance": float(
                group.validation_global_concordance.mean()
            ),
            "mean_validation_cross_profile_concordance": float(
                group.validation_cross_profile_concordance.mean()
            ),
            "mean_validation_within_patient_concordance": float(
                group.validation_within_patient_concordance.mean()
            ),
            "mean_validation_ndcg": float(group.validation_ndcg.mean()),
            "mean_validation_topk_dr_value": float(
                group.validation_topk_dr_value.mean()
            ),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "asl_ablation_summary.csv", index=False)

    paired_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_frame = results.loc[results.scenario.eq(scenario)].pivot(
            index="run_id",
            columns="variant",
            values="normalized_value",
        )
        for variant in variants:
            if variant == "full":
                continue
            diff = (scenario_frame["full"] - scenario_frame[variant]).dropna().to_numpy(float)
            low, high = _interval(diff, seed=204_000 + len(paired_rows))
            paired_rows.append({
                "scenario": scenario,
                "comparison": f"full minus {variant}",
                "runs": int(len(diff)),
                "mean_difference": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "full_wins": int((diff > 0).sum()),
                "full_losses": int((diff < 0).sum()),
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "asl_ablation_paired_differences.csv", index=False)

    validation_paired_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_frame = results.loc[results.scenario.eq(scenario)].pivot(
            index="run_id",
            columns="variant",
            values="validation_selection_metric",
        )
        for variant in variants:
            if variant == "full":
                continue
            diff = (
                scenario_frame["full"] - scenario_frame[variant]
            ).dropna().to_numpy(float)
            low, high = _interval(
                diff,
                seed=205_000 + len(validation_paired_rows),
            )
            validation_paired_rows.append({
                "scenario": scenario,
                "comparison": f"full minus {variant}",
                "runs": int(len(diff)),
                "mean_difference": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "full_wins": int((diff > 0).sum()),
                "full_losses": int((diff < 0).sum()),
                "metric": "nonoracle_validation_selection_metric",
            })
    validation_paired = pd.DataFrame(validation_paired_rows)
    validation_paired.to_csv(
        root / "asl_ablation_paired_validation_differences.csv",
        index=False,
    )

    lines = [
        "# ASL PROMETHEUS paired ablation",
        "",
        "All variants share the same ASL sample, DGP, split and repeated DR supervision within each run.",
        "A positive `full minus variant` difference favors the complete model.",
        "",
    ]
    lines.append("## Oracle policy-value comparison (evaluation only)")
    lines.append("")
    for row in paired.itertuples(index=False):
        lines.append(
            f"- **{row.scenario} — {row.comparison}**: "
            f"{row.mean_difference:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"wins/losses {row.full_wins}/{row.full_losses}."
        )
    lines.extend([
        "",
        "## Non-oracle validation comparison",
        "",
        "Use this section for discovery and hyperparameter selection.",
        "",
    ])
    for row in validation_paired.itertuples(index=False):
        lines.append(
            f"- **{row.scenario} — {row.comparison}**: "
            f"{row.mean_difference:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}], "
            f"wins/losses {row.full_wins}/{row.full_losses}."
        )
    (root / "asl_ablation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return root
