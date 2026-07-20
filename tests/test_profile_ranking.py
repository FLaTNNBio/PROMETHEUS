import numpy as np
import pandas as pd
import pytest
import torch

from causal_population_ranking.diagnostics import (
    derive_diagnostic_seeds,
    finalize_diagnostic_campaign,
    run_nonoracle_ranker_diagnostics,
)
from causal_population_ranking.experiments import (
    derive_experiment_seeds,
    summarize_experiment_metrics,
)
from causal_population_ranking.ranking import (
    pairwise_ranking_loss,
    sample_profile_pairs,
    train_profile_rankers,
)


PROFILES = ("profile_a", "profile_b")


def test_phase9_seed_derivation_reuses_only_fixed_data_and_changes_learners():
    first = derive_experiment_seeds(1901, dataset_seed=5003)
    repeated = derive_experiment_seeds(1901, dataset_seed=5003)
    second_model = derive_experiment_seeds(1931, dataset_seed=5003)
    assert first == repeated
    data_keys = {
        "population_seed", "reference_seed", "need_model_seed", "need_split_seed",
        "current_care_seed", "effect_seed", "assignment_seed", "outcome_seed",
    }
    assert {key: first[key] for key in data_keys} == {
        key: second_model[key] for key in data_keys
    }
    assert first["ranker_model_seed"] != second_model["ranker_model_seed"]
    assert len(first.values()) == len(set(first.values()))


def test_phase9_summary_has_run_paired_and_stability_intervals():
    rows = []
    for run_seed, selected, comparator in ((1601, 0.70, 0.60), (1627, 0.74, 0.61)):
        for variant, value in (
            ("global_rank_plus_contrastive", selected),
            ("global_rank_only", comparator),
        ):
            rows.append({
                "record_type": "run_metric",
                "phase": "confirmation",
                "run_seed": run_seed,
                "dataset_seed": np.nan,
                "layer": "causal_ranking",
                "variant": variant,
                "metric": "global_pairwise_concordance",
                "value": value,
                "uses_oracle": True,
                "oracle_unblinded_after_freeze": True,
                "n_runs": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "paired_comparator": "",
                "notes": "",
            })
    rows.append({
        **rows[0],
        "record_type": "stability_pair",
        "phase": "fixed_dataset_stability",
        "run_seed": "1901|1931",
        "dataset_seed": 5003,
        "layer": "causal_ranking",
        "variant": "global_rank_plus_contrastive",
        "metric": "score_spearman",
        "value": 0.91,
        "uses_oracle": False,
    })
    summary = summarize_experiment_metrics(
        pd.DataFrame(rows),
        selected_variant="global_rank_plus_contrastive",
        comparator_variant="global_rank_only",
        bootstrap_samples=100,
        bootstrap_seed=2039,
    )
    paired = summary.loc[summary.record_type.eq("paired_uncertainty_summary")]
    stability = summary.loc[summary.record_type.eq("stability_summary")]
    assert len(paired) == 1 and np.isclose(paired.iloc[0].value, 0.115)
    assert len(stability) == 1 and stability.iloc[0].value == 0.91
    assert summary.loc[
        summary.record_type.isin((
            "uncertainty_summary", "paired_uncertainty_summary", "stability_summary"
        ))
    ].ci_lower.notna().all()


def ranking_inputs(seed: int = 41):
    rng = np.random.default_rng(seed)
    patients = np.asarray([f"patient_{index:03d}" for index in range(80)])
    split = np.asarray(
        ["rank_train"] * 40 + ["validation"] * 24 + ["test"] * 16
    )
    severity = rng.normal(size=len(patients))
    learner = pd.DataFrame(
        {
            "patient_id": patients,
            "severity": severity,
            "baseline_need_score": np.clip(3.0 + severity, 1.0, 6.0),
            "prior_inpatient": np.maximum(0.0, severity + 1.0),
            "prior_emergency": np.maximum(0.0, severity + 0.5),
            "condition_distinct": np.maximum(0.0, 2.0 + severity),
            "frailty_index": np.clip(0.4 + 0.1 * severity, 0.0, 1.0),
            "functional_limitation_score": np.clip(
                0.3 + 0.1 * severity, 0.0, 1.0
            ),
            "social_fragility_score": np.clip(0.2 + 0.1 * severity, 0.0, 1.0),
            "current_care_profile": np.where(
                np.arange(len(patients)) % 2, "current_b", "current_a"
            ),
        }
    )
    rows = []
    opportunities = []
    for patient_index, patient_id in enumerate(patients):
        for profile_index, profile in enumerate(PROFILES):
            signal = severity[patient_index] + (0.8 if profile_index else -0.4)
            rows.append(
                {
                    "patient_id": patient_id,
                    "care_profile_id": profile,
                    "split": split[patient_index],
                    "dr_pseudo_outcome": signal,
                    "dr_reliability_weight": 1.0,
                    "dr_pseudo_outcome_repeat_0": signal - 0.03,
                    "dr_pseudo_outcome_repeat_1": signal + 0.03,
                    "causal_supervision_status": "supported",
                }
            )
            opportunities.append(
                {
                    "patient_id": patient_id,
                    "care_profile_id": profile,
                    "care_profile_index": profile_index,
                    "empirical_support": True,
                    "baseline_need_score": learner.loc[
                        patient_index, "baseline_need_score"
                    ],
                    "current_care_profile": learner.loc[
                        patient_index, "current_care_profile"
                    ],
                }
            )
    return (
        learner,
        pd.DataFrame(rows),
        pd.DataFrame(opportunities),
        pd.DataFrame({"patient_id": patients, "split": split}),
    )


def ranker_settings():
    return {
        "primary_variant": "global_rank_plus_contrastive",
        "variants": [
            "independent_profile_rankers",
            "global_rank_only",
            "global_rank_plus_contrastive",
            "direct_pairwise_gbdt",
            "risk_based_ranking",
            "baseline_need_based_ranking",
            "random",
        ],
        "evaluation_only_variants": ["oracle_evaluation_only"],
        "hidden_dim": 10,
        "profile_embedding_dim": 4,
        "projection_dim": 3,
        "epochs": 2,
        "patience": 2,
        "minimum_improvement": 1e-8,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "training_pairs": 96,
        "validation_pairs": 64,
        "within_profile_fraction": 0.5,
        "minimum_signal_gap": 0.05,
        "minimum_direction_agreement": 0.5,
        "maximum_pair_weight": 5.0,
        "allow_same_patient_cross_profile_pairs": False,
        "contrastive_pairs": 64,
        "contrastive_response_bins": 4,
        "contrastive_weight": 0.1,
        "contrastive_margin": 1.0,
        "gbdt_max_iter": 5,
        "gbdt_max_depth": 2,
        "gbdt_learning_rate": 0.1,
        "gbdt_reference_rows": 12,
    }


def fit_rankers():
    learner, supervision, opportunities, splits = ranking_inputs()
    return train_profile_rankers(
        learner,
        supervision,
        opportunities,
        splits,
        ranker_settings(),
        numeric_features=["severity", "baseline_need_score"],
        categorical_features=["current_care_profile"],
        profile_ids=PROFILES,
        model_seed=101,
        pair_seed=103,
        validation_pair_seed=107,
        contrastive_seed=109,
        gbdt_seed=113,
        random_seed=127,
    )


def test_profile_pair_sampling_is_seeded_stable_and_contains_valid_cross_pairs():
    _, supervision, _, _ = ranking_inputs()
    frame = supervision.loc[supervision.split.eq("rank_train")].reset_index(drop=True)
    first = sample_profile_pairs(
        frame,
        pairs=80,
        seed=17,
        minimum_signal_gap=0.05,
        minimum_direction_agreement=0.5,
        within_profile_fraction=0.5,
        allow_cross_profile=True,
    )
    second = sample_profile_pairs(
        frame,
        pairs=80,
        seed=17,
        minimum_signal_gap=0.05,
        minimum_direction_agreement=0.5,
        within_profile_fraction=0.5,
        allow_cross_profile=True,
    )
    np.testing.assert_array_equal(first.left, second.left)
    np.testing.assert_array_equal(first.right, second.right)
    assert first.is_cross_profile.any() and (~first.is_cross_profile).any()
    assert np.all(frame.patient_id.to_numpy()[first.left] != frame.patient_id.to_numpy()[first.right])
    difference = (
        frame.dr_pseudo_outcome.to_numpy()[first.left]
        - frame.dr_pseudo_outcome.to_numpy()[first.right]
    )
    np.testing.assert_array_equal(first.direction, np.sign(difference))


def test_pairwise_loss_rewards_the_correct_direct_order():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "b"],
            "care_profile_id": ["profile_a", "profile_a"],
            "dr_pseudo_outcome": [1.0, 0.0],
            "dr_reliability_weight": [1.0, 1.0],
            "dr_pseudo_outcome_repeat_0": [1.0, 0.0],
            "dr_pseudo_outcome_repeat_1": [1.1, -0.1],
        }
    )
    pairs = sample_profile_pairs(
        frame,
        pairs=1,
        seed=3,
        minimum_signal_gap=0.0,
        minimum_direction_agreement=1.0,
        within_profile_fraction=1.0,
        allow_cross_profile=False,
    )
    correct = pairwise_ranking_loss(torch.tensor([2.0, 0.0]), pairs)
    reversed_score = pairwise_ranking_loss(torch.tensor([0.0, 2.0]), pairs)
    assert correct < reversed_score


def test_all_phase5_variants_are_non_oracle_and_primary_score_is_separate():
    result = fit_rankers()
    assert result.audit["status"] == "trained"
    assert result.audit["individual_cate_estimated_then_sorted"] is False
    assert result.audit["oracle_used_for_training"] is False
    assert result.audit["validation_within_profile_pairs"] > 0
    assert result.audit["validation_cross_profile_pairs"] > 0
    assert set(result.scores.method) == set(ranker_settings()["variants"])
    primary = result.scores.method.eq("global_rank_plus_contrastive")
    assert result.scores.loc[primary, "raw_priority_score"].notna().all()
    assert result.scores.loc[~primary, "raw_priority_score"].isna().all()
    assert result.primary_model_bundle["score_semantics"].endswith("no_causal_zero")
    rank_only = result.audit["variants"]["global_rank_only"]
    contrastive = result.audit["variants"]["global_rank_plus_contrastive"]
    assert rank_only["projection_head_last_gradient_norm"] == 0.0
    assert contrastive["projection_head_last_gradient_norm"] > 0.0
    assert contrastive["profile_embedding_last_gradient_norm"] > 0.0
    assert result.audit["variants"]["independent_profile_rankers"][
        "globally_comparable"
    ] is False


def test_restart_ensemble_is_seeded_ordinal_and_reproducible():
    learner, supervision, opportunities, splits = ranking_inputs()
    settings = ranker_settings()
    settings.update({
        "primary_variant": "global_rank_only",
        "variants": ["global_rank_only"],
        "model_restarts": 3,
        "epochs": 3,
    })

    def fit():
        return train_profile_rankers(
            learner,
            supervision,
            opportunities,
            splits,
            settings,
            numeric_features=["severity", "baseline_need_score"],
            categorical_features=["current_care_profile"],
            profile_ids=PROFILES,
            model_seed=101,
            pair_seed=103,
            validation_pair_seed=107,
            contrastive_seed=109,
            gbdt_seed=113,
            random_seed=127,
        )

    first = fit()
    second = fit()
    pd.testing.assert_frame_equal(first.scores, second.scores)
    audit = first.audit["variants"]["global_rank_only"]
    assert audit["model_restarts"] == 3
    assert len(set(audit["model_restart_seeds"])) == 3
    assert audit["restart_aggregation"] == "mean_percentile_rank"
    assert np.isfinite(audit["best_validation_pairwise_loss"])
    assert first.scores.method_score.between(0.0, 1.0).all()
    assert len(first.primary_model_bundle["members"]) == 3
    assert first.audit["individual_cate_estimated_then_sorted"] is False


def test_validation_pairs_are_frozen_and_oracle_columns_are_rejected():
    first = fit_rankers()
    second = fit_rankers()
    assert first.audit["fixed_validation_pair_hash"] == second.audit[
        "fixed_validation_pair_hash"
    ]
    pd.testing.assert_frame_equal(first.validation_pairs, second.validation_pairs)
    learner, supervision, opportunities, splits = ranking_inputs()
    learner["true_profile_benefit"] = 1.0
    with pytest.raises(ValueError, match="Oracle columns"):
        train_profile_rankers(
            learner,
            supervision,
            opportunities,
            splits,
            ranker_settings(),
            numeric_features=["severity", "baseline_need_score"],
            categorical_features=["current_care_profile"],
            profile_ids=PROFILES,
            model_seed=101,
            pair_seed=103,
            validation_pair_seed=107,
            contrastive_seed=109,
            gbdt_seed=113,
            random_seed=127,
        )


def test_unknown_candidate_profile_is_rejected():
    learner, supervision, opportunities, splits = ranking_inputs()
    opportunities.loc[0, "care_profile_id"] = "unknown_profile"
    with pytest.raises(ValueError, match="Unknown care_profile_id"):
        train_profile_rankers(
            learner,
            supervision,
            opportunities,
            splits,
            ranker_settings(),
            numeric_features=["severity", "baseline_need_score"],
            categorical_features=["current_care_profile"],
            profile_ids=PROFILES,
            model_seed=101,
            pair_seed=103,
            validation_pair_seed=107,
            contrastive_seed=109,
            gbdt_seed=113,
            random_seed=127,
        )


def test_phase8_ranker_diagnostics_are_seeded_nonoracle_and_reproducible():
    learner, supervision, opportunities, splits = ranking_inputs()
    settings = ranker_settings()
    campaign = {
        "heldout_test_pairs": 48,
        "ranker_variants": ["global_rank_only", "global_rank_plus_contrastive"],
    }
    derived = derive_diagnostic_seeds(1201)
    assert len(derived) == len(set(derived.values()))
    result = run_nonoracle_ranker_diagnostics(
        learner,
        supervision,
        opportunities,
        splits,
        settings,
        scenario="baseline_identifiable",
        conditions=[
            "standard",
            "within_profile_training_only",
            "permuted_training_pair_labels",
        ],
        base_seeds=[1201],
        campaign_settings=campaign,
        numeric_features=["severity", "baseline_need_score"],
        categorical_features=["current_care_profile"],
        profile_ids=PROFILES,
    )
    assert result.audit["evaluation_partition"] == "test"
    assert result.audit["test_partition_used_for_fitting_or_early_stopping"] is False
    assert result.audit["oracle_used"] is False
    assert not result.rows.oracle_used.astype(bool).any()
    permutation = result.rows.loc[
        result.rows.training_condition.eq("permuted_training_pair_labels")
    ]
    assert (permutation.training_label_changes.astype(float) > 0).all()
    reproduction = result.rows.loc[
        result.rows.metric.eq("exact_score_hash_match")
    ]
    assert len(reproduction) == 1 and bool(reproduction.iloc[0].passed)


def test_phase8_exit_gates_use_nonoracle_heldout_and_structural_diagnostics():
    rows = []

    def metric(scenario, condition, variant, value, seed=1201, changes=0):
        rows.append({
            "row_type": "replicate_metric",
            "family": "negative_control_or_ablation",
            "diagnostic": "heldout_pair_concordance",
            "scenario": scenario,
            "base_seed": seed,
            "training_condition": condition,
            "variant": variant,
            "pair_scope": "all",
            "partition": "test",
            "metric": "heldout_pair_concordance",
            "value": value,
            "training_label_changes": changes,
            "oracle_used": False,
            "eligible_for_model_selection": False,
        })

    for scenario in ("sharp_null", "placebo_outcome"):
        metric(scenario, "standard", "global_rank_only", 0.51)
        metric(scenario, "standard", "global_rank_plus_contrastive", 0.49)
    metric("baseline_identifiable", "standard", "global_rank_only", 0.70)
    metric("baseline_identifiable", "standard", "global_rank_plus_contrastive", 0.71)
    metric(
        "baseline_identifiable",
        "permuted_training_pair_labels",
        "global_rank_only",
        0.50,
        changes=20,
    )
    metric(
        "baseline_identifiable",
        "within_profile_training_only",
        "global_rank_only",
        0.66,
    )
    metric("no_shared_response", "standard", "global_rank_only", 0.52)
    metric(
        "no_shared_response", "standard", "global_rank_plus_contrastive", 0.51
    )
    metric(
        "no_shared_response",
        "within_profile_training_only",
        "global_rank_only",
        0.53,
    )
    metric(
        "baseline_identifiable_supervised_need",
        "standard",
        "global_rank_only",
        0.68,
    )
    rows.extend([
        {
            "row_type": "replicate_check",
            "family": "reproducibility",
            "metric": "exact_score_hash_match",
            "passed": True,
            "oracle_used": False,
            "eligible_for_model_selection": False,
        },
        {
            "row_type": "sensitivity_declaration",
            "family": "identification_sensitivity",
            "metric": "identification_failure_declared",
            "value": 1.0,
            "passed": True,
            "oracle_used": False,
            "eligible_for_model_selection": False,
        },
    ])
    for threshold, rate in ((0.0, 0.50), (2.0, 0.40), (5.0, 0.20)):
        rows.append({
            "row_type": "ablation_metric",
            "family": "threshold_ablation",
            "metric": "recommendation_rate",
            "threshold": threshold,
            "value": rate,
            "oracle_used": False,
            "eligible_for_model_selection": False,
        })
    recommendation_summary = pd.DataFrame({
        "scenario": ["sharp_null", "placebo_outcome"],
        "patients": [100, 100],
        "patients_recommended": [4, 2],
    })
    allocation_summary = pd.DataFrame({
        "constraint_violations": [0, 0],
        "oracle_used": [False, False],
    })
    result = finalize_diagnostic_campaign(
        pd.DataFrame(rows),
        settings={
            "controls": {
                "null_max_median_concordance_distance_from_chance": 0.15,
                "null_max_recommendation_rate": 0.10,
                "permuted_max_median_concordance": 0.60,
                "minimum_unpermuted_minus_permuted_concordance": 0.0,
            },
            "threshold_sensitivity_days": [0.0, 2.0, 5.0],
        },
        recommendation_summary=recommendation_summary,
        allocation_summary=allocation_summary,
    )
    assert result.audit["status"] == "passed"
    assert result.audit["required_gate_count"] == 8
    gates = result.rows.loc[result.rows.row_type.eq("gate")]
    assert gates.passed.astype(bool).all()
    assert not result.rows.oracle_used.fillna(False).astype(bool).any()
