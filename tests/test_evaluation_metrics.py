import numpy as np
import pandas as pd

from causal_population_ranking.evaluation import (
    evaluate_profile_ranking_scores,
    evaluate_profile_recommendations,
    linear_calibration_metrics,
    observational_ranking_metrics,
    oracle_ranking_metrics,
    pairwise_concordance,
)


def test_oracle_metrics_preserve_orientation_across_profiles():
    benefit = np.arange(20, dtype=float)
    profiles = np.where(benefit % 2 == 0, "profile_a", "profile_b")
    good = oracle_ranking_metrics(
        benefit, benefit, profiles, fractions=(0.10,), seed=17
    )
    bad = oracle_ranking_metrics(
        -benefit, benefit, profiles, fractions=(0.10,), seed=17
    )
    assert good["global_pairwise_concordance"] == 1.0
    assert good["within_profile_pairwise_concordance"] == 1.0
    assert good["cross_profile_pairwise_concordance"] == 1.0
    assert bad["global_pairwise_concordance"] == 0.0
    assert good["oracle_top_10pct_recovery"] == 1.0
    assert good["regret_at_10pct"] == 0.0
    assert good["oracle_evaluation_only"] is True


def test_pairwise_concordance_does_not_penalize_oracle_ties():
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    score = np.array([0.0, 0.1, 0.9, 1.0])
    assert pairwise_concordance(score, truth, max_pairs=100000, seed=3) == 1.0


def test_observational_metrics_use_heldout_signal_and_report_overlap():
    signal = np.arange(20, dtype=float)
    result = observational_ranking_metrics(
        score=signal,
        heldout_dr_signal=signal,
        propensity=np.full(20, 0.5),
        care_profile=np.where(signal % 2 == 0, "profile_a", "profile_b"),
        patient_ids=np.arange(20),
        fractions=(0.10,),
        bootstrap_samples=4,
        seed=23,
    )
    assert result["global_pairwise_concordance"] == 1.0
    assert result["overlap_coverage"] == 1.0
    assert result["oracle_target"] is False
    assert result["heldout_dr_rate_bootstrap_samples"] == 4


def test_profile_ranking_table_separates_global_and_within_profile_scopes():
    patient_ids = [f"p{index}" for index in range(8)]
    profiles = ["profile_a"] * 4 + ["profile_b"] * 4
    signal = np.arange(8, dtype=float)
    base = pd.DataFrame({
        "patient_id": patient_ids,
        "care_profile_id": profiles,
        "split": "test",
    })
    scores = pd.concat([
        base.assign(
            method="global_rank_only",
            method_score=signal,
            score_role="globally_comparable_priority_score",
            oracle_used=False,
        ),
        base.assign(
            method="independent_profile_rankers",
            method_score=signal,
            score_role="within_profile_only_ordinal_baseline",
            oracle_used=False,
        ),
    ], ignore_index=True)
    supervision = base.assign(
        dr_pseudo_outcome=signal,
        e_hat=0.5,
        causal_supervision_status="supported",
    )
    truth = base[["patient_id", "care_profile_id"]].assign(
        true_profile_benefit=signal,
        evaluation_only=True,
    )
    metrics = evaluate_profile_ranking_scores(
        scores,
        supervision,
        truth,
        fractions=(0.25,),
        seed=29,
    )
    global_metrics = metrics.loc[metrics.method.eq("global_rank_only")]
    independent = metrics.loc[metrics.method.eq("independent_profile_rankers")]
    assert "global_pairwise_concordance" in set(global_metrics.metric)
    assert "global_pairwise_concordance" not in set(independent.metric)
    assert "within_profile_pairwise_concordance" in set(independent.metric)
    assert global_metrics.loc[
        global_metrics.metric.eq("global_pairwise_concordance") & global_metrics.uses_oracle
    ].value.iloc[0] == 1.0
    assert independent.globally_comparable.eq(False).all()


def test_linear_calibration_metrics_have_expected_orientation():
    prediction = np.arange(10, dtype=float)
    target = 2.0 * prediction + 3.0
    metrics = linear_calibration_metrics(prediction, target, "validation")
    assert np.isclose(metrics["validation_calibration_slope"], 2.0)
    assert np.isclose(metrics["validation_calibration_intercept"], 3.0)


def test_recommendation_metrics_are_explicitly_oracle_evaluation_only():
    recommendations = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "split": ["test", "test"],
        "baseline_need_level": [2, 3],
        "current_care_profile": ["current", "current"],
        "current_care_profile_level": [2, 3],
        "recommended_actionable_level": [3, 3],
        "recommended_profile_id": ["profile_a", None],
        "recommendation_abstained": [False, True],
        "eligible_supported_candidate_count": [1, 1],
        "score_range_supported_candidate_count": [1, 1],
    })
    opportunities = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "care_profile_id": ["profile_a", "profile_a"],
        "care_profile_level": [3, 4],
        "eligibility": [True, True],
        "discretionary_rank_candidate": [True, True],
        "empirical_support": [True, True],
        "current_care_profile_level": [2, 3],
    })
    truth = pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "care_profile_id": ["profile_a", "profile_a"],
        "true_profile_benefit": [5.0, 1.0],
        "evaluation_only": [True, True],
    })
    metrics = evaluate_profile_recommendations(
        recommendations,
        truth,
        opportunities,
        benefit_threshold_days=2.0,
    ).set_index("metric")
    assert metrics.loc["recommendation_rate", "value"] == 0.5
    assert metrics.loc["expected_recommended_true_benefit_days", "value"] == 2.5
    assert metrics.loc["mean_recommendation_regret_days", "value"] == 0.0
    assert metrics.oracle_evaluation_only.all()
    assert metrics.loc["recommendation_rate", "uses_oracle"] == False
    assert metrics.loc["oracle_profile_exact_agreement", "uses_oracle"] == True
