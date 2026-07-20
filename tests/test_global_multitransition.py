from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from causal_population_ranking.allocation.global_allocator import allocate_global_opportunities
from causal_population_ranking.causal_utilization.global_prometheus_runner import run_global_prometheus
from causal_population_ranking.causal_utilization.global_support import TransitionSupportOutcomePredictor
from causal_population_ranking.causal_utilization.global_simulation import (
    TRANSITION_NAMES,
    simulate_multivalued_care,
)
from causal_population_ranking.data.validation import assert_no_oracle_columns
from causal_population_ranking.evaluation.global_metrics import global_pairwise_concordance
from causal_population_ranking.ranking.calibration import MonotoneScoreCalibrator
from causal_population_ranking.ranking.global_pairs import (
    sample_global_contrastive_pairs,
    sample_global_ranking_pairs,
)
from causal_population_ranking.ranking.global_ranker import GlobalPrometheusRanker
from causal_population_ranking.ranking.losses import (
    causal_contrastive_loss,
    pairwise_causal_ranking_loss,
)
from causal_population_ranking.ranking.opportunities import OpportunityArrays
from causal_population_ranking.ranking.prometheus_ranker import _TransitionConditionedRanker


def _cohort(n=600, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"p{index:05d}" for index in range(n)],
        "age": rng.uniform(18, 92, n),
        "female": rng.integers(0, 2, n),
        "condition_distinct": rng.poisson(3.0, n),
        "prior_inpatient": rng.poisson(0.8, n),
        "prior_emergency": rng.poisson(1.1, n),
        "medication_distinct": rng.poisson(4.0, n),
        "recent_utilization_trend": rng.normal(0.0, 1.0, n),
        "multimorbidity": rng.integers(0, 2, n),
        "polypharmacy": rng.integers(0, 2, n),
        "encounter_count": rng.poisson(5.0, n),
        "days_since_encounter": rng.uniform(0, 365, n),
        "procedure_count": rng.poisson(1.0, n),
        "careplan_count": rng.poisson(0.5, n),
    })


@pytest.fixture(scope="module")
def simulation():
    return simulate_multivalued_care(_cohort(), seed=31, eligibility_mode="synthetic_clinical")


def test_multivalued_dgp_has_one_treatment_common_outcome_and_separate_oracle(simulation):
    learner, truth = simulation.learner, simulation.ground_truth.set_index("patient_id")
    assert len(learner) == learner.patient_id.nunique()
    assert set(learner.treatment_level) <= set(range(1, 7))
    assert learner.treatment_level.between(1, 6).all()
    assert learner.observed_outcome.between(0, 365).all()
    assert_no_oracle_columns(learner)
    assert not {"baseline_care_level", "current_care_level"}.intersection(learner.columns)
    selected_mean = np.asarray([
        truth.loc[patient, f"potential_outcome_{treatment}"]
        for patient, treatment in zip(learner.patient_id, learner.treatment_level)
    ])
    noise = truth.loc[learner.patient_id, "observed_outcome_noise"].to_numpy(float)
    np.testing.assert_allclose(learner.observed_outcome, selected_mean + noise)
    for index, name in enumerate(TRANSITION_NAMES, start=1):
        difference = truth[f"potential_outcome_{index + 1}"] - truth[f"potential_outcome_{index}"]
        np.testing.assert_allclose(difference, truth[f"true_benefit_{name}"], atol=1e-10)


def test_eligibility_is_nested_pre_treatment_and_adjacent_populations_are_correct(simulation):
    learner = simulation.learner
    eligibility = learner[[f"eligible_{name}" for name in TRANSITION_NAMES]].to_numpy(bool)
    assert not np.any(eligibility[:, 1:] & ~eligibility[:, :-1])
    eligibility_before = eligibility.copy()
    mutated_treatment = learner.treatment_level.sample(frac=1.0, random_state=99).to_numpy()
    assert not np.array_equal(mutated_treatment, learner.treatment_level)
    np.testing.assert_array_equal(eligibility, eligibility_before)
    different_assignment = simulate_multivalued_care(
        _cohort(), seed=32, eligibility_mode="synthetic_clinical"
    ).learner
    assert not np.array_equal(different_assignment.treatment_level, learner.treatment_level)
    np.testing.assert_array_equal(
        different_assignment[[f"eligible_{name}" for name in TRANSITION_NAMES]],
        eligibility,
    )
    for index, name in enumerate(TRANSITION_NAMES, start=1):
        frame = simulation.transition_learners[name]
        assert frame.treatment_level.isin((index, index + 1)).all()
        assert frame[f"eligible_{name}"].all()
        np.testing.assert_array_equal(
            frame.transition_treatment,
            (frame.treatment_level == index + 1).astype(int),
        )
    assert learner.groupby("patient_id").split.nunique().max() == 1


def test_multivalued_simulation_is_seed_reproducible_and_oracle_mutation_is_isolated():
    first = simulate_multivalued_care(_cohort(300), seed=17, eligibility_mode="all")
    second = simulate_multivalued_care(_cohort(300), seed=17, eligibility_mode="all")
    pd.testing.assert_frame_equal(first.learner, second.learner)
    original = first.learner.copy(deep=True)
    first.ground_truth.loc[:, "true_benefit_1_to_2"] += 10_000
    pd.testing.assert_frame_equal(first.learner, original)


def test_test_treatment_and_outcome_do_not_change_non_test_support_models(simulation):
    frame = simulation.transition_learners["1_to_2"].copy()
    features = simulation.feature_columns
    first = TransitionSupportOutcomePredictor("linear", 101).fit(frame, features)
    x = frame.loc[:20, features].to_numpy(float)
    first_prediction = first.predict(x)
    test = frame.split.astype(str) == "test"
    mutated = frame.copy()
    mutated.loc[test, "transition_treatment"] = 1 - mutated.loc[test, "transition_treatment"]
    mutated.loc[test, "observed_outcome"] += 1000.0
    second = TransitionSupportOutcomePredictor("linear", 101).fit(mutated, features)
    for left, right in zip(first_prediction, second.predict(x)):
        np.testing.assert_allclose(left, right)


def _opportunities(rows_per_transition=24, seed=3, patient_prefix="train"):
    rng = np.random.default_rng(seed)
    transition = np.repeat(np.arange(5), rows_per_transition)
    n = len(transition)
    within = np.tile(np.linspace(-4, 4, rows_per_transition), 5)
    signal = within + 3.0 * transition
    repeated = np.column_stack([
        signal + rng.normal(0, 0.05, n) for _ in range(3)
    ])
    patient = np.asarray([f"{patient_prefix}_{index % rows_per_transition:03d}" for index in range(n)])
    return OpportunityArrays(rng.normal(size=(n, 6)), transition, patient, signal, repeated)


def test_global_pair_sampler_contains_balanced_real_cross_transition_pairs():
    opportunities = _opportunities()
    signal_before = opportunities.signal.copy()
    pairs = sample_global_ranking_pairs(
        opportunities, within_pairs=100, cross_pairs=200,
        min_signal_gap_days=0.2, min_direction_agreement=0.66, seed=19,
    )
    assert pairs.diagnostics["valid_within_pairs"] == 100
    assert pairs.diagnostics["valid_cross_pairs"] == 200
    cross = pairs.is_cross
    assert np.all(
        opportunities.transition_index[pairs.left[cross]]
        != opportunities.transition_index[pairs.right[cross]]
    )
    cross_counts = [
        count for block, count in pairs.diagnostics["pair_counts_per_transition_pair"].items()
        if block.split(":")[0] != block.split(":")[1]
    ]
    assert max(cross_counts) - min(cross_counts) <= 1
    np.testing.assert_array_equal(opportunities.signal, signal_before)
    contrastive = sample_global_contrastive_pairs(
        opportunities, 150, 29, positive_max_gap_days=0.5,
        negative_min_gap_days=8.0, source="causal",
    )
    assert contrastive.is_cross.any()
    assert contrastive.diagnostics["threshold_scale"] == "pooled_outcome_days"
    assert contrastive.diagnostics["positive_pairs"] > 0
    assert contrastive.diagnostics["negative_pairs"] > 0


def test_contrastive_sampler_balances_positive_and_negative_pairs_when_available():
    signal = np.asarray([0.0, 0.1, 0.2, 20.0, 40.0, 60.0])
    opportunities = OpportunityArrays(
        x=np.arange(12, dtype=float).reshape(6, 2),
        transition_index=np.zeros(6, dtype=np.int64),
        patient_ids=np.asarray([f"p{index}" for index in range(6)]),
        signal=signal,
        repeated_signal=np.repeat(signal[:, None], 3, axis=1),
    )
    pairs = sample_global_contrastive_pairs(
        opportunities,
        pairs=6,
        seed=31,
        positive_max_gap_days=0.5,
        negative_min_gap_days=10.0,
        source="causal",
        pair_scope="within_action",
        require_stable_direction=True,
        min_direction_agreement=0.8,
    )
    assert pairs.diagnostics["positive_pairs"] == 3
    assert pairs.diagnostics["negative_pairs"] == 3
    assert pairs.diagnostics["positive_pairs_per_transition_pair"] == {"0:0": 3}
    assert pairs.diagnostics["negative_pairs_per_transition_pair"] == {"0:0": 3}


def test_v2_contrastive_sampler_balances_within_cross_and_tracks_reliability():
    opportunities = _opportunities(rows_per_transition=20, seed=37)
    pairs = sample_global_contrastive_pairs(
        opportunities,
        pairs=210,
        seed=41,
        positive_max_gap_days=0.5,
        negative_min_gap_days=4.0,
        source="causal",
        pair_scope="balanced_mixed",
        require_stable_direction=True,
        min_direction_agreement=0.66,
        reliability_weighting=True,
    )
    assert pairs.diagnostics["requested_within_pairs"] == 105
    assert pairs.diagnostics["requested_cross_pairs"] == 105
    assert pairs.diagnostics["within_pairs"] == 105
    assert pairs.diagnostics["cross_pairs"] == 105
    assert pairs.diagnostics["positive_pairs"] > 0
    assert pairs.diagnostics["negative_pairs"] > 0
    assert pairs.diagnostics["reliability_weighting"] is True
    assert np.all((pairs.weights >= 0.05) & (pairs.weights <= 1.0))
    assert np.all((pairs.direction_agreement >= 0.66) & (pairs.direction_agreement <= 1.0))


def test_v2_separate_contrastive_head_does_not_backpropagate_into_scoring_head():
    torch.manual_seed(43)
    model = _TransitionConditionedRanker(
        4, 3, 8, 5, separate_contrastive_head=True
    )
    x = torch.randn(12, 4)
    transition = torch.tensor([0, 1, 2] * 4)
    _, rank_representation = model(x, transition)
    contrastive_representation = model.contrastive_embedding(x, transition)
    assert rank_representation.shape == contrastive_representation.shape == (12, 5)
    assert model.contrastive_projection_head is not None

    model.zero_grad()
    loss = causal_contrastive_loss(
        contrastive_representation[:6],
        contrastive_representation[6:],
        torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32),
    )
    loss.backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in model.contrastive_projection_head.parameters()
        if parameter.grad is not None
    ) > 0
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in model.clinical_encoder.parameters()
        if parameter.grad is not None
    ) > 0
    assert all(parameter.grad is None for parameter in model.scoring_head.parameters())
    assert all(parameter.grad is None for parameter in model.projection_head.parameters())


def test_forward_accepts_mixed_transition_indices_and_both_pair_losses_have_gradients():
    torch.manual_seed(5)
    model = _TransitionConditionedRanker(4, 5, 8, 3)
    x = torch.randn(8, 4)
    transition = torch.tensor([0, 0, 1, 2, 3, 4, 1, 3])
    score, representation = model(x, transition)
    assert score.shape == (8,)
    assert representation.shape == (8, 3)
    for left, right in ((torch.tensor([0, 2]), torch.tensor([1, 6])),
                        (torch.tensor([0, 2]), torch.tensor([3, 5]))):
        model.zero_grad()
        current_score, _ = model(x, transition)
        loss = pairwise_causal_ranking_loss(
            current_score[left], current_score[right], torch.ones(2)
        )
        loss.backward()
        assert sum(float(parameter.grad.abs().sum()) for parameter in model.parameters() if parameter.grad is not None) > 0


def test_global_ranker_trains_with_within_and_cross_losses_and_mixed_prediction():
    training = _opportunities(18, seed=7, patient_prefix="train")
    validation = _opportunities(10, seed=9, patient_prefix="validation")
    ranker = GlobalPrometheusRanker(
        training_mode="unified_global_ranker",
        within_pairs_per_epoch=50,
        cross_pairs_per_epoch=100,
        beta_cross=0.5,
        min_signal_gap_days=0.1,
        min_pair_direction_agreement=0.66,
        lambda_con=0.0,
        epochs=1,
        batch_size=32,
        hidden_dim=12,
        latent_dim=5,
        patience=1,
        seed=23,
    ).fit(training, validation)
    row = ranker.history[0]
    assert row["valid_within_pairs"] > 0 and row["valid_cross_pairs"] > 0
    assert np.isfinite(row["within_ranking_loss"]) and np.isfinite(row["cross_ranking_loss"])
    score = ranker.predict_opportunity_scores(validation.x, validation.transition_index)
    assert score.shape == (len(validation.x),)
    assert ranker.training_diagnostics["cross_transition_pairs"] is True


def test_permuted_pair_label_negative_control_is_seeded_and_train_only():
    training = _opportunities(18, seed=17, patient_prefix="permuted_train")
    validation = _opportunities(10, seed=19, patient_prefix="permuted_validation")
    settings = dict(
        training_mode="unified_global_ranker",
        within_pairs_per_epoch=50,
        cross_pairs_per_epoch=100,
        min_signal_gap_days=0.1,
        min_pair_direction_agreement=0.66,
        lambda_con=0.0,
        epochs=1,
        batch_size=32,
        hidden_dim=12,
        latent_dim=5,
        patience=1,
        permute_training_pair_labels=True,
        seed=29,
    )
    first = GlobalPrometheusRanker(**settings).fit(training, validation)
    second = GlobalPrometheusRanker(**settings).fit(training, validation)
    np.testing.assert_allclose(
        first.predict_opportunity_scores(validation.x, validation.transition_index),
        second.predict_opportunity_scores(validation.x, validation.transition_index),
    )
    assert first.training_diagnostics["training_pair_labels_permuted"] is True
    assert first.training_diagnostics["fixed_validation_pair_diagnostics"].get(
        "training_pair_labels_permuted", False
    ) is False


def test_global_causal_contrastive_v2_uses_separate_head_and_mixed_pairs():
    training = _opportunities(18, seed=47, patient_prefix="train_v2")
    validation = _opportunities(10, seed=53, patient_prefix="validation_v2")
    ranker = GlobalPrometheusRanker(
        training_mode="prometheus_global_causal_contrastive_v2",
        within_pairs_per_epoch=50,
        cross_pairs_per_epoch=100,
        beta_cross=0.5,
        min_signal_gap_days=0.1,
        min_pair_direction_agreement=0.66,
        contrastive_pairs_per_epoch=210,
        positive_max_gap_days=0.5,
        negative_min_gap_days=4.0,
        lambda_con=0.01,
        contrastive_pair_scope="balanced_mixed",
        contrastive_require_stable_direction=True,
        contrastive_reliability_weighting=True,
        epochs=1,
        batch_size=32,
        hidden_dim=12,
        latent_dim=5,
        patience=1,
        seed=59,
    ).fit(training, validation)
    diagnostics = ranker.training_diagnostics
    contrastive = diagnostics["best_contrastive_diagnostics"]
    assert diagnostics["separate_contrastive_head"] is True
    assert diagnostics["contrastive_pair_scope"] == "balanced_mixed"
    assert diagnostics["contrastive_reliability_weighting"] is True
    assert contrastive["within_pairs"] == 105
    assert contrastive["cross_pairs"] == 105
    assert contrastive["reliability_weighting"] is True
    assert ranker.history[0]["gradient_norm_contrastive_head"] > 0


def test_pooled_isotonic_calibrator_is_monotone_and_clips_out_of_bounds():
    score = np.asarray([-2, -1, 0, 1, 2, 3], dtype=float)
    signal = np.asarray([-4, -1, -2, 3, 2, 8], dtype=float)
    calibrator = MonotoneScoreCalibrator().fit(score, signal)
    prediction = calibrator.predict(np.linspace(-10, 10, 101))
    assert np.all(np.diff(prediction) >= -1e-12)
    assert calibrator.diagnostics["oracle_target"] is False
    assert calibrator.diagnostics["out_of_bounds"] == "clip"


def _allocation_frame():
    rows = []
    benefits = {
        "a": [8, 7, 6, 5, 4],
        "b": [6, 20, 1, 1, 1],
        "c": [5, -1, 30, 1, 1],
    }
    for patient, values in benefits.items():
        for transition, benefit in enumerate(values):
            rows.append({
                "patient_id": patient,
                "transition_index": transition,
                "calibrated_benefit": benefit,
                "cost": transition + 1.0,
                "eligible": not (patient == "c" and transition >= 4),
                "supported": not (patient == "b" and transition >= 3),
            })
    return pd.DataFrame(rows)


def test_exact_allocator_respects_every_constraint_and_dominates_greedy_on_objective():
    frame = _allocation_frame()
    exact = allocate_global_opportunities(
        frame, shared_budget=12.0, transition_capacities={"1_to_2": 3, "2_to_3": 2},
        require_empirical_support=True, allocate_negative_predicted_benefit=False,
        exact_solver=True,
    )
    greedy = allocate_global_opportunities(
        frame, shared_budget=12.0, transition_capacities={"1_to_2": 3, "2_to_3": 2},
        require_empirical_support=True, allocate_negative_predicted_benefit=False,
        exact_solver=False,
    )
    diagnostics = exact.diagnostics
    assert diagnostics["budget_violation"] == 0
    assert diagnostics["eligibility_violations"] == 0
    assert diagnostics["support_violations"] == 0
    assert diagnostics["precedence_violations"] == 0
    exact_value = float(frame.loc[exact.selected, "calibrated_benefit"].sum())
    greedy_value = float(frame.loc[greedy.selected, "calibrated_benefit"].sum())
    assert exact_value >= greedy_value


def test_global_cross_transition_concordance_is_correct_for_perfect_and_inverse_scores():
    transition = np.repeat(np.arange(5), 6)
    benefit = np.linspace(-5, 12, len(transition)) + 0.1 * transition
    perfect = global_pairwise_concordance(benefit, benefit, transition, seed=3)
    inverse = global_pairwise_concordance(-benefit, benefit, transition, seed=3)
    assert perfect["within_transition_concordance"] == 1.0
    assert perfect["cross_transition_concordance"] == 1.0
    assert inverse["global_concordance"] == 0.0


def test_legacy_local_ranker_remains_importable():
    from causal_population_ranking.ranking.prometheus_ranker import PrometheusRanker

    assert PrometheusRanker is not None


def test_end_to_end_global_smoke_writes_required_oracle_separated_artifacts(tmp_path):
    config = yaml.safe_load(Path("configs/prometheus/global_smoke.yaml").read_text(encoding="utf-8"))
    config["run"]["output_root"] = str(tmp_path)
    config["data"]["cohort_cache"] = "this/path/must/not/be/read.csv"
    output = run_global_prometheus(config)
    required = {
        "resolved_config.json", "learner_multivalued_dataset.csv",
        "evaluation_multilevel_ground_truth.csv", "rank_training_opportunities.csv",
        "validation_opportunities.csv", "test_global_opportunities.csv",
        "test_global_allocation.csv", "patient_final_packages.csv",
        "prometheus_training_history.csv", "prometheus_training_diagnostics.json",
        "calibration_diagnostics.json", "global_metrics.json", "transition_metrics.json",
        "allocation_diagnostics.json", "policy_evaluation.csv", "manifest.json", "report.md",
        "prometheus_global_ranker.pt",
        "dm77_patient_assessments.csv", "dm77_population_summary.csv",
        "dm77_manual_review_queue.csv", "dm77_intervention_eligibility.csv",
        "dm77_audit_report.json",
        "synthetic_patient_features.csv", "synthetic_monthly_history.csv",
        "synthetic_generation_metadata.json", "synthetic_validation_checks.csv",
        "synthetic_validation_report.json", "synthetic_privacy_audit.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    operational = pd.read_csv(output / "test_global_opportunities.csv")
    assert_no_oracle_columns(operational)
    assert not any(column.startswith("potential_") for column in operational.columns)
    assert {"dm77_need_level", "dm77_need_label", "recommended_pathway"} <= set(operational.columns)
    assessments = pd.read_csv(output / "dm77_patient_assessments.csv")
    represented_levels = set(
        assessments.dm77_need_level.dropna().astype(int).unique().tolist()
    )
    assert represented_levels == set(range(1, 7))
    protected = set(assessments.loc[assessments.dm77_protected_pathway, "patient_id"].astype(str))
    protected_opportunities = operational.loc[operational.patient_id.astype(str).isin(protected)]
    assert not protected_opportunities.eligibility.any()
    assert not protected_opportunities.selected.any()
    manifest = yaml.safe_load((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cross_transition_pairs"] is True
    assert manifest["oracle_used_for_training"] is False
    assert manifest["dm77_level_used_as_historical_treatment"] is False
    assert manifest["dm77_level_used_as_ranker_feature"] is False
    assert manifest["source_population_fully_synthetic"] is True
    assert manifest["real_patient_records_used"] is False
    assert manifest["synthetic_validation_critical_failures"] == 0
