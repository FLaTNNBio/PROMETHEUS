import numpy as np
import pytest
import torch

from causal_population_ranking.config import load_prometheus_config
from causal_population_ranking.ranking.losses import (
    apply_response_bin_boundaries,
    causal_contrastive_loss,
    fit_response_bin_boundaries,
    permute_response_labels,
    prometheus_objective,
    sample_contrastive_pairs,
    sample_ranking_pairs,
)
from causal_population_ranking.ranking.prometheus_ranker import (
    PrometheusRanker,
    TRAINING_MODES,
    TransitionArrays,
)


TRANSITIONS = ("1_to_2", "2_to_3", "3_to_4", "4_to_5", "5_to_6")


def transition_data(seed=7, n=72):
    rng = np.random.default_rng(seed)
    result = {}
    validation = {}
    for transition_index, name in enumerate(TRANSITIONS):
        x = rng.normal(size=(n, 4))
        benefit = (0.3 + 0.1 * transition_index) * x[:, 0] - 0.15 * x[:, 1]
        d = rng.binomial(1, 0.5, n)
        outcome = d * benefit + rng.normal(0, 0.03, n)
        nuisance = np.c_[np.full(n, 0.5), np.zeros(n), benefit]
        result[name] = TransitionArrays(
            x[:48], d[:48], outcome[:48], nuisance[:48],
            np.array([f"train_{i}" for i in range(48)]),
        )
        validation[name] = TransitionArrays(
            x[48:], d[48:], outcome[48:], nuisance[48:],
            np.array([f"validation_{i}" for i in range(24)]),
        )
    return result, validation


def test_response_bin_boundaries_are_fitted_only_on_training_signal():
    training = np.linspace(-2, 2, 100)
    boundaries = fit_response_bin_boundaries(training, 5)
    test = np.array([-1000.0, 1000.0])
    labels = apply_response_bin_boundaries(test, boundaries)
    np.testing.assert_allclose(boundaries, fit_response_bin_boundaries(training.copy(), 5))
    assert labels.tolist() == [0, 4]


def test_positive_contrastive_pairs_are_pulled_closer():
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    close = anchor.clone()
    far = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    assert causal_contrastive_loss(anchor, close, torch.ones(2)) < causal_contrastive_loss(
        anchor, far, torch.ones(2)
    )


def test_negative_pairs_beyond_margin_contribute_zero_loss():
    left = torch.zeros((2, 2))
    right = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    loss = causal_contrastive_loss(left, right, torch.zeros(2), margin=1.0)
    assert loss.item() == 0.0


def test_random_labels_preserve_counts_but_break_patient_correspondence():
    labels = np.repeat(np.arange(5), 10)
    permuted = permute_response_labels(labels, seed=13)
    np.testing.assert_array_equal(np.bincount(permuted), np.bincount(labels))
    assert not np.array_equal(permuted, labels)


def test_pair_samplers_never_cross_transition_boundaries():
    for transition_index in range(3):
        signal = np.linspace(-1, 1, 30) + transition_index
        rank_pairs = sample_ranking_pairs(signal, 40, 0.0, seed=transition_index)
        bins = apply_response_bin_boundaries(signal, fit_response_bin_boundaries(signal, 5))
        contrastive_pairs = sample_contrastive_pairs(bins, 40, seed=10 + transition_index)
        transition = np.full(len(signal), transition_index)
        assert np.all(transition[rank_pairs.left] == transition[rank_pairs.right])
        assert np.all(transition[contrastive_pairs.left] == transition[contrastive_pairs.right])


def test_total_objective_reduces_to_ranking_when_optional_terms_are_zero():
    ranking = torch.tensor(2.0, requires_grad=True)
    contrastive = torch.tensor(3.0, requires_grad=True)
    regularization = torch.tensor(4.0, requires_grad=True)
    total = prometheus_objective(ranking, contrastive, regularization, 0.0, 0.0)
    assert total.item() == ranking.item()
    total.backward()
    assert contrastive.grad.item() == 0.0


def test_contrastive_branch_receives_gradients_only_when_enabled():
    ranking = torch.tensor(1.0, requires_grad=True)
    contrastive = torch.tensor(2.0, requires_grad=True)
    regularization = torch.tensor(0.0)
    enabled = prometheus_objective(ranking, contrastive, regularization, 0.5, 0.0)
    enabled.backward()
    assert contrastive.grad.item() == pytest.approx(0.5)

    ranking2 = torch.tensor(1.0, requires_grad=True)
    contrastive2 = torch.tensor(2.0, requires_grad=True)
    disabled = prometheus_objective(ranking2, contrastive2, regularization, 0.0, 0.0)
    disabled.backward()
    assert contrastive2.grad.item() == 0.0


def _fit(mode, seed=19):
    training, validation = transition_data(seed=seed)
    names = ("2_to_3",) if mode == "independent_rankers" else TRANSITIONS
    training = {name: training[name] for name in names}
    validation = {name: validation[name] for name in names}
    return PrometheusRanker(
        transition_names=names,
        training_mode=mode,
        pairs_per_transition=48,
        contrastive_pairs_per_transition=32,
        epochs=1,
        batch_size=48,
        hidden_dim=12,
        latent_dim=6,
        patience=1,
        seed=seed,
        device="cpu",
        lambda_con=0.0 if mode in {"independent_rankers", "unified_rank_only"} else 0.05,
        lambda_reg=0.0,
    ).fit(training, validation), validation


@pytest.mark.parametrize("mode", TRAINING_MODES)
def test_all_five_training_modes_produce_expected_output_shapes(mode):
    model, validation = _fit(mode)
    name = next(iter(validation))
    score = model.predict_score(validation[name].x, name)
    representation = model.predict_representation(validation[name].x, name)
    assert score.shape == (24,)
    assert representation.shape == (24, 6)
    assert np.isfinite(score).all()


def test_scores_use_the_requested_transition_identifier():
    model, validation = _fit("prometheus_causal_contrastive", seed=23)
    x = validation["2_to_3"].x
    score_23 = model.predict_score(x, "2_to_3")
    score_45 = model.predict_score(x, "4_to_5")
    assert not np.allclose(score_23, score_45)
    with pytest.raises(ValueError, match="Unknown transition"):
        model.predict_score(x, "not_a_transition")


def test_patient_level_train_validation_leakage_is_detected():
    training, validation = transition_data()
    leaked = validation["2_to_3"]
    validation["2_to_3"] = TransitionArrays(
        leaked.x, leaked.transition_treatment, leaked.outcome, leaked.nuisance,
        np.array(["train_0", *leaked.patient_ids[1:]]),
    )
    model = PrometheusRanker(
        training_mode="unified_rank_only", lambda_con=0.0,
        pairs_per_transition=16, epochs=1, batch_size=8, patience=1,
    )
    with pytest.raises(ValueError, match="Patient-level train/validation leakage"):
        model.fit(training, validation)


def test_epoch_logging_contains_required_losses_pairs_distances_and_gradients():
    model, _ = _fit("prometheus_causal_contrastive", seed=31)
    row = model.history[0]
    required = {
        "total_loss", "ranking_loss", "contrastive_loss", "regularization_loss",
        "gradient_norm_shared_encoder", "gradient_norm_transition_embeddings",
        "gradient_norm_projection_head", "gradient_norm_scoring_head",
    }
    assert required.issubset(row)
    for name in TRANSITIONS:
        assert row[f"valid_ranking_pairs_{name}"] > 0
        assert row[f"positive_contrastive_pairs_{name}"] > 0
        assert row[f"negative_contrastive_pairs_{name}"] > 0
        assert 0 <= row[f"discarded_ambiguous_pair_fraction_{name}"] <= 1
        assert 0 <= row[f"discarded_unstable_pair_fraction_{name}"] <= 1
        assert 0 <= row[f"mean_pair_direction_agreement_{name}"] <= 1


def test_current_score_cannot_be_selected_as_a_contrastive_label_source():
    with pytest.raises(ValueError, match="Unknown training mode"):
        PrometheusRanker(training_mode="score_generated_pairs")


def test_pair_reliability_weights_are_optional_and_mean_normalized():
    signal = np.linspace(-2, 2, 40)
    unweighted = sample_ranking_pairs(signal, 60, 0.0, seed=5)
    weighted = sample_ranking_pairs(
        signal, 60, 0.0, seed=5, reliability_weighting=True, max_weight=0.5
    )
    np.testing.assert_array_equal(unweighted.left, weighted.left)
    assert np.all(unweighted.weights == 1.0)
    assert weighted.weights.mean() == pytest.approx(1.0)
    assert not np.allclose(weighted.weights, 1.0)


def test_all_five_configuration_examples_expose_required_settings():
    required = {
        "lambda_con", "lambda_reg", "contrastive_margin", "num_response_bins",
        "negative_bin_separation", "ranking_min_signal_gap", "propensity_clip_epsilon",
        "pairs_per_transition", "positive_negative_pair_ratio",
        "normalize_contrastive_embeddings", "pair_reliability_weighting",
        "dr_signal_aggregation", "min_pair_direction_agreement",
    }
    observed_modes = set()
    for mode in TRAINING_MODES:
        config = load_prometheus_config(f"configs/prometheus/{mode}.yaml")
        observed_modes.add(config["prometheus"]["training_mode"])
        assert required.issubset(config["prometheus"])
        assert config["nuisance"]["repeats"] >= 1
        assert config["nuisance"]["repeat_seed_stride"] >= 1
    assert observed_modes == set(TRAINING_MODES)
