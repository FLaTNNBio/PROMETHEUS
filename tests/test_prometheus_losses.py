import numpy as np
import pytest
import torch

from causal_population_ranking.ranking.causal_signals import (
    aggregate_repeated_signals,
    doubly_robust_signal,
    repeated_doubly_robust_signals,
)
from causal_population_ranking.ranking.losses import (
    pairwise_causal_ranking_loss,
    sample_ranking_pairs,
)
from causal_population_ranking.ranking.pair_sampler import PairSampler


def test_doubly_robust_signal_matches_formula_and_clips_propensity():
    d = torch.tensor([1.0, 0.0])
    y = torch.tensor([2.0, -0.2])
    e = torch.tensor([0.0, 1.0])
    mu0 = torch.tensor([0.0, -0.1])
    mu1 = torch.tensor([0.5, 0.4])
    got = doubly_robust_signal(d, y, e, mu0, mu1, 0.1)
    clipped = torch.tensor([0.1, 0.9])
    expected = mu1 - mu0 + d / clipped * (y - mu1) - (1 - d) / (1 - clipped) * (y - mu0)
    assert torch.allclose(got, expected)
    assert torch.isfinite(got).all() and float(got.abs().max()) < 100


def test_ranking_loss_decreases_as_correct_score_gap_increases():
    direction = torch.ones(1)
    small = pairwise_causal_ranking_loss(torch.tensor([0.2]), torch.tensor([0.0]), direction)
    large = pairwise_causal_ranking_loss(torch.tensor([2.0]), torch.tensor([0.0]), direction)
    assert large < small


def test_ranking_loss_increases_when_pair_order_is_reversed():
    direction = torch.ones(1)
    correct = pairwise_causal_ranking_loss(torch.tensor([1.0]), torch.tensor([0.0]), direction)
    reversed_order = pairwise_causal_ranking_loss(torch.tensor([0.0]), torch.tensor([1.0]), direction)
    assert reversed_order > correct


def test_tied_and_below_threshold_signal_pairs_are_excluded():
    signal = np.array([0.0, 0.0, 0.05, 1.0])
    pairs = sample_ranking_pairs(signal, 12, min_signal_gap=0.1, seed=4)
    assert len(pairs.left) > 0
    assert np.all(np.abs(signal[pairs.left] - signal[pairs.right]) > 0.1)
    assert pairs.discarded_fraction > 0


def test_pair_sampler_is_deterministic_unique_and_has_no_self_pairs():
    sampler = PairSampler(20, 100, 7)
    i, j = sampler.sample(2)
    i2, j2 = sampler.sample(2)
    assert np.array_equal(i, i2) and np.array_equal(j, j2) and np.all(i != j)
    assert len(set(zip(i, j))) == 100


def test_repeated_dr_signal_is_aggregated_rowwise_without_oracle_inputs():
    treatment = np.array([1.0, 0.0])
    outcome = np.array([2.0, -0.2])
    nuisance = np.array([
        [[0.4, 0.0, 0.5], [0.5, 0.1, 0.6], [0.6, 0.2, 0.7]],
        [[0.4, -0.1, 0.4], [0.5, -0.2, 0.3], [0.6, -0.3, 0.2]],
    ])
    repeated = repeated_doubly_robust_signals(treatment, outcome, nuisance, 0.02)
    assert repeated.shape == (2, 3)
    np.testing.assert_allclose(
        aggregate_repeated_signals(repeated, "median"), np.median(repeated, axis=1)
    )


def test_unstable_ranking_pair_directions_are_rejected():
    signal = np.array([1.0, 0.0])
    repeated = np.array([[1.0, -1.0, 1.0], [0.0, 0.0, 0.0]])
    rejected = sample_ranking_pairs(
        signal, pairs=2, min_signal_gap=0.0, seed=3,
        repeated_signal=repeated, min_direction_agreement=0.8,
    )
    accepted = sample_ranking_pairs(
        signal, pairs=2, min_signal_gap=0.0, seed=3,
        repeated_signal=repeated, min_direction_agreement=0.6,
    )
    assert len(rejected.left) == 0
    assert len(accepted.left) == 2
    assert accepted.mean_direction_agreement == pytest.approx(2 / 3)
    assert rejected.stability_discarded_fraction == pytest.approx(1.0)
