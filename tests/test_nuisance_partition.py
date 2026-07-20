import numpy as np
import pandas as pd
import pytest

from causal_population_ranking.nuisance.cross_fitting import (
    fit_partitioned_nuisance,
    fit_repeated_partitioned_nuisance,
    grouped_patient_folds,
)


FEATURES = ["x1", "x2", "x3"]


def partitioned_learner(n=240):
    rng = np.random.default_rng(17)
    split = np.array(
        ["nuisance_train"] * 84
        + ["rank_train"] * 72
        + ["validation"] * 36
        + ["test"] * 48,
        dtype=object,
    )
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    treatment = np.arange(n) % 2
    outcome = 0.4 * x1 - 0.2 * x2 + treatment * (0.3 + 0.1 * x3) + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "treatment": treatment,
            "observed_outcome": outcome,
            "split": split,
        }
    )


def test_partitioned_nuisance_is_finite_and_uses_only_dedicated_training_split():
    learner = partitioned_learner()
    predictions, diagnostics = fit_partitioned_nuisance(
        learner, FEATURES, folds=3, model="gradient_boosting", seed=31, clip=(0.02, 0.98)
    )

    assert predictions.patient_id.tolist() == learner.patient_id.tolist()
    assert np.isfinite(predictions[["e_hat", "mu0_hat", "mu1_hat"]]).all().all()
    training = learner.split.eq("nuisance_train").to_numpy()
    assert (predictions.loc[training, "nuisance_fold"] >= 0).all()
    assert (predictions.loc[~training, "nuisance_fold"] == -1).all()
    assert diagnostics["protocol"] == "strict_partition_holdout_v1_1"
    assert diagnostics["training_n"] == int(training.sum())
    assert diagnostics["grouped_by_patient"] is True
    assert diagnostics["self_prediction_violations"] == 0


def test_rank_validation_and_test_outcomes_cannot_change_nuisance_predictions():
    learner = partitioned_learner()
    original, _ = fit_partitioned_nuisance(
        learner, FEATURES, folds=3, model="gradient_boosting", seed=31, clip=(0.02, 0.98)
    )

    perturbed = learner.copy()
    closed = ~perturbed.split.eq("nuisance_train")
    perturbed.loc[closed, "treatment"] = 1 - perturbed.loc[closed, "treatment"]
    perturbed.loc[closed, "observed_outcome"] = (
        10_000 + np.arange(int(closed.sum()), dtype=float)
    )
    repeated, _ = fit_partitioned_nuisance(
        perturbed, FEATURES, folds=3, model="gradient_boosting", seed=31, clip=(0.02, 0.98)
    )

    np.testing.assert_allclose(
        original[["e_hat", "mu0_hat", "mu1_hat"]],
        repeated[["e_hat", "mu0_hat", "mu1_hat"]],
        rtol=0,
        atol=0,
    )


def test_partitioned_nuisance_rejects_missing_validation_split():
    learner = partitioned_learner()
    learner.loc[learner.split.eq("validation"), "split"] = "rank_train"
    with pytest.raises(ValueError, match="Missing required splits"):
        fit_partitioned_nuisance(
            learner, FEATURES, folds=3, model="gradient_boosting", seed=31
        )


def test_grouped_cross_fitting_never_uses_a_patient_for_its_own_prediction():
    patient_ids = np.repeat([f"p{i}" for i in range(12)], 2)
    seen = np.zeros(len(patient_ids), dtype=int)
    for fit, holdout in grouped_patient_folds(patient_ids, folds=3, seed=11):
        assert set(patient_ids[fit]).isdisjoint(patient_ids[holdout])
        seen[holdout] += 1
    assert np.all(seen == 1)


def test_repeated_partitioned_nuisance_is_deterministic_and_preserves_repetitions():
    learner = partitioned_learner()
    first, first_repetitions, diagnostics = fit_repeated_partitioned_nuisance(
        learner, FEATURES, folds=3, model="gradient_boosting", seed=31, repeats=3
    )
    second, second_repetitions, _ = fit_repeated_partitioned_nuisance(
        learner, FEATURES, folds=3, model="gradient_boosting", seed=31, repeats=3
    )

    assert diagnostics["protocol"] == "repeated_strict_partition_holdout_v1"
    assert diagnostics["repeats"] == 3
    assert diagnostics["repeat_seeds"] == [31, 1040, 2049]
    assert len(first_repetitions) == 3
    np.testing.assert_allclose(
        first[["e_hat", "mu0_hat", "mu1_hat"]],
        second[["e_hat", "mu0_hat", "mu1_hat"]],
        rtol=0,
        atol=0,
    )
    for left, right in zip(first_repetitions, second_repetitions):
        np.testing.assert_allclose(
            left[["e_hat", "mu0_hat", "mu1_hat"]],
            right[["e_hat", "mu0_hat", "mu1_hat"]],
            rtol=0,
            atol=0,
        )


def test_repeated_nuisance_never_opens_rank_validation_or_test_outcomes():
    learner = partitioned_learner()
    _, original, _ = fit_repeated_partitioned_nuisance(
        learner, FEATURES, folds=3, model="gradient_boosting", seed=41, repeats=2
    )
    perturbed = learner.copy()
    closed = ~perturbed.split.eq("nuisance_train")
    perturbed.loc[closed, "treatment"] = 1 - perturbed.loc[closed, "treatment"]
    perturbed.loc[closed, "observed_outcome"] = 50_000 + np.arange(closed.sum())
    _, repeated, _ = fit_repeated_partitioned_nuisance(
        perturbed, FEATURES, folds=3, model="gradient_boosting", seed=41, repeats=2
    )
    for left, right in zip(original, repeated):
        np.testing.assert_allclose(
            left[["e_hat", "mu0_hat", "mu1_hat"]],
            right[["e_hat", "mu0_hat", "mu1_hat"]],
            rtol=0,
            atol=0,
        )
