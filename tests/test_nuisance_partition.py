import numpy as np
import pandas as pd
import pytest

from causal_population_ranking.nuisance.cross_fitting import (
    fit_partitioned_nuisance,
    fit_repeated_partitioned_nuisance,
    grouped_patient_folds,
)
from causal_population_ranking.nuisance.profile_supervision import (
    build_profile_causal_supervision,
)
from causal_population_ranking.nuisance.signals import (
    aggregate_repeated_signals,
    repeated_doubly_robust_signals,
    robustify_repeated_signals,
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


def profile_learner(n=320):
    rng = np.random.default_rng(73)
    observed = np.array(
        ["no_new_profile"] * 160
        + ["profile_a"] * 80
        + ["profile_b"] * 80,
        dtype=object,
    )
    rng.shuffle(observed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    baseline_need = rng.integers(1, 6, size=n)
    current = np.where(np.arange(n) % 2, "current_low", "current_high")
    outcome = (
        10.0 + 0.5 * x1 - 0.3 * x2
        + 1.5 * (observed == "profile_a")
        + 0.8 * (observed == "profile_b")
        + rng.normal(0.0, 0.5, size=n)
    )
    return pd.DataFrame({
        "patient_id": [f"profile_patient_{index:04d}" for index in range(n)],
        "x1": x1,
        "x2": x2,
        "baseline_need_level": baseline_need,
        "current_care_profile": current,
        "observed_treatment_profile": observed,
        "observed_outcome": outcome,
    })


def profile_opportunities(learner):
    return pd.DataFrame([
        {
            "patient_id": patient_id,
            "care_profile_id": profile,
            "care_profile_index": profile_index,
            "empirical_support": pd.NA,
            "empirical_support_status": "pending",
        }
        for patient_id in learner.patient_id
        for profile_index, profile in enumerate(("profile_a", "profile_b"))
    ])


def profile_settings():
    return {
        "model": "linear",
        "folds": 3,
        "repeats": 2,
        "repeat_seed_stride": 101,
        "split_fractions": {
            "nuisance_train": 0.40,
            "rank_train": 0.30,
            "validation": 0.15,
            "test": 0.15,
        },
        "propensity_clip": [0.03, 0.97],
        "overlap_support_bounds": [0.05, 0.95],
        "minimum_arm_count_per_split": 2,
        "minimum_effective_sample_size_per_split": 3.0,
        "minimum_profile_effective_sample_size": 10.0,
        "minimum_overlap_fraction_per_split": 0.50,
        "aggregation": "median",
        "winsorize": True,
        "winsorize_quantiles": [0.01, 0.99],
        "numeric_features": ["x1", "x2", "baseline_need_level"],
        "categorical_features": ["current_care_profile"],
    }


def test_profile_supervision_is_deterministic_exact_and_non_oracle():
    learner = profile_learner()
    opportunities = profile_opportunities(learner)
    first = build_profile_causal_supervision(
        learner, opportunities, profile_settings(), split_seed=2213, nuisance_seed=2237
    )
    second = build_profile_causal_supervision(
        learner, opportunities, profile_settings(), split_seed=2213, nuisance_seed=2237
    )

    pd.testing.assert_frame_equal(first.patient_splits, second.patient_splits)
    pd.testing.assert_frame_equal(first.supervision, second.supervision)
    assert first.patient_splits.patient_id.is_unique
    assert first.audit["profiles_supported"] == 2
    assert first.audit["nuisance_repeat_seeds"] == [2237, 2338]
    assert first.audit["oracle_used"] is False
    assert not any(
        column.startswith(("true_", "oracle_", "potential_outcome", "latent_"))
        for column in first.supervision
    )
    for profile, rows in first.supervision.groupby("care_profile_id"):
        assert set(rows.observed_treatment_profile) <= {profile, "no_new_profile"}
        assert rows.profile_treatment.eq(
            rows.observed_treatment_profile.eq(profile).astype(int)
        ).all()


def test_profile_nuisance_predictions_ignore_downstream_outcome_mutations():
    learner = profile_learner()
    opportunities = profile_opportunities(learner)
    original = build_profile_causal_supervision(
        learner, opportunities, profile_settings(), split_seed=2213, nuisance_seed=2237
    )
    mutated = learner.copy()
    split_map = original.patient_splits.set_index("patient_id").split
    downstream = mutated.patient_id.map(split_map).ne("nuisance_train")
    mutated.loc[downstream, "observed_outcome"] = 100_000 + np.arange(downstream.sum())
    repeated = build_profile_causal_supervision(
        mutated, opportunities, profile_settings(), split_seed=2213, nuisance_seed=2237
    )

    keys = ["care_profile_id", "nuisance_repeat", "patient_id"]
    columns = ["e_hat", "mu0_hat", "mu1_hat"]
    left = original.nuisance_predictions.sort_values(keys).reset_index(drop=True)
    right = repeated.nuisance_predictions.sort_values(keys).reset_index(drop=True)
    np.testing.assert_allclose(left[columns], right[columns], rtol=0, atol=0)


def test_profile_supervision_rejects_oracle_inputs_and_marks_sparse_profiles():
    learner = profile_learner()
    opportunities = profile_opportunities(learner)
    contaminated = learner.assign(true_profile_benefit=1.0)
    with pytest.raises(ValueError, match="Oracle columns"):
        build_profile_causal_supervision(
            contaminated,
            opportunities,
            profile_settings(),
            split_seed=2213,
            nuisance_seed=2237,
        )

    sparse = learner.copy()
    sparse.loc[sparse.observed_treatment_profile.eq("profile_b"), "observed_treatment_profile"] = (
        "profile_a"
    )
    sparse.loc[:2, "observed_treatment_profile"] = "profile_b"
    result = build_profile_causal_supervision(
        sparse,
        opportunities,
        profile_settings(),
        split_seed=2213,
        nuisance_seed=2237,
        profile_candidates=[("profile_a", 0), ("profile_b", 1), ("profile_c", 2)],
    )
    assert result.audit["profile_support"]["profile_b"] is False
    assert result.audit["nuisance_diagnostics"]["profile_b"]["status"].startswith(
        "not_fitted"
    )
    assert not result.supported_opportunities.loc[
        result.supported_opportunities.care_profile_id.eq("profile_b"),
        "empirical_support",
    ].any()
    assert result.audit["profiles_considered"] == 3
    assert result.audit["profile_support"]["profile_c"] is False


def test_repeated_dr_signal_matches_the_clipped_profile_comparison_formula():
    treatment = np.array([1.0, 0.0])
    outcome = np.array([2.0, -0.2])
    nuisance = np.array([
        [[0.0, 0.0, 0.5]],
        [[1.0, -0.1, 0.4]],
    ])
    result = repeated_doubly_robust_signals(
        treatment, outcome, nuisance, propensity_clip_epsilon=0.1
    )
    propensity = np.array([[0.1], [0.9]])
    expected = (
        nuisance[:, :, 2]
        - nuisance[:, :, 1]
        + treatment[:, None]
        / propensity
        * (outcome[:, None] - nuisance[:, :, 2])
        - (1.0 - treatment[:, None])
        / (1.0 - propensity)
        * (outcome[:, None] - nuisance[:, :, 1])
    )
    np.testing.assert_allclose(result, expected)


def test_signal_robustification_is_split_local_and_oracle_free():
    repeated = np.array([
        [0.0, 0.1, 100.0],
        [0.2, 0.3, 0.4],
        [-100.0, -0.2, -0.1],
        [0.1, 0.2, 0.3],
    ])
    split = np.array(["rank_train", "rank_train", "validation", "validation"])
    result = robustify_repeated_signals(
        repeated,
        split,
        aggregation="median",
        winsorize_quantiles=(0.1, 0.9),
    )
    np.testing.assert_allclose(
        result.raw_aggregate, aggregate_repeated_signals(repeated, "median")
    )
    assert set(result.winsor_bounds) == {"rank_train", "validation"}
    assert np.isclose(result.reliability_weight.mean(), 1.0)
    assert np.isfinite(result.robust_aggregate).all()
