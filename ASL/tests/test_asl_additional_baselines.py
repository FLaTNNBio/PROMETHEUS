import numpy as np
import pandas as pd

from causal_population_ranking.applications.asl_semisynthetic_case import (
    ASL_FEATURE_COLUMNS,
    _fit_pooled_dr_learner,
    _fit_r_learner,
    _fit_s_learner,
    _processor,
)


def _feature_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "age": rng.normal(68, 10, n),
        "female": rng.integers(0, 2, n),
        "morbidity_proxy": rng.gamma(2.0, 1.0, n),
        "frailty_proxy": rng.uniform(0, 1, n),
        "prior_admissions": rng.poisson(1.0, n),
        "log_los_days": rng.normal(1.0, 0.5, n),
        "log_hospital_reimbursement": rng.normal(8.0, 0.7, n),
        "log_exemption_records": rng.normal(1.0, 0.5, n),
        "log_distinct_diagnoses": rng.normal(1.5, 0.5, n),
        "income_exemption": rng.integers(0, 2, n),
        "district": rng.choice(["A", "B", "C"], n),
    })
    return frame[list(ASL_FEATURE_COLUMNS)]


def test_additional_asl_baselines_return_finite_profile_scores():
    rng = np.random.default_rng(42)
    train = _feature_frame(320, 1)
    treatment = rng.integers(0, 4, len(train))
    base = 0.04 * train.age.to_numpy() + train.morbidity_proxy.to_numpy()
    effect = np.column_stack([
        np.zeros(len(train)),
        0.5 + 0.8 * train.frailty_proxy.to_numpy(),
        0.2 + 0.5 * train.morbidity_proxy.to_numpy(),
        0.4 + 0.3 * train.income_exemption.to_numpy(),
    ])
    outcome = base + effect[np.arange(len(train)), treatment] + rng.normal(0, 0.2, len(train))
    train = train.assign(assigned_profile=treatment, observed_outcome=outcome)

    test = _feature_frame(90, 2)
    test_profile = np.tile(np.array([1, 2, 3]), 30)
    processor = _processor(train)
    train_x = np.asarray(processor.fit_transform(train[list(ASL_FEATURE_COLUMNS)]), dtype=float)
    test_x = np.asarray(processor.transform(test[list(ASL_FEATURE_COLUMNS)]), dtype=float)

    s_score = _fit_s_learner(
        train_x,
        treatment,
        outcome,
        test_x,
        test_profile,
        max_iter=20,
        seed=10,
    )
    r_scores = {}
    for profile in (1, 2, 3):
        r_scores[profile] = _fit_r_learner(
            train,
            test,
            processor,
            profile,
            max_iter=20,
            n_splits=3,
            propensity_clip=0.05,
            seed=20 + profile,
        )
    r_score = np.asarray([r_scores[p][i] for i, p in enumerate(test_profile)])

    opportunity_train = pd.concat([
        _feature_frame(120, 100 + profile).assign(
            profile_id=profile,
            _dr_repeat_0=lambda f, profile=profile: (
                0.2 * profile
                + 0.4 * f["frailty_proxy"]
                + 0.1 * f["morbidity_proxy"]
            ),
        )
        for profile in (1, 2, 3)
    ], ignore_index=True)
    opportunity_test = test.assign(profile_id=test_profile)
    pooled_score = _fit_pooled_dr_learner(
        opportunity_train,
        opportunity_test,
        ("_dr_repeat_0",),
        max_iter=20,
        seed=30,
    )

    assert s_score.shape == (len(test),)
    assert r_score.shape == (len(test),)
    assert pooled_score.shape == (len(test),)
    assert np.isfinite(s_score).all()
    assert np.isfinite(r_score).all()
    assert np.isfinite(pooled_score).all()
