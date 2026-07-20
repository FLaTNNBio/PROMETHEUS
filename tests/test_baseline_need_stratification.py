from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from causal_population_ranking.dm77 import (
    BaselineNeedStratifier,
    cross_fit_baseline_need,
)
from causal_population_ranking.evaluation import evaluate_baseline_need
from causal_population_ranking.synthetic import (
    generate_synthetic_need_reference,
    generate_synthetic_population,
)


def _balanced_reference_frame(rows_per_level: int = 60, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    level = np.repeat(np.arange(1, 7), rows_per_level)
    severity = (level - 1) / 5.0
    frame = pd.DataFrame({
        "patient_id": [f"need_{index:05d}" for index in range(len(level))],
        "age": 28 + 62 * severity + rng.normal(0, 4, len(level)),
        "condition_distinct": np.clip(
            np.rint(7 * severity + rng.normal(0, 0.7, len(level))), 0, 8
        ),
        "prior_inpatient": np.clip(
            np.rint(3 * severity + rng.normal(0, 0.5, len(level))), 0, 5
        ),
        "prior_emergency": np.clip(
            np.rint(4 * severity + rng.normal(0, 0.7, len(level))), 0, 7
        ),
        "medication_distinct": np.clip(
            np.rint(10 * severity + rng.normal(0, 1.0, len(level))), 0, 14
        ),
        "frailty_index": np.clip(severity + rng.normal(0, 0.08, len(level)), 0, 1),
        "functional_limitation_score": np.clip(
            severity + rng.normal(0, 0.10, len(level)), 0, 1
        ),
        "social_fragility_score": np.clip(
            0.8 * severity + rng.normal(0, 0.12, len(level)), 0, 1
        ),
        "cognitive_impairment": (level >= 5).astype(float),
        "non_self_sufficiency": (level == 5).astype(float),
        "caregiver_available": (level < 5).astype(float),
        "housing_instability": ((level >= 4) & (rng.random(len(level)) < 0.3)).astype(float),
        "palliative_need": (level == 6).astype(float),
        "clinician_assigned_need_level": level,
        "need_reference_source": "synthetic_test_panel_v1",
    })
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_rules_and_supervised_modes_emit_the_same_need_schema():
    frame = _balanced_reference_frame()
    rules = BaselineNeedStratifier("rules", seed=101).predict(frame).assessments
    model = BaselineNeedStratifier("supervised", seed=103).fit(
        frame.iloc[:240], frame.iloc[240:], "clinician_assigned_need_level"
    )
    supervised = model.predict(frame).assessments

    assert list(rules.columns) == list(supervised.columns)
    required = {
        "patient_id", "baseline_need_level", "baseline_need_score",
        "baseline_need_probabilities", "baseline_need_explanation",
        "baseline_need_mode", "baseline_need_status", "baseline_need_model_version",
    }
    assert required <= set(rules.columns)
    for output in (rules, supervised):
        probability = output[[f"baseline_need_probability_{level}" for level in range(1, 7)]]
        assert np.allclose(probability.sum(axis=1), 1.0)
        assert output.baseline_need_level.between(1, 6).all()
        assert all(set(json.loads(value)) == set(map(str, range(1, 7))) for value in output.baseline_need_probabilities)


def test_need_predictions_ignore_oracle_capacity_and_downstream_columns():
    frame = _balanced_reference_frame()
    training, calibration = frame.iloc[:240], frame.iloc[240:]
    model = BaselineNeedStratifier("supervised", seed=107).fit(
        training, calibration, "clinician_assigned_need_level"
    )
    clean = model.predict(frame).assessments
    contaminated = frame.copy()
    contaminated["true_baseline_need_level"] = np.arange(len(frame)) % 6 + 1
    contaminated["oracle_profile"] = "forbidden"
    contaminated["raw_priority_score"] = np.linspace(-100, 100, len(frame))
    contaminated["recommended_actionable_level"] = 6
    contaminated["allocated_care_level"] = 1
    changed = model.predict(contaminated).assessments
    pd.testing.assert_frame_equal(clean, changed)

    rules_clean = BaselineNeedStratifier("rules", seed=109).predict(frame)
    rules_changed = BaselineNeedStratifier("rules", seed=109).predict(contaminated)
    pd.testing.assert_frame_equal(rules_clean.assessments, rules_changed.assessments)
    assert "true_baseline_need_level" in rules_changed.audit["forbidden_columns_present_but_ignored"]
    assert rules_changed.audit["capacity_used"] is False
    assert rules_changed.audit["causal_ranking_used"] is False


def test_supervised_cross_fitting_is_reproducible_complete_and_patient_disjoint():
    frame = _balanced_reference_frame(rows_per_level=70)
    first = cross_fit_baseline_need(
        frame, "supervised", model_seed=113, split_seed=127
    )
    second = cross_fit_baseline_need(
        frame, "supervised", model_seed=113, split_seed=127
    )
    pd.testing.assert_frame_equal(first.assessments, second.assessments)
    pd.testing.assert_frame_equal(first.fold_assignments, second.fold_assignments)
    assert first.assessments.patient_id.tolist() == frame.patient_id.tolist()
    assert not first.fold_assignments.patient_id.duplicated().any()
    assert first.fold_assignments.baseline_need_fold.nunique() == 5
    assert all(item["patient_partitions_disjoint"] for item in first.audit["fold_audit"])
    assert first.audit["every_patient_predicted_once"] is True
    assert first.audit["reference_provenance_values"] == ["synthetic_test_panel_v1"]
    assert first.audit["full_frozen_model_partitions_disjoint"] is True
    assert first.audit["oracle_used"] is False


def test_supervised_need_rejects_wrong_label_overlap_and_missing_predictors():
    frame = _balanced_reference_frame()
    model = BaselineNeedStratifier("supervised", seed=131)
    with pytest.raises(ValueError, match="label must be"):
        model.fit(frame.iloc[:240], frame.iloc[240:], "true_baseline_need_level")
    with pytest.raises(ValueError, match="patient leakage"):
        model.fit(frame.iloc[:240], frame.iloc[200:], "clinician_assigned_need_level")
    without_provenance = frame.drop(columns="need_reference_source")
    with pytest.raises(ValueError, match="provenance field"):
        model.fit(
            without_provenance.iloc[:240],
            without_provenance.iloc[240:],
            "clinician_assigned_need_level",
        )
    fractional = frame.copy()
    fractional["clinician_assigned_need_level"] = fractional[
        "clinician_assigned_need_level"
    ].astype(float)
    fractional.loc[0, "clinician_assigned_need_level"] = 1.5
    with pytest.raises(ValueError, match="integer levels"):
        model.fit(
            fractional.iloc[:240], fractional.iloc[240:],
            "clinician_assigned_need_level",
        )

    fitted = BaselineNeedStratifier("supervised", seed=137).fit(
        frame.iloc[:240], frame.iloc[240:], "clinician_assigned_need_level"
    )
    incomplete = frame.iloc[:2].copy()
    incomplete.loc[0, "frailty_index"] = np.nan
    prediction = fitted.predict(incomplete).assessments
    assert prediction.loc[0, "baseline_need_status"] == "manual_review"
    assert pd.isna(prediction.loc[0, "baseline_need_level"])
    assert "frailty_index" in prediction.loc[0, "baseline_need_explanation"]


def test_synthetic_need_truth_is_separate_deterministic_and_nontrivially_noisy():
    patients = generate_synthetic_population(1000, seed=149).patients
    first = generate_synthetic_need_reference(patients, seed=151)
    second = generate_synthetic_need_reference(patients, seed=151)
    pd.testing.assert_frame_equal(first.learner_labels, second.learner_labels)
    pd.testing.assert_frame_equal(first.ground_truth, second.ground_truth)
    assert not any(
        column.startswith(("true_", "oracle_", "latent_"))
        for column in first.learner_labels.columns
    )
    assert {"true_baseline_need_level", "true_need_latent_score"} <= set(
        first.ground_truth.columns
    )
    joined = first.learner_labels.merge(first.ground_truth, on="patient_id")
    assert (joined.clinician_assigned_need_level != joined.true_baseline_need_level).any()
    assert first.metadata["oracle_available_to_learner"] is False


def test_baseline_need_metrics_report_ordinal_and_subgroup_diagnostics():
    reference = pd.DataFrame({
        "patient_id": [f"p{index}" for index in range(60)],
        "true_baseline_need_level": np.tile(np.arange(1, 7), 10),
    })
    prediction = reference.rename(
        columns={"true_baseline_need_level": "baseline_need_level"}
    )
    groups = pd.DataFrame({
        "patient_id": reference.patient_id,
        "sex": np.repeat(("female", "male"), 30),
    })
    result = evaluate_baseline_need(
        reference,
        prediction,
        subgroup_frame=groups,
        subgroup_columns=("sex",),
        minimum_subgroup_size=20,
    )
    assert result.summary["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert result.summary["macro_f1"] == pytest.approx(1.0)
    assert result.summary["ordinal_mae"] == pytest.approx(0.0)
    assert result.summary["within_one_level_accuracy"] == pytest.approx(1.0)
    assert result.summary["spearman"] == pytest.approx(1.0)
    assert result.confusion_matrix.filter(like="predicted_level").to_numpy().sum() == 60
    assert len(result.subgroup_metrics) == 2
    assert result.subgroup_metrics.status.eq("reported").all()
