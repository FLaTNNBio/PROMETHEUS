"""Synthetic baseline-need truth and noisy learner-reference generation.

The returned tables deliberately separate evaluation-only truth from the reference
label that a supervised need stratifier is allowed to consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NEED_REFERENCE_VERSION = "synthetic_noisy_need_panel_v1"


@dataclass(frozen=True)
class SyntheticNeedReferenceResult:
    learner_labels: pd.DataFrame
    ground_truth: pd.DataFrame
    metadata: dict


def _standardize(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="raise").to_numpy(float)
    return (array - array.mean()) / max(float(array.std()), 1e-8)


def generate_synthetic_need_reference(
    patients: pd.DataFrame,
    seed: int,
    panel_noise_scale: float = 0.75,
    palliative_panel_sensitivity: float = 0.82,
    palliative_panel_false_positive_rate: float = 0.015,
) -> SyntheticNeedReferenceResult:
    """Create physically separate truth and noisy panel-reference tables.

    The synthetic truth uses a nonlinear latent construct only for final evaluation.
    The learner receives a noisy panel label and never receives that construct, its
    cut points, or the true level.
    """

    if seed is None:
        raise ValueError("Synthetic need-reference generation requires an explicit seed")
    required = {
        "patient_id", "age", "condition_distinct", "prior_inpatient",
        "prior_emergency", "medication_distinct", "frailty_index",
        "functional_limitation_score", "social_fragility_score",
        "cognitive_impairment", "non_self_sufficiency", "caregiver_available",
        "housing_instability", "palliative_need",
    }
    missing = sorted(required.difference(patients.columns))
    if missing:
        raise ValueError(f"Synthetic need reference is missing inputs: {missing}")
    identifiers = patients.patient_id.astype(str)
    if identifiers.duplicated().any():
        raise ValueError("Synthetic need reference requires unique patients")
    if not 0.0 < panel_noise_scale or not 0.0 <= palliative_panel_sensitivity <= 1.0:
        raise ValueError("Invalid synthetic need panel settings")
    if not 0.0 <= palliative_panel_false_positive_rate <= 1.0:
        raise ValueError("Invalid palliative false-positive rate")

    rng = np.random.default_rng(int(seed))
    age = _standardize(patients.age)
    chronicity = _standardize(patients.condition_distinct)
    inpatient = _standardize(np.log1p(patients.prior_inpatient))
    emergency = _standardize(np.log1p(patients.prior_emergency))
    medications = _standardize(np.log1p(patients.medication_distinct))
    frailty = pd.to_numeric(patients.frailty_index).to_numpy(float)
    functional = pd.to_numeric(patients.functional_limitation_score).to_numpy(float)
    social = pd.to_numeric(patients.social_fragility_score).to_numpy(float)
    cognitive = pd.to_numeric(patients.cognitive_impairment).to_numpy(float)
    dependency = pd.to_numeric(patients.non_self_sufficiency).to_numpy(float)
    caregiver_absent = 1.0 - pd.to_numeric(patients.caregiver_available).to_numpy(float)
    housing = pd.to_numeric(patients.housing_instability).to_numpy(float)
    palliative = pd.to_numeric(patients.palliative_need).to_numpy(float) >= 0.5

    # This latent construct is evaluation-only. Interactions and independent noise
    # prevent it from being a restatement of the transparent rules implementation.
    latent = (
        0.30 * age + 0.52 * chronicity + 0.32 * inpatient + 0.18 * emergency
        + 0.22 * medications + 1.25 * frailty + 1.10 * functional
        + 0.58 * social + 0.36 * cognitive + 0.54 * dependency
        + 0.24 * caregiver_absent + 0.16 * housing
        + 0.48 * frailty * functional + 0.22 * social * caregiver_absent
        + rng.normal(0.0, 0.32, len(patients))
    )
    non_palliative = ~palliative
    if int(non_palliative.sum()) < 10:
        raise ValueError("Synthetic need reference requires non-palliative patients")
    cut_points = np.quantile(latent[non_palliative], (0.20, 0.40, 0.60, 0.80))
    true_level = 1 + np.searchsorted(cut_points, latent, side="right")
    true_level[palliative] = 6

    # The learner label represents an imperfect panel process with different weights,
    # independent noise, occasional adjacent disagreement, and imperfect pathway
    # ascertainment. It is intentionally not an encoded copy of true_level.
    panel_score = (
        0.25 * age + 0.47 * chronicity + 0.29 * inpatient + 0.22 * emergency
        + 0.18 * medications + 1.05 * frailty + 0.92 * functional
        + 0.48 * social + 0.28 * cognitive + 0.42 * dependency
        + 0.20 * caregiver_absent + 0.12 * housing
        + rng.normal(0.0, float(panel_noise_scale), len(patients))
    )
    panel_level = 1 + np.searchsorted(cut_points, panel_score, side="right")
    disagreement = rng.random(len(patients)) < 0.10
    direction = rng.choice((-1, 1), size=len(patients))
    panel_level = np.clip(panel_level + disagreement * direction, 1, 5)
    panel_palliative = (
        (palliative & (rng.random(len(patients)) < palliative_panel_sensitivity))
        | (
            (~palliative)
            & (panel_level == 5)
            & (rng.random(len(patients)) < palliative_panel_false_positive_rate)
        )
    )
    panel_level[panel_palliative] = 6

    learner = pd.DataFrame({
        "patient_id": identifiers.to_numpy(),
        "clinician_assigned_need_level": panel_level.astype(np.int64),
        "need_reference_source": NEED_REFERENCE_VERSION,
    })
    truth = pd.DataFrame({
        "patient_id": identifiers.to_numpy(),
        "true_baseline_need_level": true_level.astype(np.int64),
        "true_need_latent_score": latent,
    })
    exact_agreement = float(np.mean(panel_level == true_level))
    metadata = {
        "generator_version": NEED_REFERENCE_VERSION,
        "seed": int(seed),
        "learner_reference_field": "clinician_assigned_need_level",
        "evaluation_only_fields": [
            "true_baseline_need_level", "true_need_latent_score",
        ],
        "panel_noise_scale": float(panel_noise_scale),
        "palliative_panel_sensitivity": float(palliative_panel_sensitivity),
        "palliative_panel_false_positive_rate": float(
            palliative_panel_false_positive_rate
        ),
        "panel_truth_exact_agreement_evaluation_only": exact_agreement,
        "oracle_available_to_learner": False,
        "capacity_used": False,
        "causal_ranking_used": False,
    }
    return SyntheticNeedReferenceResult(learner, truth, metadata)
