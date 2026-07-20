"""Metrics for future DM77 rule validation against an adjudicated panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class DM77ClinicalValidationResult:
    summary: dict
    confusion_matrix: pd.DataFrame
    critical_level_metrics: pd.DataFrame


def weighted_fleiss_kappa(ratings, minimum_level: int = 1, maximum_level: int = 6) -> float:
    """Quadratic weighted multi-rater kappa for ordinal panel ratings."""

    values = np.asarray(ratings, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Panel ratings must have shape [patients, raters>=2]")
    if np.any(np.isfinite(values) & (
        (values < int(minimum_level)) | (values > int(maximum_level))
    )):
        raise ValueError("Panel ratings contain an invalid DM77 level")
    scale = float(maximum_level - minimum_level)
    observed_disagreement = []
    pooled = []
    for row in values:
        valid = row[np.isfinite(row)]
        if len(valid) < 2:
            continue
        pair_disagreement = []
        for left in range(len(valid)):
            for right in range(left + 1, len(valid)):
                pair_disagreement.append(((valid[left] - valid[right]) / scale) ** 2)
        observed_disagreement.append(float(np.mean(pair_disagreement)))
        pooled.extend(valid.tolist())
    if not observed_disagreement:
        raise ValueError("No patient has at least two finite panel ratings")
    pooled = np.asarray(pooled, dtype=float)
    expected = np.mean(
        np.square((pooled[:, None] - pooled[None, :]) / scale)
    )
    if expected <= 1e-12:
        return 1.0
    return float(1.0 - np.mean(observed_disagreement) / expected)


def _binary_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    positives = int(reference.sum())
    negatives = int((~reference).sum())
    return {
        "sensitivity": float(np.mean(predicted[reference])) if positives else float("nan"),
        "specificity": float(np.mean(~predicted[~reference])) if negatives else float("nan"),
        "false_negative_rate": float(np.mean(~predicted[reference]))
        if positives else float("nan"),
        "reference_positives": positives,
    }


def _actions(value) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item.strip() for item in str(value).split("|") if item.strip()}


def evaluate_dm77_against_adjudicated_panel(
    rule_assessments: pd.DataFrame,
    adjudicated_reference: pd.DataFrame,
    critical_levels=(5, 6),
) -> DM77ClinicalValidationResult:
    """Compare computable rules with a panel-adjudicated reference standard.

    This evaluates the DM77 rule layer only; it does not evaluate causal ranking or
    treatment effectiveness.
    """

    required_rule = {
        "patient_id", "dm77_need_level", "dm77_assessment_status",
        "dm77_protected_pathway",
    }
    required_reference = {
        "patient_id", "reference_dm77_need_level",
        "reference_manual_review", "reference_protected_pathway",
    }
    if not required_rule.issubset(rule_assessments) or not required_reference.issubset(
        adjudicated_reference
    ):
        raise ValueError("DM77 panel validation input schema is incomplete")
    if rule_assessments.patient_id.astype(str).duplicated().any() or (
        adjudicated_reference.patient_id.astype(str).duplicated().any()
    ):
        raise ValueError("DM77 panel validation requires one row per patient")
    frame = rule_assessments.copy()
    frame["patient_id"] = frame.patient_id.astype(str)
    reference = adjudicated_reference.copy()
    reference["patient_id"] = reference.patient_id.astype(str)
    frame = frame.merge(reference, on="patient_id", how="inner", validate="one_to_one")
    if not len(frame):
        raise ValueError("DM77 panel validation has no shared patients")

    comparable = frame.dm77_need_level.notna() & frame.reference_dm77_need_level.notna()
    predicted_level = frame.loc[comparable, "dm77_need_level"].to_numpy(int)
    reference_level = frame.loc[comparable, "reference_dm77_need_level"].to_numpy(int)
    if not len(predicted_level):
        raise ValueError("No finite adjudicated DM77 levels are available")
    error = np.abs(predicted_level - reference_level)
    confusion = pd.crosstab(
        pd.Series(reference_level, name="reference_level"),
        pd.Series(predicted_level, name="rule_level"),
        dropna=False,
    ).reindex(index=range(1, 7), columns=range(1, 7), fill_value=0)

    critical_rows = []
    for level in map(int, critical_levels):
        critical_rows.append({
            "critical_level": level,
            **_binary_metrics(reference_level == level, predicted_level == level),
        })
    predicted_protected = frame.dm77_protected_pathway.astype(bool).to_numpy()
    reference_protected = frame.reference_protected_pathway.astype(bool).to_numpy()
    predicted_manual = frame.dm77_assessment_status.eq("manual_review").to_numpy()
    reference_manual = frame.reference_manual_review.astype(bool).to_numpy()
    protected = _binary_metrics(reference_protected, predicted_protected)
    manual = _binary_metrics(reference_manual, predicted_manual)

    summary = {
        "reference_standard": "panel_adjudicated_not_single_reviewer",
        "patients_compared": int(len(frame)),
        "patients_with_comparable_levels": int(comparable.sum()),
        "raw_level_agreement": float(np.mean(predicted_level == reference_level)),
        "quadratic_weighted_cohen_kappa": float(cohen_kappa_score(
            reference_level, predicted_level, weights="quadratic", labels=list(range(1, 7))
        )),
        "ordinal_mean_absolute_error": float(np.mean(error)),
        "error_greater_than_one_level_fraction": float(np.mean(error > 1)),
        "protected_pathway_sensitivity": protected["sensitivity"],
        "protected_pathway_false_negative_rate": protected["false_negative_rate"],
        "manual_review_sensitivity": manual["sensitivity"],
        "manual_review_false_negative_rate": manual["false_negative_rate"],
        "causal_ranker_evaluated": False,
        "clinical_effectiveness_evaluated": False,
    }
    action_columns = {"eligible_action_ids", "reference_eligible_action_ids"}
    if action_columns.issubset(frame):
        agreements, jaccards = [], []
        for predicted, expected in zip(
            frame.eligible_action_ids, frame.reference_eligible_action_ids
        ):
            predicted_set, expected_set = _actions(predicted), _actions(expected)
            agreements.append(predicted_set == expected_set)
            union = predicted_set | expected_set
            jaccards.append(len(predicted_set & expected_set) / len(union) if union else 1.0)
        summary.update({
            "eligible_action_exact_agreement": float(np.mean(agreements)),
            "eligible_action_mean_jaccard": float(np.mean(jaccards)),
        })
    return DM77ClinicalValidationResult(
        summary=summary,
        confusion_matrix=confusion,
        critical_level_metrics=pd.DataFrame(critical_rows),
    )
