"""Evaluation of the ordinal baseline-need stratification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


LEVELS = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class BaselineNeedEvaluationResult:
    summary: dict
    confusion_matrix: pd.DataFrame
    subgroup_metrics: pd.DataFrame


def _ordinal_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(reference):
        raise ValueError("Baseline-need evaluation requires at least one assessed row")
    return {
        "evaluated_patients": int(len(reference)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(reference, prediction, labels=LEVELS, weights="quadratic")
        ),
        "macro_f1": float(
            f1_score(reference, prediction, labels=LEVELS, average="macro")
        ),
        "balanced_accuracy": float(balanced_accuracy_score(reference, prediction)),
        "ordinal_mae": float(np.mean(np.abs(reference - prediction))),
        "within_one_level_accuracy": float(
            np.mean(np.abs(reference - prediction) <= 1)
        ),
        "spearman": float(spearmanr(reference, prediction).statistic),
    }


def evaluate_baseline_need(
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    subgroup_frame: pd.DataFrame | None = None,
    subgroup_columns: tuple[str, ...] = (),
    minimum_subgroup_size: int = 30,
    reference_field: str = "true_baseline_need_level",
    prediction_field: str = "baseline_need_level",
) -> BaselineNeedEvaluationResult:
    """Join frozen predictions to evaluation truth and compute ordinal diagnostics."""

    if minimum_subgroup_size < 1:
        raise ValueError("minimum_subgroup_size must be positive")
    for frame, field in ((reference, reference_field), (prediction, prediction_field)):
        if "patient_id" not in frame or field not in frame:
            raise ValueError(f"Baseline-need evaluation requires patient_id and {field}")
        if frame.patient_id.astype(str).duplicated().any():
            raise ValueError("Baseline-need evaluation requires unique patient rows")

    left = reference[["patient_id", reference_field]].copy()
    right = prediction[["patient_id", prediction_field]].copy()
    left["patient_id"] = left.patient_id.astype(str)
    right["patient_id"] = right.patient_id.astype(str)
    merged = left.merge(right, on="patient_id", how="inner", validate="one_to_one")
    if len(merged) != len(reference) or len(merged) != len(prediction):
        raise ValueError(
            "Reference and baseline-need predictions must cover the same patients"
        )

    reference_values = pd.to_numeric(merged[reference_field], errors="coerce")
    prediction_values = pd.to_numeric(merged[prediction_field], errors="coerce")
    assessed = reference_values.notna() & prediction_values.notna()
    valid_reference = reference_values.loc[assessed].astype(int).to_numpy()
    valid_prediction = prediction_values.loc[assessed].astype(int).to_numpy()
    observed_levels = set(np.unique(np.r_[valid_reference, valid_prediction]))
    if not observed_levels.issubset(LEVELS):
        raise ValueError("Baseline-need evaluation levels must be 1 through 6")

    summary = _ordinal_metrics(valid_reference, valid_prediction)
    summary.update(
        {
            "total_patients": int(len(merged)),
            "manual_review_or_missing_predictions": int((~assessed).sum()),
            "coverage": float(assessed.mean()),
            "reference_field": reference_field,
            "prediction_field": prediction_field,
            "evaluation_only": True,
        }
    )
    matrix = pd.DataFrame(
        confusion_matrix(valid_reference, valid_prediction, labels=LEVELS),
        index=pd.Index(LEVELS, name="reference_level"),
        columns=[f"predicted_level_{level}" for level in LEVELS],
    ).reset_index()

    subgroup_rows: list[dict] = []
    if subgroup_columns:
        if subgroup_frame is None or "patient_id" not in subgroup_frame:
            raise ValueError("Subgroup diagnostics require a patient_id subgroup frame")
        missing = sorted(set(subgroup_columns).difference(subgroup_frame.columns))
        if missing:
            raise ValueError(f"Missing subgroup columns: {missing}")
        groups = subgroup_frame[["patient_id", *subgroup_columns]].copy()
        groups["patient_id"] = groups.patient_id.astype(str)
        if groups.patient_id.duplicated().any():
            raise ValueError("Subgroup diagnostics require unique patient rows")
        enriched = merged.merge(groups, on="patient_id", how="left", validate="one_to_one")
        enriched["_assessed"] = assessed.to_numpy()
        for column in subgroup_columns:
            for value, subset in enriched.groupby(column, dropna=False, sort=True):
                valid_subset = subset.loc[subset._assessed]
                row = {
                    "subgroup_column": column,
                    "subgroup_value": str(value),
                    "evaluated_patients": int(len(valid_subset)),
                }
                if len(valid_subset) < int(minimum_subgroup_size):
                    row["status"] = "insufficient_sample"
                else:
                    row.update(
                        _ordinal_metrics(
                            valid_subset[reference_field].astype(int).to_numpy(),
                            valid_subset[prediction_field].astype(int).to_numpy(),
                        )
                    )
                    row["status"] = "reported"
                subgroup_rows.append(row)

    return BaselineNeedEvaluationResult(
        summary=summary,
        confusion_matrix=matrix,
        subgroup_metrics=pd.DataFrame(subgroup_rows),
    )
