"""Baseline-need models and patient-disjoint cross-fitting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .need_rules import assess_dm77_population


DEFAULT_CONTRACT = Path("configs/pipeline.yaml")


@dataclass(frozen=True)
class BaselineNeedPrediction:
    assessments: pd.DataFrame
    audit: dict


@dataclass(frozen=True)
class CrossFittedBaselineNeedPrediction:
    assessments: pd.DataFrame
    fold_assignments: pd.DataFrame
    fitted_stratifier: "BaselineNeedStratifier"
    audit: dict


def load_baseline_need_contract(source: str | Path | Mapping | None = None) -> dict:
    if source is None:
        source = DEFAULT_CONTRACT
    if isinstance(source, Mapping):
        contract = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Baseline-need contract does not exist: {path}")
        contract = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(contract.get("baseline_need"), Mapping):
            embedded = contract["baseline_need"].get("contract")
            if isinstance(embedded, Mapping):
                contract = dict(embedded)
    required = {
        "contract_version", "levels", "output_fields", "allowed_preindex_features",
        "forbidden_exact_fields", "forbidden_prefixes", "modes",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(f"Baseline-need contract is missing fields: {missing}")
    if tuple(map(int, contract["levels"])) != (1, 2, 3, 4, 5, 6):
        raise ValueError("Baseline-need levels must be exactly 1 through 6")
    if set(contract["modes"]) != {"rules", "supervised"}:
        raise ValueError("Baseline-need contract requires rules and supervised modes")
    return contract


def _forbidden_present(frame: pd.DataFrame, contract: Mapping) -> list[str]:
    exact = set(map(str, contract["forbidden_exact_fields"]))
    prefixes = tuple(map(str, contract["forbidden_prefixes"]))
    return sorted(
        column for column in frame.columns
        if column in exact or column.startswith(prefixes)
    )


def _patient_ids(frame: pd.DataFrame) -> pd.Series:
    if "patient_id" not in frame:
        raise ValueError("Baseline-need stratification requires patient_id")
    values = frame.patient_id.astype(str)
    if values.duplicated().any():
        raise ValueError("Baseline-need stratification requires unique patients")
    return values


def _probability_json(probability: np.ndarray, levels: tuple[int, ...]) -> list[str]:
    return [
        json.dumps(
            {str(level): float(value) for level, value in zip(levels, row)},
            sort_keys=True,
        )
        for row in probability
    ]


def _validated_reference(values: pd.Series, levels: tuple[int, ...]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Supervised need fitting requires complete reference labels")
    if not np.allclose(numeric, np.rint(numeric)):
        raise ValueError("Supervised need labels must be integer levels 1 through 6")
    integer = numeric.astype(int)
    if not set(np.unique(integer)).issubset(levels):
        raise ValueError("Supervised need labels must be integer levels 1 through 6")
    return integer


class BaselineNeedStratifier:
    """Rules or supervised ordinal baseline-need model.

    The implementation projects every input onto the allowed pre-index schema before
    fitting or prediction. Downstream and oracle columns may be present for audit, but
    cannot affect the output.
    """

    def __init__(
        self,
        mode: str,
        seed: int,
        contract: str | Path | Mapping | None = None,
    ):
        if seed is None:
            raise ValueError("Baseline-need stratification requires an explicit seed")
        self.contract = load_baseline_need_contract(contract)
        self.mode = str(mode)
        self.seed = int(seed)
        if self.mode not in self.contract["modes"]:
            raise ValueError(f"Unknown baseline-need mode {self.mode!r}")
        self.levels = tuple(map(int, self.contract["levels"]))
        self.features = tuple(map(str, self.contract["allowed_preindex_features"]))
        self.settings = dict(self.contract["modes"][self.mode])
        self.model_version = str(self.settings["implementation"])
        self._fitted = self.mode == "rules"

    def _feature_frame(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        identifiers = _patient_ids(frame)
        missing = sorted(set(self.features).difference(frame.columns))
        if missing:
            safe = frame.reindex(columns=self.features).copy()
        else:
            safe = frame.loc[:, self.features].copy()
        for column in self.features:
            safe[column] = pd.to_numeric(safe[column], errors="coerce")
        return identifiers, safe

    def fit(
        self,
        training: pd.DataFrame,
        calibration: pd.DataFrame | None = None,
        reference_label: str | None = None,
    ) -> "BaselineNeedStratifier":
        if self.mode == "rules":
            self.fit_audit = {
                "mode": "rules",
                "training_required": False,
                "oracle_used": False,
                "seed": self.seed,
            }
            self._fitted = True
            return self
        if calibration is None:
            raise ValueError("Supervised need fitting requires a held-out calibration frame")
        label = str(reference_label or self.settings["reference_label"])
        if label != str(self.settings["reference_label"]):
            raise ValueError(
                f"Supervised need label must be {self.settings['reference_label']!r}"
            )
        train_ids, train_x = self._feature_frame(training)
        calibration_ids, calibration_x = self._feature_frame(calibration)
        overlap = set(train_ids).intersection(set(calibration_ids))
        if overlap:
            raise ValueError(
                f"Need model-fit/calibration patient leakage: {sorted(overlap)[:5]}"
            )
        if label not in training or label not in calibration:
            raise ValueError(f"Supervised need fitting requires {label}")
        provenance_field = str(
            self.settings.get("reference_provenance_field", "need_reference_source")
        )
        if bool(self.settings.get("reference_label_provenance_required", False)):
            if provenance_field not in training or provenance_field not in calibration:
                raise ValueError(
                    f"Supervised need fitting requires provenance field {provenance_field}"
                )
            provenance = pd.concat(
                [training[provenance_field], calibration[provenance_field]],
                ignore_index=True,
            )
            if provenance.isna().any() or provenance.astype(str).str.strip().eq("").any():
                raise ValueError("Supervised need reference provenance must be complete")
            provenance_values = sorted(provenance.astype(str).unique())
        else:
            provenance_values = []
        train_y = _validated_reference(training[label], self.levels)
        calibration_y = _validated_reference(calibration[label], self.levels)
        if train_x.isna().any().any() or calibration_x.isna().any().any():
            raise ValueError("Supervised need fitting requires complete allowed predictors")
        minimum = int(self.settings["minimum_training_rows_per_level"])
        combined = np.r_[train_y, calibration_y]
        inadequate = {
            level: int(np.sum(combined == level))
            for level in self.levels if int(np.sum(combined == level)) < minimum
        }
        if inadequate:
            raise ValueError(
                "Insufficient supervised need references by level: "
                f"{inadequate}; minimum={minimum}"
            )

        self.threshold_models = []
        self.threshold_calibrators = []
        calibration_diagnostics = []
        train_array = train_x.to_numpy(float)
        calibration_array = calibration_x.to_numpy(float)
        for threshold in map(int, self.settings["thresholds"]):
            train_binary = (train_y > threshold).astype(int)
            calibration_binary = (calibration_y > threshold).astype(int)
            if len(np.unique(train_binary)) != 2 or len(np.unique(calibration_binary)) != 2:
                raise ValueError(
                    f"Ordinal threshold {threshold} requires both classes in fit and calibration"
                )
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2_000,
                    random_state=self.seed + threshold,
                ),
            ).fit(train_array, train_binary)
            raw = model.predict_proba(calibration_array)[:, 1]
            calibrator = IsotonicRegression(
                increasing=True, out_of_bounds="clip"
            ).fit(raw, calibration_binary)
            calibrated = calibrator.predict(raw)
            self.threshold_models.append(model)
            self.threshold_calibrators.append(calibrator)
            calibration_diagnostics.append({
                "threshold": threshold,
                "fit_rows": int(len(train_y)),
                "calibration_rows": int(len(calibration_y)),
                "calibration_brier": float(
                    np.mean(np.square(calibrated - calibration_binary))
                ),
                "fit_positive_fraction": float(train_binary.mean()),
                "calibration_positive_fraction": float(calibration_binary.mean()),
            })
        self.fit_audit = {
            "mode": "supervised",
            "model_version": self.model_version,
            "reference_label": label,
            "reference_provenance_field": provenance_field,
            "reference_provenance_values": provenance_values,
            "fit_partition": "model_fit",
            "calibration_partition": "validation_calibration",
            "fit_rows": int(len(training)),
            "calibration_rows": int(len(calibration)),
            "fit_patient_ids": sorted(map(str, train_ids)),
            "calibration_patient_ids": sorted(map(str, calibration_ids)),
            "threshold_calibration": calibration_diagnostics,
            "allowed_features": list(self.features),
            "forbidden_columns_present_but_ignored": sorted(set(
                _forbidden_present(training, self.contract)
                + _forbidden_present(calibration, self.contract)
            )),
            "oracle_used": False,
            "capacity_used": False,
            "causal_ranking_used": False,
            "seed": self.seed,
        }
        self._fitted = True
        return self

    def _supervised_probabilities(self, x: np.ndarray) -> np.ndarray:
        cumulative = np.column_stack([
            calibrator.predict(model.predict_proba(x)[:, 1])
            for model, calibrator in zip(
                self.threshold_models, self.threshold_calibrators
            )
        ])
        cumulative = np.minimum.accumulate(np.clip(cumulative, 0.0, 1.0), axis=1)
        probability = np.column_stack((
            1.0 - cumulative[:, 0],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1],
        ))
        probability = np.clip(probability, 0.0, 1.0)
        probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
        return probability

    def predict(self, frame: pd.DataFrame) -> BaselineNeedPrediction:
        if not self._fitted:
            raise RuntimeError("Baseline-need stratifier has not been fitted")
        identifiers, safe = self._feature_frame(frame)
        forbidden = _forbidden_present(frame, self.contract)
        if self.mode == "rules":
            rules_input = pd.concat(
                [identifiers.rename("patient_id").reset_index(drop=True), safe.reset_index(drop=True)],
                axis=1,
            )
            assessments, rules_audit = assess_dm77_population(
                rules_input,
                self.settings["settings"],
            )
            probability = np.zeros((len(assessments), len(self.levels)), dtype=float)
            valid = assessments.dm77_need_level.notna().to_numpy()
            levels = assessments.dm77_need_level.fillna(1).astype(int).to_numpy()
            probability[np.flatnonzero(valid), levels[valid] - 1] = 1.0
            result = pd.DataFrame({
                "patient_id": assessments.patient_id.astype(str),
                "baseline_need_level": assessments.dm77_need_level.astype("Int64"),
                "baseline_need_score": assessments.dm77_need_level.astype(float),
                "baseline_need_probabilities": [
                    value if is_valid else "{}"
                    for value, is_valid in zip(
                        _probability_json(probability, self.levels), valid
                    )
                ],
                "baseline_need_explanation": assessments.dm77_reason_codes.astype(str),
                "baseline_need_mode": "rules",
                "baseline_need_status": assessments.dm77_assessment_status.astype(str),
                "baseline_need_model_version": self.model_version,
                "baseline_need_protected_pathway": assessments.dm77_protected_pathway.to_numpy(bool),
            })
            for index, level in enumerate(self.levels):
                result[f"baseline_need_probability_{level}"] = probability[:, index]
            audit = {
                "contract_version": self.contract["contract_version"],
                "mode": "rules",
                "model_version": self.model_version,
                "patients": int(len(result)),
                "forbidden_columns_present_but_ignored": forbidden,
                "probability_semantics": self.settings["probability_semantics"],
                "oracle_used": False,
                "capacity_used": False,
                "causal_ranking_used": False,
                "seed": self.seed,
                "rules_audit": rules_audit,
            }
            return BaselineNeedPrediction(result, audit)

        valid = ~safe.isna().any(axis=1).to_numpy()
        probability = np.full((len(safe), len(self.levels)), np.nan, dtype=float)
        if valid.any():
            probability[valid] = self._supervised_probabilities(
                safe.loc[valid].to_numpy(float)
            )
        score = np.full(len(safe), np.nan, dtype=float)
        predicted = np.full(len(safe), np.nan, dtype=float)
        if valid.any():
            score[valid] = probability[valid] @ np.asarray(self.levels, dtype=float)
            cumulative = np.cumsum(probability[valid], axis=1)
            predicted[valid] = 1 + np.sum(cumulative[:, :-1] < 0.5, axis=1)
        result = pd.DataFrame({
            "patient_id": identifiers.to_numpy(str),
            "baseline_need_level": pd.array(predicted, dtype="Int64"),
            "baseline_need_score": score,
            "baseline_need_probabilities": [
                _probability_json(row[None, :], self.levels)[0] if is_valid else "{}"
                for row, is_valid in zip(probability, valid)
            ],
            "baseline_need_explanation": [
                (
                    f"SUPERVISED_ORDINAL_MEDIAN_LEVEL:{int(predicted[index])};"
                    f"EXPECTED_LEVEL:{score[index]:.6f}"
                    if is_valid else
                    "MANUAL_REVIEW_MISSING_BASELINE_INPUT:"
                    + "|".join(safe.columns[safe.iloc[index].isna()])
                )
                for index, is_valid in enumerate(valid)
            ],
            "baseline_need_mode": "supervised",
            "baseline_need_status": np.where(valid, "assessed", "manual_review"),
            "baseline_need_model_version": self.model_version,
            "baseline_need_protected_pathway": np.where(
                valid, predicted == 6, False
            ),
        })
        for index, level in enumerate(self.levels):
            result[f"baseline_need_probability_{level}"] = probability[:, index]
        audit = {
            "contract_version": self.contract["contract_version"],
            "mode": "supervised",
            "model_version": self.model_version,
            "patients": int(len(result)),
            "assessed_patients": int(valid.sum()),
            "manual_review_patients": int((~valid).sum()),
            "forbidden_columns_present_but_ignored": forbidden,
            "allowed_features": list(self.features),
            "oracle_used": False,
            "capacity_used": False,
            "causal_ranking_used": False,
            "seed": self.seed,
        }
        return BaselineNeedPrediction(result, audit)


def cross_fit_baseline_need(
    frame: pd.DataFrame,
    mode: str,
    model_seed: int,
    split_seed: int,
    contract: str | Path | Mapping | None = None,
    reference_label: str | None = None,
) -> CrossFittedBaselineNeedPrediction:
    """Return leakage-safe out-of-fold need predictions plus a frozen full model."""

    if model_seed is None or split_seed is None:
        raise ValueError("Cross-fitted baseline need requires model and split seeds")
    base = BaselineNeedStratifier(mode, int(model_seed), contract)
    if mode == "rules":
        prediction = base.predict(frame)
        assignments = pd.DataFrame({
            "patient_id": frame.patient_id.astype(str),
            "baseline_need_fold": "rules_no_training",
        })
        return CrossFittedBaselineNeedPrediction(
            prediction.assessments,
            assignments,
            base,
            {
                "mode": "rules", "cross_fitting_required": False,
                "model_seed": int(model_seed), "split_seed": int(split_seed),
                "oracle_used": False,
            },
        )

    label = str(reference_label or base.settings["reference_label"])
    if label not in frame:
        raise ValueError(f"Cross-fitted supervised need requires {label}")
    identifiers = _patient_ids(frame)
    target = pd.to_numeric(frame[label], errors="coerce")
    target = pd.Series(_validated_reference(target, base.levels), index=frame.index)
    folds = int(base.settings["cross_fitting_folds"])
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(split_seed)
    )
    fold_values = np.full(len(frame), -1, dtype=int)
    parts = []
    fold_audit = []
    for fold, (development_index, heldout_index) in enumerate(
        splitter.split(np.zeros(len(frame)), target.astype(int))
    ):
        development = frame.iloc[development_index].reset_index(drop=True)
        heldout = frame.iloc[heldout_index].reset_index(drop=True)
        development_target = pd.to_numeric(development[label]).astype(int)
        internal = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=int(split_seed) + 10_007 * (fold + 1),
        )
        model_index, calibration_index = next(
            internal.split(np.zeros(len(development)), development_target)
        )
        model_frame = development.iloc[model_index].reset_index(drop=True)
        calibration_frame = development.iloc[calibration_index].reset_index(drop=True)
        estimator = BaselineNeedStratifier(
            "supervised", int(model_seed) + 1009 * (fold + 1), contract
        ).fit(model_frame, calibration_frame, label)
        predicted = estimator.predict(heldout).assessments
        parts.append(predicted)
        fold_values[heldout_index] = fold
        fit_ids = set(model_frame.patient_id.astype(str))
        calibration_ids = set(calibration_frame.patient_id.astype(str))
        heldout_ids = set(heldout.patient_id.astype(str))
        if fit_ids & calibration_ids or fit_ids & heldout_ids or calibration_ids & heldout_ids:
            raise AssertionError("Baseline-need cross-fitting patient leakage")
        fold_audit.append({
            "fold": fold,
            "fit_rows": int(len(model_frame)),
            "calibration_rows": int(len(calibration_frame)),
            "heldout_rows": int(len(heldout)),
            "patient_partitions_disjoint": True,
            "model_seed": estimator.seed,
            "split_seed": int(split_seed) + 10_007 * (fold + 1),
        })
    if np.any(fold_values < 0):
        raise AssertionError("Every patient must receive one out-of-fold prediction")
    out_of_fold = pd.concat(parts, ignore_index=True).set_index("patient_id").loc[
        identifiers
    ].reset_index()
    assignments = pd.DataFrame({
        "patient_id": identifiers,
        "baseline_need_fold": fold_values,
    })

    full_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=int(split_seed) + 900_001
    )
    full_model_index, full_calibration_index = next(
        full_split.split(np.zeros(len(frame)), target.astype(int))
    )
    full_model = BaselineNeedStratifier(
        "supervised", int(model_seed) + 900_001, contract
    ).fit(
        frame.iloc[full_model_index].reset_index(drop=True),
        frame.iloc[full_calibration_index].reset_index(drop=True),
        label,
    )
    provenance_field = str(
        base.settings.get("reference_provenance_field", "need_reference_source")
    )
    audit = {
        "contract_version": base.contract["contract_version"],
        "mode": "supervised",
        "prediction_semantics": "out_of_fold_for_same_observational_population",
        "folds": folds,
        "model_seed": int(model_seed),
        "split_seed": int(split_seed),
        "fold_audit": fold_audit,
        "every_patient_predicted_once": True,
        "reference_label": label,
        "reference_provenance_field": provenance_field,
        "reference_provenance_values": sorted(
            frame[provenance_field].astype(str).unique()
        ) if provenance_field in frame else [],
        "full_frozen_model_fit_rows": int(len(full_model_index)),
        "full_frozen_model_calibration_rows": int(len(full_calibration_index)),
        "full_frozen_model_partitions_disjoint": not bool(
            set(frame.iloc[full_model_index].patient_id.astype(str)).intersection(
                set(frame.iloc[full_calibration_index].patient_id.astype(str))
            )
        ),
        "oracle_used": False,
        "capacity_used": False,
        "causal_ranking_used": False,
    }
    return CrossFittedBaselineNeedPrediction(
        out_of_fold, assignments, full_model, audit
    )
