"""Governed Phase-10 evaluation of baseline need on external cohorts.

This module evaluates only frozen ``baseline_need_level`` outputs. It never trains a
model and never reads causal recommendation or allocation outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import yaml

from .baseline import (
    BaselineNeedEvaluationResult,
    _ordinal_metrics,
    evaluate_baseline_need,
)


PROTOCOL_VERSION = "prometheus_phase10_external_validation_v1"
SCIENTIFIC_CONTRACT_VERSION = "prometheus_profile_stratification_v2"
BASELINE_NEED_CONTRACT_VERSION = "prometheus_baseline_need_stratifier_v1"
AUTHORIZED_EXTERNAL = "authorized_external_clinical"
SYNTHETIC_SMOKE = "synthetic_smoke_only"
LEVELS = (1, 2, 3, 4, 5, 6)

FORBIDDEN_OUTPUT_COLUMNS = {
    "recommended_profile_id",
    "recommended_actionable_level",
    "recommendation_abstained",
    "allocated_profile_id",
    "allocated_care_level",
    "raw_priority_score",
    "calibrated_incremental_benefit",
    "dr_pseudo_outcome",
    "observed_outcome",
    "outcome",
    "future_resource_use",
}
FORBIDDEN_PREFIXES = (
    "true_",
    "oracle_",
    "potential_outcome",
    "latent_",
    "postindex_",
    "post_index_",
    "future_",
)


@dataclass(frozen=True)
class ExternalBaselineValidationResult:
    clinical_reference: BaselineNeedEvaluationResult
    clinical_uncertainty: pd.DataFrame
    acg_agreement: BaselineNeedEvaluationResult | None
    acg_uncertainty: pd.DataFrame | None
    audit: dict


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(map(str, required)).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _string_patient_ids(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    result["patient_id"] = result.patient_id.astype(str).str.strip()
    if result.patient_id.eq("").any():
        raise ValueError(f"{label} contains blank patient identifiers")
    return result


def _utc(values: pd.Series, field: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{field} must contain complete parseable timestamps")
    return parsed


def _assert_no_forbidden_columns(frame: pd.DataFrame, label: str) -> None:
    bad = []
    for column in map(str, frame.columns):
        lowered = column.lower()
        if lowered in FORBIDDEN_OUTPUT_COLUMNS or lowered.startswith(FORBIDDEN_PREFIXES):
            bad.append(column)
    if bad:
        raise ValueError(f"{label} contains forbidden causal, post-index or oracle fields: {sorted(bad)}")


def _validate_levels(values: pd.Series, field: str, *, allow_missing: bool) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"{field} contains non-numeric values")
    if not allow_missing and numeric.isna().any():
        raise ValueError(f"{field} must be complete")
    present = numeric.dropna()
    if not np.allclose(present, np.round(present)):
        raise ValueError(f"{field} must contain integer ordinal levels")
    if not set(present.astype(int)).issubset(LEVELS):
        raise ValueError(f"{field} must contain only levels 1 through 6")


def _all_explicit_true(values: pd.Series) -> bool:
    normalized = values.map(
        lambda value: value is True
        or (isinstance(value, (int, np.integer)) and int(value) == 1)
        or (isinstance(value, str) and value.strip().lower() == "true")
    )
    return bool(normalized.all())


def _patient_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(frame, ("patient_id", "index_date"), label)
    result = _string_patient_ids(frame, label)
    result["_index_date"] = _utc(result.index_date, f"{label}.index_date")
    if result.patient_id.duplicated().any():
        raise ValueError(f"{label} requires exactly one row per patient")
    return result


def _assert_same_patient_index(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    observed_label: str,
) -> None:
    left = expected[["patient_id", "_index_date"]]
    right = observed[["patient_id", "_index_date"]]
    merged = left.merge(
        right,
        on="patient_id",
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged._merge.eq("both").all():
        raise ValueError(f"{observed_label} must cover exactly the frozen prediction cohort")
    mismatch = merged._index_date_expected.ne(merged._index_date_observed)
    if mismatch.any():
        raise ValueError(f"{observed_label} must use the same patient-specific index date")


def validate_external_protocol(config: Mapping) -> dict:
    protocol = config.get("protocol", {})
    if protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"External-validation protocol must be {PROTOCOL_VERSION}")
    if protocol.get("scientific_contract_version") != SCIENTIFIC_CONTRACT_VERSION:
        raise ValueError("External validation requires the frozen scientific contract v2")
    if protocol.get("baseline_need_contract_version") != BASELINE_NEED_CONTRACT_VERSION:
        raise ValueError("External validation requires the frozen baseline-need contract v1")
    if protocol.get("primary_output") != "baseline_need_level":
        raise ValueError("Phase 10 validates baseline_need_level only")
    if protocol.get("causal_policy_outputs_in_scope") is not False:
        raise ValueError("Causal recommendation and allocation outputs must remain out of scope")
    if protocol.get("output_policy") != "aggregate_only":
        raise ValueError("External validation may export aggregate results only")

    evaluation = config.get("evaluation", {})
    if tuple(map(int, evaluation.get("levels", ()))) != LEVELS:
        raise ValueError("External-validation ordinal levels must be 1 through 6")
    if evaluation.get("acg_comparison_role") != (
        "descriptive_ordinal_agreement_not_accuracy_or_superiority"
    ):
        raise ValueError("ACG comparison must remain descriptive ordinal agreement")
    if int(evaluation.get("minimum_subgroup_size", 0)) < 30:
        raise ValueError("External subgroup reporting requires a minimum cell size of 30")
    if int(evaluation.get("minimum_reportable_cell_size", 0)) < 10:
        raise ValueError("Aggregate external reporting requires cell suppression below 10")
    uncertainty = evaluation.get("uncertainty", {})
    if uncertainty.get("method") != "patient_level_percentile_bootstrap":
        raise ValueError("External uncertainty must use the frozen patient bootstrap")
    if int(uncertainty.get("samples", 0)) < 500:
        raise ValueError("External uncertainty requires at least 500 bootstrap samples")
    if float(uncertainty.get("confidence_level", 0.0)) != 0.95:
        raise ValueError("External confidence level is frozen at 0.95")
    bootstrap_seeds = (
        int(uncertainty["clinical_reference_seed"]),
        int(uncertainty["acg_agreement_seed"]),
    )
    if min(bootstrap_seeds) < 0 or len(set(bootstrap_seeds)) != 2:
        raise ValueError("External bootstrap seeds must be explicit, non-negative and distinct")

    governance = config.get("governance", {})
    if governance.get("authorized_external_classification") != AUTHORIZED_EXTERNAL:
        raise ValueError("Authorized external data classification is frozen")
    if governance.get("smoke_classification") != SYNTHETIC_SMOKE:
        raise ValueError("Synthetic smoke data classification is frozen")
    if governance.get("external_claims_allowed_by_software_run") is not False:
        raise ValueError("Software runs cannot authorize external claims")

    required_features = tuple(map(
        str, config.get("preindex_contract", {}).get("required_feature_names", ())
    ))
    if not required_features or len(required_features) != len(set(required_features)):
        raise ValueError("Required pre-index features must be non-empty and unique")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": str(protocol.get("status")),
        "required_preindex_features": list(required_features),
        "causal_policy_outputs_in_scope": False,
        "output_policy": "aggregate_only",
    }


def validate_governance_manifest(
    manifest: Mapping,
    config: Mapping,
    *,
    include_acg: bool,
) -> dict:
    classification = str(manifest.get("data_classification", ""))
    governance = config["governance"]
    allowed = {
        str(governance["authorized_external_classification"]),
        str(governance["smoke_classification"]),
    }
    if classification not in allowed:
        raise ValueError(f"Unsupported external-validation data classification: {classification!r}")

    if manifest.get("external_claims_allowed") is not False:
        raise ValueError("A software evaluation run cannot authorize external claims")
    if classification == AUTHORIZED_EXTERNAL:
        for field in governance["required_external_attestations"]:
            if manifest.get(str(field)) is not True:
                raise ValueError(f"Missing required governance attestation: {field}")
        if not str(manifest.get("approval_id", "")).strip():
            raise ValueError("Authorized external validation requires an approval_id")
    else:
        if manifest.get("synthetic_smoke_only") is not True:
            raise ValueError("Synthetic smoke inputs must be explicitly marked smoke-only")

    if include_acg:
        if classification == AUTHORIZED_EXTERNAL:
            for field in governance["required_acg_attestations"]:
                if manifest.get(str(field)) is not True:
                    raise ValueError(f"Missing required ACG attestation: {field}")
            if not str(manifest.get("acg_approval_id", "")).strip():
                raise ValueError("Authorized ACG comparison requires an acg_approval_id")
        elif manifest.get("synthetic_acg_smoke_only") is not True:
            raise ValueError("Synthetic ACG input must be explicitly marked smoke-only")

    return {
        "data_classification": classification,
        "authorized_external_evaluation": classification == AUTHORIZED_EXTERNAL,
        "synthetic_smoke_only": classification == SYNTHETIC_SMOKE,
        "acg_included": bool(include_acg),
        "external_claims_allowed": False,
    }


def _validate_prediction_and_reference(
    predictions: pd.DataFrame,
    clinical_reference: pd.DataFrame,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = config["inputs"]
    _require_columns(
        predictions, inputs["prediction_required_columns"], "Frozen predictions"
    )
    _require_columns(
        clinical_reference,
        inputs["clinical_reference_required_columns"],
        "Clinical reference",
    )
    _assert_no_forbidden_columns(predictions, "Frozen predictions")
    _assert_no_forbidden_columns(clinical_reference, "Clinical reference")
    if any(column in predictions for column in (
        "clinical_reference_level", "acg_risk_group"
    )):
        raise ValueError("Frozen predictions must not contain evaluation references")
    if any(column in clinical_reference for column in (
        "baseline_need_level", "acg_risk_group", "recommended_actionable_level",
        "allocated_care_level",
    )):
        raise ValueError("Clinical reference must not contain PROMETHEUS or ACG outputs")

    prediction = _patient_index(predictions, "Frozen predictions")
    reference = _patient_index(clinical_reference, "Clinical reference")
    _assert_same_patient_index(prediction, reference, "Clinical reference")
    _validate_levels(prediction.baseline_need_level, "baseline_need_level", allow_missing=True)
    _validate_levels(
        reference.clinical_reference_level,
        "clinical_reference_level",
        allow_missing=False,
    )
    if not _all_explicit_true(reference.adjudication_independent_of_prometheus):
        raise ValueError("Every clinical reference must be independently adjudicated")
    if reference.adjudication_panel_version.astype(str).str.strip().eq("").any():
        raise ValueError("Clinical adjudication panel version must be recorded")
    if reference.adjudication_panel_version.astype(str).nunique() != 1:
        raise ValueError("One frozen adjudication panel version is required per run")
    if prediction.baseline_need_model_version.astype(str).str.strip().eq("").any():
        raise ValueError("Baseline-need model version must be recorded")
    if prediction.baseline_need_model_version.astype(str).nunique() != 1:
        raise ValueError("One frozen baseline-need model version is required per run")

    reference_cutoff = _utc(
        reference.reference_data_cutoff, "clinical_reference.reference_data_cutoff"
    )
    adjudicated = _utc(reference.adjudicated_at, "clinical_reference.adjudicated_at")
    released = _utc(reference.reference_released_at, "clinical_reference.reference_released_at")
    frozen = _utc(prediction.prediction_frozen_at, "predictions.prediction_frozen_at")
    if not reference_cutoff.lt(reference._index_date).all():
        raise ValueError("Clinical reference inputs must be strictly pre-index")
    if not adjudicated.le(released).all():
        raise ValueError("Clinical reference cannot be released before adjudication")

    freeze_frame = prediction[["patient_id"]].copy()
    freeze_frame["_frozen"] = frozen
    release_frame = reference[["patient_id"]].copy()
    release_frame["_released"] = released
    timing = freeze_frame.merge(release_frame, on="patient_id", validate="one_to_one")
    if not timing._frozen.lt(timing._released).all():
        raise ValueError("Predictions must be frozen before clinical-reference release")
    return prediction, reference


def _validate_feature_provenance(
    predictions: pd.DataFrame,
    feature_provenance: pd.DataFrame,
    config: Mapping,
) -> dict:
    required_columns = config["inputs"]["feature_provenance_required_columns"]
    _require_columns(feature_provenance, required_columns, "Feature provenance")
    provenance = _string_patient_ids(feature_provenance, "Feature provenance")
    provenance["_index_date"] = _utc(
        provenance.index_date, "feature_provenance.index_date"
    )
    provenance["_latest_source_time"] = _utc(
        provenance.latest_source_time, "feature_provenance.latest_source_time"
    )
    if provenance[["patient_id", "feature_name"]].duplicated().any():
        raise ValueError("Feature provenance requires one row per patient and feature")
    if not provenance._latest_source_time.lt(provenance._index_date).all():
        raise ValueError("Every feature must be measured strictly before the index date")

    patient_index = provenance[["patient_id", "_index_date"]].drop_duplicates()
    if patient_index.patient_id.duplicated().any():
        raise ValueError("Feature provenance has inconsistent index dates within patient")
    _assert_same_patient_index(predictions, patient_index, "Feature provenance")

    feature_names = provenance.feature_name.astype(str).str.strip()
    if feature_names.eq("").any():
        raise ValueError("Feature provenance contains blank feature names")
    forbidden = sorted({
        value for value in feature_names
        if value.lower() in FORBIDDEN_OUTPUT_COLUMNS
        or value.lower().startswith(FORBIDDEN_PREFIXES)
    })
    if forbidden:
        raise ValueError(f"Feature provenance contains forbidden features: {forbidden}")

    required_features = set(map(
        str, config["preindex_contract"]["required_feature_names"]
    )) | set(map(str, config["evaluation"]["subgroup_columns"]))
    observed = provenance.groupby("patient_id").feature_name.agg(
        lambda values: set(map(str, values))
    )
    incomplete = observed.map(lambda values: not required_features.issubset(values))
    if incomplete.any():
        raise ValueError(
            "Feature provenance is incomplete for patients: "
            + ", ".join(observed.index[incomplete].astype(str)[:5])
        )
    return {
        "provenance_rows": int(len(provenance)),
        "required_feature_count": int(len(required_features)),
        "all_features_strictly_preindex": True,
        "forbidden_features_present": False,
    }


def _validate_subgroups(
    predictions: pd.DataFrame,
    subgroup_frame: pd.DataFrame | None,
    subgroup_columns: tuple[str, ...],
) -> pd.DataFrame | None:
    if not subgroup_columns:
        return None
    if subgroup_frame is None:
        raise ValueError("The frozen protocol requires a subgroup frame")
    _require_columns(
        subgroup_frame, ("patient_id", "index_date", *subgroup_columns), "Subgroups"
    )
    _assert_no_forbidden_columns(subgroup_frame, "Subgroups")
    groups = _patient_index(subgroup_frame, "Subgroups")
    _assert_same_patient_index(predictions, groups, "Subgroups")
    return groups


def _validate_acg(
    predictions: pd.DataFrame,
    acg: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    _require_columns(acg, config["inputs"]["acg_required_columns"], "ACG comparison")
    _assert_no_forbidden_columns(acg, "ACG comparison")
    if any(column in acg for column in (
        "baseline_need_level", "clinical_reference_level", "recommended_actionable_level",
        "recommended_profile_id", "allocated_care_level",
    )):
        raise ValueError("ACG comparison cannot contain PROMETHEUS or clinical-reference outputs")
    frame = _patient_index(acg, "ACG comparison")
    _assert_same_patient_index(predictions, frame, "ACG comparison")
    _validate_levels(frame.acg_risk_group, "acg_risk_group", allow_missing=False)
    cutoff = _utc(frame.acg_data_cutoff, "acg.acg_data_cutoff")
    if not cutoff.lt(frame._index_date).all():
        raise ValueError("ACG inputs must be strictly pre-index")
    if frame.acg_mapping_version.astype(str).str.strip().eq("").any():
        raise ValueError("ACG mapping version must be recorded")
    if frame.acg_mapping_version.astype(str).nunique() != 1:
        raise ValueError("One frozen ACG mapping version is required per run")
    return frame


def _bootstrap_ordinal_uncertainty(
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    reference_field: str,
    prediction_field: str,
    point_summary: Mapping,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    left = reference[["patient_id", reference_field]].copy()
    right = prediction[["patient_id", prediction_field]].copy()
    left["patient_id"] = left.patient_id.astype(str)
    right["patient_id"] = right.patient_id.astype(str)
    merged = left.merge(right, on="patient_id", validate="one_to_one")
    reference_values = pd.to_numeric(merged[reference_field], errors="coerce").to_numpy()
    prediction_values = pd.to_numeric(merged[prediction_field], errors="coerce").to_numpy()
    metrics = (
        "quadratic_weighted_kappa",
        "macro_f1",
        "balanced_accuracy",
        "ordinal_mae",
        "within_one_level_accuracy",
        "spearman",
        "coverage",
    )
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(int(seed))
    for _ in range(int(samples)):
        indices = rng.integers(0, len(merged), size=len(merged))
        sampled_reference = reference_values[indices]
        sampled_prediction = prediction_values[indices]
        assessed = np.isfinite(sampled_reference) & np.isfinite(sampled_prediction)
        draws["coverage"].append(float(assessed.mean()))
        if not assessed.any():
            for metric in metrics[:-1]:
                draws[metric].append(np.nan)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = _ordinal_metrics(
                sampled_reference[assessed].astype(int),
                sampled_prediction[assessed].astype(int),
            )
        for metric in metrics[:-1]:
            draws[metric].append(float(values[metric]))

    alpha = (1.0 - float(confidence_level)) / 2.0
    rows = []
    for metric in metrics:
        values = np.asarray(draws[metric], dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "metric": metric,
            "estimate": float(point_summary[metric]),
            "ci_lower": float(np.quantile(finite, alpha)) if len(finite) else np.nan,
            "ci_upper": float(np.quantile(finite, 1.0 - alpha)) if len(finite) else np.nan,
            "bootstrap_samples": int(samples),
            "finite_bootstrap_samples": int(len(finite)),
            "bootstrap_seed": int(seed),
            "confidence_level": float(confidence_level),
        })
    return pd.DataFrame(rows)


def evaluate_external_baseline_validation(
    predictions: pd.DataFrame,
    clinical_reference: pd.DataFrame,
    feature_provenance: pd.DataFrame,
    governance_manifest: Mapping,
    config: Mapping,
    *,
    subgroup_frame: pd.DataFrame | None = None,
    acg: pd.DataFrame | None = None,
) -> ExternalBaselineValidationResult:
    """Validate governed inputs and compute aggregate ordinal diagnostics only."""

    protocol_audit = validate_external_protocol(config)
    governance_audit = validate_governance_manifest(
        governance_manifest, config, include_acg=acg is not None
    )
    prediction, reference = _validate_prediction_and_reference(
        predictions, clinical_reference, config
    )
    provenance_audit = _validate_feature_provenance(
        prediction, feature_provenance, config
    )
    subgroup_columns = tuple(map(str, config["evaluation"]["subgroup_columns"]))
    groups = _validate_subgroups(prediction, subgroup_frame, subgroup_columns)
    acg_frame = _validate_acg(prediction, acg, config) if acg is not None else None

    clinical = evaluate_baseline_need(
        reference=reference,
        prediction=prediction,
        subgroup_frame=groups,
        subgroup_columns=subgroup_columns,
        minimum_subgroup_size=int(config["evaluation"]["minimum_subgroup_size"]),
        reference_field=str(config["evaluation"]["clinical_reference_field"]),
        prediction_field="baseline_need_level",
    )
    uncertainty = config["evaluation"]["uncertainty"]
    clinical_uncertainty = _bootstrap_ordinal_uncertainty(
        reference,
        prediction,
        reference_field=str(config["evaluation"]["clinical_reference_field"]),
        prediction_field="baseline_need_level",
        point_summary=clinical.summary,
        samples=int(uncertainty["samples"]),
        seed=int(uncertainty["clinical_reference_seed"]),
        confidence_level=float(uncertainty["confidence_level"]),
    )
    acg_result = None
    acg_uncertainty = None
    if acg_frame is not None:
        acg_result = evaluate_baseline_need(
            reference=acg_frame,
            prediction=prediction,
            subgroup_frame=groups,
            subgroup_columns=subgroup_columns,
            minimum_subgroup_size=int(config["evaluation"]["minimum_subgroup_size"]),
            reference_field=str(config["evaluation"]["acg_field"]),
            prediction_field="baseline_need_level",
        )
        acg_uncertainty = _bootstrap_ordinal_uncertainty(
            acg_frame,
            prediction,
            reference_field=str(config["evaluation"]["acg_field"]),
            prediction_field="baseline_need_level",
            point_summary=acg_result.summary,
            samples=int(uncertainty["samples"]),
            seed=int(uncertainty["acg_agreement_seed"]),
            confidence_level=float(uncertainty["confidence_level"]),
        )

    audit = {
        **protocol_audit,
        **governance_audit,
        **provenance_audit,
        "patients": int(len(prediction)),
        "clinical_reference_role": "independent_adjudicated_ordinal_reference",
        "acg_comparison_role": (
            config["evaluation"]["acg_comparison_role"] if acg is not None else "not_run"
        ),
        "causal_actionability_evaluated": False,
        "recommendation_or_allocation_fields_read": False,
        "row_level_output_written": False,
        "uncertainty_method": uncertainty["method"],
        "bootstrap_samples": int(uncertainty["samples"]),
        "clinical_reference_bootstrap_seed": int(
            uncertainty["clinical_reference_seed"]
        ),
        "acg_agreement_bootstrap_seed": int(uncertainty["acg_agreement_seed"]),
        "external_evidence_status": (
            "authorized_external_evaluation_completed_requires_governance_review"
            if governance_audit["authorized_external_evaluation"]
            else "synthetic_smoke_only_not_external_evidence"
        ),
    }
    return ExternalBaselineValidationResult(
        clinical_reference=clinical,
        clinical_uncertainty=clinical_uncertainty,
        acg_agreement=acg_result,
        acg_uncertainty=acg_uncertainty,
        audit=audit,
    )


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_pipeline_contract(config: Mapping, config_path: Path) -> dict:
    protocol = config["protocol"]
    pipeline_path = Path(str(protocol["pipeline_contract_path"]))
    if not pipeline_path.is_absolute():
        pipeline_path = config_path.resolve().parent / pipeline_path
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Pipeline contract not found: {pipeline_path}")
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    scientific = pipeline["scientific_contract"]
    baseline = pipeline["baseline_need"]["contract"]
    if scientific["contract_version"] != protocol["scientific_contract_version"]:
        raise ValueError("External protocol and scientific contract versions disagree")
    if baseline["contract_version"] != protocol["baseline_need_contract_version"]:
        raise ValueError("External protocol and baseline-need contract versions disagree")
    expected_features = set(map(str, baseline["allowed_preindex_features"]))
    external_features = set(map(
        str, config["preindex_contract"]["required_feature_names"]
    ))
    if external_features != expected_features:
        raise ValueError("External protocol must freeze the current pre-index feature set")
    return {
        "pipeline_contract_path": str(pipeline_path),
        "scientific_contract_version": scientific["contract_version"],
        "baseline_need_contract_version": baseline["contract_version"],
        "preindex_feature_contract_matches": True,
    }


def _suppress_confusion_cells(frame: pd.DataFrame, minimum_cell_size: int) -> pd.DataFrame:
    result = frame.copy()
    value_columns = [column for column in result.columns if column != "reference_level"]
    for column in value_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result.loc[values.gt(0) & values.lt(minimum_cell_size), column] = pd.NA
    return result


def _suppress_small_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if result.empty or "status" not in result:
        return result
    identifiers = {"subgroup_column", "subgroup_value", "status"}
    suppressed = result.status.astype(str).ne("reported")
    for column in result.columns:
        if column not in identifiers:
            result.loc[suppressed, column] = pd.NA
    return result


def _json_compatible(value):
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def run_external_validation(
    config_path: str | Path,
    predictions_path: str | Path,
    clinical_reference_path: str | Path,
    feature_provenance_path: str | Path,
    governance_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    subgroup_path: str | Path | None = None,
    acg_path: str | Path | None = None,
) -> Path:
    """Run the aggregate-only Phase-10 evaluator from governed CSV inputs."""

    paths = {
        "config": Path(config_path),
        "predictions": Path(predictions_path),
        "clinical_reference": Path(clinical_reference_path),
        "feature_provenance": Path(feature_provenance_path),
        "governance_manifest": Path(governance_manifest_path),
    }
    if subgroup_path is not None:
        paths["subgroups"] = Path(subgroup_path)
    if acg_path is not None:
        paths["acg"] = Path(acg_path)
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing external-validation inputs: {missing}")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"External-validation output already exists: {output}")

    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    validate_external_protocol(config)
    pipeline_contract_audit = _validate_pipeline_contract(config, paths["config"])
    governance = json.loads(paths["governance_manifest"].read_text(encoding="utf-8"))
    result = evaluate_external_baseline_validation(
        predictions=pd.read_csv(paths["predictions"]),
        clinical_reference=pd.read_csv(paths["clinical_reference"]),
        feature_provenance=pd.read_csv(paths["feature_provenance"]),
        governance_manifest=governance,
        config=config,
        subgroup_frame=(pd.read_csv(paths["subgroups"]) if "subgroups" in paths else None),
        acg=(pd.read_csv(paths["acg"]) if "acg" in paths else None),
    )

    output.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    minimum_cell_size = int(config["evaluation"]["minimum_reportable_cell_size"])

    def write_json(name: str, payload: Mapping) -> None:
        target = output / name
        target.write_text(
            json.dumps(
                _json_compatible(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        written.append(target)

    def write_csv(name: str, frame: pd.DataFrame) -> None:
        target = output / name
        frame.to_csv(target, index=False)
        written.append(target)

    write_json("external_validation_audit.json", {
        **result.audit,
        **pipeline_contract_audit,
        "minimum_reportable_cell_size": minimum_cell_size,
        "small_cell_suppression_applied": True,
    })
    write_json("clinical_reference_summary.json", result.clinical_reference.summary)
    write_csv("clinical_reference_uncertainty.csv", result.clinical_uncertainty)
    write_csv(
        "clinical_reference_confusion_matrix.csv",
        _suppress_confusion_cells(result.clinical_reference.confusion_matrix, minimum_cell_size),
    )
    write_csv(
        "clinical_reference_subgroup_metrics.csv",
        _suppress_small_subgroups(result.clinical_reference.subgroup_metrics),
    )
    if result.acg_agreement is not None:
        write_json("acg_agreement_summary.json", result.acg_agreement.summary)
        write_csv("acg_agreement_uncertainty.csv", result.acg_uncertainty)
        write_csv(
            "acg_agreement_confusion_matrix.csv",
            _suppress_confusion_cells(result.acg_agreement.confusion_matrix, minimum_cell_size),
        )
        write_csv(
            "acg_agreement_subgroup_metrics.csv",
            _suppress_small_subgroups(result.acg_agreement.subgroup_metrics),
        )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": result.audit["external_evidence_status"],
        "input_sha256": {name: _checksum(path) for name, path in paths.items()},
        "output_sha256": {path.name: _checksum(path) for path in written},
        "row_level_output_written": False,
        "external_claims_allowed": False,
    }
    write_json("external_validation_manifest.json", manifest)
    return output
