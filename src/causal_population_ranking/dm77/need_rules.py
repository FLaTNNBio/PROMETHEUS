"""Auditable rules for DM 77-aligned baseline-need stratification.

The six level descriptions come from the DM 77 population-stratification model.
The numeric thresholds in this module are a configurable research
operationalization: they are not official national cut-offs and require local
clinical, social-care, and governance validation before any real-world use.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DM77_LEVEL_LABELS = {
    1: "Persona in salute",
    2: "Complessita minima o limitata nel tempo",
    3: "Complessita media",
    4: "Complessita medio-alta con possibile fragilita sociale",
    5: "Complessita elevata con possibile fragilita sociale e non autosufficienza",
    6: "Cure palliative",
}

# Only pre-index fields belong to this contract. Treatment, outcomes, priority
# scores, and evaluation-only truth are intentionally absent.
DM77_INPUT_FIELDS = (
    "age",
    "condition_distinct",
    "prior_inpatient",
    "prior_emergency",
    "medication_distinct",
    "frailty_index",
    "functional_limitation_score",
    "social_fragility_score",
    "cognitive_impairment",
    "non_self_sufficiency",
    "caregiver_available",
    "housing_instability",
    "palliative_need",
)

DM77_FORBIDDEN_INPUT_PREFIXES = ("true_", "oracle_", "potential_outcome_")
DM77_FORBIDDEN_INPUT_COLUMNS = {
    "treatment",
    "treatment_level",
    "observed_outcome",
    "outcome",
    "causal_priority",
    "raw_score",
    "calibrated_benefit",
    "latent_rank",
}


DEFAULT_THRESHOLDS = {
    "clinical_high": 0.60,
    "functional_moderate": 0.35,
    "functional_high": 0.75,
    "social_fragility": 0.45,
    "frailty_moderate": 0.25,
    "frailty_high": 0.70,
    "chronicity_min_conditions": 2,
    "high_condition_count": 5,
    "high_prior_inpatient": 2,
}


def load_dm77_settings(section: Mapping | None = None) -> dict:
    """Load a DM 77 YAML profile and merge explicit run-level overrides."""

    section = dict(section or {})
    settings: dict = {}
    config_path = section.pop("config_path", None)
    if config_path:
        path = Path(str(config_path))
        if not path.exists():
            raise FileNotFoundError(f"DM 77 configuration does not exist: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("The DM 77 configuration root must be a mapping")
        settings.update(loaded)
    for key, value in section.items():
        if key == "thresholds" and isinstance(value, Mapping):
            settings[key] = {**settings.get(key, {}), **dict(value)}
        else:
            settings[key] = value
    settings["thresholds"] = {**DEFAULT_THRESHOLDS, **settings.get("thresholds", {})}
    settings.setdefault("assessment_version", "dm77_research_operationalization_v1")
    settings.setdefault("threshold_profile", "research_default_not_official")
    settings.setdefault("missing_data_policy", "manual_review")
    settings.setdefault("required_fields", list(DM77_INPUT_FIELDS))
    settings.setdefault("exclude_level_vi_from_standard_allocation", True)
    settings.setdefault("exclude_manual_review_from_standard_allocation", True)
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Mapping) -> None:
    if settings.get("missing_data_policy") != "manual_review":
        raise ValueError("DM 77 missing_data_policy must be 'manual_review'")
    required = tuple(settings.get("required_fields", ()))
    unknown = sorted(set(required).difference(DM77_INPUT_FIELDS))
    if unknown:
        raise ValueError(f"Unknown DM 77 required fields: {unknown}")
    thresholds = settings["thresholds"]
    probability_thresholds = (
        "clinical_high",
        "functional_moderate",
        "functional_high",
        "social_fragility",
        "frailty_moderate",
        "frailty_high",
    )
    if any(not 0.0 <= float(thresholds[name]) <= 1.0 for name in probability_thresholds):
        raise ValueError("DM 77 score thresholds must be in [0, 1]")
    if float(thresholds["functional_moderate"]) >= float(thresholds["functional_high"]):
        raise ValueError("functional_moderate must be lower than functional_high")
    if float(thresholds["frailty_moderate"]) >= float(thresholds["frailty_high"]):
        raise ValueError("frailty_moderate must be lower than frailty_high")


def _bounded(value: object) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _dimension_label(value: float) -> str:
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "medium"
    return "high"


def _clinical_score(row: pd.Series) -> float:
    age_component = np.clip((float(row.age) - 45.0) / 45.0, 0.0, 1.0)
    return _bounded(
        0.14 * age_component
        + 0.25 * min(float(row.condition_distinct) / 6.0, 1.0)
        + 0.15 * min(float(row.medication_distinct) / 10.0, 1.0)
        + 0.22 * min(float(row.prior_inpatient) / 3.0, 1.0)
        + 0.12 * min(float(row.prior_emergency) / 4.0, 1.0)
        + 0.12 * _bounded(row.frailty_index)
    )


def _functional_score(row: pd.Series) -> float:
    return _bounded(
        0.55 * _bounded(row.functional_limitation_score)
        + 0.25 * _bounded(row.frailty_index)
        + 0.10 * float(bool(row.cognitive_impairment))
        + 0.10 * float(bool(row.non_self_sufficiency))
    )


def _social_score(row: pd.Series) -> float:
    return _bounded(
        0.70 * _bounded(row.social_fragility_score)
        + 0.18 * float(not bool(row.caregiver_available))
        + 0.12 * float(bool(row.housing_instability))
    )


def _utilization_score(row: pd.Series) -> float:
    return _bounded(
        0.65 * min(float(row.prior_inpatient) / 3.0, 1.0)
        + 0.35 * min(float(row.prior_emergency) / 4.0, 1.0)
    )


def _assign_level(row: pd.Series, clinical: float, functional: float, social: float, thresholds: Mapping) -> tuple[int, list[str]]:
    if bool(row.palliative_need):
        return 6, ["PALLIATIVE_NEED_RECORDED", "PROTECTED_PALLIATIVE_PATHWAY"]

    high_functional = functional >= float(thresholds["functional_high"])
    high_frailty = float(row.frailty_index) >= float(thresholds["frailty_high"])
    high_clinical = clinical >= float(thresholds["clinical_high"])
    social_fragility = social >= float(thresholds["social_fragility"])
    non_self_sufficient = bool(row.non_self_sufficiency)
    if non_self_sufficient or (high_functional and (high_clinical or high_frailty or social_fragility)):
        reasons = ["HIGH_COMPLEXITY_OR_NON_SELF_SUFFICIENCY"]
        if non_self_sufficient:
            reasons.append("NON_SELF_SUFFICIENCY_RECORDED")
        if social_fragility:
            reasons.append("SOCIAL_FRAGILITY_RECORDED")
        return 5, reasons

    high_burden = (
        float(row.condition_distinct) >= int(thresholds["high_condition_count"])
        or float(row.prior_inpatient) >= int(thresholds["high_prior_inpatient"])
    )
    moderate_functional = functional >= float(thresholds["functional_moderate"])
    if (high_clinical and (moderate_functional or social_fragility)) or (high_burden and social_fragility):
        reasons = ["MEDIUM_HIGH_MULTIDIMENSIONAL_COMPLEXITY"]
        if social_fragility:
            reasons.append("SOCIAL_FRAGILITY_RECORDED")
        return 4, reasons

    chronicity = float(row.condition_distinct) >= int(thresholds["chronicity_min_conditions"])
    moderate_frailty = float(row.frailty_index) >= float(thresholds["frailty_moderate"])
    if chronicity or moderate_frailty or moderate_functional or bool(row.cognitive_impairment):
        reasons = ["CHRONICITY_OR_EARLY_FRAGILITY"]
        if moderate_functional:
            reasons.append("FUNCTIONAL_LIMITATION_RECORDED")
        return 3, reasons

    episodic_need = (
        float(row.condition_distinct) > 0
        or float(row.prior_inpatient) > 0
        or float(row.prior_emergency) > 0
        or float(row.medication_distinct) > 0
    )
    if episodic_need:
        return 2, ["MINIMAL_OR_TIME_LIMITED_COMPLEXITY"]
    return 1, ["NO_RECORDED_COMPLEXITY_INDICATOR"]


def assess_dm77_population(frame: pd.DataFrame, settings: Mapping | None = None) -> tuple[pd.DataFrame, dict]:
    """Assign DM 77 need levels without reading treatment, outcomes, or oracle data."""

    settings = load_dm77_settings(settings)
    if "patient_id" not in frame:
        raise ValueError("DM 77 assessment requires patient_id")
    identifiers = frame.patient_id.astype(str)
    if identifiers.duplicated().any():
        raise ValueError("DM 77 assessment requires one row per patient")

    required = tuple(settings["required_fields"])
    missing_columns = sorted(set(required).difference(frame.columns))
    safe = frame.reindex(columns=["patient_id", *DM77_INPUT_FIELDS]).copy()
    rows: list[dict] = []
    for position, row in safe.iterrows():
        missing_fields = [name for name in required if name in missing_columns or pd.isna(row[name])]
        base = {
            "patient_id": str(row.patient_id),
            "dm77_assessment_version": str(settings["assessment_version"]),
            "dm77_threshold_profile": str(settings["threshold_profile"]),
        }
        if missing_fields:
            rows.append({
                **base,
                "dm77_need_level": pd.NA,
                "dm77_need_label": "Valutazione multidimensionale richiesta",
                "dm77_clinical_complexity": np.nan,
                "dm77_functional_complexity": np.nan,
                "dm77_social_complexity": np.nan,
                "dm77_utilization_intensity": np.nan,
                "dm77_clinical_band": "unknown",
                "dm77_functional_band": "unknown",
                "dm77_social_band": "unknown",
                "dm77_assessment_status": "manual_review",
                "dm77_needs_multidimensional_assessment": True,
                "dm77_protected_pathway": False,
                "dm77_data_completeness": float((len(required) - len(missing_fields)) / max(len(required), 1)),
                "dm77_reason_codes": "|".join(f"MISSING_INPUT:{name}" for name in missing_fields),
            })
            continue

        clinical = _clinical_score(row)
        functional = _functional_score(row)
        social = _social_score(row)
        utilization = _utilization_score(row)
        level, reasons = _assign_level(row, clinical, functional, social, settings["thresholds"])
        protected = level == 6
        rows.append({
            **base,
            "dm77_need_level": level,
            "dm77_need_label": DM77_LEVEL_LABELS[level],
            "dm77_clinical_complexity": clinical,
            "dm77_functional_complexity": functional,
            "dm77_social_complexity": social,
            "dm77_utilization_intensity": utilization,
            "dm77_clinical_band": _dimension_label(clinical),
            "dm77_functional_band": _dimension_label(functional),
            "dm77_social_band": _dimension_label(social),
            "dm77_assessment_status": "protected_pathway" if protected else "assessed",
            "dm77_needs_multidimensional_assessment": level >= 3,
            "dm77_protected_pathway": protected,
            "dm77_data_completeness": 1.0,
            "dm77_reason_codes": "|".join(reasons),
        })

    assessments = pd.DataFrame(rows)
    assessments["dm77_need_level"] = assessments["dm77_need_level"].astype("Int64")
    forbidden_present = sorted(
        column for column in frame.columns
        if column in DM77_FORBIDDEN_INPUT_COLUMNS
        or any(column.startswith(prefix) for prefix in DM77_FORBIDDEN_INPUT_PREFIXES)
    )
    audit = {
        "assessment_version": str(settings["assessment_version"]),
        "threshold_profile": str(settings["threshold_profile"]),
        "operationalization_status": "research_configurable_not_official_national_algorithm",
        "patients": int(len(assessments)),
        "assessed_patients": int(assessments.dm77_need_level.notna().sum()),
        "manual_review_patients": int((assessments.dm77_assessment_status == "manual_review").sum()),
        "protected_level_vi_patients": int(assessments.dm77_protected_pathway.sum()),
        "represented_need_levels": sorted(
            int(level) for level in assessments.dm77_need_level.dropna().unique()
        ),
        "need_level_counts": {
            str(level): int((assessments.dm77_need_level == level).sum())
            for level in DM77_LEVEL_LABELS
        },
        "input_fields": list(DM77_INPUT_FIELDS),
        "forbidden_columns_present_but_ignored": forbidden_present,
        "treatment_used": False,
        "outcome_used": False,
        "oracle_used": False,
        "causal_priority_used": False,
        "level_is_treatment": False,
        "level_is_ranker_feature": False,
    }
    return assessments, audit


def summarize_dm77_population(assessments: pd.DataFrame) -> pd.DataFrame:
    """Create an auditable population count table, including abstentions."""

    total = max(len(assessments), 1)
    rows = []
    for level, label in DM77_LEVEL_LABELS.items():
        count = int((assessments.dm77_need_level == level).sum())
        rows.append({
            "dm77_need_level": level,
            "dm77_need_label": label,
            "patient_count": count,
            "population_fraction": count / total,
        })
    manual = int(assessments.dm77_need_level.isna().sum())
    rows.append({
        "dm77_need_level": pd.NA,
        "dm77_need_label": "Valutazione multidimensionale richiesta",
        "patient_count": manual,
        "population_fraction": manual / total,
    })
    summary = pd.DataFrame(rows)
    summary["dm77_need_level"] = summary["dm77_need_level"].astype("Int64")
    return summary
