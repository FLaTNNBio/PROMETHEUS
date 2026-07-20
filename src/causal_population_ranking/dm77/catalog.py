"""Governed intervention catalogue linked to need levels, not to causal effects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import yaml


def load_intervention_catalog(source: str | Path | Mapping) -> dict:
    if isinstance(source, Mapping):
        catalog = dict(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"DM 77 intervention catalogue does not exist: {path}")
        catalog = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    interventions = catalog.get("interventions")
    if not isinstance(interventions, list) or not interventions:
        raise ValueError("The DM 77 catalogue requires a non-empty interventions list")
    identifiers = []
    for item in interventions:
        required = {"intervention_id", "label", "eligible_levels", "protected_only"}
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"Intervention is missing fields: {missing}")
        levels = tuple(int(level) for level in item["eligible_levels"])
        if not levels or any(level not in range(1, 7) for level in levels):
            raise ValueError(f"Invalid eligible levels for {item['intervention_id']}")
        identifiers.append(str(item["intervention_id"]))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("DM 77 intervention identifiers must be unique")
    catalog.setdefault("catalog_version", "dm77_intervention_catalog_v1")
    catalog.setdefault("causal_effectiveness_claim", False)
    return catalog


def derive_intervention_eligibility(assessments: pd.DataFrame, catalog_source: str | Path | Mapping) -> pd.DataFrame:
    """Map assessed need to candidate pathways without claiming effectiveness."""

    catalog = load_intervention_catalog(catalog_source)
    rows = []
    for assessment in assessments.itertuples(index=False):
        level = assessment.dm77_need_level
        missing = pd.isna(level)
        for intervention in catalog["interventions"]:
            protected_only = bool(intervention["protected_only"])
            eligible = False if missing else int(level) in set(map(int, intervention["eligible_levels"]))
            if protected_only:
                eligible = eligible and bool(assessment.dm77_protected_pathway)
            elif not missing and int(level) == 6:
                eligible = False
            reason = (
                "manual_multidimensional_review_required" if missing
                else "need_level_in_catalog_scope" if eligible
                else "need_level_outside_catalog_scope"
            )
            rows.append({
                "patient_id": str(assessment.patient_id),
                "dm77_need_level": pd.NA if missing else int(level),
                "intervention_id": str(intervention["intervention_id"]),
                "intervention_label": str(intervention["label"]),
                "catalog_eligible": bool(eligible),
                "requires_professional_review": bool(missing or intervention.get("requires_professional_review", True)),
                "protected_pathway": bool(protected_only),
                "eligibility_reason": reason,
                "catalog_version": str(catalog["catalog_version"]),
                "causal_effectiveness_established": False,
            })
    result = pd.DataFrame(rows)
    result["dm77_need_level"] = result["dm77_need_level"].astype("Int64")
    return result
