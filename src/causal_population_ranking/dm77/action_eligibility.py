"""Authoritative DM77-plus-current-state care-action eligibility."""

from __future__ import annotations

import pandas as pd

from .action_models import CareAction, CareActionCatalog


def _clinical_rule_reasons(patient: pd.Series, action: CareAction) -> list[str]:
    reasons = []
    for rule, value in action.clinical_rules.items():
        if rule.startswith("min_"):
            column = rule.removeprefix("min_")
            if column not in patient or float(patient[column]) < float(value):
                reasons.append(f"CLINICAL_RULE_FAILED:{rule}")
        elif rule.startswith("max_"):
            column = rule.removeprefix("max_")
            if column not in patient or float(patient[column]) > float(value):
                reasons.append(f"CLINICAL_RULE_FAILED:{rule}")
        elif rule.startswith("requires_"):
            column = rule.removeprefix("requires_")
            if bool(value) and (column not in patient or not bool(patient[column])):
                reasons.append(f"CLINICAL_RULE_FAILED:{rule}")
        else:
            raise ValueError(f"Unsupported clinical rule {rule!r} in {action.action_id}")
    return reasons


def derive_dm77_action_eligibility(
    patients: pd.DataFrame,
    assessments: pd.DataFrame,
    current_states: pd.DataFrame,
    catalog: CareActionCatalog,
) -> pd.DataFrame:
    """Return one governed eligibility decision per patient and catalogue action."""

    context = (
        patients.assign(patient_id=patients.patient_id.astype(str))
        .merge(
            assessments[[
                "patient_id", "dm77_need_level", "dm77_need_label",
                "dm77_assessment_status", "dm77_reason_codes",
                "dm77_protected_pathway",
            ]].assign(patient_id=lambda value: value.patient_id.astype(str)),
            on="patient_id", validate="one_to_one",
        )
        .merge(
            current_states[["patient_id", "current_care_state", "active_services"]]
            .assign(patient_id=lambda value: value.patient_id.astype(str)),
            on="patient_id", validate="one_to_one",
        )
    )
    rows = []
    for patient in context.to_dict(orient="records"):
        patient_series = pd.Series(patient)
        manual_review = patient["dm77_assessment_status"] == "manual_review"
        protected_pathway = bool(patient["dm77_protected_pathway"])
        level = patient["dm77_need_level"]
        active_services = set(str(patient["active_services"]).split("|"))
        for action in catalog.actions:
            eligibility_reasons = []
            contraindication_reasons = []
            if manual_review:
                eligibility_reasons.append("MANUAL_REVIEW_REQUIRED")
            if protected_pathway and not action.protected:
                eligibility_reasons.append("PROTECTED_PATHWAY_EXCLUDED_FROM_DISCRETIONARY_RANKING")
            if not protected_pathway and action.protected:
                eligibility_reasons.append("PROTECTED_ACTION_NOT_INDICATED")
            if patient["current_care_state"] not in action.from_states:
                eligibility_reasons.append("CURRENT_CARE_STATE_NOT_COMPATIBLE")
            if pd.isna(level) or int(level) not in action.eligible_dm77_levels:
                eligibility_reasons.append("DM77_NEED_LEVEL_NOT_IN_ACTION_SCOPE")
            eligibility_reasons.extend(_clinical_rule_reasons(patient_series, action))
            for contraindication in action.contraindications:
                if contraindication not in patient:
                    contraindication_reasons.append(
                        f"CONTRAINDICATION_INPUT_MISSING:{contraindication}"
                    )
                elif bool(patient[contraindication]):
                    contraindication_reasons.append(
                        f"CONTRAINDICATION_PRESENT:{contraindication}"
                    )
            for prerequisite in action.prerequisites:
                if prerequisite not in active_services:
                    eligibility_reasons.append(f"PREREQUISITE_NOT_ACTIVE:{prerequisite}")
            eligible = not eligibility_reasons and not contraindication_reasons
            rows.append({
                "patient_id": str(patient["patient_id"]),
                "dm77_need_level": pd.NA if pd.isna(level) else int(level),
                "dm77_need_label": str(patient["dm77_need_label"]),
                "current_care_state": str(patient["current_care_state"]),
                "action_id": action.action_id,
                "transition_index": action.action_index,
                "eligible": bool(eligible),
                "eligibility_reasons": "|".join(eligibility_reasons) or "ELIGIBLE_BY_DM77_CATALOG",
                "contraindication_reasons": "|".join(contraindication_reasons),
                "manual_review": bool(manual_review),
                "protected_pathway": bool(protected_pathway),
                "mandatory": action.mandatory,
                "protected_action": action.protected,
                "from_states": "|".join(action.from_states),
                "to_state": action.to_state,
                "prerequisites": "|".join(action.prerequisites),
                "mutually_exclusive_with": "|".join(action.mutually_exclusive_with),
                "capacity_pool": action.capacity_pool,
                "cost": action.cost,
                "outcome_name": action.outcome_name,
                "followup_horizon": action.followup_horizon,
                "outcome_direction": action.outcome_direction,
                "outcome_unit": action.outcome_unit,
                "discretionary_rank_candidate": bool(
                    eligible and not action.mandatory and not action.protected
                ),
            })
    result = pd.DataFrame(rows)
    result["dm77_need_level"] = result.dm77_need_level.astype("Int64")
    return result


def build_patient_action_opportunities(action_eligibility: pd.DataFrame) -> pd.DataFrame:
    """Build candidates only from the authoritative eligibility table."""

    required = {
        "patient_id", "action_id", "transition_index", "eligible",
        "discretionary_rank_candidate", "current_care_state",
    }
    missing = sorted(required.difference(action_eligibility.columns))
    if missing:
        raise ValueError(f"Action eligibility is missing opportunity fields: {missing}")
    candidates = action_eligibility.loc[
        action_eligibility.discretionary_rank_candidate.to_numpy(bool)
    ].copy()
    if candidates[["patient_id", "action_id"]].duplicated().any():
        raise ValueError("Authoritative eligibility contains duplicate patient-action rows")
    return candidates.reset_index(drop=True)
