"""Current care, governed eligibility, and patient-profile opportunities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .care_profiles import (
    CareAction,
    CareProfile,
    CareProfileCatalog,
    ProfileBundleApplication,
)


def resolve_current_care_profiles(
    current_states: pd.DataFrame,
    catalog: CareProfileCatalog,
    explicit_profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resolve exact pre-index profiles without guessing ambiguous legacy states."""

    required = {"patient_id", "current_care_state"}
    missing = sorted(required.difference(current_states.columns))
    if missing:
        raise ValueError(f"Current-care profile resolution is missing fields: {missing}")
    states = current_states.copy()
    states["patient_id"] = states.patient_id.astype(str)
    if states.patient_id.duplicated().any():
        raise ValueError("Current-care profile resolution requires unique patients")

    explicit_by_patient: dict[str, str] = {}
    if explicit_profiles is not None:
        required_explicit = {"patient_id", "current_care_profile"}
        missing_explicit = sorted(
            required_explicit.difference(explicit_profiles.columns)
        )
        if missing_explicit:
            raise ValueError(f"Explicit current profiles are missing: {missing_explicit}")
        explicit = explicit_profiles.loc[
            :, ["patient_id", "current_care_profile"]
        ].copy()
        explicit["patient_id"] = explicit.patient_id.astype(str)
        if explicit.patient_id.duplicated().any():
            raise ValueError("Explicit current profiles require unique patients")
        unknown_patients = sorted(set(explicit.patient_id).difference(states.patient_id))
        if unknown_patients:
            raise ValueError(
                f"Explicit profiles contain unknown patients: {unknown_patients[:5]}"
            )
        explicit_by_patient = dict(
            zip(explicit.patient_id, explicit.current_care_profile.astype(str))
        )

    profiles = catalog.profile_by_id
    profiles_by_state = catalog.profiles_by_state
    rows = []
    for row in states.to_dict(orient="records"):
        patient_id = str(row["patient_id"])
        state = str(row["current_care_state"])
        candidates = profiles_by_state.get(state, ())
        explicit_id = explicit_by_patient.get(patient_id)
        profile = None
        status = "manual_review"
        if explicit_id is not None:
            if explicit_id not in profiles:
                reason = f"UNKNOWN_EXPLICIT_CURRENT_PROFILE:{explicit_id}"
            elif profiles[explicit_id].resulting_care_state != state:
                reason = "EXPLICIT_PROFILE_STATE_MISMATCH"
            else:
                profile = profiles[explicit_id]
                status = "resolved_explicit_profile"
                reason = "EXPLICIT_PREINDEX_PROFILE_RECORD"
        elif len(candidates) == 1:
            profile = candidates[0]
            status = "resolved_unique_state_mapping"
            reason = "UNIQUE_LEGACY_STATE_TO_PROFILE_MAPPING"
        elif len(candidates) > 1:
            reason = "AMBIGUOUS_LEGACY_STATE_REQUIRES_EXPLICIT_PROFILE:" + "|".join(
                candidate.care_profile_id for candidate in candidates
            )
        else:
            reason = f"UNMATCHED_CURRENT_CARE_STATE:{state}"
        active = catalog.action_catalog.active_services_by_state.get(state, ())
        rows.append(
            {
                "patient_id": patient_id,
                "current_care_profile": (
                    pd.NA if profile is None else profile.care_profile_id
                ),
                "current_care_profile_level": (
                    pd.NA if profile is None else profile.care_profile_level
                ),
                "current_care_state": state,
                "active_services": "|".join(active),
                "current_care_profile_status": status,
                "current_care_profile_reason_codes": reason,
                "current_care_profile_manual_review": profile is None,
                "profile_catalog_version": catalog.catalog_version,
            }
        )
    result = pd.DataFrame(rows)
    result["current_care_profile_level"] = result[
        "current_care_profile_level"
    ].astype("Int64")
    return result


FORBIDDEN_EXACT = {
    "observed_outcome", "outcome", "future_utilization", "raw_priority_score",
    "calibrated_incremental_benefit", "recommended_profile_id",
    "allocated_profile_id", "dr_pseudo_outcome",
}
FORBIDDEN_PREFIXES = (
    "true_", "oracle_", "potential_outcome", "latent_", "postindex_", "future_",
)


def _forbidden_present(*frames: pd.DataFrame) -> list[str]:
    return sorted({
        column
        for frame in frames
        for column in frame.columns
        if column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)
    })


def _bool_or_false(value) -> bool:
    return False if pd.isna(value) else bool(value)


def _clinical_rule_reasons(patient: pd.Series, action: CareAction) -> list[str]:
    reasons = []
    for rule, threshold in action.clinical_rules.items():
        if rule.startswith("min_"):
            field = rule.removeprefix("min_")
            value = pd.to_numeric(pd.Series([patient.get(field, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(value) or float(value) < float(threshold):
                reasons.append(f"CLINICAL_RULE_FAILED:{rule}")
        elif rule.startswith("max_"):
            field = rule.removeprefix("max_")
            value = pd.to_numeric(pd.Series([patient.get(field, np.nan)]), errors="coerce").iloc[0]
            if pd.isna(value) or float(value) > float(threshold):
                reasons.append(f"CLINICAL_RULE_FAILED:{rule}")
        elif rule.startswith("requires_"):
            field = rule.removeprefix("requires_")
            value = patient.get(field, pd.NA)
            if bool(threshold) and (pd.isna(value) or not bool(value)):
                reasons.append(f"CLINICAL_RULE_FAILED:{rule}")
        else:
            raise ValueError(f"Unsupported clinical rule {rule!r} in {action.action_id}")
    return reasons


def apply_profile_bundle(
    patient: pd.Series,
    current_profile_id: str,
    target_profile_id: str,
    catalog: CareProfileCatalog,
) -> ProfileBundleApplication:
    """Evaluate ordered missing components and incremental resources for one patient."""

    profiles = catalog.profile_by_id
    if current_profile_id not in profiles or target_profile_id not in profiles:
        raise ValueError("Profile bundle application requires explicit known profile IDs")
    current = profiles[current_profile_id]
    target = profiles[target_profile_id]
    actions = catalog.action_catalog.action_by_id
    state = current.resulting_care_state
    active = set(catalog.action_catalog.active_services_by_state[state])
    missing_actions = []
    deactivated_actions = []
    cost = 0.0
    capacities: dict[str, float] = {}
    feasibility_reasons = []
    clinical_reasons = []
    contraindication_reasons = []

    for prerequisite in target.clinical_prerequisites:
        if prerequisite in {
            service
            for services in catalog.action_catalog.active_services_by_state.values()
            for service in services
        }:
            if prerequisite not in active:
                feasibility_reasons.append(
                    f"PROFILE_PREREQUISITE_NOT_ACTIVE:{prerequisite}"
                )
        else:
            value = patient.get(prerequisite, pd.NA)
            if pd.isna(value) or not bool(value):
                clinical_reasons.append(
                    f"PROFILE_CLINICAL_PREREQUISITE_FAILED:{prerequisite}"
                )

    for action_id in target.action_application_order:
        action = actions[action_id]
        target_services = set(
            catalog.action_catalog.active_services_by_state[action.to_state]
        )
        component_explicitly_active = action_id in current.included_care_actions
        upstream_service_already_active = (
            target_services < active
            and action.to_state != current.resulting_care_state
        )
        if component_explicitly_active or upstream_service_already_active:
            continue
        missing_actions.append(action_id)
        deactivated_actions.extend(
            current_action
            for current_action in current.included_care_actions
            if current_action in action.mutually_exclusive_with
        )
        if state not in action.from_states:
            compatible_bases = [
                candidate_state for candidate_state in action.from_states
                if set(
                    catalog.action_catalog.active_services_by_state[candidate_state]
                ).issubset(active)
            ]
            if not compatible_bases:
                feasibility_reasons.append(
                    f"ACTION_FROM_STATE_INCOMPATIBLE:{action_id}:{state}"
                )
                continue
            state = max(
                compatible_bases,
                key=lambda value: len(
                    catalog.action_catalog.active_services_by_state[value]
                ),
            )
        for prerequisite in action.prerequisites:
            if prerequisite not in active:
                feasibility_reasons.append(
                    f"ACTION_PREREQUISITE_NOT_ACTIVE:{action_id}:{prerequisite}"
                )
        clinical_reasons.extend(_clinical_rule_reasons(patient, action))
        for contraindication in action.contraindications:
            value = patient.get(contraindication, pd.NA)
            if pd.isna(value):
                contraindication_reasons.append(
                    f"CONTRAINDICATION_INPUT_MISSING:{action_id}:{contraindication}"
                )
            elif bool(value):
                contraindication_reasons.append(
                    f"CONTRAINDICATION_PRESENT:{action_id}:{contraindication}"
                )
        state = action.to_state
        active = target_services
        cost += action.cost
        if action.capacity_pool != "none":
            capacities[action.capacity_pool] = capacities.get(action.capacity_pool, 0.0) + 1.0
    if state != target.resulting_care_state:
        feasibility_reasons.append(
            f"BUNDLE_RESULTING_STATE_MISMATCH:{state}:{target.resulting_care_state}"
        )
    reasons = tuple(dict.fromkeys(feasibility_reasons))
    clinical = tuple(dict.fromkeys(clinical_reasons))
    contraindications = tuple(dict.fromkeys(contraindication_reasons))
    return ProfileBundleApplication(
        source_profile_id=current_profile_id,
        target_profile_id=target_profile_id,
        feasible=not reasons and not clinical and not contraindications,
        missing_action_ids=tuple(missing_actions),
        deactivated_action_ids=tuple(dict.fromkeys(deactivated_actions)),
        incremental_cost=float(cost),
        incremental_capacity_requirements=capacities,
        resulting_care_state=state,
        reason_codes=reasons,
        clinical_rule_reasons=clinical,
        contraindication_reasons=contraindications,
    )


def _need_columns(assessments: pd.DataFrame) -> tuple[str, str, str]:
    if {
        "baseline_need_level", "baseline_need_status", "baseline_need_protected_pathway"
    }.issubset(assessments.columns):
        return (
            "baseline_need_level", "baseline_need_status",
            "baseline_need_protected_pathway",
        )
    raise ValueError("Profile eligibility requires the versioned baseline-need schema")


def derive_care_profile_eligibility(
    patients: pd.DataFrame,
    assessments: pd.DataFrame,
    current_profiles: pd.DataFrame,
    catalog: CareProfileCatalog,
) -> tuple[pd.DataFrame, dict]:
    """Return one pre-index governed decision per patient/profile key."""

    for name, frame in (
        ("patients", patients), ("assessments", assessments),
        ("current_profiles", current_profiles),
    ):
        if "patient_id" not in frame:
            raise ValueError(f"Profile eligibility {name} requires patient_id")
        if frame.patient_id.astype(str).duplicated().any():
            raise ValueError(f"Profile eligibility {name} requires unique patients")
    level_field, status_field, protected_field = _need_columns(assessments)
    required_current = {
        "current_care_profile", "current_care_profile_level",
        "current_care_profile_status", "current_care_profile_manual_review",
    }
    missing_current = sorted(required_current.difference(current_profiles.columns))
    if missing_current:
        raise ValueError(f"Current profiles are missing eligibility fields: {missing_current}")

    patient_safe = patients.copy()
    patient_safe["patient_id"] = patient_safe.patient_id.astype(str)
    need_safe = assessments[[
        "patient_id", level_field, status_field, protected_field,
    ]].copy()
    need_safe["patient_id"] = need_safe.patient_id.astype(str)
    current_safe = current_profiles[[
        "patient_id", "current_care_profile", "current_care_profile_level",
        "current_care_profile_status", "current_care_profile_manual_review",
    ]].copy()
    current_safe["patient_id"] = current_safe.patient_id.astype(str)
    context = patient_safe.merge(
        need_safe, on="patient_id", how="inner", validate="one_to_one"
    ).merge(current_safe, on="patient_id", how="inner", validate="one_to_one")
    if len(context) != len(patients) or len(context) != len(assessments) or len(context) != len(current_profiles):
        raise ValueError("Profile eligibility inputs must contain the same patients")

    rows = []
    for record in context.to_dict(orient="records"):
        patient = pd.Series(record)
        patient_id = str(record["patient_id"])
        level = record[level_field]
        current_id = record["current_care_profile"]
        need_manual = str(record[status_field]) == "manual_review" or pd.isna(level)
        current_manual = bool(record["current_care_profile_manual_review"])
        manual_review = need_manual or current_manual
        protected_pathway = _bool_or_false(record[protected_field])
        urgent_case = _bool_or_false(record.get("urgent_case", False))
        patient_mandatory = _bool_or_false(record.get("mandatory_care", False))
        for profile in catalog.profiles:
            clinical_reasons = []
            ranking_exclusions = []
            bundle = None
            if manual_review:
                clinical_reasons.append("MANUAL_REVIEW_REQUIRED")
            if pd.isna(current_id):
                clinical_reasons.append("CURRENT_CARE_PROFILE_UNRESOLVED")
            elif str(current_id) == profile.care_profile_id:
                clinical_reasons.append("TARGET_PROFILE_ALREADY_ACTIVE")
            elif str(current_id) not in profile.admissible_current_profile_ids:
                clinical_reasons.append("CURRENT_PROFILE_NOT_ADMISSIBLE_FOR_TARGET")
            if pd.isna(level) or int(level) not in profile.eligible_baseline_need_levels:
                clinical_reasons.append("BASELINE_NEED_LEVEL_NOT_IN_PROFILE_SCOPE")
            if protected_pathway and not profile.protected:
                clinical_reasons.append("PROTECTED_PATHWAY_EXCLUDED_FROM_STANDARD_PROFILE")
            if not protected_pathway and profile.protected:
                clinical_reasons.append("PROTECTED_PROFILE_NOT_INDICATED")
            if not clinical_reasons and not pd.isna(current_id):
                bundle = apply_profile_bundle(
                    patient, str(current_id), profile.care_profile_id, catalog
                )
                clinical_reasons.extend(bundle.reason_codes)
                clinical_reasons.extend(bundle.clinical_rule_reasons)
                clinical_reasons.extend(bundle.contraindication_reasons)
            eligibility = not clinical_reasons
            if urgent_case:
                ranking_exclusions.append("URGENT_CASE_BYPASSES_DISCRETIONARY_RANKING")
            if patient_mandatory:
                ranking_exclusions.append("MANDATORY_CARE_BYPASSES_DISCRETIONARY_RANKING")
            if profile.mandatory:
                ranking_exclusions.append("MANDATORY_PROFILE_NOT_DISCRETIONARY")
            if profile.protected:
                ranking_exclusions.append("PROTECTED_PROFILE_NOT_DISCRETIONARY")
            if profile.maintenance_reference_only:
                ranking_exclusions.append("MAINTENANCE_REFERENCE_NOT_TREATMENT_CANDIDATE")
            if not profile.automatic_rank_candidate:
                ranking_exclusions.append("PROFILE_NOT_AUTOMATIC_RANK_CANDIDATE")
            discretionary = bool(
                eligibility and profile.automatic_rank_candidate
                and not urgent_case and not patient_mandatory
            )
            missing_actions = () if bundle is None else bundle.missing_action_ids
            deactivated_actions = () if bundle is None else bundle.deactivated_action_ids
            incremental_cost = np.nan if bundle is None else bundle.incremental_cost
            capacities = {} if bundle is None else bundle.incremental_capacity_requirements
            rows.append({
                "patient_id": patient_id,
                "care_profile_id": profile.care_profile_id,
                "care_profile_index": profile.care_profile_index,
                "care_profile_level": profile.care_profile_level,
                "baseline_need_level": pd.NA if pd.isna(level) else int(level),
                "current_care_profile": current_id,
                "current_care_profile_level": record["current_care_profile_level"],
                "eligibility": bool(eligibility),
                "eligibility_reasons": "|".join(dict.fromkeys(clinical_reasons)) or "ELIGIBLE_BY_PROFILE_CATALOG",
                "ranking_exclusion_reasons": "|".join(dict.fromkeys(ranking_exclusions)),
                "protected_pathway": protected_pathway,
                "manual_review": manual_review,
                "urgent_case": urgent_case,
                "mandatory_care": bool(patient_mandatory or profile.mandatory),
                "protected_profile": profile.protected,
                "automatic_rank_candidate": profile.automatic_rank_candidate,
                "discretionary_rank_candidate": discretionary,
                "common_outcome_contract": profile.ranking_domain == catalog.primary_ranking_domain,
                "outcome_contract_id": "|".join(map(str, profile.outcome_contract)),
                "ranking_domain": profile.ranking_domain,
                "empirical_support": pd.NA,
                "empirical_support_status": "pending_phase4_causal_supervision",
                "missing_component_actions": "|".join(missing_actions),
                "deactivated_component_actions": "|".join(deactivated_actions),
                "incremental_resource_cost": incremental_cost,
                "incremental_capacity_requirements": json.dumps(capacities, sort_keys=True),
                "profile_catalog_version": catalog.catalog_version,
                "component_action_catalog_version": catalog.action_catalog.catalog_version,
            })
    result = pd.DataFrame(rows)
    result["baseline_need_level"] = result.baseline_need_level.astype("Int64")
    result["current_care_profile_level"] = result.current_care_profile_level.astype("Int64")
    if result[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Profile eligibility produced duplicate patient-profile keys")
    audit = {
        "decision_unit": "care_profile",
        "key": ["patient_id", "care_profile_id"],
        "patients": int(len(context)),
        "profiles": int(len(catalog.profiles)),
        "rows": int(len(result)),
        "eligible_rows": int(result.eligibility.sum()),
        "structural_discretionary_candidates": int(
            result.discretionary_rank_candidate.sum()
        ),
        "empirical_support_applied": False,
        "empirical_support_stage": "phase4",
        "forbidden_columns_present_but_ignored": _forbidden_present(
            patients, assessments, current_profiles
        ),
        "oracle_used": False,
        "outcome_used": False,
        "priority_score_used": False,
        "recommendation_used": False,
        "allocation_used": False,
        "implicit_action_to_profile_conversion": False,
        "profile_catalog_version": catalog.catalog_version,
    }
    return result, audit


def build_patient_profile_opportunities(
    profile_eligibility: pd.DataFrame,
    require_empirical_support: bool = True,
) -> pd.DataFrame:
    """Build explicit patient-profile candidates from the authoritative table."""

    required = {
        "patient_id", "care_profile_id", "eligibility",
        "discretionary_rank_candidate", "empirical_support",
        "empirical_support_status", "ranking_domain", "common_outcome_contract",
    }
    missing = sorted(required.difference(profile_eligibility.columns))
    if missing:
        raise ValueError(f"Profile eligibility is missing opportunity fields: {missing}")
    if profile_eligibility[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Authoritative eligibility contains duplicate patient-profile rows")
    candidate = profile_eligibility.discretionary_rank_candidate.fillna(False).to_numpy(bool)
    if require_empirical_support:
        support = profile_eligibility.empirical_support.fillna(False).to_numpy(bool)
        candidate = candidate & support
    opportunities = profile_eligibility.loc[candidate].copy().reset_index(drop=True)
    opportunities["opportunity_status"] = np.where(
        require_empirical_support,
        "eligible_and_empirically_supported",
        "structurally_eligible_support_pending",
    )
    if not opportunities.common_outcome_contract.to_numpy(bool).all():
        raise ValueError("Automatic opportunities contain an incompatible outcome contract")
    return opportunities
