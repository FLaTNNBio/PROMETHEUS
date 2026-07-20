"""Transparent DM 77 need stratification, kept separate from causal ranking."""

from .catalog import derive_intervention_eligibility, load_intervention_catalog
from .action_catalog import (
    are_actions_comparable,
    load_care_action_catalog,
    validate_ranked_action_comparability,
)
from .action_eligibility import (
    build_patient_action_opportunities,
    derive_dm77_action_eligibility,
)
from .action_models import (
    AllocationDecision,
    CareAction,
    CurrentCareState,
    DM77NeedAssessment,
    PatientActionOpportunity,
)
from .current_care import attach_current_care_state_features, derive_current_care_states
from .schema import DM77_INPUT_FIELDS, DM77_LEVEL_LABELS
from .stratification import (
    assess_dm77_population,
    load_dm77_settings,
    summarize_dm77_population,
)

__all__ = [
    "DM77_INPUT_FIELDS",
    "DM77_LEVEL_LABELS",
    "assess_dm77_population",
    "derive_intervention_eligibility",
    "load_dm77_settings",
    "load_intervention_catalog",
    "summarize_dm77_population",
    "AllocationDecision",
    "CareAction",
    "CurrentCareState",
    "DM77NeedAssessment",
    "PatientActionOpportunity",
    "are_actions_comparable",
    "attach_current_care_state_features",
    "build_patient_action_opportunities",
    "derive_current_care_states",
    "derive_dm77_action_eligibility",
    "load_care_action_catalog",
    "validate_ranked_action_comparability",
]
