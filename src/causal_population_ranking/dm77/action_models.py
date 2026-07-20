"""Explicit need, care-state, action, opportunity, and decision contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ROMAN_TO_LEVEL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
LEVEL_TO_ROMAN = {value: key for key, value in ROMAN_TO_LEVEL.items()}


@dataclass(frozen=True)
class DM77NeedAssessment:
    patient_id: str
    need_level: int | None
    need_label: str
    reason_codes: tuple[str, ...] = ()
    manual_review: bool = False
    protected_pathway: bool = False


@dataclass(frozen=True)
class CurrentCareState:
    patient_id: str
    state_id: str
    active_services: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CareAction:
    action_id: str
    action_index: int
    from_states: tuple[str, ...]
    to_state: str
    eligible_dm77_levels: tuple[int, ...]
    clinical_rules: dict[str, Any]
    contraindications: tuple[str, ...]
    prerequisites: tuple[str, ...]
    mutually_exclusive_with: tuple[str, ...]
    capacity_pool: str
    cost: float
    mandatory: bool
    protected: bool
    outcome_name: str
    followup_horizon: int
    outcome_direction: str
    outcome_unit: str
    synthetic_effect_mean_days: float = 0.0
    synthetic_effect_scale_days: float = 1.0


@dataclass(frozen=True)
class PatientActionOpportunity:
    patient_id: str
    dm77_need_level: int | None
    current_care_state: str
    action_id: str
    transition_index: int
    eligible: bool
    eligibility_reasons: tuple[str, ...] = ()
    contraindication_reasons: tuple[str, ...] = ()
    manual_review: bool = False
    protected_pathway: bool = False
    cost: float = 0.0
    capacity_pool: str = ""


@dataclass(frozen=True)
class AllocationDecision:
    patient_id: str
    dm77_need_level: int | None
    current_care_state: str
    recommended_action: str | None
    allocated_action: str | None
    calibrated_incremental_benefit: float | None
    cost: float
    capacity_pool: str | None
    allocation_status: str
    resulting_care_state: str
    allocation_reason: str
    candidate_action_id: str | None = None
    causal_priority_score: float | None = None
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CareActionCatalog:
    catalog_version: str
    care_states: tuple[str, ...]
    active_services_by_state: dict[str, tuple[str, ...]]
    actions: tuple[CareAction, ...]

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.action_id for action in self.actions)

    @property
    def ranked_actions(self) -> tuple[CareAction, ...]:
        return tuple(
            action for action in self.actions if not action.protected and not action.mandatory
        )

    @property
    def action_by_id(self) -> dict[str, CareAction]:
        return {action.action_id: action for action in self.actions}
