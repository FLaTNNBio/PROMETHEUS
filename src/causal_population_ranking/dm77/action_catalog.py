"""Loading and validation for the computable DM 77 care-action catalogue."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from .action_models import CareAction, CareActionCatalog, ROMAN_TO_LEVEL


ACTION_REQUIRED_FIELDS = {
    "action_id", "from_states", "to_state", "eligible_dm77_levels",
    "clinical_rules", "contraindications", "prerequisites",
    "mutually_exclusive_with", "capacity_pool", "cost", "mandatory",
    "protected", "outcome_name", "followup_horizon", "outcome_direction",
    "outcome_unit",
}


def load_care_action_catalog(source: str | Path | Mapping) -> CareActionCatalog:
    if isinstance(source, Mapping):
        document = dict(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Care-action catalogue does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    state_items = document.get("care_states")
    action_items = document.get("actions")
    if not isinstance(state_items, list) or not state_items:
        raise ValueError("Care-action catalogue requires care_states")
    if not isinstance(action_items, list) or not action_items:
        raise ValueError("Care-action catalogue requires actions")

    care_states: list[str] = []
    active_services: dict[str, tuple[str, ...]] = {}
    for item in state_items:
        if isinstance(item, str):
            state_id, services = item, (item,)
        else:
            state_id = str(item.get("state_id", ""))
            services = tuple(map(str, item.get("active_services", (state_id,))))
        if not state_id or state_id in active_services:
            raise ValueError("Care-state IDs must be non-empty and unique")
        care_states.append(state_id)
        active_services[state_id] = services

    actions: list[CareAction] = []
    for action_index, item in enumerate(action_items):
        missing = sorted(ACTION_REQUIRED_FIELDS.difference(item))
        if missing:
            raise ValueError(f"Care action is missing required fields: {missing}")
        levels = []
        for value in item["eligible_dm77_levels"]:
            if isinstance(value, int):
                level = value
            else:
                level = ROMAN_TO_LEVEL.get(str(value).upper(), -1)
            if level not in range(1, 7):
                raise ValueError(f"Invalid DM 77 level {value!r} in {item['action_id']}")
            levels.append(level)
        from_states = tuple(map(str, item["from_states"]))
        to_state = str(item["to_state"])
        if not from_states or not set(from_states).issubset(care_states) or to_state not in care_states:
            raise ValueError(f"Unknown care state in action {item['action_id']}")
        action = CareAction(
            action_id=str(item["action_id"]),
            action_index=action_index,
            from_states=from_states,
            to_state=to_state,
            eligible_dm77_levels=tuple(levels),
            clinical_rules=dict(item["clinical_rules"] or {}),
            contraindications=tuple(map(str, item["contraindications"] or ())),
            prerequisites=tuple(map(str, item["prerequisites"] or ())),
            mutually_exclusive_with=tuple(map(str, item["mutually_exclusive_with"] or ())),
            capacity_pool=str(item["capacity_pool"]),
            cost=float(item["cost"]),
            mandatory=bool(item["mandatory"]),
            protected=bool(item["protected"]),
            outcome_name=str(item["outcome_name"]),
            followup_horizon=int(item["followup_horizon"]),
            outcome_direction=str(item["outcome_direction"]),
            outcome_unit=str(item["outcome_unit"]),
            synthetic_effect_mean_days=float(item.get("synthetic_effect_mean_days", 0.0)),
            synthetic_effect_scale_days=float(item.get("synthetic_effect_scale_days", 1.0)),
        )
        if action.cost < 0 or action.followup_horizon < 1 or not action.capacity_pool:
            raise ValueError(f"Invalid cost/capacity/outcome settings for {action.action_id}")
        if action.outcome_direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"Unknown outcome direction for {action.action_id}")
        actions.append(action)

    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Care-action IDs must be unique")
    known_actions = set(action_ids)
    known_services = {service for values in active_services.values() for service in values}
    for action in actions:
        unknown_exclusions = set(action.mutually_exclusive_with).difference(known_actions)
        unknown_prerequisites = set(action.prerequisites).difference(known_services | known_actions)
        if unknown_exclusions or unknown_prerequisites:
            raise ValueError(
                f"Unknown exclusions/prerequisites for {action.action_id}: "
                f"{sorted(unknown_exclusions | unknown_prerequisites)}"
            )
        for excluded in action.mutually_exclusive_with:
            reciprocal = next(item for item in actions if item.action_id == excluded)
            if action.action_id not in reciprocal.mutually_exclusive_with:
                raise ValueError(
                    "Mutual exclusions must be symmetric: "
                    f"{action.action_id} -> {excluded}"
                )
    return CareActionCatalog(
        catalog_version=str(document.get("catalog_version", "dm77_care_actions_v1")),
        care_states=tuple(care_states),
        active_services_by_state=active_services,
        actions=tuple(actions),
    )


def are_actions_comparable(action_a: CareAction, action_b: CareAction) -> bool:
    return (
        action_a.outcome_name == action_b.outcome_name
        and action_a.followup_horizon == action_b.followup_horizon
        and action_a.outcome_direction == action_b.outcome_direction
        and action_a.outcome_unit == action_b.outcome_unit
    )


def validate_ranked_action_comparability(catalog: CareActionCatalog) -> dict:
    incompatible = []
    actions = catalog.ranked_actions
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1:]:
            if not are_actions_comparable(left, right):
                incompatible.append((left.action_id, right.action_id))
    if incompatible:
        raise ValueError(
            "Cross-action ranking requires common outcome semantics; incompatible pairs: "
            f"{incompatible[:10]}"
        )
    first = actions[0] if actions else None
    return {
        "ranked_actions": len(actions),
        "all_ranked_actions_comparable": True,
        "outcome_name": None if first is None else first.outcome_name,
        "followup_horizon": None if first is None else first.followup_horizon,
        "outcome_direction": None if first is None else first.outcome_direction,
        "outcome_unit": None if first is None else first.outcome_unit,
    }
