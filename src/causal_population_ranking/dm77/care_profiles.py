"""Care-profile contracts, component actions, and catalogue validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROMAN_TO_LEVEL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
CARE_PROFILE = "care_profile"


@dataclass(frozen=True)
class CareAction:
    """Concrete component used inside a versioned care-profile bundle."""

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
            action
            for action in self.actions
            if not action.protected and not action.mandatory
        )

    @property
    def action_by_id(self) -> dict[str, CareAction]:
        return {action.action_id: action for action in self.actions}


@dataclass(frozen=True)
class CareProfile:
    care_profile_id: str
    care_profile_index: int
    care_profile_level: int
    care_profile_name: str
    resulting_care_state: str
    included_care_actions: tuple[str, ...]
    action_application_order: tuple[str, ...]
    admissible_current_profile_ids: tuple[str, ...]
    eligible_baseline_need_levels: tuple[int, ...]
    outcome_name: str
    outcome_horizon: int
    outcome_unit: str
    benefit_direction: str
    ranking_domain: str
    resource_cost: float
    capacity_pool: str
    capacity_requirements: dict[str, float]
    clinical_prerequisites: tuple[str, ...]
    mandatory: bool
    protected: bool
    maintenance_reference_only: bool
    automatic_rank_candidate: bool

    @property
    def outcome_contract(self) -> tuple[str, int, str, str, str]:
        return (
            self.outcome_name,
            self.outcome_horizon,
            self.outcome_unit,
            self.benefit_direction,
            self.ranking_domain,
        )


@dataclass(frozen=True)
class CareProfileCatalog:
    catalog_version: str
    catalog_status: str
    profile_composition_semantics: str
    resource_cost_semantics: str
    patient_cost_semantics: str
    primary_ranking_domain: str
    action_catalog: CareActionCatalog
    profiles: tuple[CareProfile, ...]

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(profile.care_profile_id for profile in self.profiles)

    @property
    def profile_by_id(self) -> dict[str, CareProfile]:
        return {profile.care_profile_id: profile for profile in self.profiles}

    @property
    def automatic_rank_profiles(self) -> tuple[CareProfile, ...]:
        return tuple(
            profile for profile in self.profiles if profile.automatic_rank_candidate
        )

    @property
    def maintenance_profile(self) -> CareProfile:
        values = tuple(
            profile for profile in self.profiles if profile.maintenance_reference_only
        )
        if len(values) != 1:
            raise ValueError("Care-profile catalogue requires one maintenance reference")
        return values[0]

    @property
    def profiles_by_state(self) -> dict[str, tuple[CareProfile, ...]]:
        return {
            state: tuple(
                profile
                for profile in self.profiles
                if profile.resulting_care_state == state
            )
            for state in self.action_catalog.care_states
        }


@dataclass(frozen=True)
class ProfileBundleApplication:
    source_profile_id: str
    target_profile_id: str
    feasible: bool
    missing_action_ids: tuple[str, ...]
    deactivated_action_ids: tuple[str, ...]
    incremental_cost: float
    incremental_capacity_requirements: dict[str, float]
    resulting_care_state: str
    reason_codes: tuple[str, ...]
    clinical_rule_reasons: tuple[str, ...] = ()
    contraindication_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionUnitResolution:
    decision_unit: str
    resolution_source: str
    metadata: dict[str, Any]


ACTION_REQUIRED_FIELDS = {
    "action_id",
    "from_states",
    "to_state",
    "eligible_dm77_levels",
    "clinical_rules",
    "contraindications",
    "prerequisites",
    "mutually_exclusive_with",
    "capacity_pool",
    "cost",
    "mandatory",
    "protected",
    "outcome_name",
    "followup_horizon",
    "outcome_direction",
    "outcome_unit",
}


def load_care_action_catalog(source: str | Path | Mapping) -> CareActionCatalog:
    """Load the component actions referenced by care-profile bundles."""

    if isinstance(source, Mapping):
        document = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
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
            level = value if isinstance(value, int) else ROMAN_TO_LEVEL.get(
                str(value).upper(), -1
            )
            if level not in range(1, 7):
                raise ValueError(
                    f"Invalid DM 77 level {value!r} in {item['action_id']}"
                )
            levels.append(level)
        from_states = tuple(map(str, item["from_states"]))
        to_state = str(item["to_state"])
        if (
            not from_states
            or not set(from_states).issubset(care_states)
            or to_state not in care_states
        ):
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
            mutually_exclusive_with=tuple(
                map(str, item["mutually_exclusive_with"] or ())
            ),
            capacity_pool=str(item["capacity_pool"]),
            cost=float(item["cost"]),
            mandatory=bool(item["mandatory"]),
            protected=bool(item["protected"]),
            outcome_name=str(item["outcome_name"]),
            followup_horizon=int(item["followup_horizon"]),
            outcome_direction=str(item["outcome_direction"]),
            outcome_unit=str(item["outcome_unit"]),
            synthetic_effect_mean_days=float(
                item.get("synthetic_effect_mean_days", 0.0)
            ),
            synthetic_effect_scale_days=float(
                item.get("synthetic_effect_scale_days", 1.0)
            ),
        )
        if action.cost < 0 or action.followup_horizon < 1 or not action.capacity_pool:
            raise ValueError(
                f"Invalid cost/capacity/outcome settings for {action.action_id}"
            )
        if action.outcome_direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"Unknown outcome direction for {action.action_id}")
        actions.append(action)

    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("Care-action IDs must be unique")
    known_actions = set(action_ids)
    known_services = {
        service for values in active_services.values() for service in values
    }
    for action in actions:
        unknown_exclusions = set(action.mutually_exclusive_with).difference(
            known_actions
        )
        unknown_prerequisites = set(action.prerequisites).difference(
            known_services | known_actions
        )
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
        catalog_version=str(
            document.get(
                "component_action_catalog_version",
                document.get("catalog_version", "dm77_care_actions_v1"),
            )
        ),
        care_states=tuple(care_states),
        active_services_by_state=active_services,
        actions=tuple(actions),
    )


def are_actions_comparable(left: CareAction, right: CareAction) -> bool:
    return (
        left.outcome_name == right.outcome_name
        and left.followup_horizon == right.followup_horizon
        and left.outcome_direction == right.outcome_direction
        and left.outcome_unit == right.outcome_unit
    )


def validate_ranked_action_comparability(catalog: CareActionCatalog) -> dict:
    incompatible = []
    actions = catalog.ranked_actions
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1 :]:
            if not are_actions_comparable(left, right):
                incompatible.append((left.action_id, right.action_id))
    if incompatible:
        raise ValueError(
            "Cross-action ranking requires common outcome semantics; "
            f"incompatible pairs: {incompatible[:10]}"
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


def resolve_decision_unit(config: Mapping) -> DecisionUnitResolution:
    """Require the explicit care-profile contract without compatibility inference."""

    if config.get("decision_unit") != CARE_PROFILE:
        raise ValueError("An explicit decision_unit: care_profile is required")
    care = config.get("care", {})
    if not isinstance(care, Mapping) or not care.get("profile_catalog"):
        raise ValueError("care_profile decision unit requires care.profile_catalog")
    if str(config.get("opportunity_source", "care_profile_catalog")) != (
        "care_profile_catalog"
    ):
        raise ValueError(
            "care_profile decision unit requires opportunity_source: care_profile_catalog"
        )
    return DecisionUnitResolution(
        decision_unit=CARE_PROFILE,
        resolution_source="explicit_configuration",
        metadata={"implicit_action_to_profile_conversion": False},
    )


PROFILE_REQUIRED_FIELDS = {
    "care_profile_id", "care_profile_level", "care_profile_name",
    "resulting_care_state", "included_care_actions", "action_application_order",
    "admissible_current_profile_ids", "eligible_baseline_need_levels",
    "outcome_name", "outcome_horizon", "outcome_unit", "benefit_direction",
    "ranking_domain", "resource_cost", "capacity_pool", "capacity_requirements",
    "clinical_prerequisites", "mandatory", "protected",
    "maintenance_reference_only", "automatic_rank_candidate",
}


def _load_document(source: str | Path | Mapping) -> tuple[dict, Path | None]:
    if isinstance(source, Mapping):
        return dict(source), None
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Care-profile catalogue does not exist: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}, path


def _resolve_action_catalog_source(
    document: Mapping,
    profile_path: Path | None,
    override: str | Path | Mapping | CareActionCatalog | None,
):
    if override is not None:
        return override
    if document.get("care_states") and document.get("actions"):
        return document
    declared = document.get("action_catalog")
    if not declared:
        raise ValueError("Care-profile catalogue must declare its component action catalogue")
    declared_path = Path(str(declared))
    if declared_path.is_file():
        return declared_path
    if profile_path is not None:
        relative = profile_path.parent / declared_path
        if relative.is_file():
            return relative
    return declared_path


def _validate_bundle_path(
    profile: CareProfile,
    source: CareProfile,
    action_catalog: CareActionCatalog,
) -> None:
    state = source.resulting_care_state
    active = set(action_catalog.active_services_by_state[state])
    actions = action_catalog.action_by_id
    for action_id in profile.action_application_order:
        action = actions[action_id]
        target_services = set(action_catalog.active_services_by_state[action.to_state])
        component_explicitly_active = action_id in source.included_care_actions
        upstream_service_already_active = (
            target_services < active and action.to_state != source.resulting_care_state
        )
        if component_explicitly_active or upstream_service_already_active:
            continue
        if state not in action.from_states:
            compatible_bases = [
                candidate_state for candidate_state in action.from_states
                if set(action_catalog.active_services_by_state[candidate_state]).issubset(active)
            ]
            if not compatible_bases:
                raise ValueError(
                    f"Profile bundle {profile.care_profile_id} cannot apply {action_id} "
                    f"from {source.care_profile_id}/{state}"
                )
            state = max(
                compatible_bases,
                key=lambda value: len(action_catalog.active_services_by_state[value]),
            )
        missing_prerequisites = set(action.prerequisites).difference(active)
        if missing_prerequisites:
            raise ValueError(
                f"Profile bundle {profile.care_profile_id} has unmet prerequisites "
                f"from {source.care_profile_id}: {sorted(missing_prerequisites)}"
            )
        state = action.to_state
        active = target_services
    if state != profile.resulting_care_state:
        raise ValueError(
            f"Profile bundle {profile.care_profile_id} ends at {state}, expected "
            f"{profile.resulting_care_state}, from {source.care_profile_id}"
        )


def load_care_profile_catalog(
    source: str | Path | Mapping,
    action_catalog_source: str | Path | Mapping | CareActionCatalog | None = None,
) -> CareProfileCatalog:
    """Load profiles without ever treating a component action as a profile alias."""

    document, profile_path = _load_document(source)
    items = document.get("profiles")
    if not isinstance(items, list) or not items:
        raise ValueError("Care-profile catalogue requires a non-empty profiles list")
    action_source = _resolve_action_catalog_source(
        document, profile_path, action_catalog_source
    )
    action_catalog = (
        action_source
        if isinstance(action_source, CareActionCatalog)
        else load_care_action_catalog(action_source)
    )
    actions = action_catalog.action_by_id
    known_services = {
        service
        for services in action_catalog.active_services_by_state.values()
        for service in services
    }
    known_clinical_inputs = {
        rule.removeprefix("min_").removeprefix("max_").removeprefix("requires_")
        for action in action_catalog.actions
        for rule in action.clinical_rules
    } | {
        field for action in action_catalog.actions for field in action.contraindications
    }
    profiles: list[CareProfile] = []
    for index, item in enumerate(items):
        missing = sorted(PROFILE_REQUIRED_FIELDS.difference(item))
        if missing:
            raise ValueError(f"Care profile is missing required fields: {missing}")
        profile_id = str(item["care_profile_id"])
        included = tuple(map(str, item["included_care_actions"] or ()))
        order = tuple(map(str, item["action_application_order"] or ()))
        if included != order:
            raise ValueError(
                f"Profile {profile_id} must declare one unambiguous component order"
            )
        if len(included) != len(set(included)):
            raise ValueError(f"Profile {profile_id} repeats a component action")
        unknown_actions = sorted(set(included).difference(actions))
        if unknown_actions:
            raise ValueError(
                f"Profile {profile_id} references unknown actions: {unknown_actions}"
            )
        level = int(item["care_profile_level"])
        eligible_levels = tuple(map(int, item["eligible_baseline_need_levels"] or ()))
        if level not in range(1, 7) or not eligible_levels or not set(eligible_levels).issubset(range(1, 7)):
            raise ValueError(f"Profile {profile_id} has invalid baseline/profile levels")
        requirements = {
            str(pool): float(value)
            for pool, value in dict(item["capacity_requirements"] or {}).items()
        }
        profile = CareProfile(
            care_profile_id=profile_id,
            care_profile_index=index,
            care_profile_level=level,
            care_profile_name=str(item["care_profile_name"]),
            resulting_care_state=str(item["resulting_care_state"]),
            included_care_actions=included,
            action_application_order=order,
            admissible_current_profile_ids=tuple(
                map(str, item["admissible_current_profile_ids"] or ())
            ),
            eligible_baseline_need_levels=eligible_levels,
            outcome_name=str(item["outcome_name"]),
            outcome_horizon=int(item["outcome_horizon"]),
            outcome_unit=str(item["outcome_unit"]),
            benefit_direction=str(item["benefit_direction"]),
            ranking_domain=str(item["ranking_domain"]),
            resource_cost=float(item["resource_cost"]),
            capacity_pool=str(item["capacity_pool"]),
            capacity_requirements=requirements,
            clinical_prerequisites=tuple(map(str, item["clinical_prerequisites"] or ())),
            mandatory=bool(item["mandatory"]),
            protected=bool(item["protected"]),
            maintenance_reference_only=bool(item["maintenance_reference_only"]),
            automatic_rank_candidate=bool(item["automatic_rank_candidate"]),
        )
        if not profile_id or profile_id in actions:
            raise ValueError("Profile IDs must be explicit and distinct from action IDs")
        if profile.resulting_care_state not in action_catalog.care_states:
            raise ValueError(f"Profile {profile_id} has an unknown resulting care state")
        if profile.outcome_horizon < 1 or profile.resource_cost < 0:
            raise ValueError(f"Profile {profile_id} has invalid outcome/cost values")
        unknown_prerequisites = set(profile.clinical_prerequisites).difference(
            known_services | known_clinical_inputs
        )
        if unknown_prerequisites:
            raise ValueError(
                f"Profile {profile_id} has unknown prerequisites: "
                f"{sorted(unknown_prerequisites)}"
            )
        if profile.benefit_direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"Profile {profile_id} has invalid benefit direction")
        if any(value <= 0 for value in requirements.values()):
            raise ValueError(f"Profile {profile_id} has non-positive capacity requirements")
        if profile.maintenance_reference_only and included:
            raise ValueError("Maintenance reference cannot contain activation actions")
        if not profile.maintenance_reference_only and not included:
            raise ValueError(f"Target profile {profile_id} requires explicit component actions")
        if profile.automatic_rank_candidate and (
            profile.mandatory or profile.protected or profile.maintenance_reference_only
        ):
            raise ValueError(f"Profile {profile_id} cannot be an automatic rank candidate")
        profiles.append(profile)

    profile_ids = [profile.care_profile_id for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("Care-profile IDs must be unique")
    known_profiles = set(profile_ids)
    maintenance = [profile for profile in profiles if profile.maintenance_reference_only]
    if len(maintenance) != 1:
        raise ValueError("Care-profile catalogue requires exactly one maintenance reference")
    for profile in profiles:
        unknown_sources = set(profile.admissible_current_profile_ids).difference(known_profiles)
        if unknown_sources or profile.care_profile_id in profile.admissible_current_profile_ids:
            raise ValueError(
                f"Profile {profile.care_profile_id} has invalid source profiles: "
                f"{sorted(unknown_sources)}"
            )
        expected_cost = sum(actions[action_id].cost for action_id in profile.included_care_actions)
        if abs(profile.resource_cost - expected_cost) > 1e-9:
            raise ValueError(
                f"Profile {profile.care_profile_id} resource cost does not equal its components"
            )
        expected_requirements = Counter(
            actions[action_id].capacity_pool for action_id in profile.included_care_actions
        )
        expected_requirements.pop("none", None)
        if requirements := profile.capacity_requirements:
            normalized = {key: float(value) for key, value in expected_requirements.items()}
            if requirements != normalized:
                raise ValueError(
                    f"Profile {profile.care_profile_id} capacity requirements do not match components"
                )
        elif expected_requirements:
            raise ValueError(
                f"Profile {profile.care_profile_id} is missing capacity requirements"
            )
        if expected_requirements and profile.capacity_pool not in expected_requirements:
            raise ValueError(f"Profile {profile.care_profile_id} has an invalid primary capacity pool")
        if not expected_requirements and profile.capacity_pool != "none":
            raise ValueError(f"Profile {profile.care_profile_id} requires capacity_pool none")
        for action_id in profile.included_care_actions:
            action = actions[action_id]
            if action.protected != profile.protected or action.mandatory != profile.mandatory:
                raise ValueError(
                    f"Profile {profile.care_profile_id} protection/mandatory semantics "
                    f"do not match component {action_id}"
                )
            if (
                action.outcome_name != profile.outcome_name
                or action.followup_horizon != profile.outcome_horizon
                or action.outcome_unit != profile.outcome_unit
                or action.outcome_direction != profile.benefit_direction
            ):
                raise ValueError(
                    f"Profile {profile.care_profile_id} and action {action_id} have incompatible outcomes"
                )
            conflicts = set(action.mutually_exclusive_with).intersection(
                profile.included_care_actions
            )
            if conflicts:
                raise ValueError(
                    f"Profile {profile.care_profile_id} contains mutually exclusive actions"
                )
        for source_id in profile.admissible_current_profile_ids:
            _validate_bundle_path(profile, profiles[profile_ids.index(source_id)], action_catalog)

    return CareProfileCatalog(
        catalog_version=str(document.get("catalog_version", "care_profiles_v1")),
        catalog_status=str(document.get("catalog_status", "unspecified")),
        profile_composition_semantics=str(document.get("profile_composition_semantics", "")),
        resource_cost_semantics=str(document.get("resource_cost_semantics", "")),
        patient_cost_semantics=str(document.get("patient_cost_semantics", "")),
        primary_ranking_domain=str(document.get("primary_ranking_domain", "")),
        action_catalog=action_catalog,
        profiles=tuple(profiles),
    )


def are_profiles_comparable(left: CareProfile, right: CareProfile) -> bool:
    return left.outcome_contract == right.outcome_contract


def validate_ranked_profile_comparability(catalog: CareProfileCatalog) -> dict:
    domains: dict[str, list[CareProfile]] = {}
    for profile in catalog.profiles:
        domains.setdefault(profile.ranking_domain, []).append(profile)
    incompatible_within_domain = []
    for domain, profiles in domains.items():
        for left_index, left in enumerate(profiles):
            for right in profiles[left_index + 1:]:
                if not are_profiles_comparable(left, right):
                    incompatible_within_domain.append(
                        (domain, left.care_profile_id, right.care_profile_id)
                    )
    if incompatible_within_domain:
        raise ValueError(
            "Profiles in one ranking domain have incompatible outcomes: "
            f"{incompatible_within_domain[:10]}"
        )
    automatic = catalog.automatic_rank_profiles
    if any(profile.ranking_domain != catalog.primary_ranking_domain for profile in automatic):
        raise ValueError("Automatic profile candidates must remain in the primary ranking domain")
    return {
        "catalog_version": catalog.catalog_version,
        "ranking_domains": {
            domain: {
                "profile_ids": [profile.care_profile_id for profile in profiles],
                "outcome_contract": list(profiles[0].outcome_contract),
            }
            for domain, profiles in sorted(domains.items())
        },
        "automatic_rank_profiles": len(automatic),
        "all_automatic_profiles_share_primary_domain": True,
        "cross_domain_pairs_allowed": False,
    }


def require_comparable_profile_pair(
    catalog: CareProfileCatalog,
    left_profile_id: str,
    right_profile_id: str,
) -> None:
    profiles = catalog.profile_by_id
    if left_profile_id not in profiles or right_profile_id not in profiles:
        raise ValueError("Profile pair references an unknown care_profile_id")
    if not are_profiles_comparable(profiles[left_profile_id], profiles[right_profile_id]):
        raise ValueError(
            "Cross-profile pair blocked by incompatible outcome/ranking domains: "
            f"{left_profile_id}, {right_profile_id}"
        )
