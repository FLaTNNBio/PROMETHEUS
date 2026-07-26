"""DM 77-aligned need and governed care-profile contracts."""

from .care_profiles import (
    CARE_PROFILE,
    CareProfile,
    CareProfileCatalog,
    DecisionUnitResolution,
    ProfileBundleApplication,
    are_profiles_comparable,
    load_care_profile_catalog,
    require_comparable_profile_pair,
    resolve_decision_unit,
    validate_ranked_profile_comparability,
)
from .need import (
    BaselineNeedPrediction,
    BaselineNeedStratifier,
    CrossFittedBaselineNeedPrediction,
    cross_fit_baseline_need,
    load_baseline_need_contract,
)
from .need_rules import (
    DM77_INPUT_FIELDS,
    DM77_LEVEL_LABELS,
    assess_dm77_population,
    load_dm77_settings,
    summarize_dm77_population,
)
from .opportunities import (
    apply_profile_bundle,
    build_patient_profile_opportunities,
    derive_care_profile_eligibility,
    resolve_current_care_profiles,
)

__all__ = [
    "CARE_PROFILE",
    "BaselineNeedPrediction",
    "BaselineNeedStratifier",
    "CareProfile",
    "CareProfileCatalog",
    "CrossFittedBaselineNeedPrediction",
    "DM77_INPUT_FIELDS",
    "DM77_LEVEL_LABELS",
    "DecisionUnitResolution",
    "ProfileBundleApplication",
    "apply_profile_bundle",
    "are_profiles_comparable",
    "assess_dm77_population",
    "build_patient_profile_opportunities",
    "cross_fit_baseline_need",
    "derive_care_profile_eligibility",
    "load_baseline_need_contract",
    "load_care_profile_catalog",
    "load_dm77_settings",
    "require_comparable_profile_pair",
    "resolve_current_care_profiles",
    "resolve_decision_unit",
    "summarize_dm77_population",
    "validate_ranked_profile_comparability",
]
