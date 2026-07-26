"""Fully synthetic longitudinal population generation and validation."""

from .population import (
    SYNTHETIC_POPULATION_SCENARIOS,
    SyntheticPopulationResult,
    generate_synthetic_population,
)
from .validation import validate_synthetic_population
from .need_reference import (
    NEED_REFERENCE_VERSION,
    SyntheticNeedReferenceResult,
    generate_synthetic_need_reference,
)
from .profile_dgp import (
    DEFAULT_DGP_STRESS_SETTINGS,
    NO_NEW_PROFILE,
    PROFILE_DGP_SCENARIOS,
    PROFILE_DGP_VERSION,
    SyntheticProfileDGPResult,
    generate_exact_current_care_profiles,
    generate_synthetic_profile_dgp,
)
from .profile_validation import validate_synthetic_profile_dgp

__all__ = [
    "SyntheticPopulationResult",
    "SYNTHETIC_POPULATION_SCENARIOS",
    "generate_synthetic_population",
    "validate_synthetic_population",
    "NEED_REFERENCE_VERSION",
    "SyntheticNeedReferenceResult",
    "generate_synthetic_need_reference",
    "NO_NEW_PROFILE",
    "DEFAULT_DGP_STRESS_SETTINGS",
    "PROFILE_DGP_SCENARIOS",
    "PROFILE_DGP_VERSION",
    "SyntheticProfileDGPResult",
    "generate_exact_current_care_profiles",
    "generate_synthetic_profile_dgp",
    "validate_synthetic_profile_dgp",
]
