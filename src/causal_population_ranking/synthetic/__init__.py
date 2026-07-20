"""Fully synthetic longitudinal population generation and validation."""

from .population import (
    SYNTHETIC_POPULATION_SCENARIOS,
    SyntheticPopulationResult,
    generate_synthetic_population,
)
from .validation import validate_synthetic_population

__all__ = [
    "SyntheticPopulationResult",
    "SYNTHETIC_POPULATION_SCENARIOS",
    "generate_synthetic_population",
    "validate_synthetic_population",
]
