"""Stable public evaluation API for PROMETHEUS."""

from .baseline import BaselineNeedEvaluationResult, evaluate_baseline_need
from .external import (
    ExternalBaselineValidationResult,
    evaluate_external_baseline_validation,
    run_external_validation,
    validate_external_protocol,
    validate_governance_manifest,
)
from .ranking import (
    evaluate_profile_allocation,
    evaluate_profile_ranking_scores,
    evaluate_profile_recommendations,
    linear_calibration_metrics,
    observational_ranking_metrics,
    oracle_ranking_metrics,
    pairwise_concordance,
    profile_concordance_metrics,
    rank_weighted_metrics,
)

__all__ = [
    "BaselineNeedEvaluationResult",
    "ExternalBaselineValidationResult",
    "evaluate_baseline_need",
    "evaluate_external_baseline_validation",
    "evaluate_profile_allocation",
    "evaluate_profile_ranking_scores",
    "evaluate_profile_recommendations",
    "linear_calibration_metrics",
    "observational_ranking_metrics",
    "oracle_ranking_metrics",
    "pairwise_concordance",
    "profile_concordance_metrics",
    "rank_weighted_metrics",
    "run_external_validation",
    "validate_external_protocol",
    "validate_governance_manifest",
]
