"""Contrastive direct ordinal ranking for EMS escalation opportunities."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from causal_population_ranking.applications.contrastive_ranker import (
    ContrastiveCausalRanker,
    fit_contrastive_causal_ranker,
)
from .synthetic import LEARNER_FEATURE_COLUMNS

RANKING_FEATURE_COLUMNS = (*LEARNER_FEATURE_COLUMNS,)
EMSDirectRanker = ContrastiveCausalRanker


def fit_ems_direct_ranker(
    opportunities: pd.DataFrame,
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    config: Mapping[str, Any],
) -> EMSDirectRanker:
    signal_columns = tuple(
        c for c in opportunities.columns if c.startswith("_dr_repeat_")
    )
    if not signal_columns:
        raise ValueError("EMS contrastive ranker requires repeated DR signals")
    return fit_contrastive_causal_ranker(
        opportunities,
        train_pairs,
        validation_pairs,
        feature_columns=RANKING_FEATURE_COLUMNS,
        treatment_column="opportunity_type",
        signal_columns=signal_columns,
        unit_id_column="mission_id",
        config=config,
    )
