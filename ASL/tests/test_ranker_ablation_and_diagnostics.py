import numpy as np
import pandas as pd

from causal_population_ranking.applications.contrastive_ranker import (
    fit_contrastive_causal_ranker,
)
from causal_population_ranking.ems.policy_benchmark import (
    EMSPolicyInputs,
    evaluate_ems_policy_score_diagnostics,
)


def _opportunities() -> pd.DataFrame:
    rows = []
    for split, offset in (("rank_train", 0), ("validation", 100)):
        for unit in range(10):
            for treatment in ("a", "b"):
                base = float(unit) + (0.5 if treatment == "b" else 0.0)
                rows.append({
                    "unit_id": f"{split}_{unit}",
                    "split": split,
                    "x": base + offset / 1000.0,
                    "treatment": treatment,
                    "_dr_repeat_0": base,
                    "_dr_repeat_1": base + 0.02,
                })
    return pd.DataFrame(rows)


def _ordered_pairs(count: int) -> pd.DataFrame:
    high = np.arange(1, count, dtype=int)
    low = np.arange(0, count - 1, dtype=int)
    return pd.DataFrame({
        "high_index": high,
        "low_index": low,
        "weight": np.ones(len(high), dtype=float),
        "pair_type": "within_treatment",
    })


def test_no_contrastive_ablation_skips_triplets():
    opportunities = _opportunities()
    train_count = int(opportunities.split.eq("rank_train").sum())
    validation_count = int(opportunities.split.eq("validation").sum())
    ranker = fit_contrastive_causal_ranker(
        opportunities,
        _ordered_pairs(train_count),
        _ordered_pairs(validation_count),
        feature_columns=("x",),
        treatment_column="treatment",
        signal_columns=("_dr_repeat_0", "_dr_repeat_1"),
        unit_id_column="unit_id",
        config={
            "model_seed": 7,
            "epochs": 2,
            "patience": 1,
            "batch_size": 16,
            "hidden_width": 16,
            "treatment_embedding_dim": 4,
            "projection_dim": 4,
            "triplet": {
                "enabled": False,
                "weight": 0.0,
                "maximum_triplets": 0,
            },
        },
    )
    assert ranker.audit["triplet_enabled"] is False
    assert ranker.audit["triplets"] == 0
    assert ranker.audit["checkpoint_received_contrastive_updates"] is False


def test_policy_score_diagnostics_separates_tiers_and_global_scope():
    mission_ids = pd.Series(["m1", "m2", "m3", "m4"])
    allocator_values = pd.DataFrame({
        "policy": ["PROMETHEUS"] * 4,
        "mission_id": mission_ids,
        "nurse_value": [0.1, 0.2, 0.3, 0.4],
        "medicalized_value": [0.2, 0.4, 0.6, 0.8],
    })
    inputs = EMSPolicyInputs(
        allocator_values=allocator_values,
        mission_metadata=pd.DataFrame({
            "mission_id": mission_ids,
            "severity_score": [0, 1, 2, 3],
            "simulated_arrival_order": [0, 1, 2, 3],
        }),
        methodology=pd.DataFrame(),
        audit={},
    )
    truth = pd.DataFrame({
        "mission_id": mission_ids,
        "true_increment_nurse_supported": [0.1, 0.2, 0.3, 0.4],
        "true_increment_medicalized": [0.1, 0.2, 0.3, 0.4],
    })
    summary, detail, audit = evaluate_ems_policy_score_diagnostics(inputs, truth)
    assert set(summary.scope) == {
        "nurse_supported",
        "medicalized_total",
        "global_opportunities",
        "within_mission_tier_choice",
    }
    assert len(detail) == 4
    assert audit["oracle_inputs_used_for_policy_construction"] is False
    global_row = summary.loc[summary.scope.eq("global_opportunities")].iloc[0]
    assert global_row.spearman > 0.99


def test_calibration_and_within_unit_weighting_are_configurable():
    opportunities = _opportunities()
    train_count = int(opportunities.split.eq("rank_train").sum())
    validation_count = int(opportunities.split.eq("validation").sum())
    train_pairs = _ordered_pairs(train_count)
    train_pairs.loc[train_pairs.index[:4], "pair_type"] = "within_unit"
    ranker = fit_contrastive_causal_ranker(
        opportunities,
        train_pairs,
        _ordered_pairs(validation_count),
        feature_columns=("x",),
        treatment_column="treatment",
        signal_columns=("_dr_repeat_0", "_dr_repeat_1"),
        unit_id_column="unit_id",
        config={
            "model_seed": 13,
            "epochs": 3,
            "patience": 2,
            "batch_size": 16,
            "hidden_width": 16,
            "treatment_embedding_dim": 4,
            "projection_dim": 4,
            "pair_type_loss_multipliers": {
                "within_unit": 2.0,
                "within_treatment": 1.0,
            },
            "calibration": {
                "enabled": True,
                "weight": 0.1,
                "batch_size": 16,
                "huber_beta": 1.0,
            },
            "triplet": {
                "enabled": False,
                "weight": 0.0,
                "maximum_triplets": 0,
            },
        },
    )
    assert ranker.audit["calibration_enabled"] is True
    assert ranker.audit["calibration_weight"] == 0.1
    assert ranker.audit["pair_type_loss_multipliers"]["within_unit"] == 2.0
    assert "validation_calibration_rmse" in ranker.audit["history"][-1]
