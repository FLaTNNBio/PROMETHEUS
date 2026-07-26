import numpy as np
import pandas as pd
import torch

from causal_population_ranking.applications.contrastive_ranker import (
    _build_causal_contrastive_index,
    _sample_uact_triplets,
    _weighted_cosine_triplet_loss,
    causal_supervised_contrastive_loss,
    fit_contrastive_causal_ranker,
)


def test_optional_info_nce_rewards_responder_aware_geometry():
    effects = torch.tensor([-1.1, -1.0, 0.0, 0.05, 1.0, 1.1])
    classes = torch.tensor([-1, -1, 0, 0, 1, 1])
    reliability = torch.ones(6)
    good = torch.nn.functional.normalize(
        torch.tensor(
            [
                [-1.0, 0.0],
                [-0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],
            ]
        ),
        dim=1,
    )
    bad = good[[0, 4, 2, 5, 1, 3]]
    good_loss = causal_supervised_contrastive_loss(
        good,
        effects,
        classes,
        reliability,
        positive_radius=0.2,
        temperature=0.1,
    )
    bad_loss = causal_supervised_contrastive_loss(
        bad,
        effects,
        classes,
        reliability,
        positive_radius=0.2,
        temperature=0.1,
    )
    assert float(good_loss) < float(bad_loss)


def _index_fixture():
    frame = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3", "p3"],
            "treatment": ["a", "b", "c"] * 3,
        }
    )
    repeats = np.asarray(
        [
            [1.00, 1.05, 0.95],
            [0.10, 0.12, 0.08],
            [-1.00, -0.95, -1.05],
            [0.90, 0.95, 0.85],
            [0.05, 0.02, 0.08],
            [-0.90, -0.85, -0.95],
            [1.10, 1.08, 1.12],
            [0.00, 0.03, -0.03],
            [-1.10, -1.08, -1.12],
        ]
    )
    index = _build_causal_contrastive_index(
        frame,
        repeats,
        unit_id_column="patient",
        treatment_column="treatment",
        settings={
            "neutral_effect_threshold": 0.2,
            "positive_radius": 0.35,
            "negative_radius": 0.75,
            "uncertainty_filtering": False,
            "hard_negative_effect_gap": 0.75,
        },
        seed=7,
    )
    return frame, repeats, index


def test_uact_index_separates_positive_negative_and_ambiguous_pairs():
    _, _, index = _index_fixture()
    assert len(index.eligible_anchors) > 0
    assert index.audit["positive_pair_count"] > 0
    assert index.audit["clear_negative_pair_count"] > 0
    assert index.negative_radius > index.positive_radius
    assert index.audit["within_unit_hard_negative_anchors"] > 0


def test_uact_sampler_uses_same_patient_negatives_and_reliability():
    _, _, index = _index_fixture()
    anchors = index.eligible_anchors[: min(8, len(index.eligible_anchors))]
    anchor, positive, negative, weight, diagnostic = _sample_uact_triplets(
        index,
        anchors,
        same_patient_negative_probability=1.0,
        uncertainty_weighting=True,
        rng=np.random.default_rng(11),
    )
    assert len(anchor) > 0
    assert len(anchor) == len(positive) == len(negative) == len(weight)
    assert diagnostic["same_patient_negative_count"] > 0
    assert np.all(weight > 0.0)
    assert diagnostic["mean_negative_effect_gap"] > diagnostic["mean_positive_effect_gap"]


def test_weighted_uact_loss_rewards_ordered_geometry():
    good_anchor = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1
    )
    good_positive = torch.nn.functional.normalize(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]), dim=1
    )
    good_negative = torch.nn.functional.normalize(
        torch.tensor([[-1.0, 0.0], [0.0, -1.0]]), dim=1
    )
    bad_positive = good_negative
    bad_negative = good_positive
    weight = torch.ones(2)
    good_loss = _weighted_cosine_triplet_loss(
        good_anchor,
        good_positive,
        good_negative,
        weight,
        margin=0.15,
    )
    bad_loss = _weighted_cosine_triplet_loss(
        good_anchor,
        bad_positive,
        bad_negative,
        weight,
        margin=0.15,
    )
    assert float(good_loss) < float(bad_loss)


def test_ranker_audit_reports_uact_training():
    rows = []
    for split, offset in (("rank_train", 0.0), ("validation", 0.1)):
        for patient in range(12):
            for treatment_index, treatment in enumerate(("a", "b", "c")):
                # Include responder, neutral and harmed opportunities per patient.
                effect = (treatment_index - 1) * 1.0 + patient * 0.02
                rows.append(
                    {
                        "patient": f"{split}_{patient}",
                        "split": split,
                        "x": effect + offset,
                        "treatment": treatment,
                        "_dr_repeat_0": effect,
                        "_dr_repeat_1": effect + 0.03,
                        "_dr_repeat_2": effect - 0.02,
                    }
                )
    opportunities = pd.DataFrame(rows)

    def pairs(count: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high_index": np.arange(1, count, dtype=int),
                "low_index": np.arange(0, count - 1, dtype=int),
                "weight": np.ones(count - 1),
                "pair_type": "global_cross_treatment",
            }
        )

    train_n = int(opportunities.split.eq("rank_train").sum())
    val_n = int(opportunities.split.eq("validation").sum())
    ranker = fit_contrastive_causal_ranker(
        opportunities,
        pairs(train_n),
        pairs(val_n),
        feature_columns=("x",),
        treatment_column="treatment",
        signal_columns=("_dr_repeat_0", "_dr_repeat_1", "_dr_repeat_2"),
        unit_id_column="patient",
        config={
            "model_seed": 19,
            "epochs": 5,
            "patience": 2,
            "minimum_training_epochs": 1,
            "batch_size": 16,
            "hidden_width": 16,
            "treatment_embedding_dim": 4,
            "projection_dim": 4,
            "triplet": {"enabled": False, "weight": 0.0, "maximum_triplets": 0},
            "causal_contrastive": {
                "enabled": True,
                "mode": "uncertainty_aware_ordinal_triplet",
                "weight": 0.0025,
                "margin": 0.15,
                "neutral_effect_threshold": 0.2,
                "positive_radius": 0.25,
                "negative_radius": 0.75,
                "uncertainty_filtering": False,
                "maximum_anchors": 24,
                "anchor_batch_size": 8,
                "batch_size": 24,
                "same_patient_hard_negatives": True,
                "same_patient_negative_probability": 1.0,
                "require_same_patient_negatives": True,
                "warmup_epochs": 0,
                "ramp_epochs": 1,
                "minimum_post_ramp_epochs": 0,
                "strict_diagnostics": True,
            },
        },
    )
    assert ranker.audit["causal_contrastive_enabled"] is True
    assert ranker.audit["contrastive_objective"] == "uncertainty_aware_ordinal_triplet"
    assert ranker.audit["causal_contrastive_index"]["eligible_anchors"] > 0
    assert ranker.audit["causal_contrastive_total_sampled_triplets"] > 0
    assert ranker.audit["causal_contrastive_total_same_patient_negatives"] > 0
    assert ranker.audit["causal_contrastive_max_gradient_norm"] > 0.0
    assert any(
        row["effective_causal_contrastive_weight"] > 0
        for row in ranker.audit["history"]
    )


def _small_pretraining_problem():
    rows = []
    for split, offset in (("rank_train", 0.0), ("validation", 0.05)):
        for patient in range(15):
            for treatment_index, treatment in enumerate(("a", "b", "c")):
                effect = (treatment_index - 1) * 1.1 + patient * 0.015
                rows.append(
                    {
                        "patient": f"{split}_{patient}",
                        "split": split,
                        "x": effect + offset,
                        "treatment": treatment,
                        "_dr_repeat_0": effect,
                        "_dr_repeat_1": effect + 0.02,
                        "_dr_repeat_2": effect - 0.02,
                    }
                )
    opportunities = pd.DataFrame(rows)

    def pairs(count: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "high_index": np.arange(1, count, dtype=int),
                "low_index": np.arange(0, count - 1, dtype=int),
                "weight": np.ones(count - 1),
                "pair_type": "global_cross_treatment",
            }
        )

    train_n = int(opportunities.split.eq("rank_train").sum())
    val_n = int(opportunities.split.eq("validation").sum())
    return opportunities, pairs(train_n), pairs(val_n)


def test_contrastive_pretraining_updates_encoder_and_is_audited():
    opportunities, train_pairs, validation_pairs = _small_pretraining_problem()
    ranker = fit_contrastive_causal_ranker(
        opportunities,
        train_pairs,
        validation_pairs,
        feature_columns=("x",),
        treatment_column="treatment",
        signal_columns=("_dr_repeat_0", "_dr_repeat_1", "_dr_repeat_2"),
        unit_id_column="patient",
        config={
            "model_seed": 41,
            "epochs": 4,
            "patience": 2,
            "minimum_training_epochs": 1,
            "batch_size": 16,
            "hidden_width": 16,
            "treatment_embedding_dim": 4,
            "projection_dim": 4,
            "contrastive_pretraining": {
                "enabled": True,
                "epochs": 4,
                "patience": 2,
                "learning_rate": 0.001,
                "margin": 0.10,
                "neutral_effect_threshold": 0.2,
                "positive_radius": 0.30,
                "negative_radius": 0.75,
                "uncertainty_filtering": False,
                "uncertainty_weighting": True,
                "maximum_anchors": 36,
                "maximum_validation_anchors": 30,
                "maximum_validation_triplets": 30,
                "anchor_batch_size": 12,
                "same_patient_hard_negatives": False,
                "strict_diagnostics": True,
            },
            "ranking_finetuning": {
                "freeze_encoder_epochs": 1,
                "encoder_learning_rate": 0.0001,
                "head_learning_rate": 0.0005,
            },
            "legacy_triplet": {
                "enabled": False,
                "weight": 0.0,
                "maximum_triplets": 0,
            },
            "causal_contrastive": {
                "enabled": False,
                "weight": 0.0,
            },
        },
    )
    audit = ranker.audit
    assert audit["contrastive_pretraining_enabled"] is True
    assert audit["architecture_family"].startswith("contrastively_pretrained")
    pretraining = audit["contrastive_pretraining"]
    assert pretraining["best_epoch"] >= 0
    assert pretraining["total_sampled_triplets"] > 0
    assert pretraining["maximum_gradient_norm"] > 0.0
    assert audit["ranking_finetuning_freeze_encoder_epochs"] == 1


def test_random_initialization_disables_pretraining_audit():
    opportunities, train_pairs, validation_pairs = _small_pretraining_problem()
    ranker = fit_contrastive_causal_ranker(
        opportunities,
        train_pairs,
        validation_pairs,
        feature_columns=("x",),
        treatment_column="treatment",
        signal_columns=("_dr_repeat_0", "_dr_repeat_1", "_dr_repeat_2"),
        unit_id_column="patient",
        config={
            "model_seed": 42,
            "epochs": 3,
            "patience": 2,
            "minimum_training_epochs": 1,
            "batch_size": 16,
            "hidden_width": 16,
            "treatment_embedding_dim": 4,
            "projection_dim": 4,
            "contrastive_pretraining": {"enabled": False},
            "legacy_triplet": {
                "enabled": False,
                "weight": 0.0,
                "maximum_triplets": 0,
            },
            "causal_contrastive": {"enabled": False, "weight": 0.0},
        },
    )
    assert ranker.audit["contrastive_pretraining_enabled"] is False
    assert ranker.audit["contrastive_pretraining"] == {"enabled": False}
