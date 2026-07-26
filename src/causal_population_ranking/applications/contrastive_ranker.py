"""Shared treatment-conditioned Siamese causal ranker.

The model is used unchanged in both application domains.  It learns an ordinal
patient/mission--treatment score from doubly robust supervision and optionally
regularizes the latent geometry with a causal-response-guided contrastive loss.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as F


class TreatmentConditionedSiameseNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        treatment_count: int,
        hidden_dim: int = 96,
        treatment_embedding_dim: int = 12,
        projection_dim: int = 24,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.clinical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.treatment_embedding = nn.Embedding(
            treatment_count, treatment_embedding_dim
        )
        joint_dim = hidden_dim + treatment_embedding_dim
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, projection_dim),
        )

    def forward(
        self, x: torch.Tensor, treatment_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clinical = self.clinical_encoder(x)
        treatment = self.treatment_embedding(treatment_index)
        representation = self.joint_encoder(torch.cat([clinical, treatment], dim=1))
        score = self.score_head(representation).squeeze(1)
        projection = F.normalize(self.projection_head(representation), dim=1)
        return score, projection


@dataclass(frozen=True)
class ContrastiveCausalRanker:
    processor: ColumnTransformer
    model: TreatmentConditionedSiameseNetwork
    feature_columns: tuple[str, ...]
    treatment_column: str
    treatment_mapping: dict[str, int]
    audit: dict[str, Any]

    def score(self, opportunities: pd.DataFrame) -> np.ndarray:
        matrix = np.asarray(
            self.processor.transform(opportunities[list(self.feature_columns)]),
            dtype=np.float32,
        )
        treatment = opportunities[self.treatment_column].astype(str).map(
            self.treatment_mapping
        )
        if treatment.isna().any():
            unknown = sorted(
                opportunities.loc[treatment.isna(), self.treatment_column]
                .astype(str)
                .unique()
            )
            raise ValueError(f"Unknown treatments at scoring time: {unknown}")
        self.model.eval()
        with torch.no_grad():
            score, _ = self.model(
                torch.from_numpy(matrix),
                torch.from_numpy(treatment.to_numpy(np.int64)),
            )
        return score.cpu().numpy()

    def representation(self, opportunities: pd.DataFrame) -> np.ndarray:
        matrix = np.asarray(
            self.processor.transform(opportunities[list(self.feature_columns)]),
            dtype=np.float32,
        )
        treatment = opportunities[self.treatment_column].astype(str).map(
            self.treatment_mapping
        )
        self.model.eval()
        with torch.no_grad():
            _, projection = self.model(
                torch.from_numpy(matrix),
                torch.from_numpy(treatment.to_numpy(np.int64)),
            )
        return projection.cpu().numpy()


def _processor(frame: pd.DataFrame, feature_columns: Sequence[str]) -> ColumnTransformer:
    categorical = [
        c for c in feature_columns
        if (not pd.api.types.is_numeric_dtype(frame[c]))
        or pd.api.types.is_bool_dtype(frame[c])
    ]
    numeric = [c for c in feature_columns if c not in categorical]
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("categorical", one_hot, categorical),
         ("numeric", StandardScaler(), numeric)],
        remainder="drop",
    )


def _sample_contrastive_pairs(
    signal: np.ndarray,
    unit_id: np.ndarray,
    treatment: np.ndarray,
    *,
    maximum_pairs: int,
    positive_quantile: float,
    negative_quantile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(signal) < 3:
        raise ValueError("At least three opportunities are required")
    rng = np.random.default_rng(seed)
    candidate_count = max(maximum_pairs * 12, 10_000)
    left = rng.integers(0, len(signal), size=candidate_count)
    right = rng.integers(0, len(signal), size=candidate_count)
    valid = (left != right) & (unit_id[left] != unit_id[right])
    left, right = left[valid], right[valid]
    gap = np.abs(signal[left] - signal[right])
    finite = np.isfinite(gap)
    left, right, gap = left[finite], right[finite], gap[finite]
    if not len(gap):
        raise ValueError("No valid contrastive candidates")
    positive_threshold = float(np.quantile(gap, positive_quantile))
    negative_threshold = float(np.quantile(gap, negative_quantile))
    positive = np.flatnonzero(gap <= positive_threshold)
    negative = np.flatnonzero(gap >= negative_threshold)
    half = maximum_pairs // 2
    positive = rng.choice(positive, min(half, len(positive)), replace=False)
    negative = rng.choice(
        negative, min(maximum_pairs - len(positive), len(negative)), replace=False
    )
    selected = np.concatenate([positive, negative])
    labels = np.concatenate([
        np.ones(len(positive), dtype=np.float32),
        np.zeros(len(negative), dtype=np.float32),
    ])
    order = rng.permutation(len(selected))
    return left[selected][order], right[selected][order], labels[order]


def _pair_loss(
    scores: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(weight * F.softplus(-(scores[high] - scores[low])))


def _contrastive_loss(
    projection: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    similar: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(projection[left] - projection[right], dim=1)
    positive = similar * distance.square()
    negative = (1.0 - similar) * F.relu(margin - distance).square()
    return torch.mean(positive + negative)


def fit_contrastive_causal_ranker(
    opportunities: pd.DataFrame,
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    treatment_column: str,
    signal_columns: Sequence[str],
    unit_id_column: str,
    config: Mapping[str, Any],
    train_split: str = "rank_train",
    validation_split: str = "validation",
) -> ContrastiveCausalRanker:
    """Train the shared rank+contrastive model without oracle information."""
    settings = config["ranking"] if "ranking" in config else config
    seed = int(settings.get("model_seed", 77))
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    train = opportunities.loc[opportunities.split.eq(train_split)].reset_index(drop=True)
    validation = opportunities.loc[
        opportunities.split.eq(validation_split)
    ].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Ranker requires non-empty train and validation opportunities")

    feature_columns = tuple(feature_columns)
    processor = _processor(train, feature_columns)
    train_x = torch.from_numpy(np.asarray(
        processor.fit_transform(train[list(feature_columns)]), dtype=np.float32
    ))
    validation_x = torch.from_numpy(np.asarray(
        processor.transform(validation[list(feature_columns)]), dtype=np.float32
    ))
    treatments = sorted(opportunities[treatment_column].astype(str).unique())
    treatment_mapping = {value: index for index, value in enumerate(treatments)}
    train_t = torch.from_numpy(
        train[treatment_column].astype(str).map(treatment_mapping).to_numpy(np.int64)
    )
    validation_t = torch.from_numpy(
        validation[treatment_column].astype(str).map(treatment_mapping).to_numpy(np.int64)
    )

    model = TreatmentConditionedSiameseNetwork(
        input_dim=int(train_x.shape[1]),
        treatment_count=len(treatment_mapping),
        hidden_dim=int(settings.get("hidden_width", settings.get("hidden_dim", 96))),
        treatment_embedding_dim=int(settings.get("treatment_embedding_dim", 12)),
        projection_dim=int(settings.get("projection_dim", 24)),
        dropout=float(settings.get("dropout", 0.10)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )

    train_high = torch.from_numpy(train_pairs.high_index.to_numpy(np.int64))
    train_low = torch.from_numpy(train_pairs.low_index.to_numpy(np.int64))
    train_weight = torch.from_numpy(train_pairs.weight.to_numpy(np.float32))
    val_high = torch.from_numpy(validation_pairs.high_index.to_numpy(np.int64))
    val_low = torch.from_numpy(validation_pairs.low_index.to_numpy(np.int64))
    val_weight = torch.from_numpy(validation_pairs.weight.to_numpy(np.float32))

    signal = train[list(signal_columns)].mean(axis=1).to_numpy(float)
    con_left, con_right, con_label = _sample_contrastive_pairs(
        signal,
        train[unit_id_column].astype(str).to_numpy(),
        train[treatment_column].astype(str).to_numpy(),
        maximum_pairs=int(settings.get("maximum_contrastive_pairs", 4000)),
        positive_quantile=float(settings.get("contrastive_positive_quantile", 0.20)),
        negative_quantile=float(settings.get("contrastive_negative_quantile", 0.80)),
        seed=seed + 19,
    )
    con_left = torch.from_numpy(con_left.astype(np.int64))
    con_right = torch.from_numpy(con_right.astype(np.int64))
    con_label = torch.from_numpy(con_label.astype(np.float32))

    batch_size = int(settings.get("batch_size", 256))
    con_batch_size = int(settings.get("contrastive_batch_size", batch_size))
    lambda_con = float(settings.get("contrastive_weight", 0.10))
    margin = float(settings.get("contrastive_margin", 1.0))
    warmup = int(settings.get("contrastive_warmup_epochs", 2))
    minimum_active = int(settings.get("minimum_contrastive_epochs", 3))
    earliest_checkpoint_epoch = warmup + minimum_active - 1
    epochs = int(settings.get("epochs", 50))
    patience = int(settings.get("patience", 8))
    generator = torch.Generator().manual_seed(seed + 1)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = -1
    remaining = patience
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        rank_order = torch.randperm(len(train_pairs), generator=generator)
        con_order = torch.randperm(len(con_left), generator=generator)
        steps = max(
            int(np.ceil(len(rank_order) / batch_size)),
            int(np.ceil(len(con_order) / con_batch_size)),
        )
        epoch_rank = 0.0
        epoch_con = 0.0
        for step in range(steps):
            rank_sel = rank_order[
                (step * batch_size) % len(rank_order):
                min((step * batch_size) % len(rank_order) + batch_size, len(rank_order))
            ]
            if len(rank_sel) == 0:
                rank_sel = rank_order[:batch_size]
            con_sel = con_order[
                (step * con_batch_size) % len(con_order):
                min((step * con_batch_size) % len(con_order) + con_batch_size, len(con_order))
            ]
            if len(con_sel) == 0:
                con_sel = con_order[:con_batch_size]
            optimizer.zero_grad()
            scores, projection = model(train_x, train_t)
            rank_loss = _pair_loss(
                scores,
                train_high[rank_sel],
                train_low[rank_sel],
                train_weight[rank_sel],
            )
            con_loss = _contrastive_loss(
                projection,
                con_left[con_sel],
                con_right[con_sel],
                con_label[con_sel],
                margin,
            )
            active_weight = 0.0 if epoch < warmup else lambda_con
            total = rank_loss + active_weight * con_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_rank += float(rank_loss.detach())
            epoch_con += float(con_loss.detach())

        model.eval()
        with torch.no_grad():
            train_scores, train_projection = model(train_x, train_t)
            val_scores, _ = model(validation_x, validation_t)
            train_loss = float(_pair_loss(
                train_scores, train_high, train_low, train_weight
            ))
            val_loss = float(_pair_loss(val_scores, val_high, val_low, val_weight))
            full_con = float(_contrastive_loss(
                train_projection, con_left, con_right, con_label, margin
            ))
        history.append({
            "epoch": epoch,
            "train_pairwise_loss": train_loss,
            "validation_pairwise_loss": val_loss,
            "contrastive_loss": full_con,
            "effective_contrastive_weight": 0.0 if epoch < warmup else lambda_con,
        })
        # The selected checkpoint must have received actual contrastive updates.
        # Otherwise a warm-up checkpoint would silently turn the final model into
        # a rank-only network even when contrastive training was requested.
        if epoch >= earliest_checkpoint_epoch:
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                remaining = patience
            else:
                remaining -= 1
                if remaining <= 0:
                    break

    if best_epoch < earliest_checkpoint_epoch:
        best_epoch = int(history[-1]["epoch"])
        best_loss = float(history[-1]["validation_pairwise_loss"])
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return ContrastiveCausalRanker(
        processor=processor,
        model=model,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        treatment_mapping=treatment_mapping,
        audit={
            "architecture_family": "treatment_conditioned_siamese_contrastive_ranker",
            "primary_objective": "direct_pairwise_causal_ranking",
            "contrastive_role": "causal_response_guided_latent_regularizer",
            "score_semantics": "global_ordinal_priority_not_individual_cate",
            "oracle_inputs_used": False,
            "individual_cate_estimated_then_sorted": False,
            "early_stopping_target": "nonoracle_validation_pairwise_loss",
            "model_seed": seed,
            "encoded_feature_count": int(train_x.shape[1]),
            "treatments": treatment_mapping,
            "train_pairs": int(len(train_pairs)),
            "validation_pairs": int(len(validation_pairs)),
            "contrastive_pairs": int(len(con_left)),
            "contrastive_weight": lambda_con,
            "contrastive_margin": margin,
            "contrastive_warmup_epochs": warmup,
            "minimum_contrastive_epochs": minimum_active,
            "checkpoint_received_contrastive_updates": bool(best_epoch >= earliest_checkpoint_epoch),
            "best_epoch": best_epoch,
            "best_validation_pairwise_loss": best_loss,
            "history": history,
        },
    )
