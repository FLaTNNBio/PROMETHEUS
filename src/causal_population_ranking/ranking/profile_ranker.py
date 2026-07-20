"""Direct causal ranking over supported patient-care-profile opportunities."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.nn import functional as F


PROFILE_RANKING_VARIANTS = (
    "independent_profile_rankers",
    "global_rank_only",
    "global_rank_plus_contrastive",
    "direct_pairwise_gbdt",
    "risk_based_ranking",
    "baseline_need_based_ranking",
    "random",
)
ORACLE_EVALUATION_VARIANT = "oracle_evaluation_only"
ORACLE_PREFIXES = ("true_", "oracle_", "potential_outcome", "latent_")
TRAIN_SPLIT = "rank_train"
VALIDATION_SPLIT = "validation"


@dataclass(frozen=True)
class PairBatch:
    left: np.ndarray
    right: np.ndarray
    direction: np.ndarray
    weight: np.ndarray
    is_cross_profile: np.ndarray
    direction_agreement: np.ndarray

    def __len__(self) -> int:
        return len(self.left)


@dataclass(frozen=True)
class SimilarityBatch:
    left: np.ndarray
    right: np.ndarray
    similar: np.ndarray
    weight: np.ndarray
    is_cross_profile: np.ndarray

    def __len__(self) -> int:
        return len(self.left)


@dataclass(frozen=True)
class ProfileRankingResult:
    scores: pd.DataFrame
    training_history: pd.DataFrame
    validation_pairs: pd.DataFrame
    audit: dict[str, Any]
    primary_model_bundle: dict[str, Any] | None


def _empty_pairs() -> PairBatch:
    return PairBatch(
        left=np.array([], dtype=np.int64),
        right=np.array([], dtype=np.int64),
        direction=np.array([], dtype=float),
        weight=np.array([], dtype=float),
        is_cross_profile=np.array([], dtype=bool),
        direction_agreement=np.array([], dtype=float),
    )


def _empty_similarity() -> SimilarityBatch:
    return SimilarityBatch(
        left=np.array([], dtype=np.int64),
        right=np.array([], dtype=np.int64),
        similar=np.array([], dtype=float),
        weight=np.array([], dtype=float),
        is_cross_profile=np.array([], dtype=bool),
    )


def _repeat_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = [
        column
        for column in frame.columns
        if str(column).startswith("dr_pseudo_outcome_repeat_")
    ]
    return tuple(sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1])))


def _sample_balanced_indices(
    within: np.ndarray,
    cross: np.ndarray,
    total: int,
    within_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    within_target = min(len(within), int(round(total * within_fraction)))
    cross_target = min(len(cross), total - within_target)
    selected = []
    if within_target:
        selected.extend(rng.choice(within, within_target, replace=False).tolist())
    if cross_target:
        selected.extend(rng.choice(cross, cross_target, replace=False).tolist())
    missing = total - len(selected)
    if missing:
        available = np.setdiff1d(
            np.concatenate((within, cross)),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False,
        )
        if len(available):
            selected.extend(
                rng.choice(available, min(missing, len(available)), replace=False).tolist()
            )
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def sample_profile_pairs(
    frame: pd.DataFrame,
    *,
    pairs: int,
    seed: int,
    minimum_signal_gap: float,
    minimum_direction_agreement: float,
    within_profile_fraction: float,
    allow_cross_profile: bool,
    allow_same_patient_cross_profile: bool = False,
    maximum_pair_weight: float = 5.0,
) -> PairBatch:
    """Sample stable within/cross-profile pairs from repeated DR directions."""

    required = {
        "patient_id",
        "care_profile_id",
        "dr_pseudo_outcome",
        "dr_reliability_weight",
    }
    repeat_columns = _repeat_columns(frame)
    missing = sorted(required.difference(frame.columns))
    if missing or not repeat_columns:
        raise ValueError(
            f"Ranking pairs require repeated DR supervision: missing={missing}"
        )
    if int(pairs) < 1 or float(minimum_signal_gap) < 0:
        raise ValueError("Pair count must be positive and signal gap non-negative")
    if not 0.0 <= float(minimum_direction_agreement) <= 1.0:
        raise ValueError("minimum_direction_agreement must be in [0, 1]")
    if not 0.0 <= float(within_profile_fraction) <= 1.0:
        raise ValueError("within_profile_fraction must be in [0, 1]")
    if float(maximum_pair_weight) <= 0:
        raise ValueError("maximum_pair_weight must be positive")
    if len(frame) < 2:
        return _empty_pairs()

    signal = frame.dr_pseudo_outcome.to_numpy(float)
    repeated = frame.loc[:, repeat_columns].to_numpy(float)
    reliability = frame.dr_reliability_weight.to_numpy(float)
    profiles = frame.care_profile_id.astype(str).to_numpy()
    patients = frame.patient_id.astype(str).to_numpy()
    if not all(np.isfinite(value).all() for value in (signal, repeated, reliability)):
        raise ValueError("Pair construction requires finite non-oracle supervision")
    left, right = np.triu_indices(len(frame), k=1)
    signal_difference = signal[left] - signal[right]
    comparable = np.abs(signal_difference) > float(minimum_signal_gap)
    same_profile = profiles[left] == profiles[right]
    same_patient = patients[left] == patients[right]
    if not allow_cross_profile:
        comparable &= same_profile
    elif not allow_same_patient_cross_profile:
        comparable &= ~(~same_profile & same_patient)
    repeat_difference = repeated[left] - repeated[right]
    direction = np.sign(signal_difference)
    agreement = np.mean(
        repeat_difference * direction[:, None] > 0.0,
        axis=1,
    )
    comparable &= agreement >= float(minimum_direction_agreement)
    candidates = np.flatnonzero(comparable)
    if not len(candidates):
        return _empty_pairs()
    within = candidates[same_profile[candidates]]
    cross = candidates[~same_profile[candidates]]
    rng = np.random.default_rng(int(seed))
    selected = _sample_balanced_indices(
        within,
        cross,
        min(int(pairs), len(candidates)),
        float(within_profile_fraction),
        rng,
    )
    pair_weight = np.sqrt(reliability[left[selected]] * reliability[right[selected]])
    pair_weight *= agreement[selected]
    pair_weight = np.clip(pair_weight, 1e-8, float(maximum_pair_weight))
    pair_weight /= float(pair_weight.mean())
    return PairBatch(
        left=left[selected].astype(np.int64),
        right=right[selected].astype(np.int64),
        direction=direction[selected].astype(float),
        weight=pair_weight.astype(float),
        is_cross_profile=(~same_profile[selected]).astype(bool),
        direction_agreement=agreement[selected].astype(float),
    )


def _sample_contrastive_pairs(
    frame: pd.DataFrame,
    *,
    pairs: int,
    bins: int,
    seed: int,
    minimum_label_agreement: float,
    allow_same_patient_cross_profile: bool,
) -> SimilarityBatch:
    if int(pairs) < 1 or int(bins) < 3 or len(frame) < 2:
        return _empty_similarity()
    repeat_columns = _repeat_columns(frame)
    signal = frame.dr_pseudo_outcome.to_numpy(float)
    repeated = frame.loc[:, repeat_columns].to_numpy(float)
    patients = frame.patient_id.astype(str).to_numpy()
    profiles = frame.care_profile_id.astype(str).to_numpy()
    boundaries = np.unique(
        np.quantile(signal, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
    )
    if len(boundaries) < 2:
        return _empty_similarity()
    label = np.digitize(signal, boundaries)
    repeated_label = np.stack(
        [np.digitize(repeated[:, repeat], boundaries) for repeat in range(repeated.shape[1])],
        axis=1,
    )
    left, right = np.triu_indices(len(frame), k=1)
    cross = profiles[left] != profiles[right]
    valid = np.ones(len(left), dtype=bool)
    if not allow_same_patient_cross_profile:
        valid &= ~(cross & (patients[left] == patients[right]))
    label_gap = np.abs(label[left] - label[right])
    positive = label_gap == 0
    negative = label_gap >= 2
    positive_agreement = np.mean(
        repeated_label[left] == repeated_label[right], axis=1
    )
    negative_agreement = np.mean(
        np.abs(repeated_label[left] - repeated_label[right]) >= 2, axis=1
    )
    valid_positive = np.flatnonzero(
        valid & positive & (positive_agreement >= minimum_label_agreement)
    )
    valid_negative = np.flatnonzero(
        valid & negative & (negative_agreement >= minimum_label_agreement)
    )
    if not len(valid_positive) and not len(valid_negative):
        return _empty_similarity()
    rng = np.random.default_rng(int(seed))
    half = int(pairs) // 2
    chosen_positive = (
        rng.choice(valid_positive, min(half, len(valid_positive)), replace=False)
        if len(valid_positive)
        else np.array([], dtype=int)
    )
    negative_target = int(pairs) - len(chosen_positive)
    chosen_negative = (
        rng.choice(valid_negative, min(negative_target, len(valid_negative)), replace=False)
        if len(valid_negative)
        else np.array([], dtype=int)
    )
    selected = np.concatenate((chosen_positive, chosen_negative)).astype(np.int64)
    rng.shuffle(selected)
    similar = positive[selected].astype(float)
    agreement = np.where(
        similar.astype(bool), positive_agreement[selected], negative_agreement[selected]
    )
    weight = agreement / max(float(agreement.mean()), 1e-8)
    return SimilarityBatch(
        left=left[selected].astype(np.int64),
        right=right[selected].astype(np.int64),
        similar=similar,
        weight=weight.astype(float),
        is_cross_profile=cross[selected].astype(bool),
    )


def pairwise_ranking_loss(
    score: torch.Tensor,
    pairs: PairBatch,
) -> torch.Tensor:
    """Weighted logistic loss on direct pair directions."""

    if not len(pairs):
        return score.sum() * 0.0
    left = torch.as_tensor(pairs.left, dtype=torch.long, device=score.device)
    right = torch.as_tensor(pairs.right, dtype=torch.long, device=score.device)
    direction = torch.as_tensor(
        pairs.direction, dtype=score.dtype, device=score.device
    )
    weight = torch.as_tensor(pairs.weight, dtype=score.dtype, device=score.device)
    return torch.mean(weight * F.softplus(-direction * (score[left] - score[right])))


def _contrastive_loss(
    representation: torch.Tensor,
    pairs: SimilarityBatch,
    margin: float,
) -> torch.Tensor:
    if not len(pairs):
        return representation.sum() * 0.0
    left = torch.as_tensor(pairs.left, dtype=torch.long, device=representation.device)
    right = torch.as_tensor(pairs.right, dtype=torch.long, device=representation.device)
    similar = torch.as_tensor(
        pairs.similar, dtype=representation.dtype, device=representation.device
    )
    weight = torch.as_tensor(
        pairs.weight, dtype=representation.dtype, device=representation.device
    )
    distance = torch.linalg.vector_norm(
        representation[left] - representation[right], dim=1
    )
    loss = similar * distance.square() + (1.0 - similar) * F.relu(
        float(margin) - distance
    ).square()
    return torch.mean(weight * loss)


class _FeatureEncoder:
    def __init__(self, numeric_features: Sequence[str], categorical_features: Sequence[str]):
        self.numeric_features = tuple(map(str, numeric_features))
        self.categorical_features = tuple(map(str, categorical_features))

    def fit(self, frame: pd.DataFrame) -> "_FeatureEncoder":
        numeric = frame.loc[:, self.numeric_features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        self.median = np.nanmedian(numeric, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(numeric), numeric, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        self.levels = {
            column: tuple(sorted(frame[column].fillna("__missing__").astype(str).unique()))
            for column in self.categorical_features
        }
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.loc[:, self.numeric_features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        numeric = np.where(np.isfinite(numeric), numeric, self.median)
        blocks = [(numeric - self.mean) / self.scale]
        for column in self.categorical_features:
            values = frame[column].fillna("__missing__").astype(str).to_numpy()
            levels = self.levels[column]
            index = {value: position for position, value in enumerate(levels)}
            encoded = np.zeros((len(frame), len(levels) + 1), dtype=float)
            positions = np.asarray([index.get(value, len(levels)) for value in values])
            encoded[np.arange(len(frame)), positions] = 1.0
            blocks.append(encoded)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def payload(self) -> dict[str, Any]:
        return {
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "categorical_levels": {
                key: list(value) for key, value in self.levels.items()
            },
        }


class _ProfileRankNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        profile_count: int,
        hidden_dim: int,
        profile_embedding_dim: int,
        projection_dim: int,
    ):
        super().__init__()
        self.clinical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.profile_embedding = nn.Embedding(profile_count, profile_embedding_dim)
        joint_dim = hidden_dim + profile_embedding_dim
        self.scoring_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def joint_representation(
        self, features: torch.Tensor, profile_index: torch.Tensor
    ) -> torch.Tensor:
        clinical = self.clinical_encoder(features)
        profile = self.profile_embedding(profile_index)
        return torch.cat((clinical, profile), dim=1)

    def forward(
        self, features: torch.Tensor, profile_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint = self.joint_representation(features, profile_index)
        return self.scoring_head(joint).squeeze(1), self.projection_head(joint)


def _gradient_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum().cpu())
    return float(np.sqrt(squared))


def _profile_indices(frame: pd.DataFrame, mapping: Mapping[str, int]) -> np.ndarray:
    unknown = sorted(set(frame.care_profile_id.astype(str)).difference(mapping))
    if unknown:
        raise ValueError(f"Unknown care_profile_id values: {unknown}")
    return frame.care_profile_id.astype(str).map(mapping).to_numpy(np.int64).copy()


def _restart_model_seeds(model_seed: int, count: int) -> tuple[int, ...]:
    if int(count) < 1:
        raise ValueError("model_restarts must be positive")
    if int(count) == 1:
        return (int(model_seed),)
    state = np.random.SeedSequence(
        int(model_seed), spawn_key=(73,)
    ).generate_state(int(count), dtype=np.uint32)
    seeds = tuple(1 + int(value) % 2_000_000_000 for value in state)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Ranker restart-seed derivation produced a collision")
    return seeds


def _mean_percentile_rank(score_columns: Sequence[np.ndarray]) -> np.ndarray:
    if not score_columns:
        raise ValueError("At least one restart score is required")
    lengths = {len(np.asarray(values)) for values in score_columns}
    if len(lengths) != 1:
        raise ValueError("Restart scores must have identical lengths")
    percentile_scores = [
        pd.Series(np.asarray(values, dtype=float)).rank(
            method="average", pct=True
        ).to_numpy(float)
        for values in score_columns
    ]
    result = np.mean(np.column_stack(percentile_scores), axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("Restart aggregation produced non-finite priority scores")
    return result


def _align_scores_to_validation(
    score_frame: pd.DataFrame,
    validation: pd.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    lookup = score_frame[["patient_id", "care_profile_id"]].copy()
    lookup["ensemble_score"] = np.asarray(scores, dtype=float)
    if lookup[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Score opportunities must be unique for ensemble validation")
    aligned = validation[["patient_id", "care_profile_id"]].merge(
        lookup,
        on=["patient_id", "care_profile_id"],
        how="left",
        validate="one_to_one",
    )
    if aligned.ensemble_score.isna().any():
        raise ValueError("Every validation opportunity requires an ensemble score")
    return aligned.ensemble_score.to_numpy(float).copy()


def _fit_network(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    score_frame: pd.DataFrame,
    train_pairs: PairBatch,
    validation_pairs: PairBatch,
    contrastive_pairs: SimilarityBatch,
    profile_mapping: Mapping[str, int],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    settings: Mapping[str, Any],
    *,
    model_seed: int,
    contrastive_weight: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not len(train_pairs) or not len(validation_pairs):
        raise ValueError("Ranker requires non-empty training and fixed validation pairs")
    torch.manual_seed(int(model_seed))
    torch.use_deterministic_algorithms(True)
    encoder = _FeatureEncoder(numeric_features, categorical_features).fit(train)
    train_x = torch.as_tensor(encoder.transform(train), dtype=torch.float32)
    validation_x = torch.as_tensor(encoder.transform(validation), dtype=torch.float32)
    score_x = torch.as_tensor(encoder.transform(score_frame), dtype=torch.float32)
    train_profile = torch.as_tensor(
        _profile_indices(train, profile_mapping), dtype=torch.long
    )
    validation_profile = torch.as_tensor(
        _profile_indices(validation, profile_mapping), dtype=torch.long
    )
    score_profile = torch.as_tensor(
        _profile_indices(score_frame, profile_mapping), dtype=torch.long
    )
    model = _ProfileRankNetwork(
        input_dim=train_x.shape[1],
        profile_count=len(profile_mapping),
        hidden_dim=int(settings["hidden_dim"]),
        profile_embedding_dim=int(settings["profile_embedding_dim"]),
        projection_dim=int(settings["projection_dim"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    patience_used = 0
    history = []
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad()
        train_score, projection = model(train_x, train_profile)
        ranking_loss = pairwise_ranking_loss(train_score, train_pairs)
        contrastive_loss = _contrastive_loss(
            projection, contrastive_pairs, float(settings["contrastive_margin"])
        )
        total_loss = ranking_loss + float(contrastive_weight) * contrastive_loss
        total_loss.backward()
        gradients = {
            "gradient_norm_clinical_encoder": _gradient_norm(model.clinical_encoder),
            "gradient_norm_profile_embedding": _gradient_norm(model.profile_embedding),
            "gradient_norm_scoring_head": _gradient_norm(model.scoring_head),
            "gradient_norm_projection_head": _gradient_norm(model.projection_head),
        }
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_score, _ = model(validation_x, validation_profile)
            validation_loss = float(
                pairwise_ranking_loss(validation_score, validation_pairs).cpu()
            )
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "validation_pairwise_loss": validation_loss,
                **gradients,
            }
        )
        if validation_loss < best_loss - float(settings["minimum_improvement"]):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Ranker did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(score_x, score_profile)[0].cpu().numpy().astype(float)
    history_frame = pd.DataFrame(history)
    audit = {
        "best_epoch": int(best_epoch),
        "best_validation_pairwise_loss": float(best_loss),
        "epochs_completed": int(len(history)),
        "model_seed": int(model_seed),
        "contrastive_weight": float(contrastive_weight),
        "train_pairs": int(len(train_pairs)),
        "train_within_profile_pairs": int((~train_pairs.is_cross_profile).sum()),
        "train_cross_profile_pairs": int(train_pairs.is_cross_profile.sum()),
        "fixed_validation_pairs": int(len(validation_pairs)),
        "validation_within_profile_pairs": int(
            (~validation_pairs.is_cross_profile).sum()
        ),
        "validation_cross_profile_pairs": int(
            validation_pairs.is_cross_profile.sum()
        ),
        "contrastive_pairs": int(len(contrastive_pairs)),
        "projection_head_separate": True,
        "projection_head_last_gradient_norm": float(
            history_frame.iloc[-1].gradient_norm_projection_head
        ),
        "profile_embedding_last_gradient_norm": float(
            history_frame.iloc[-1].gradient_norm_profile_embedding
        ),
        "early_stopping_target": "fixed_validation_dr_pairwise_loss",
        "oracle_used_for_early_stopping": False,
    }
    bundle = {
        "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "feature_encoder": encoder.payload(),
        "profile_mapping": dict(profile_mapping),
        "architecture": {
            "input_dim": int(train_x.shape[1]),
            "hidden_dim": int(settings["hidden_dim"]),
            "profile_embedding_dim": int(settings["profile_embedding_dim"]),
            "projection_dim": int(settings["projection_dim"]),
        },
        "score_semantics": "globally_comparable_ordinal_priority_no_causal_zero",
    }
    return scores, history_frame, audit, bundle


def _pair_frame(
    validation: pd.DataFrame,
    pairs: PairBatch,
    method: str,
) -> pd.DataFrame:
    if not len(pairs):
        return pd.DataFrame()
    left = validation.iloc[pairs.left].reset_index(drop=True)
    right = validation.iloc[pairs.right].reset_index(drop=True)
    return pd.DataFrame(
        {
            "method": method,
            "partition": VALIDATION_SPLIT,
            "left_patient_id": left.patient_id.astype(str),
            "left_care_profile_id": left.care_profile_id.astype(str),
            "right_patient_id": right.patient_id.astype(str),
            "right_care_profile_id": right.care_profile_id.astype(str),
            "direction": pairs.direction,
            "pair_weight": pairs.weight,
            "direction_agreement": pairs.direction_agreement,
            "is_cross_profile": pairs.is_cross_profile,
        }
    )


def _validation_pair_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    payload = frame.sort_values(list(frame.columns)).to_dict(orient="records")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pairwise_gbdt_features(
    features: np.ndarray,
    profile_index: np.ndarray,
    profile_count: int,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_profile = np.eye(profile_count, dtype=float)[profile_index[left]]
    right_profile = np.eye(profile_count, dtype=float)[profile_index[right]]
    return np.concatenate(
        (features[left] - features[right], left_profile, right_profile), axis=1
    )


def _fit_pairwise_gbdt(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    score_frame: pd.DataFrame,
    train_pairs: PairBatch,
    validation_pairs: PairBatch,
    profile_mapping: Mapping[str, int],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    settings: Mapping[str, Any],
    *,
    model_seed: int,
    reference_seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    encoder = _FeatureEncoder(numeric_features, categorical_features).fit(train)
    train_x = encoder.transform(train)
    validation_x = encoder.transform(validation)
    score_x = encoder.transform(score_frame)
    train_profile = _profile_indices(train, profile_mapping)
    validation_profile = _profile_indices(validation, profile_mapping)
    score_profile = _profile_indices(score_frame, profile_mapping)
    pair_x = _pairwise_gbdt_features(
        train_x,
        train_profile,
        len(profile_mapping),
        train_pairs.left,
        train_pairs.right,
    )
    label = (train_pairs.direction > 0).astype(int)
    augmented_x = np.concatenate(
        (
            pair_x,
            _pairwise_gbdt_features(
                train_x,
                train_profile,
                len(profile_mapping),
                train_pairs.right,
                train_pairs.left,
            ),
        )
    )
    augmented_y = np.concatenate((label, 1 - label))
    augmented_weight = np.tile(train_pairs.weight, 2)
    model = HistGradientBoostingClassifier(
        max_iter=int(settings["gbdt_max_iter"]),
        max_depth=int(settings["gbdt_max_depth"]),
        learning_rate=float(settings["gbdt_learning_rate"]),
        random_state=int(model_seed),
    ).fit(augmented_x, augmented_y, sample_weight=augmented_weight)
    rng = np.random.default_rng(int(reference_seed))
    reference_count = min(int(settings["gbdt_reference_rows"]), len(train))
    reference = rng.choice(len(train), reference_count, replace=False)

    def score(values: np.ndarray, profiles: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=float)
        for row in range(len(values)):
            repeated_values = np.repeat(values[row : row + 1], reference_count, axis=0)
            repeated_profiles = np.repeat(profiles[row], reference_count)
            combined_x = np.concatenate((repeated_values, train_x[reference]), axis=0)
            combined_profile = np.concatenate(
                (repeated_profiles, train_profile[reference]), axis=0
            )
            left = np.arange(reference_count)
            right = np.arange(reference_count, 2 * reference_count)
            pair_features = _pairwise_gbdt_features(
                combined_x,
                combined_profile,
                len(profile_mapping),
                left,
                right,
            )
            probability = np.clip(model.predict_proba(pair_features)[:, 1], 1e-6, 1 - 1e-6)
            result[row] = float(np.mean(np.log(probability / (1.0 - probability))))
        return result

    validation_score = torch.as_tensor(score(validation_x, validation_profile))
    validation_loss = float(
        pairwise_ranking_loss(validation_score, validation_pairs).cpu()
    )
    scores = score(score_x, score_profile)
    history = pd.DataFrame(
        [
            {
                "epoch": 0,
                "total_loss": np.nan,
                "ranking_loss": np.nan,
                "contrastive_loss": 0.0,
                "validation_pairwise_loss": validation_loss,
            }
        ]
    )
    audit = {
        "model_seed": int(model_seed),
        "reference_seed": int(reference_seed),
        "training_objective": "direct_pairwise_classification",
        "individual_effect_regression_used": False,
        "train_pairs": int(len(train_pairs)),
        "fixed_validation_pairs": int(len(validation_pairs)),
        "validation_pairwise_loss": validation_loss,
        "reference_rows": int(reference_count),
        "oracle_used": False,
    }
    return scores, history, audit


def _oracle_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in map(str, frame.columns)
        if column.startswith(ORACLE_PREFIXES)
    )


def _prepare_frames(
    learner: pd.DataFrame,
    supervision: pd.DataFrame,
    opportunities: pd.DataFrame,
    patient_splits: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    profile_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaked = {
        "learner": _oracle_columns(learner),
        "supervision": _oracle_columns(supervision),
        "opportunities": _oracle_columns(opportunities),
    }
    if any(leaked.values()):
        raise ValueError(f"Oracle columns cannot enter profile ranking: {leaked}")
    features = tuple(map(str, numeric_features)) + tuple(map(str, categorical_features))
    missing_features = sorted(set(features).difference(learner.columns))
    if missing_features:
        raise ValueError(f"Profile ranker is missing pre-index features: {missing_features}")
    supervision_required = {
        "patient_id",
        "care_profile_id",
        "split",
        "dr_pseudo_outcome",
        "dr_reliability_weight",
        "causal_supervision_status",
    }
    opportunity_required = {
        "patient_id",
        "care_profile_id",
        "care_profile_index",
        "empirical_support",
    }
    if missing := sorted(supervision_required.difference(supervision.columns)):
        raise ValueError(f"Profile ranking supervision is missing: {missing}")
    if missing := sorted(opportunity_required.difference(opportunities.columns)):
        raise ValueError(f"Profile ranking opportunities are missing: {missing}")
    repeat_columns = _repeat_columns(supervision)
    if not repeat_columns:
        raise ValueError("Profile ranker requires repeated DR supervision")
    learner_safe = learner[["patient_id", *features]].copy()
    learner_safe["patient_id"] = learner_safe.patient_id.astype(str)
    if learner_safe.patient_id.duplicated().any():
        raise ValueError("Profile ranker requires one feature row per patient")
    safe_supervision = supervision.loc[
        supervision.causal_supervision_status.eq("supported")
        & supervision.split.isin((TRAIN_SPLIT, VALIDATION_SPLIT)),
        [
            "patient_id",
            "care_profile_id",
            "split",
            "dr_pseudo_outcome",
            "dr_reliability_weight",
            *repeat_columns,
        ],
    ].copy()
    safe_supervision["patient_id"] = safe_supervision.patient_id.astype(str)
    safe_supervision["care_profile_id"] = safe_supervision.care_profile_id.astype(str)
    ranking = safe_supervision.merge(
        learner_safe, on="patient_id", how="inner", validate="many_to_one"
    )
    if len(ranking) != len(safe_supervision):
        raise ValueError("Every ranking-supervision row requires pre-index features")
    rank_train = ranking.loc[ranking.split.eq(TRAIN_SPLIT)].reset_index(drop=True)
    validation = ranking.loc[ranking.split.eq(VALIDATION_SPLIT)].reset_index(drop=True)
    if set(rank_train.patient_id).intersection(validation.patient_id):
        raise ValueError("Rank-training and validation patients must be disjoint")

    splits = patient_splits[["patient_id", "split"]].copy()
    splits["patient_id"] = splits.patient_id.astype(str)
    if splits.patient_id.duplicated().any():
        raise ValueError("Patient split table must contain unique patients")
    supported = opportunities.loc[
        opportunities.empirical_support.astype(bool),
        [
            "patient_id",
            "care_profile_id",
            "care_profile_index",
            "empirical_support",
        ],
    ].copy()
    supported["patient_id"] = supported.patient_id.astype(str)
    supported["care_profile_id"] = supported.care_profile_id.astype(str)
    if supported[["patient_id", "care_profile_id"]].duplicated().any():
        raise ValueError("Supported profile opportunities must be unique")
    score_frame = (
        supported.merge(splits, on="patient_id", how="inner", validate="many_to_one")
        .merge(learner_safe, on="patient_id", how="inner", validate="many_to_one")
        .reset_index(drop=True)
    )
    known_profiles = set(map(str, profile_ids))
    observed_profiles = set(ranking.care_profile_id) | set(score_frame.care_profile_id)
    unknown = sorted(observed_profiles.difference(known_profiles))
    if unknown:
        raise ValueError(f"Unknown care_profile_id values: {unknown}")
    return rank_train, validation, score_frame


def _method_score_frame(
    score_frame: pd.DataFrame,
    method: str,
    score: np.ndarray,
    primary_variant: str,
    score_role: str,
) -> pd.DataFrame:
    if len(score) != len(score_frame):
        raise ValueError("Method scores must align with supported opportunities")
    result = score_frame[
        ["patient_id", "care_profile_id", "care_profile_index", "split"]
    ].copy()
    result["method"] = str(method)
    result["method_score"] = np.asarray(score, dtype=float)
    result["score_role"] = str(score_role)
    result["raw_priority_score"] = (
        result.method_score if method == primary_variant else np.nan
    )
    result["oracle_used"] = False
    return result


def _risk_score(frame: pd.DataFrame) -> np.ndarray:
    coefficients = {
        "prior_inpatient": 2.0,
        "prior_emergency": 1.0,
        "condition_distinct": 0.5,
        "frailty_index": 3.0,
        "functional_limitation_score": 2.0,
        "social_fragility_score": 1.0,
    }
    score = np.zeros(len(frame), dtype=float)
    for column, coefficient in coefficients.items():
        if column in frame:
            score += coefficient * pd.to_numeric(
                frame[column], errors="coerce"
            ).fillna(0.0).to_numpy(float)
    return score


def _random_score(frame: pd.DataFrame, seed: int) -> np.ndarray:
    values = []
    for patient, profile in zip(frame.patient_id, frame.care_profile_id):
        digest = hashlib.sha256(
            f"{int(seed)}|{patient}|{profile}".encode("utf-8")
        ).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64))
    return np.asarray(values, dtype=float)


def train_profile_rankers(
    learner: pd.DataFrame,
    supervision: pd.DataFrame,
    opportunities: pd.DataFrame,
    patient_splits: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    profile_ids: Sequence[str],
    model_seed: int,
    pair_seed: int,
    validation_pair_seed: int,
    contrastive_seed: int,
    gbdt_seed: int,
    random_seed: int,
    training_pair_label_permutation_seed: int | None = None,
) -> ProfileRankingResult:
    """Train configured non-oracle variants and score supported opportunities."""

    variants = tuple(map(str, settings["variants"]))
    unknown = sorted(set(variants).difference(PROFILE_RANKING_VARIANTS))
    if unknown or len(set(variants)) != len(variants):
        raise ValueError(f"Unknown or duplicate profile-ranking variants: {unknown}")
    primary = str(settings["primary_variant"])
    if primary not in variants or primary not in {
        "global_rank_only",
        "global_rank_plus_contrastive",
    }:
        raise ValueError("Primary ranker must be one configured global direct ranker")
    evaluation_only = tuple(map(str, settings.get("evaluation_only_variants", ())))
    if evaluation_only != (ORACLE_EVALUATION_VARIANT,):
        raise ValueError("oracle_evaluation_only must remain the sole deferred variant")
    seeds = {
        "model_seed": int(model_seed),
        "pair_seed": int(pair_seed),
        "validation_pair_seed": int(validation_pair_seed),
        "contrastive_seed": int(contrastive_seed),
        "gbdt_seed": int(gbdt_seed),
        "random_seed": int(random_seed),
    }
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("Every ranker model and sampler seed must be distinct")

    profile_ids = tuple(map(str, profile_ids))
    if not profile_ids or len(set(profile_ids)) != len(profile_ids):
        raise ValueError("Profile ranker requires unique configured profile IDs")
    train, validation, score_frame = _prepare_frames(
        learner,
        supervision,
        opportunities,
        patient_splits,
        numeric_features,
        categorical_features,
        profile_ids,
    )
    profile_mapping = {profile: index for index, profile in enumerate(profile_ids)}
    score_outputs = []
    histories = []
    variant_audits: dict[str, Any] = {}
    primary_bundle = None
    primary_pairs_frame = pd.DataFrame()

    if train.empty or validation.empty or score_frame.empty:
        audit = {
            "status": "not_trained_no_supported_rank_or_validation_opportunities",
            "primary_variant": primary,
            "requested_variants": list(variants),
            "evaluation_only_variants_deferred": list(evaluation_only),
            "seeds": seeds,
            "oracle_columns_detected": {
                "learner": _oracle_columns(learner),
                "supervision": _oracle_columns(supervision),
                "opportunities": _oracle_columns(opportunities),
            },
            "oracle_used": False,
            "test_outcomes_used": False,
        }
        return ProfileRankingResult(
            scores=pd.DataFrame(),
            training_history=pd.DataFrame(),
            validation_pairs=pd.DataFrame(),
            audit=audit,
            primary_model_bundle=None,
        )

    common_pair_arguments = {
        "minimum_signal_gap": float(settings["minimum_signal_gap"]),
        "minimum_direction_agreement": float(
            settings["minimum_direction_agreement"]
        ),
        "within_profile_fraction": float(settings["within_profile_fraction"]),
        "allow_same_patient_cross_profile": bool(
            settings["allow_same_patient_cross_profile_pairs"]
        ),
        "maximum_pair_weight": float(settings["maximum_pair_weight"]),
    }
    train_pairs = sample_profile_pairs(
        train,
        pairs=int(settings["training_pairs"]),
        seed=int(pair_seed),
        allow_cross_profile=bool(
            settings.get("allow_cross_profile_training_pairs", True)
        ),
        **common_pair_arguments,
    )
    validation_pairs = sample_profile_pairs(
        validation,
        pairs=int(settings["validation_pairs"]),
        seed=int(validation_pair_seed),
        allow_cross_profile=True,
        **common_pair_arguments,
    )
    contrastive_pairs = _sample_contrastive_pairs(
        train,
        pairs=int(settings["contrastive_pairs"]),
        bins=int(settings["contrastive_response_bins"]),
        seed=int(contrastive_seed),
        minimum_label_agreement=float(settings["minimum_direction_agreement"]),
        allow_same_patient_cross_profile=bool(
            settings["allow_same_patient_cross_profile_pairs"]
        ),
    )
    if not len(train_pairs) or not len(validation_pairs):
        raise ValueError("Supported ranking data produced no stable train/validation pairs")
    training_pair_label_changes = 0
    if training_pair_label_permutation_seed is not None:
        permutation_rng = np.random.default_rng(
            int(training_pair_label_permutation_seed)
        )
        permuted_direction = permutation_rng.permutation(train_pairs.direction)
        training_pair_label_changes = int(
            np.sum(permuted_direction != train_pairs.direction)
        )
        train_pairs = PairBatch(
            left=train_pairs.left.copy(),
            right=train_pairs.right.copy(),
            direction=permuted_direction.astype(float),
            weight=train_pairs.weight.copy(),
            is_cross_profile=train_pairs.is_cross_profile.copy(),
            direction_agreement=train_pairs.direction_agreement.copy(),
        )

    for variant in variants:
        if variant in {"global_rank_only", "global_rank_plus_contrastive"}:
            contrastive_weight = (
                float(settings["contrastive_weight"])
                if variant == "global_rank_plus_contrastive"
                else 0.0
            )
            restart_seeds = _restart_model_seeds(
                int(model_seed), int(settings.get("model_restarts", 1))
            )
            restart_scores = []
            restart_histories = []
            restart_audits = []
            restart_bundles = []
            for restart_index, restart_seed in enumerate(restart_seeds):
                restart_score, history, audit, bundle = _fit_network(
                    train,
                    validation,
                    score_frame,
                    train_pairs,
                    validation_pairs,
                    contrastive_pairs,
                    profile_mapping,
                    numeric_features,
                    categorical_features,
                    settings,
                    model_seed=int(restart_seed),
                    contrastive_weight=contrastive_weight,
                )
                history.insert(0, "restart_seed", int(restart_seed))
                history.insert(0, "restart_index", int(restart_index))
                history.insert(0, "method", variant)
                restart_scores.append(restart_score)
                restart_histories.append(history)
                restart_audits.append(audit)
                restart_bundles.append(bundle)
            scores = _mean_percentile_rank(restart_scores)
            validation_scores = _align_scores_to_validation(
                score_frame, validation, scores
            )
            ensemble_validation_loss = float(pairwise_ranking_loss(
                torch.as_tensor(validation_scores, dtype=torch.float32),
                validation_pairs,
            ))
            best_restart_index = int(np.argmin([
                item["best_validation_pairwise_loss"] for item in restart_audits
            ]))
            audit = {
                **restart_audits[best_restart_index],
                "best_validation_pairwise_loss": ensemble_validation_loss,
                "model_seed": int(model_seed),
                "model_restarts": int(len(restart_seeds)),
                "model_restart_seeds": list(map(int, restart_seeds)),
                "restart_aggregation": "mean_percentile_rank",
                "restart_best_validation_pairwise_losses": [
                    float(item["best_validation_pairwise_loss"])
                    for item in restart_audits
                ],
                "restart_best_epochs": [
                    int(item["best_epoch"]) for item in restart_audits
                ],
                "projection_head_last_gradient_norm": float(np.mean([
                    item["projection_head_last_gradient_norm"]
                    for item in restart_audits
                ])),
                "profile_embedding_last_gradient_norm": float(np.mean([
                    item["profile_embedding_last_gradient_norm"]
                    for item in restart_audits
                ])),
            }
            bundle = {
                "members": restart_bundles,
                "model_seed": int(model_seed),
                "model_restart_seeds": list(map(int, restart_seeds)),
                "restart_aggregation": "mean_percentile_rank",
                "profile_mapping": dict(profile_mapping),
                "score_semantics": (
                    "globally_comparable_ordinal_priority_no_causal_zero"
                ),
            }
            histories.extend(restart_histories)
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    scores,
                    primary,
                    "direct_causal_ordinal_priority",
                )
            )
            variant_audits[variant] = audit
            if variant == primary:
                primary_bundle = {"method": variant, **bundle}
                primary_pairs_frame = _pair_frame(validation, validation_pairs, variant)
        elif variant == "independent_profile_rankers":
            independent_scores = np.full(len(score_frame), np.nan, dtype=float)
            profile_audits = {}
            profile_histories = []
            for profile in profile_ids:
                train_mask = train.care_profile_id.eq(profile).to_numpy()
                validation_mask = validation.care_profile_id.eq(profile).to_numpy()
                score_mask = score_frame.care_profile_id.eq(profile).to_numpy()
                if not train_mask.any() or not validation_mask.any() or not score_mask.any():
                    profile_audits[profile] = {"status": "not_trained_missing_partition"}
                    continue
                profile_train = train.loc[train_mask].reset_index(drop=True)
                profile_validation = validation.loc[validation_mask].reset_index(drop=True)
                profile_score = score_frame.loc[score_mask].reset_index(drop=True)
                profile_train_pairs = sample_profile_pairs(
                    profile_train,
                    pairs=max(1, int(settings["training_pairs"]) // len(profile_ids)),
                    seed=int(pair_seed),
                    allow_cross_profile=False,
                    **common_pair_arguments,
                )
                profile_validation_pairs = sample_profile_pairs(
                    profile_validation,
                    pairs=max(1, int(settings["validation_pairs"]) // len(profile_ids)),
                    seed=int(validation_pair_seed),
                    allow_cross_profile=False,
                    **common_pair_arguments,
                )
                if not len(profile_train_pairs) or not len(profile_validation_pairs):
                    profile_audits[profile] = {"status": "not_trained_no_stable_pairs"}
                    continue
                profile_scores, history, audit, _ = _fit_network(
                    profile_train,
                    profile_validation,
                    profile_score,
                    profile_train_pairs,
                    profile_validation_pairs,
                    _empty_similarity(),
                    {profile: 0},
                    numeric_features,
                    categorical_features,
                    settings,
                    model_seed=int(model_seed),
                    contrastive_weight=0.0,
                )
                independent_scores[score_mask] = profile_scores
                history.insert(0, "care_profile_id", profile)
                history.insert(0, "method", variant)
                profile_histories.append(history)
                profile_audits[profile] = {"status": "trained", **audit}
            finite = np.isfinite(independent_scores)
            if finite.any():
                score_outputs.append(
                    _method_score_frame(
                        score_frame.loc[finite].reset_index(drop=True),
                        variant,
                        independent_scores[finite],
                        primary,
                        "within_profile_only_ordinal_baseline",
                    )
                )
            histories.extend(profile_histories)
            variant_audits[variant] = {
                "profiles": profile_audits,
                "globally_comparable": False,
                "oracle_used": False,
            }
        elif variant == "direct_pairwise_gbdt":
            scores, history, audit = _fit_pairwise_gbdt(
                train,
                validation,
                score_frame,
                train_pairs,
                validation_pairs,
                profile_mapping,
                numeric_features,
                categorical_features,
                settings,
                model_seed=int(gbdt_seed),
                reference_seed=int(pair_seed),
            )
            history.insert(0, "method", variant)
            histories.append(history)
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    scores,
                    primary,
                    "direct_pairwise_tree_baseline",
                )
            )
            variant_audits[variant] = audit
        elif variant == "risk_based_ranking":
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    _risk_score(score_frame),
                    primary,
                    "baseline_risk_comparator_not_causal_actionability",
                )
            )
            variant_audits[variant] = {
                "status": "deterministic_baseline",
                "risk_and_actionability_separate": True,
                "oracle_used": False,
            }
        elif variant == "baseline_need_based_ranking":
            need_score = pd.to_numeric(
                score_frame.baseline_need_score, errors="coerce"
            ).fillna(0.0).to_numpy(float)
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    need_score,
                    primary,
                    "descriptive_need_comparator_not_causal_actionability",
                )
            )
            variant_audits[variant] = {
                "status": "deterministic_baseline",
                "oracle_used": False,
            }
        elif variant == "random":
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    _random_score(score_frame, int(random_seed)),
                    primary,
                    "seeded_random_comparator",
                )
            )
            variant_audits[variant] = {
                "status": "seeded_baseline",
                "random_seed": int(random_seed),
                "oracle_used": False,
            }

    scores = pd.concat(score_outputs, ignore_index=True) if score_outputs else pd.DataFrame()
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    validation_pair_hash = _validation_pair_hash(primary_pairs_frame)
    primary_audit = variant_audits.get(primary, {})
    audit = {
        "status": "trained",
        "primary_variant": primary,
        "requested_variants": list(variants),
        "evaluation_only_variants_deferred": list(evaluation_only),
        "profile_ids": list(profile_ids),
        "numeric_features": list(map(str, numeric_features)),
        "categorical_features": list(map(str, categorical_features)),
        "rank_train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "scored_supported_opportunities": int(len(score_frame)),
        "partitions_used_for_fitting": [TRAIN_SPLIT],
        "feature_preprocessing_fit_partition": TRAIN_SPLIT,
        "contrastive_response_boundaries_fit_partition": TRAIN_SPLIT,
        "partition_used_for_early_stopping": VALIDATION_SPLIT,
        "test_outcomes_used": False,
        "pair_source": "repeated_dr_directions",
        "fixed_validation_pair_hash": validation_pair_hash,
        "fixed_validation_pairs": int(len(validation_pairs)),
        "validation_within_profile_pairs": int(
            (~validation_pairs.is_cross_profile).sum()
        ),
        "validation_cross_profile_pairs": int(
            validation_pairs.is_cross_profile.sum()
        ),
        "cross_profile_pairs_require_common_ranking_domain": True,
        "cross_profile_training_pairs_allowed": bool(
            settings.get("allow_cross_profile_training_pairs", True)
        ),
        "allow_same_patient_cross_profile_pairs": bool(
            settings["allow_same_patient_cross_profile_pairs"]
        ),
        "training_pair_labels_permuted": (
            training_pair_label_permutation_seed is not None
        ),
        "training_pair_label_permutation_seed": (
            None
            if training_pair_label_permutation_seed is None
            else int(training_pair_label_permutation_seed)
        ),
        "training_pair_label_changes": training_pair_label_changes,
        "score_field": "raw_priority_score",
        "score_semantics": "globally_comparable_ordinal_priority_no_causal_zero",
        "individual_cate_estimated_then_sorted": False,
        "oracle_columns_detected": {
            "learner": _oracle_columns(learner),
            "supervision": _oracle_columns(supervision),
            "opportunities": _oracle_columns(opportunities),
        },
        "oracle_used_for_training": False,
        "oracle_used_for_pair_construction": False,
        "oracle_used_for_early_stopping": False,
        "seeds": seeds,
        "primary_variant_audit": primary_audit,
        "variants": variant_audits,
    }
    return ProfileRankingResult(
        scores=scores,
        training_history=history,
        validation_pairs=primary_pairs_frame,
        audit=audit,
        primary_model_bundle=primary_bundle,
    )
