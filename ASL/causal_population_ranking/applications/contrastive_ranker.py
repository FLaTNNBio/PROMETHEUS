"""Direct multi-profile causal ranking with uncertainty-aware contrastive learning.

The network intentionally does *not* estimate an individual CATE. Repeated
learner-safe doubly robust signals are converted into reliable ordinal
constraints and uncertainty-aware triplets. The fitted score is an ordinal
allocation score; a separate monotone calibrator may be used downstream.
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


PAIR_TYPES = (
    "within_unit",
    "within_treatment",
    "global_cross_treatment",
)


class TreatmentConditionedSiameseNetwork(nn.Module):
    """Shared clinical encoder with treatment-conditioned FiLM modulation."""

    def __init__(
        self,
        input_dim: int,
        treatment_count: int,
        hidden_dim: int = 96,
        treatment_embedding_dim: int = 16,
        projection_dim: int = 24,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim < 1 or treatment_count < 1:
            raise ValueError("input_dim and treatment_count must be positive")
        self.clinical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.treatment_embedding = nn.Embedding(
            treatment_count, treatment_embedding_dim
        )
        self.film = nn.Sequential(
            nn.Linear(treatment_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.post_film = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, projection_dim),
        )

    def forward(
        self, x: torch.Tensor, treatment_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clinical = self.clinical_encoder(x)
        treatment = self.treatment_embedding(treatment_index)
        gamma_raw, beta = self.film(treatment).chunk(2, dim=1)
        # Keep modulation stable at initialization while allowing treatment-
        # specific amplification and suppression of clinical dimensions.
        gamma = 1.0 + 0.5 * torch.tanh(gamma_raw)
        conditioned = self.post_film(gamma * clinical + beta)
        score = self.score_head(conditioned).squeeze(1)
        projection = F.normalize(self.projection_head(conditioned), dim=1)
        return score, projection


@dataclass(frozen=True)
class ContrastiveCausalRanker:
    processor: ColumnTransformer
    model: TreatmentConditionedSiameseNetwork
    feature_columns: tuple[str, ...]
    treatment_column: str
    treatment_mapping: dict[str, int]
    audit: dict[str, Any]

    def _encoded(self, opportunities: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
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
        return (
            torch.from_numpy(matrix),
            torch.from_numpy(treatment.to_numpy(np.int64)),
        )

    def score(self, opportunities: pd.DataFrame) -> np.ndarray:
        matrix, treatment = self._encoded(opportunities)
        self.model.eval()
        with torch.no_grad():
            score, _ = self.model(matrix, treatment)
        return score.cpu().numpy()

    def representation(self, opportunities: pd.DataFrame) -> np.ndarray:
        matrix, treatment = self._encoded(opportunities)
        self.model.eval()
        with torch.no_grad():
            _, projection = self.model(matrix, treatment)
        return projection.cpu().numpy()


@dataclass(frozen=True)
class CausalContrastiveIndex:
    """Learner-safe metadata for uncertainty-aware causal contrastive training."""

    standardized_effect: np.ndarray
    uncertainty: np.ndarray
    reliability: np.ndarray
    effect_class: np.ndarray
    reliable_mask: np.ndarray
    eligible_anchors: np.ndarray
    class_sorted_indices: dict[int, np.ndarray]
    class_sorted_effects: dict[int, np.ndarray]
    within_unit_hard_negatives: dict[int, np.ndarray]
    negative_pool_by_class: dict[int, np.ndarray]
    positive_count_by_anchor: np.ndarray
    negative_count_by_anchor: np.ndarray
    ambiguous_count_by_anchor: np.ndarray
    neutral_threshold: float
    positive_radius: float
    negative_radius: float
    uncertainty_cutoff: float
    hard_negative_effect_gap: float
    audit: dict[str, Any]


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


def _repeat_agreement(
    signals: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    mean_gap: np.ndarray,
) -> np.ndarray:
    repeat_gap = signals[left] - signals[right]
    expected = np.sign(mean_gap)[:, None]
    return np.mean(np.sign(repeat_gap) == expected, axis=1)


def _sample_candidates(
    subset: pd.DataFrame,
    *,
    unit_id_column: str,
    treatment_column: str,
    pair_type: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(subset)
    units = subset[unit_id_column].astype(str).to_numpy()
    treatments = subset[treatment_column].astype(str).to_numpy()
    if pair_type == "within_unit":
        left: list[int] = []
        right: list[int] = []
        for idx in subset.groupby(unit_id_column, sort=False).indices.values():
            idx = np.asarray(idx, dtype=int)
            if len(idx) < 2:
                continue
            for a_position in range(len(idx) - 1):
                for b_position in range(a_position + 1, len(idx)):
                    a, b = int(idx[a_position]), int(idx[b_position])
                    if treatments[a] != treatments[b]:
                        left.append(a); right.append(b)
        if not left:
            return np.empty(0, int), np.empty(0, int)
        order = rng.permutation(len(left))[:count]
        return np.asarray(left, int)[order], np.asarray(right, int)[order]

    batch = max(count * 10, 8192)
    left = rng.integers(0, n, size=batch)
    right = rng.integers(0, n, size=batch)
    if pair_type == "within_treatment":
        valid = (left != right) & (units[left] != units[right]) & (
            treatments[left] == treatments[right]
        )
    elif pair_type == "global_cross_treatment":
        valid = (left != right) & (units[left] != units[right]) & (
            treatments[left] != treatments[right]
        )
    else:
        raise ValueError(f"Unknown pair type: {pair_type}")
    return left[valid][:count], right[valid][:count]


def sample_stable_ranking_pairs(
    opportunities: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    split: str,
    unit_id_column: str,
    treatment_column: str,
    maximum_pairs: int,
    minimum_signal_difference: float,
    minimum_repeat_agreement: float,
    pair_type_fractions: Mapping[str, float] | None = None,
    top_region_fraction: float = 0.25,
    top_pair_multiplier: float = 2.0,
    gap_clip_quantile: float = 0.95,
    seed: int,
) -> pd.DataFrame:
    """Create reliable ordinal constraints of three complementary types.

    The sampler first tries to respect the requested mixture. If the number of
    same-unit comparisons is structurally limited, the unused quota is safely
    reassigned to within-treatment and global cross-treatment comparisons.
    """
    subset = opportunities.loc[opportunities.split.eq(split)].reset_index(drop=True)
    if len(subset) < 3 or maximum_pairs < 1:
        raise ValueError(f"Not enough {split!r} opportunities for ranking pairs")
    signals = subset[list(signal_columns)].to_numpy(float)
    if not np.isfinite(signals).all():
        raise ValueError("Ranking signals must be finite")
    mean_signal = signals.mean(axis=1)
    fractions = dict(pair_type_fractions or {
        "within_unit": 0.30,
        "within_treatment": 0.35,
        "global_cross_treatment": 0.35,
    })
    unknown = set(fractions) - set(PAIR_TYPES)
    if unknown or any(v < 0 for v in fractions.values()) or sum(fractions.values()) <= 0:
        raise ValueError(f"Invalid pair_type_fractions: {fractions}")
    total_fraction = sum(fractions.values())
    quotas = {
        name: int(round(maximum_pairs * fractions.get(name, 0.0) / total_fraction))
        for name in PAIR_TYPES
    }
    quotas[PAIR_TYPES[-1]] += maximum_pairs - sum(quotas.values())
    threshold = float(np.quantile(mean_signal, 1.0 - top_region_fraction))
    rng = np.random.default_rng(int(seed))
    records: dict[tuple[int, int], tuple[int, int, float, float, str, bool]] = {}
    counts = {name: 0 for name in PAIR_TYPES}

    def add_candidates(pair_type: str, target: int, maximum_attempts: int = 30) -> None:
        attempts = 0
        while counts[pair_type] < target and len(records) < maximum_pairs and attempts < maximum_attempts:
            need = target - counts[pair_type]
            left, right = _sample_candidates(
                subset,
                unit_id_column=unit_id_column,
                treatment_column=treatment_column,
                pair_type=pair_type,
                count=max(need * 5, 2048),
                rng=rng,
            )
            if not len(left):
                break
            gap = mean_signal[left] - mean_signal[right]
            agreement = _repeat_agreement(signals, left, right, gap)
            valid = (
                np.abs(gap) >= float(minimum_signal_difference)
            ) & (agreement >= float(minimum_repeat_agreement)) & (gap != 0.0)
            added_this_attempt = 0
            for a, b, value, reliability in zip(
                left[valid], right[valid], gap[valid], agreement[valid]
            ):
                high, low = (int(a), int(b)) if value > 0 else (int(b), int(a))
                key = (high, low)
                if key in records:
                    continue
                top_pair = bool(max(mean_signal[high], mean_signal[low]) >= threshold)
                records[key] = (
                    high, low, float(abs(value)), float(reliability), pair_type, top_pair
                )
                counts[pair_type] += 1
                added_this_attempt += 1
                if counts[pair_type] >= target or len(records) >= maximum_pairs:
                    break
            attempts += 1
            # Enumeration-based within-unit candidates cannot change between
            # attempts; stop immediately if every admissible pair was rejected.
            if pair_type == "within_unit" and added_this_attempt == 0:
                break

    for pair_type in PAIR_TYPES:
        add_candidates(pair_type, quotas[pair_type])

    # Reassign any structurally unavailable within-unit quota instead of
    # silently training with far fewer pairs than requested.
    remaining = maximum_pairs - len(records)
    if remaining > 0:
        scalable = ("within_treatment", "global_cross_treatment")
        for index, pair_type in enumerate(scalable):
            target = counts[pair_type] + (remaining // 2)
            if index == len(scalable) - 1:
                target += remaining - 2 * (remaining // 2)
            add_candidates(pair_type, target, maximum_attempts=40)

    frame = pd.DataFrame(
        records.values(),
        columns=[
            "high_index", "low_index", "signal_gap", "repeat_agreement",
            "pair_type", "top_region_pair",
        ],
    )
    if frame.empty:
        raise ValueError(f"No stable ranking pairs found for split {split!r}")
    if len(frame) > maximum_pairs:
        frame = frame.sample(maximum_pairs, random_state=int(seed)).reset_index(drop=True)
    clip_value = float(frame.signal_gap.quantile(gap_clip_quantile))
    gap_weight = frame.signal_gap.clip(upper=max(clip_value, 1e-8))
    frame["weight"] = gap_weight * frame.repeat_agreement
    frame.loc[frame.top_region_pair, "weight"] *= float(top_pair_multiplier)
    frame["weight"] /= max(float(frame.weight.mean()), 1e-8)
    return frame


def _sample_response_guided_triplets(
    signal: np.ndarray,
    repeated_signals: np.ndarray,
    *,
    maximum_triplets: int,
    quantile_bins: int,
    positive_band_distance: int,
    negative_band_distance: int,
    minimum_repeat_agreement: float,
    positive_repeat_distance_quantile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample response-hard triplets without using oracle outcomes.

    Hard positives are the most distant candidates still belonging to nearby
    response bands. Hard negatives are the nearest candidates whose response
    bands are sufficiently separated.
    """
    if len(signal) < 4:
        raise ValueError("At least four opportunities are required for triplets")
    if not 0.0 < float(positive_repeat_distance_quantile) <= 1.0:
        raise ValueError(
            "positive_repeat_distance_quantile must be in (0, 1]"
        )
    rng = np.random.default_rng(int(seed))
    edges = np.unique(np.quantile(signal, np.linspace(0, 1, quantile_bins + 1)[1:-1]))
    band = np.digitize(signal, edges, right=True)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    order = rng.permutation(len(signal))
    pool_size = min(len(signal), 256)
    attempts = 0
    while len(anchors) < maximum_triplets and attempts < maximum_triplets * 8:
        anchor = int(order[attempts % len(order)])
        candidate = rng.choice(len(signal), size=pool_size, replace=False)
        candidate = candidate[candidate != anchor]
        band_gap = np.abs(band[candidate] - band[anchor])
        pos = candidate[band_gap <= positive_band_distance]
        neg = candidate[band_gap >= negative_band_distance]
        if len(pos) and len(neg):
            # Keep only positives that are also close across the repeated DR
            # signals.  Using the mean signal alone can pull together unstable
            # pseudo-outcomes and make the contrastive regularizer destructive.
            repeat_distance = np.mean(
                np.abs(repeated_signals[pos] - repeated_signals[anchor]),
                axis=1,
            )
            distance_cutoff = float(np.quantile(
                repeat_distance,
                positive_repeat_distance_quantile,
            ))
            stable_pos = pos[repeat_distance <= distance_cutoff + 1e-12]
            if not len(stable_pos):
                attempts += 1
                continue
            pos_gap = np.abs(signal[stable_pos] - signal[anchor])
            # Hard-but-stable positive: the furthest response within the stable
            # repeated-signal neighbourhood.
            positive = int(stable_pos[np.argmax(pos_gap)])
            neg_gap = np.abs(signal[neg] - signal[anchor])
            # Hard negative: the nearest clearly dissimilar candidate.
            negative_order = np.argsort(neg_gap)
            negative = None
            for idx in negative_order:
                candidate_negative = int(neg[idx])
                gaps = repeated_signals[anchor] - repeated_signals[candidate_negative]
                mean_gap = signal[anchor] - signal[candidate_negative]
                agreement = float(np.mean(np.sign(gaps) == np.sign(mean_gap)))
                if agreement >= minimum_repeat_agreement:
                    negative = candidate_negative
                    break
            if negative is not None:
                anchors.append(anchor); positives.append(positive); negatives.append(negative)
        attempts += 1
    if not anchors:
        raise ValueError("No stable response-guided triplets could be sampled")
    return (
        np.asarray(anchors, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(negatives, dtype=np.int64),
    )


def _sample_within_unit_auxiliary_pairs(
    opportunities: pd.DataFrame,
    repeated_signals: np.ndarray,
    *,
    unit_id_column: str,
    treatment_column: str,
    maximum_pairs: int,
    minimum_signal_difference: float,
    minimum_repeat_agreement: float,
    gap_clip_quantile: float,
    seed: int,
) -> pd.DataFrame:
    """Build a dedicated same-patient treatment-choice objective.

    The main global sampler deliberately filters hard for stable ordinal
    comparisons.  In multi-treatment applications that can leave too few
    same-patient pairs.  This auxiliary table uses its own, usually milder,
    thresholds while still requiring repeated-DR sign agreement.
    """
    if maximum_pairs <= 0:
        return pd.DataFrame(columns=["high_index", "low_index", "weight"])
    mean_signal = repeated_signals.mean(axis=1)
    treatments = opportunities[treatment_column].astype(str).to_numpy()
    records: list[tuple[int, int, float, float]] = []
    for indices in opportunities.groupby(unit_id_column, sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) < 2:
            continue
        for left_pos in range(len(indices) - 1):
            for right_pos in range(left_pos + 1, len(indices)):
                left = int(indices[left_pos])
                right = int(indices[right_pos])
                if treatments[left] == treatments[right]:
                    continue
                gap = float(mean_signal[left] - mean_signal[right])
                if abs(gap) < float(minimum_signal_difference) or gap == 0.0:
                    continue
                repeat_gap = repeated_signals[left] - repeated_signals[right]
                agreement = float(np.mean(np.sign(repeat_gap) == np.sign(gap)))
                if agreement < float(minimum_repeat_agreement):
                    continue
                high, low = (left, right) if gap > 0 else (right, left)
                records.append((high, low, abs(gap), agreement))
    if not records:
        return pd.DataFrame(columns=["high_index", "low_index", "weight"])
    frame = pd.DataFrame(
        records,
        columns=["high_index", "low_index", "signal_gap", "repeat_agreement"],
    ).drop_duplicates(["high_index", "low_index"])
    if len(frame) > maximum_pairs:
        frame = frame.sample(maximum_pairs, random_state=int(seed)).reset_index(drop=True)
    clip_value = float(frame.signal_gap.quantile(gap_clip_quantile))
    frame["weight"] = (
        frame.signal_gap.clip(upper=max(clip_value, 1e-8))
        * frame.repeat_agreement
    )
    frame["weight"] /= max(float(frame.weight.mean()), 1e-8)
    return frame


def _pair_loss(
    scores: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(weight * F.softplus(-(scores[high] - scores[low])))




def _direct_pair_loss(
    high_scores: torch.Tensor,
    low_scores: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(weight * F.softplus(-(high_scores - low_scores)))


def _weighted_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Stable pointwise auxiliary loss used only to align global score scales."""
    raw = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=float(beta),
    )
    return torch.sum(weight * raw) / torch.clamp(weight.sum(), min=1e-8)

def _triplet_loss(
    projection: torch.Tensor,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive_distance = torch.linalg.vector_norm(
        projection[anchor] - projection[positive], dim=1
    )
    negative_distance = torch.linalg.vector_norm(
        projection[anchor] - projection[negative], dim=1
    )
    return torch.mean(F.relu(positive_distance - negative_distance + margin))




def _quantile_threshold(values: np.ndarray, quantile: float, minimum: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float(minimum)
    return max(float(np.quantile(finite, float(quantile))), float(minimum))


def _build_causal_contrastive_index(
    opportunities: pd.DataFrame,
    repeated_signals: np.ndarray,
    *,
    unit_id_column: str,
    treatment_column: str,
    settings: Mapping[str, Any],
    seed: int,
) -> CausalContrastiveIndex:
    """Build responder-aware positive and clear-negative neighbourhoods.

    The index uses only repeated cross-fitted DR signals. Positive, negative and
    ambiguous relationships are mutually exclusive. Causal zero is preserved:
    effects are scaled but never mean-centred before responder classification.
    """
    repeated_signals = np.asarray(repeated_signals, dtype=float)
    if repeated_signals.ndim != 2 or repeated_signals.shape[1] < 1:
        raise ValueError("repeated_signals must have shape [opportunities, repeats]")
    if len(repeated_signals) != len(opportunities):
        raise ValueError("repeated_signals and opportunities must have equal length")
    if not np.isfinite(repeated_signals).all():
        raise ValueError("Causal contrastive signals must be finite")

    mean_effect = repeated_signals.mean(axis=1)
    location = float(np.mean(mean_effect))
    scale = float(np.std(mean_effect))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    effect = mean_effect / scale

    raw_uncertainty = np.std(repeated_signals, axis=1) / scale
    positive_uncertainty = raw_uncertainty[raw_uncertainty > 1e-12]
    uncertainty_reference = (
        float(np.median(positive_uncertainty)) if len(positive_uncertainty) else 1.0
    )
    reliability = 1.0 / (
        1.0 + raw_uncertainty / max(uncertainty_reference, 1e-8)
    )
    reliability_floor = float(settings.get("reliability_floor", 0.10))
    reliability = np.clip(reliability, reliability_floor, 1.0)

    neutral_threshold = settings.get("neutral_effect_threshold")
    if neutral_threshold is None:
        neutral_threshold = _quantile_threshold(
            np.abs(effect),
            float(settings.get("neutral_effect_quantile", 0.20)),
            float(settings.get("minimum_neutral_threshold", 0.05)),
        )
    neutral_threshold = float(neutral_threshold)
    effect_class = np.zeros(len(effect), dtype=np.int8)
    effect_class[effect > neutral_threshold] = 1
    effect_class[effect < -neutral_threshold] = -1

    uncertainty_cutoff = _quantile_threshold(
        raw_uncertainty,
        float(settings.get("uncertainty_max_quantile", 0.75)),
        0.0,
    )
    if not bool(settings.get("uncertainty_filtering", True)):
        uncertainty_cutoff = float("inf")
    reliable_mask = raw_uncertainty <= uncertainty_cutoff

    rng = np.random.default_rng(int(seed))
    class_sorted_indices: dict[int, np.ndarray] = {}
    class_sorted_effects: dict[int, np.ndarray] = {}
    within_class_distances: list[np.ndarray] = []
    for label in (-1, 0, 1):
        members = np.flatnonzero((effect_class == label) & reliable_mask)
        members = members[np.argsort(effect[members])]
        class_sorted_indices[label] = members.astype(np.int64)
        class_sorted_effects[label] = effect[members]
        if len(members) >= 2:
            sample_size = min(
                int(settings.get("distance_sample_size", 50000)),
                max(2048, len(members) * 8),
            )
            left = rng.choice(members, size=sample_size, replace=True)
            right = rng.choice(members, size=sample_size, replace=True)
            valid = left != right
            if valid.any():
                within_class_distances.append(
                    np.abs(effect[left[valid]] - effect[right[valid]])
                )

    sampled_within = (
        np.concatenate(within_class_distances)
        if within_class_distances
        else np.asarray([0.25, 0.75], dtype=float)
    )
    positive_radius = settings.get("positive_radius")
    if positive_radius is None:
        positive_radius = _quantile_threshold(
            sampled_within,
            float(settings.get("positive_radius_quantile", 0.20)),
            float(settings.get("minimum_positive_radius", 0.05)),
        )
    positive_radius = float(positive_radius)

    reliable_indices = np.flatnonzero(reliable_mask).astype(np.int64)
    all_distance_sample: np.ndarray
    if len(reliable_indices) >= 2:
        sample_size = min(
            int(settings.get("distance_sample_size", 50000)),
            max(4096, len(reliable_indices) * 8),
        )
        left = rng.choice(reliable_indices, size=sample_size, replace=True)
        right = rng.choice(reliable_indices, size=sample_size, replace=True)
        valid = left != right
        all_distance_sample = np.abs(effect[left[valid]] - effect[right[valid]])
    else:
        all_distance_sample = np.asarray([positive_radius * 2.0], dtype=float)

    negative_radius = settings.get("negative_radius")
    if negative_radius is None:
        negative_radius = _quantile_threshold(
            all_distance_sample,
            float(settings.get("negative_radius_quantile", 0.70)),
            max(
                float(settings.get("minimum_negative_radius", 0.25)),
                positive_radius + 1e-6,
            ),
        )
    negative_radius = max(float(negative_radius), positive_radius + 1e-6)

    all_reliable = reliable_indices
    negative_pool_by_class = {
        label: all_reliable[effect_class[all_reliable] != label]
        for label in (-1, 0, 1)
    }

    positive_count = np.zeros(len(effect), dtype=np.int32)
    negative_count = np.zeros(len(effect), dtype=np.int32)
    ambiguous_count = np.zeros(len(effect), dtype=np.int32)
    eligible: list[int] = []
    for label, members in class_sorted_indices.items():
        values = class_sorted_effects[label]
        different_class_count = int(len(negative_pool_by_class[label]))
        for position, anchor in enumerate(members):
            center = float(effect[anchor])
            pos_left = int(np.searchsorted(values, center - positive_radius, side="left"))
            pos_right = int(np.searchsorted(values, center + positive_radius, side="right"))
            pos_count = max(0, pos_right - pos_left - 1)

            neg_left_end = int(np.searchsorted(values, center - negative_radius, side="right"))
            neg_right_start = int(np.searchsorted(values, center + negative_radius, side="left"))
            same_class_far = neg_left_end + (len(values) - neg_right_start)
            neg_count = different_class_count + same_class_far

            same_class_total = max(0, len(values) - 1)
            amb_count = max(0, same_class_total - pos_count - same_class_far)
            positive_count[anchor] = pos_count
            negative_count[anchor] = neg_count
            ambiguous_count[anchor] = amb_count
            if pos_count > 0 and neg_count > 0:
                eligible.append(int(anchor))

    eligible_anchors = np.asarray(sorted(set(eligible)), dtype=np.int64)

    hard_negative_effect_gap = float(
        settings.get("hard_negative_effect_gap", negative_radius)
    )
    treatments = opportunities[treatment_column].astype(str).to_numpy()
    within_unit_hard_negatives: dict[int, np.ndarray] = {}
    for indices in opportunities.groupby(unit_id_column, sort=False).indices.values():
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) < 2:
            continue
        for anchor_index in indices:
            candidates = indices[
                (indices != anchor_index)
                & reliable_mask[indices]
                & (treatments[indices] != treatments[anchor_index])
            ]
            if not len(candidates):
                continue
            clear_negative = (
                effect_class[candidates] != effect_class[anchor_index]
            ) | (
                np.abs(effect[candidates] - effect[anchor_index])
                >= hard_negative_effect_gap
            )
            candidates = candidates[clear_negative]
            if len(candidates):
                within_unit_hard_negatives[int(anchor_index)] = candidates.astype(
                    np.int64
                )

    class_counts = {
        str(label): int(np.sum(effect_class == label)) for label in (-1, 0, 1)
    }
    reliable_class_counts = {
        str(label): int(np.sum((effect_class == label) & reliable_mask))
        for label in (-1, 0, 1)
    }
    eligible_positive_pairs = int(positive_count[eligible_anchors].sum())
    eligible_negative_pairs = int(negative_count[eligible_anchors].sum())
    eligible_ambiguous_pairs = int(ambiguous_count[eligible_anchors].sum())

    return CausalContrastiveIndex(
        standardized_effect=effect.astype(np.float32),
        uncertainty=raw_uncertainty.astype(np.float32),
        reliability=reliability.astype(np.float32),
        effect_class=effect_class,
        reliable_mask=reliable_mask,
        eligible_anchors=eligible_anchors,
        class_sorted_indices=class_sorted_indices,
        class_sorted_effects=class_sorted_effects,
        within_unit_hard_negatives=within_unit_hard_negatives,
        negative_pool_by_class=negative_pool_by_class,
        positive_count_by_anchor=positive_count,
        negative_count_by_anchor=negative_count,
        ambiguous_count_by_anchor=ambiguous_count,
        neutral_threshold=neutral_threshold,
        positive_radius=positive_radius,
        negative_radius=negative_radius,
        uncertainty_cutoff=uncertainty_cutoff,
        hard_negative_effect_gap=hard_negative_effect_gap,
        audit={
            "effect_location_for_audit_only": location,
            "effect_scale": scale,
            "neutral_threshold_standardized": neutral_threshold,
            "positive_radius_standardized": positive_radius,
            "negative_radius_standardized": negative_radius,
            "uncertainty_cutoff_standardized": uncertainty_cutoff,
            "uncertainty_reference_standardized": uncertainty_reference,
            "hard_negative_effect_gap_standardized": hard_negative_effect_gap,
            "class_counts": class_counts,
            "reliable_class_counts": reliable_class_counts,
            "eligible_anchors": int(len(eligible_anchors)),
            "positive_pair_count": eligible_positive_pairs,
            "clear_negative_pair_count": eligible_negative_pairs,
            "ambiguous_pair_count": eligible_ambiguous_pairs,
            "within_unit_hard_negative_anchors": int(
                len(within_unit_hard_negatives)
            ),
        },
    )


def _positive_candidates(
    index: CausalContrastiveIndex,
    anchor: int,
) -> np.ndarray:
    label = int(index.effect_class[anchor])
    members = index.class_sorted_indices[label]
    values = index.class_sorted_effects[label]
    center = float(index.standardized_effect[anchor])
    left = int(np.searchsorted(values, center - index.positive_radius, side="left"))
    right = int(np.searchsorted(values, center + index.positive_radius, side="right"))
    candidates = members[left:right]
    return candidates[candidates != anchor]


def _negative_candidates(
    index: CausalContrastiveIndex,
    anchor: int,
) -> np.ndarray:
    """Return only clear negatives; intermediate same-class pairs are ignored."""
    label = int(index.effect_class[anchor])
    different_class = index.negative_pool_by_class[label]
    members = index.class_sorted_indices[label]
    values = index.class_sorted_effects[label]
    center = float(index.standardized_effect[anchor])
    left_end = int(
        np.searchsorted(values, center - index.negative_radius, side="right")
    )
    right_start = int(
        np.searchsorted(values, center + index.negative_radius, side="left")
    )
    same_class_far = np.concatenate([members[:left_end], members[right_start:]])
    candidates = np.concatenate([different_class, same_class_far])
    candidates = candidates[candidates != anchor]
    return np.unique(candidates).astype(np.int64)


def _balanced_anchor_order(
    index: CausalContrastiveIndex,
    maximum_anchors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = [
        index.eligible_anchors[index.effect_class[index.eligible_anchors] == label]
        for label in (-1, 0, 1)
    ]
    groups = [group for group in groups if len(group)]
    if not groups:
        return np.empty(0, dtype=np.int64)
    target = min(int(maximum_anchors), int(sum(len(group) for group in groups)))
    base = target // len(groups)
    remainder = target - base * len(groups)
    sampled: list[np.ndarray] = []
    for group_index, group in enumerate(groups):
        count = min(len(group), base + (1 if group_index < remainder else 0))
        sampled.append(rng.choice(group, size=count, replace=False))
    result = np.concatenate(sampled) if sampled else np.empty(0, dtype=np.int64)
    if len(result) < target:
        remaining = np.setdiff1d(index.eligible_anchors, result, assume_unique=False)
        if len(remaining):
            extra = rng.choice(
                remaining,
                size=min(target - len(result), len(remaining)),
                replace=False,
            )
            result = np.concatenate([result, extra])
    rng.shuffle(result)
    return result.astype(np.int64)


def _sample_causal_contrastive_batch(
    index: CausalContrastiveIndex,
    anchors: np.ndarray,
    *,
    batch_size: int,
    hard_negative_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility sampler used by the optional InfoNCE ablation."""
    selected: list[int] = []
    triples_global: list[tuple[int, int, int]] = []
    for anchor in np.asarray(anchors, dtype=np.int64):
        positives = _positive_candidates(index, int(anchor))
        if not len(positives):
            continue
        positive = int(rng.choice(positives))
        selected.extend([int(anchor), positive])
        if rng.random() < float(hard_negative_fraction):
            negatives = index.within_unit_hard_negatives.get(int(anchor))
            if negatives is None or not len(negatives):
                negatives = _negative_candidates(index, int(anchor))
            if len(negatives):
                negative = int(rng.choice(negatives))
                selected.append(negative)
                triples_global.append((int(anchor), positive, negative))
    if not selected:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    unique = list(dict.fromkeys(selected))
    reliable_pool = np.flatnonzero(index.reliable_mask)
    if len(unique) < int(batch_size) and len(reliable_pool):
        remaining = np.setdiff1d(reliable_pool, np.asarray(unique, dtype=np.int64))
        if len(remaining):
            fill = rng.choice(
                remaining,
                size=min(int(batch_size) - len(unique), len(remaining)),
                replace=False,
            )
            unique.extend(int(value) for value in fill)
    unique = unique[: int(batch_size)]
    mapping = {global_index: local for local, global_index in enumerate(unique)}
    triples = [
        (mapping[a], mapping[p], mapping[n])
        for a, p, n in triples_global
        if a in mapping and p in mapping and n in mapping
    ]
    if triples:
        hard_anchor, hard_positive, hard_negative = map(
            lambda values: np.asarray(values, dtype=np.int64),
            zip(*triples),
        )
    else:
        hard_anchor = hard_positive = hard_negative = np.empty(0, dtype=np.int64)
    return (
        np.asarray(unique, dtype=np.int64),
        hard_anchor,
        hard_positive,
        hard_negative,
    )


def _sample_uact_triplets(
    index: CausalContrastiveIndex,
    anchors: np.ndarray,
    *,
    same_patient_negative_probability: float,
    uncertainty_weighting: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Sample one valid UACT triplet per anchor when possible."""
    anchor_rows: list[int] = []
    positive_rows: list[int] = []
    negative_rows: list[int] = []
    weights: list[float] = []
    positive_gaps: list[float] = []
    negative_gaps: list[float] = []
    same_patient_count = 0
    global_count = 0

    for anchor in np.asarray(anchors, dtype=np.int64):
        positives = _positive_candidates(index, int(anchor))
        if not len(positives):
            continue
        positive_gap = np.abs(
            index.standardized_effect[positives]
            - index.standardized_effect[int(anchor)]
        )
        positive_scale = max(index.positive_radius, 1e-8)
        positive_probability = np.exp(-positive_gap / positive_scale)
        positive_probability /= positive_probability.sum()
        positive = int(rng.choice(positives, p=positive_probability))

        negative: int | None = None
        use_same_patient = rng.random() < float(same_patient_negative_probability)
        if use_same_patient:
            local = index.within_unit_hard_negatives.get(int(anchor))
            if local is not None and len(local):
                negative = int(rng.choice(local))
                same_patient_count += 1
        if negative is None:
            global_candidates = _negative_candidates(index, int(anchor))
            if not len(global_candidates):
                continue
            negative_gap_values = np.abs(
                index.standardized_effect[global_candidates]
                - index.standardized_effect[int(anchor)]
            )
            # Harder clear negatives are sampled more often: they are close to
            # the negative boundary but never lie in the ambiguous region.
            hardness = 1.0 / (
                1.0
                + np.maximum(
                    negative_gap_values - index.negative_radius,
                    0.0,
                )
            )
            hardness /= hardness.sum()
            negative = int(rng.choice(global_candidates, p=hardness))
            global_count += 1

        reliability_weight = 1.0
        if uncertainty_weighting:
            reliability_weight = float(
                index.reliability[int(anchor)]
                * index.reliability[positive]
                * index.reliability[negative]
            )
        anchor_rows.append(int(anchor))
        positive_rows.append(positive)
        negative_rows.append(negative)
        weights.append(reliability_weight)
        positive_gaps.append(
            float(
                abs(
                    index.standardized_effect[int(anchor)]
                    - index.standardized_effect[positive]
                )
            )
        )
        negative_gaps.append(
            float(
                abs(
                    index.standardized_effect[int(anchor)]
                    - index.standardized_effect[negative]
                )
            )
        )

    diagnostic: dict[str, float | int] = {
        "sampled_triplets": len(anchor_rows),
        "same_patient_negative_count": same_patient_count,
        "global_negative_count": global_count,
        "mean_positive_effect_gap": float(np.mean(positive_gaps))
        if positive_gaps
        else 0.0,
        "mean_negative_effect_gap": float(np.mean(negative_gaps))
        if negative_gaps
        else 0.0,
        "mean_triplet_reliability": float(np.mean(weights)) if weights else 0.0,
    }
    return (
        np.asarray(anchor_rows, dtype=np.int64),
        np.asarray(positive_rows, dtype=np.int64),
        np.asarray(negative_rows, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        diagnostic,
    )


def _weighted_cosine_triplet_loss(
    anchor_projection: torch.Tensor,
    positive_projection: torch.Tensor,
    negative_projection: torch.Tensor,
    reliability_weight: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if len(anchor_projection) == 0:
        return anchor_projection.sum() * 0.0
    positive_similarity = torch.sum(
        anchor_projection * positive_projection,
        dim=1,
    )
    negative_similarity = torch.sum(
        anchor_projection * negative_projection,
        dim=1,
    )
    raw = F.relu(
        float(margin) + negative_similarity - positive_similarity
    )
    weight = reliability_weight.clamp_min(0.0)
    return torch.sum(weight * raw) / weight.sum().clamp_min(1e-8)


def causal_supervised_contrastive_loss(
    projection: torch.Tensor,
    standardized_effect: torch.Tensor,
    effect_class: torch.Tensor,
    reliability: torch.Tensor,
    *,
    positive_radius: float,
    temperature: float,
    distance_scale: float | None = None,
) -> torch.Tensor:
    """Uncertainty-weighted responder-aware supervised InfoNCE.

    Positive pairs must share the responder class and have nearby learner-safe
    DR effects. Every other non-self opportunity contributes to the denominator.
    """
    if projection.ndim != 2 or len(projection) < 2:
        return projection.sum() * 0.0
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    distance = torch.abs(
        standardized_effect.unsqueeze(1) - standardized_effect.unsqueeze(0)
    )
    positive_mask = (
        effect_class.unsqueeze(1).eq(effect_class.unsqueeze(0))
        & (distance <= float(positive_radius))
    )
    eye = torch.eye(len(projection), dtype=torch.bool, device=projection.device)
    positive_mask = positive_mask & ~eye
    logits = (projection @ projection.T) / float(temperature)
    logits = logits.masked_fill(eye, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # Avoid 0 * -inf on the diagonal.
    log_probability = log_probability.masked_fill(eye, 0.0)
    gamma = float(distance_scale or max(float(positive_radius), 1e-6))
    pair_weight = (
        positive_mask.float()
        * reliability.unsqueeze(1)
        * reliability.unsqueeze(0)
        * torch.exp(-distance / gamma)
    )
    positive_mass = pair_weight.sum(dim=1)
    valid = positive_mass > 1e-12
    if not bool(valid.any()):
        return projection.sum() * 0.0
    per_anchor = -(
        pair_weight * log_probability
    ).sum(dim=1) / positive_mass.clamp_min(1e-12)
    return per_anchor[valid].mean()


def _cosine_hard_negative_loss(
    projection: torch.Tensor,
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if not len(anchor):
        return projection.sum() * 0.0
    positive_similarity = torch.sum(
        projection[anchor] * projection[positive], dim=1
    )
    negative_similarity = torch.sum(
        projection[anchor] * projection[negative], dim=1
    )
    return torch.mean(F.relu(negative_similarity - positive_similarity + float(margin)))


def _ramped_auxiliary_weight(
    epoch: int,
    maximum_weight: float,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    if epoch < int(warmup_epochs):
        return 0.0
    if int(ramp_epochs) <= 0:
        return float(maximum_weight)
    progress = min(
        1.0,
        (epoch - int(warmup_epochs) + 1) / float(ramp_epochs),
    )
    return float(maximum_weight) * progress


def _pair_concordance(scores: np.ndarray, pairs: pd.DataFrame) -> float:
    if pairs.empty:
        return float("nan")
    margin = scores[pairs.high_index.to_numpy(int)] - scores[pairs.low_index.to_numpy(int)]
    weight = pairs.weight.to_numpy(float)
    return float(np.average(margin > 0.0, weights=weight))


def _ndcg_at_fraction(scores: np.ndarray, signal: np.ndarray, fraction: float) -> float:
    k = max(1, min(len(scores), int(round(len(scores) * fraction))))
    relevance = pd.Series(signal).rank(method="average", pct=True).to_numpy(float)
    predicted = np.argsort(-scores)[:k]
    ideal = np.argsort(-relevance)[:k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[predicted] * discount))
    idcg = float(np.sum(relevance[ideal] * discount))
    return dcg / idcg if idcg > 0 else 0.0


def _topk_dr_value_ratio(scores: np.ndarray, signal: np.ndarray, fraction: float) -> float:
    k = max(1, min(len(scores), int(round(len(scores) * fraction))))
    positive = np.maximum(signal, 0.0)
    denominator = float(np.sort(positive)[-k:].sum())
    if denominator <= 1e-12:
        return 0.0
    selected = np.argsort(-scores)[:k]
    return float(positive[selected].sum() / denominator)


def _validation_metrics(
    scores: np.ndarray,
    signal: np.ndarray,
    pairs: pd.DataFrame,
    settings: Mapping[str, Any],
) -> dict[str, float]:
    global_concordance = _pair_concordance(scores, pairs)
    if "pair_type" in pairs:
        cross = pairs.loc[pairs.pair_type.eq("global_cross_treatment")]
        within = pairs.loc[pairs.pair_type.eq("within_unit")]
    else:
        cross = pairs.iloc[0:0]
        within = pairs.iloc[0:0]
    cross_concordance = _pair_concordance(scores, cross) if not cross.empty else global_concordance
    within_concordance = _pair_concordance(scores, within) if not within.empty else global_concordance
    selection = settings.get("validation_selection", {})
    top_fraction = float(selection.get("ndcg_fraction", settings.get("top_region_fraction", 0.25)))
    ndcg = _ndcg_at_fraction(scores, signal, top_fraction)
    fractions = selection.get("topk_value_fractions", [0.10, 0.25, 0.50])
    topk_value = float(np.mean([
        _topk_dr_value_ratio(scores, signal, float(fraction))
        for fraction in fractions
    ]))
    weights = selection.get("weights", {
        "global_concordance": 0.30,
        "cross_treatment_concordance": 0.25,
        "within_unit_concordance": 0.15,
        "ndcg": 0.15,
        "topk_dr_value": 0.15,
    })
    values = {
        "global_concordance": global_concordance,
        "cross_treatment_concordance": cross_concordance,
        "within_unit_concordance": within_concordance,
        "ndcg": ndcg,
        "topk_dr_value": topk_value,
    }
    denominator = sum(float(weights.get(name, 0.0)) for name in values)
    if denominator <= 0:
        raise ValueError("validation_selection weights must sum to a positive value")
    composite = sum(float(weights.get(name, 0.0)) * value for name, value in values.items()) / denominator
    return {**values, "selection_metric": float(composite)}


def _batch_slice(order: torch.Tensor, step: int, batch_size: int) -> torch.Tensor:
    start = (step * batch_size) % len(order)
    stop = min(start + batch_size, len(order))
    result = order[start:stop]
    return result if len(result) else order[:batch_size]



def _encoder_parameters(
    model: TreatmentConditionedSiameseNetwork,
) -> tuple[nn.Parameter, ...]:
    """Return all representation parameters, excluding the ranking head."""
    return tuple(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("score_head")
    )


def _set_encoder_trainable(
    model: TreatmentConditionedSiameseNetwork,
    trainable: bool,
) -> None:
    """Freeze or unfreeze the shared representation and projection head."""
    for name, parameter in model.named_parameters():
        if name.startswith("score_head"):
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(bool(trainable))


def _set_finetuning_mode(
    model: TreatmentConditionedSiameseNetwork,
    *,
    encoder_trainable: bool,
) -> None:
    """Use deterministic encoder outputs while the pretrained encoder is frozen."""
    model.train()
    if encoder_trainable:
        return
    model.clinical_encoder.eval()
    model.treatment_embedding.eval()
    model.film.eval()
    model.post_film.eval()
    model.projection_head.eval()
    model.score_head.train()


def _effect_distance_embedding_correlation(
    effect_gap: np.ndarray,
    embedding_distance: np.ndarray,
) -> float:
    if len(effect_gap) < 3:
        return 0.0
    x = np.asarray(effect_gap, dtype=float)
    y = np.asarray(embedding_distance, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def _evaluate_uact_pretraining(
    model: TreatmentConditionedSiameseNetwork,
    x: torch.Tensor,
    treatment: torch.Tensor,
    index: CausalContrastiveIndex,
    *,
    settings: Mapping[str, Any],
    seed: int,
) -> dict[str, float | int]:
    """Evaluate a fixed held-out set of learner-safe UACT triplets."""
    rng = np.random.default_rng(int(seed))
    anchors = _balanced_anchor_order(
        index,
        int(settings.get("maximum_validation_anchors", 4000)),
        rng,
    )
    if not len(anchors):
        return {
            "selection_metric": -float("inf"),
            "triplet_loss": float("inf"),
            "triplet_accuracy": 0.0,
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "separation": 0.0,
            "effect_distance_correlation": 0.0,
            "sampled_triplets": 0,
        }
    anchor, positive, negative, weight, diagnostic = _sample_uact_triplets(
        index,
        anchors,
        same_patient_negative_probability=float(
            settings.get("same_patient_negative_probability", 0.0)
        ) if bool(settings.get("same_patient_hard_negatives", False)) else 0.0,
        uncertainty_weighting=bool(settings.get("uncertainty_weighting", True)),
        rng=rng,
    )
    maximum_triplets = int(settings.get("maximum_validation_triplets", 6000))
    if len(anchor) > maximum_triplets:
        chosen = rng.choice(len(anchor), size=maximum_triplets, replace=False)
        anchor, positive, negative, weight = (
            value[chosen] for value in (anchor, positive, negative, weight)
        )
    if not len(anchor):
        return {
            "selection_metric": -float("inf"),
            "triplet_loss": float("inf"),
            "triplet_accuracy": 0.0,
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "separation": 0.0,
            "effect_distance_correlation": 0.0,
            "sampled_triplets": 0,
        }
    model.eval()
    with torch.no_grad():
        _, anchor_projection = model(x[torch.from_numpy(anchor)], treatment[torch.from_numpy(anchor)])
        _, positive_projection = model(x[torch.from_numpy(positive)], treatment[torch.from_numpy(positive)])
        _, negative_projection = model(x[torch.from_numpy(negative)], treatment[torch.from_numpy(negative)])
        reliability = torch.from_numpy(weight.astype(np.float32))
        loss = float(_weighted_cosine_triplet_loss(
            anchor_projection,
            positive_projection,
            negative_projection,
            reliability,
            margin=float(settings.get("margin", 0.15)),
        ))
        positive_similarity = torch.sum(anchor_projection * positive_projection, dim=1).cpu().numpy()
        negative_similarity = torch.sum(anchor_projection * negative_projection, dim=1).cpu().numpy()
    margin = float(settings.get("margin", 0.15))
    accuracy = float(np.mean(positive_similarity >= negative_similarity + margin))
    mean_positive = float(np.mean(positive_similarity))
    mean_negative = float(np.mean(negative_similarity))
    separation = float(np.clip((mean_positive - mean_negative + 2.0) / 4.0, 0.0, 1.0))
    positive_gap = np.abs(index.standardized_effect[anchor] - index.standardized_effect[positive])
    negative_gap = np.abs(index.standardized_effect[anchor] - index.standardized_effect[negative])
    correlation = _effect_distance_embedding_correlation(
        np.concatenate([positive_gap, negative_gap]),
        np.concatenate([1.0 - positive_similarity, 1.0 - negative_similarity]),
    )
    correlation_score = float(np.clip((correlation + 1.0) / 2.0, 0.0, 1.0))
    weights = settings.get("validation_weights", {
        "triplet_accuracy": 0.40,
        "separation": 0.30,
        "effect_distance_correlation": 0.30,
    })
    denominator = sum(float(weights.get(name, 0.0)) for name in (
        "triplet_accuracy", "separation", "effect_distance_correlation"
    ))
    selection = (
        float(weights.get("triplet_accuracy", 0.0)) * accuracy
        + float(weights.get("separation", 0.0)) * separation
        + float(weights.get("effect_distance_correlation", 0.0)) * correlation_score
    ) / max(denominator, 1e-12)
    return {
        "selection_metric": float(selection),
        "triplet_loss": loss,
        "triplet_accuracy": accuracy,
        "positive_similarity": mean_positive,
        "negative_similarity": mean_negative,
        "separation": separation,
        "effect_distance_correlation": correlation,
        "sampled_triplets": int(len(anchor)),
        "mean_triplet_reliability": float(diagnostic.get("mean_triplet_reliability", 0.0)),
    }


def _run_uact_pretraining(
    model: TreatmentConditionedSiameseNetwork,
    train_x: torch.Tensor,
    train_t: torch.Tensor,
    validation_x: torch.Tensor,
    validation_t: torch.Tensor,
    train_index: CausalContrastiveIndex,
    validation_index: CausalContrastiveIndex,
    *,
    settings: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Learn the shared encoder before ranking using only causal triplets."""
    epochs = int(settings.get("epochs", 30))
    patience = int(settings.get("patience", 8))
    anchor_batch_size = int(settings.get("anchor_batch_size", 128))
    maximum_anchors = int(settings.get("maximum_anchors", 12000))
    margin = float(settings.get("margin", 0.15))
    strict = bool(settings.get("strict_diagnostics", True))
    rng = np.random.default_rng(int(seed) + 1)
    generator = torch.Generator().manual_seed(int(seed) + 2)

    _set_encoder_trainable(model, True)
    for parameter in model.score_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        _encoder_parameters(model),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 5e-4)),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    best_epoch = -1
    remaining = patience
    history: list[dict[str, float | int]] = []
    total_triplets = 0
    maximum_gradient_norm = 0.0

    for epoch in range(epochs):
        model.train()
        anchors = _balanced_anchor_order(train_index, maximum_anchors, rng)
        if not len(anchors):
            if strict:
                raise RuntimeError("Contrastive pretraining generated no valid anchors")
            break
        order = torch.randperm(len(anchors), generator=generator).numpy()
        anchors = anchors[order]
        losses: list[float] = []
        epoch_triplets = 0
        epoch_gradient_norm = 0.0
        for start in range(0, len(anchors), anchor_batch_size):
            anchor_slice = anchors[start:start + anchor_batch_size]
            anchor, positive, negative, weight, diagnostic = _sample_uact_triplets(
                train_index,
                anchor_slice,
                same_patient_negative_probability=float(
                    settings.get("same_patient_negative_probability", 0.0)
                ) if bool(settings.get("same_patient_hard_negatives", False)) else 0.0,
                uncertainty_weighting=bool(settings.get("uncertainty_weighting", True)),
                rng=rng,
            )
            if not len(anchor):
                continue
            optimizer.zero_grad()
            anchor_tensor = torch.from_numpy(anchor)
            positive_tensor = torch.from_numpy(positive)
            negative_tensor = torch.from_numpy(negative)
            _, anchor_projection = model(train_x[anchor_tensor], train_t[anchor_tensor])
            _, positive_projection = model(train_x[positive_tensor], train_t[positive_tensor])
            _, negative_projection = model(train_x[negative_tensor], train_t[negative_tensor])
            loss = _weighted_cosine_triplet_loss(
                anchor_projection,
                positive_projection,
                negative_projection,
                torch.from_numpy(weight.astype(np.float32)),
                margin=margin,
            )
            loss.backward()
            squared_norm = 0.0
            for parameter in _encoder_parameters(model):
                if parameter.grad is not None:
                    squared_norm += float(torch.sum(parameter.grad.detach() ** 2))
            epoch_gradient_norm = max(epoch_gradient_norm, squared_norm ** 0.5)
            torch.nn.utils.clip_grad_norm_(_encoder_parameters(model), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            epoch_triplets += int(diagnostic.get("sampled_triplets", 0))
        validation_metrics = _evaluate_uact_pretraining(
            model,
            validation_x,
            validation_t,
            validation_index,
            settings=settings,
            seed=int(seed) + 10_000,
        )
        history.append({
            "epoch": epoch,
            "train_triplet_loss": float(np.mean(losses)) if losses else 0.0,
            "train_sampled_triplets": int(epoch_triplets),
            "gradient_norm": float(epoch_gradient_norm),
            **validation_metrics,
        })
        total_triplets += epoch_triplets
        maximum_gradient_norm = max(maximum_gradient_norm, epoch_gradient_norm)
        metric = float(validation_metrics["selection_metric"])
        if metric > best_metric + 1e-7:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    model.load_state_dict(best_state)
    for parameter in model.score_head.parameters():
        parameter.requires_grad_(True)
    if strict:
        if best_epoch < 0:
            raise RuntimeError("Contrastive pretraining failed to select a checkpoint")
        if total_triplets <= 0:
            raise RuntimeError("Contrastive pretraining sampled no triplets")
        if maximum_gradient_norm <= 0.0:
            raise RuntimeError("Contrastive pretraining produced no encoder gradient")
    return {
        "enabled": True,
        "mode": "uncertainty_aware_ordinal_triplet",
        "best_epoch": int(best_epoch),
        "best_validation_selection_metric": float(best_metric),
        "total_sampled_triplets": int(total_triplets),
        "maximum_gradient_norm": float(maximum_gradient_norm),
        "train_index": train_index.audit,
        "validation_index": validation_index.audit,
        "history": history,
    }


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
    """Fit a contrastively pretrained ordinal ranker with optional retention."""
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
        treatment_embedding_dim=int(settings.get("treatment_embedding_dim", 16)),
        projection_dim=int(settings.get("projection_dim", 24)),
        dropout=float(settings.get("dropout", 0.10)),
    )
    pair_type_loss_multipliers = {
        str(key): float(value)
        for key, value in settings.get("pair_type_loss_multipliers", {}).items()
    }

    def pair_tensors(
        frame: pd.DataFrame,
        *,
        apply_training_multipliers: bool,
    ) -> tuple[torch.Tensor, ...]:
        required = {"high_index", "low_index", "weight"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Ranking pair table is missing columns: {sorted(missing)}")
        weights = frame.weight.to_numpy(np.float32).copy()
        if (
            apply_training_multipliers
            and pair_type_loss_multipliers
            and "pair_type" in frame
        ):
            factors = frame.pair_type.astype(str).map(
                lambda value: pair_type_loss_multipliers.get(value, 1.0)
            ).to_numpy(np.float32)
            weights *= factors
            # Preserve the overall scale of the pairwise objective while
            # changing the relative emphasis of the pair families.
            weights /= max(float(weights.mean()), 1e-8)
        return (
            torch.from_numpy(frame.high_index.to_numpy(np.int64)),
            torch.from_numpy(frame.low_index.to_numpy(np.int64)),
            torch.from_numpy(weights),
        )

    train_high, train_low, train_weight = pair_tensors(
        train_pairs,
        apply_training_multipliers=True,
    )
    val_high, val_low, val_weight = pair_tensors(
        validation_pairs,
        apply_training_multipliers=False,
    )

    train_signal_matrix = train[list(signal_columns)].to_numpy(float)
    train_signal = train_signal_matrix.mean(axis=1)
    validation_signal_matrix = validation[list(signal_columns)].to_numpy(float)
    validation_signal = validation_signal_matrix.mean(axis=1)

    pretraining_settings = settings.get("contrastive_pretraining", {})
    contrastive_pretraining_enabled = bool(
        pretraining_settings.get("enabled", False)
    )
    pretraining_audit: dict[str, Any] = {"enabled": False}
    if contrastive_pretraining_enabled:
        pretraining_mode = str(
            pretraining_settings.get(
                "mode", "uncertainty_aware_ordinal_triplet"
            )
        )
        if pretraining_mode != "uncertainty_aware_ordinal_triplet":
            raise ValueError(
                "contrastive_pretraining.mode must be "
                "'uncertainty_aware_ordinal_triplet'"
            )
        pretraining_train_index = _build_causal_contrastive_index(
            train,
            train_signal_matrix,
            unit_id_column=unit_id_column,
            treatment_column=treatment_column,
            settings=pretraining_settings,
            seed=seed + 501,
        )
        pretraining_validation_index = _build_causal_contrastive_index(
            validation,
            validation_signal_matrix,
            unit_id_column=unit_id_column,
            treatment_column=treatment_column,
            settings=pretraining_settings,
            seed=seed + 502,
        )
        pretraining_audit = _run_uact_pretraining(
            model,
            train_x,
            train_t,
            validation_x,
            validation_t,
            pretraining_train_index,
            pretraining_validation_index,
            settings=pretraining_settings,
            seed=seed + 503,
        )

    # Reset all fine-tuning stochasticity after initialization. Paired
    # ablations therefore differ in encoder weights, not in dropout streams.
    finetuning_stochastic_seed = seed + 7000
    torch.manual_seed(finetuning_stochastic_seed)
    np.random.seed(finetuning_stochastic_seed)

    finetuning_settings = settings.get("ranking_finetuning", {})
    freeze_encoder_epochs = int(
        finetuning_settings.get("freeze_encoder_epochs", 0)
    ) if contrastive_pretraining_enabled else 0
    freeze_encoder_entire_finetuning = bool(
        finetuning_settings.get("freeze_encoder_entire_finetuning", False)
    ) if contrastive_pretraining_enabled else False
    encoder_learning_rate = float(
        finetuning_settings.get(
            "encoder_learning_rate",
            settings.get("learning_rate", 3e-4),
        )
    )
    head_learning_rate = float(
        finetuning_settings.get(
            "head_learning_rate",
            settings.get("learning_rate", 3e-4),
        )
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(_encoder_parameters(model)),
                "lr": encoder_learning_rate,
            },
            {
                "params": list(model.score_head.parameters()),
                "lr": head_learning_rate,
            },
        ],
        weight_decay=float(settings.get("weight_decay", 5e-4)),
    )

    causal_settings = settings.get("causal_contrastive", {})
    lambda_causal_contrastive = float(causal_settings.get("weight", 0.0))
    causal_mode = str(
        causal_settings.get("mode", "uncertainty_aware_ordinal_triplet")
    )
    allowed_causal_modes = {
        "uncertainty_aware_ordinal_triplet",
        "causal_supervised_info_nce",
    }
    if causal_mode not in allowed_causal_modes:
        raise ValueError(
            f"Unsupported causal_contrastive mode: {causal_mode}. "
            f"Expected one of {sorted(allowed_causal_modes)}"
        )
    causal_contrastive_enabled = bool(
        causal_settings.get("enabled", lambda_causal_contrastive > 0.0)
        and lambda_causal_contrastive > 0.0
    )
    causal_contrastive_index = (
        _build_causal_contrastive_index(
            train,
            train_signal_matrix,
            unit_id_column=unit_id_column,
            treatment_column=treatment_column,
            settings=causal_settings,
            seed=seed + 43,
        )
        if causal_contrastive_enabled
        else None
    )
    causal_strict_diagnostics = bool(
        causal_settings.get("strict_diagnostics", False)
    )
    if (
        causal_contrastive_enabled
        and causal_contrastive_index is not None
        and not len(causal_contrastive_index.eligible_anchors)
    ):
        if causal_strict_diagnostics:
            raise RuntimeError(
                "Causal contrastive learning is enabled but no valid anchors "
                "were generated from learner-safe DR supervision."
            )
        causal_contrastive_enabled = False
        causal_contrastive_index = None
    causal_batch_size = int(causal_settings.get("batch_size", 256))
    causal_anchor_batch_size = int(causal_settings.get("anchor_batch_size", 96))
    causal_maximum_anchors = int(causal_settings.get("maximum_anchors", 12000))
    causal_temperature = float(causal_settings.get("temperature", 0.10))
    causal_same_patient_probability = float(
        causal_settings.get(
            "same_patient_negative_probability",
            causal_settings.get("hard_negative_fraction", 0.50),
        )
    )
    if not 0.0 <= causal_same_patient_probability <= 1.0:
        raise ValueError("same_patient_negative_probability must lie in [0, 1]")
    causal_same_patient_enabled = bool(
        causal_settings.get("same_patient_hard_negatives", True)
    )
    if not causal_same_patient_enabled:
        causal_same_patient_probability = 0.0
    causal_require_same_patient = bool(
        causal_settings.get("require_same_patient_negatives", False)
    )
    if (
        causal_contrastive_enabled
        and causal_mode == "uncertainty_aware_ordinal_triplet"
        and causal_require_same_patient
        and causal_contrastive_index is not None
        and not len(causal_contrastive_index.within_unit_hard_negatives)
    ):
        raise RuntimeError(
            "Same-patient hard negatives are required but none were generated."
        )
    causal_margin = float(
        causal_settings.get(
            "margin",
            causal_settings.get("hard_negative_margin", 0.15),
        )
    )
    causal_hard_negative_fraction = causal_same_patient_probability
    causal_hard_negative_margin = causal_margin
    causal_hard_negative_weight = float(
        causal_settings.get("hard_negative_weight", 0.25)
    )
    causal_warmup = int(causal_settings.get("warmup_epochs", 3))
    causal_ramp = int(causal_settings.get("ramp_epochs", 7))
    causal_minimum_active = int(causal_settings.get("minimum_active_epochs", 3))
    causal_minimum_post_ramp = int(
        causal_settings.get("minimum_post_ramp_epochs", 0)
    )
    causal_uncertainty_weighting = bool(
        causal_settings.get("uncertainty_weighting", True)
    )

    calibration_settings = settings.get("calibration", {})
    lambda_calibration = float(calibration_settings.get("weight", 0.0))
    calibration_enabled = bool(
        calibration_settings.get("enabled", lambda_calibration > 0.0)
        and lambda_calibration > 0.0
    )
    calibration_mean = float(np.mean(train_signal))
    calibration_scale = float(np.std(train_signal))
    if not np.isfinite(calibration_scale) or calibration_scale < 1e-6:
        calibration_scale = 1.0
    calibration_target_np = (
        (train_signal - calibration_mean) / calibration_scale
    ).astype(np.float32)
    validation_calibration_target = (
        (validation_signal - calibration_mean) / calibration_scale
    ).astype(np.float32)
    repeat_sd = np.std(train_signal_matrix, axis=1)
    positive_sd = repeat_sd[repeat_sd > 1e-12]
    sd_reference = float(np.median(positive_sd)) if len(positive_sd) else 1.0
    calibration_weight_np = 1.0 / (1.0 + repeat_sd / max(sd_reference, 1e-8))
    lower_weight = float(calibration_settings.get("minimum_reliability_weight", 0.25))
    calibration_weight_np = np.clip(calibration_weight_np, lower_weight, 1.0)
    calibration_weight_np /= max(float(calibration_weight_np.mean()), 1e-8)
    calibration_target = torch.from_numpy(calibration_target_np)
    calibration_weight = torch.from_numpy(calibration_weight_np.astype(np.float32))
    calibration_batch_size = int(
        calibration_settings.get("batch_size", settings.get("batch_size", 256))
    )
    calibration_beta = float(calibration_settings.get("huber_beta", 1.0))
    calibration_warmup = int(calibration_settings.get("warmup_epochs", 0))

    within_unit_settings = settings.get("within_unit_auxiliary", {})
    lambda_within_unit = float(within_unit_settings.get("weight", 0.0))
    within_unit_enabled = bool(
        within_unit_settings.get("enabled", lambda_within_unit > 0.0)
        and lambda_within_unit > 0.0
    )
    within_unit_pairs = (
        _sample_within_unit_auxiliary_pairs(
            train,
            train_signal_matrix,
            unit_id_column=unit_id_column,
            treatment_column=treatment_column,
            maximum_pairs=int(within_unit_settings.get("maximum_pairs", 12000)),
            minimum_signal_difference=float(
                within_unit_settings.get("minimum_signal_difference", 0.05)
            ),
            minimum_repeat_agreement=float(
                within_unit_settings.get("minimum_repeat_agreement", 0.60)
            ),
            gap_clip_quantile=float(
                within_unit_settings.get("gap_clip_quantile", 0.95)
            ),
            seed=seed + 31,
        )
        if within_unit_enabled
        else pd.DataFrame(columns=["high_index", "low_index", "weight"])
    )
    if within_unit_enabled and within_unit_pairs.empty:
        within_unit_enabled = False
    within_high = torch.from_numpy(
        within_unit_pairs.high_index.to_numpy(np.int64)
    ) if within_unit_enabled else torch.empty(0, dtype=torch.int64)
    within_low = torch.from_numpy(
        within_unit_pairs.low_index.to_numpy(np.int64)
    ) if within_unit_enabled else torch.empty(0, dtype=torch.int64)
    within_weight = torch.from_numpy(
        within_unit_pairs.weight.to_numpy(np.float32)
    ) if within_unit_enabled else torch.empty(0, dtype=torch.float32)
    within_unit_batch_size = int(
        within_unit_settings.get("batch_size", settings.get("batch_size", 256))
    )
    within_unit_warmup = int(within_unit_settings.get("warmup_epochs", 0))
    triplet_settings = settings.get("legacy_triplet", settings.get("triplet", {}))
    lambda_triplet = float(triplet_settings.get(
        "weight", settings.get("contrastive_weight", 0.02)
    ))
    maximum_triplets = int(triplet_settings.get(
        "maximum_triplets", settings.get("maximum_contrastive_pairs", 4000)
    ))
    triplet_enabled = bool(
        triplet_settings.get("enabled", True)
        and lambda_triplet > 0.0
        and maximum_triplets > 0
    )
    positive_repeat_distance_quantile = float(
        triplet_settings.get("positive_repeat_distance_quantile", 0.50)
    )
    if triplet_enabled:
        triplet_anchor_np, triplet_positive_np, triplet_negative_np = (
            _sample_response_guided_triplets(
                train_signal,
                train_signal_matrix,
                maximum_triplets=maximum_triplets,
                quantile_bins=int(triplet_settings.get("quantile_bins", 10)),
                positive_band_distance=int(
                    triplet_settings.get("positive_band_distance", 1)
                ),
                negative_band_distance=int(
                    triplet_settings.get("negative_band_distance", 4)
                ),
                minimum_repeat_agreement=float(
                    triplet_settings.get("minimum_repeat_agreement", 0.70)
                ),
                positive_repeat_distance_quantile=(
                    positive_repeat_distance_quantile
                ),
                seed=seed + 19,
            )
        )
    else:
        triplet_anchor_np = np.empty(0, dtype=np.int64)
        triplet_positive_np = np.empty(0, dtype=np.int64)
        triplet_negative_np = np.empty(0, dtype=np.int64)
    triplet_anchor = torch.from_numpy(triplet_anchor_np)
    triplet_positive = torch.from_numpy(triplet_positive_np)
    triplet_negative = torch.from_numpy(triplet_negative_np)

    batch_size = int(settings.get("batch_size", 256))
    triplet_batch_size = int(triplet_settings.get("batch_size", batch_size))
    margin = float(triplet_settings.get(
        "margin", settings.get("contrastive_margin", 0.50)
    ))
    warmup = int(settings.get("contrastive_warmup_epochs", 2)) if triplet_enabled else 0
    minimum_active = int(settings.get("minimum_contrastive_epochs", 3)) if triplet_enabled else 0
    minimum_training_epochs = int(settings.get("minimum_training_epochs", 0))
    causal_ready_epoch = (
        causal_warmup + causal_ramp + causal_minimum_post_ramp - 1
        if causal_contrastive_enabled
        else 0
    )
    earliest_checkpoint_epoch = max(
        0,
        minimum_training_epochs - 1,
        freeze_encoder_epochs,
        warmup + minimum_active - 1,
        causal_ready_epoch,
    )
    epochs = int(settings.get("epochs", 60))
    patience = int(settings.get("patience", 10))
    if epochs <= earliest_checkpoint_epoch:
        raise ValueError(
            "epochs must exceed the minimum checkpoint epoch; increase epochs "
            "or reduce minimum_training_epochs/warmup/ramp settings"
        )
    generator = torch.Generator().manual_seed(seed + 1)
    causal_rng = np.random.default_rng(seed + 44)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")
    best_epoch = -1
    remaining = patience
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        encoder_trainable = not freeze_encoder_entire_finetuning and (
            epoch >= freeze_encoder_epochs
        )
        _set_encoder_trainable(model, encoder_trainable)
        _set_finetuning_mode(
            model,
            encoder_trainable=encoder_trainable,
        )
        rank_order = torch.randperm(len(train_pairs), generator=generator)
        triplet_order = (
            torch.randperm(len(triplet_anchor), generator=generator)
            if triplet_enabled else torch.empty(0, dtype=torch.int64)
        )
        triplet_steps = (
            int(np.ceil(len(triplet_order) / triplet_batch_size))
            if triplet_enabled else 0
        )
        calibration_order = (
            torch.randperm(len(train_x), generator=generator)
            if calibration_enabled else torch.empty(0, dtype=torch.int64)
        )
        calibration_steps = (
            int(np.ceil(len(calibration_order) / calibration_batch_size))
            if calibration_enabled else 0
        )
        within_unit_order = (
            torch.randperm(len(within_high), generator=generator)
            if within_unit_enabled else torch.empty(0, dtype=torch.int64)
        )
        within_unit_steps = (
            int(np.ceil(len(within_unit_order) / within_unit_batch_size))
            if within_unit_enabled else 0
        )
        causal_anchor_order = (
            _balanced_anchor_order(
                causal_contrastive_index,
                causal_maximum_anchors,
                causal_rng,
            )
            if causal_contrastive_enabled and causal_contrastive_index is not None
            else np.empty(0, dtype=np.int64)
        )
        causal_steps = (
            int(np.ceil(len(causal_anchor_order) / causal_anchor_batch_size))
            if len(causal_anchor_order) else 0
        )
        # The primary ranking objective fixes the number of optimizer steps.
        # Auxiliary objectives cycle through their own batches and must not
        # change the optimization budget across ablation variants.
        steps = int(np.ceil(len(rank_order) / batch_size))
        epoch_causal_losses: list[float] = []
        epoch_causal_info_nce_losses: list[float] = []
        epoch_causal_hard_losses: list[float] = []
        epoch_sampled_triplets = 0
        epoch_same_patient_negatives = 0
        epoch_global_negatives = 0
        epoch_positive_gaps: list[float] = []
        epoch_negative_gaps: list[float] = []
        epoch_triplet_reliability: list[float] = []
        epoch_contrastive_gradient_norm = 0.0
        gradient_norm_measured = False
        for step in range(steps):
            rank_sel = _batch_slice(rank_order, step, batch_size)
            optimizer.zero_grad()
            high_idx = train_high[rank_sel]
            low_idx = train_low[rank_sel]
            high_score, _ = model(train_x[high_idx], train_t[high_idx])
            low_score, _ = model(train_x[low_idx], train_t[low_idx])
            rank_loss = _direct_pair_loss(
                high_score, low_score, train_weight[rank_sel]
            )
            if triplet_enabled:
                triplet_sel = _batch_slice(
                    triplet_order, step, triplet_batch_size
                )
                anchor_idx = triplet_anchor[triplet_sel]
                positive_idx = triplet_positive[triplet_sel]
                negative_idx = triplet_negative[triplet_sel]
                _, anchor_projection = model(
                    train_x[anchor_idx], train_t[anchor_idx]
                )
                _, positive_projection = model(
                    train_x[positive_idx], train_t[positive_idx]
                )
                _, negative_projection = model(
                    train_x[negative_idx], train_t[negative_idx]
                )
                positive_distance = torch.linalg.vector_norm(
                    anchor_projection - positive_projection, dim=1
                )
                negative_distance = torch.linalg.vector_norm(
                    anchor_projection - negative_projection, dim=1
                )
                triplet_loss = torch.mean(
                    F.relu(positive_distance - negative_distance + margin)
                )
            else:
                triplet_loss = rank_loss.new_zeros(())
            if calibration_enabled:
                calibration_sel = _batch_slice(
                    calibration_order,
                    step,
                    calibration_batch_size,
                )
                calibration_score, _ = model(
                    train_x[calibration_sel],
                    train_t[calibration_sel],
                )
                calibration_loss = _weighted_smooth_l1_loss(
                    calibration_score,
                    calibration_target[calibration_sel],
                    calibration_weight[calibration_sel],
                    calibration_beta,
                )
            else:
                calibration_loss = rank_loss.new_zeros(())
            if within_unit_enabled:
                within_sel = _batch_slice(
                    within_unit_order,
                    step,
                    within_unit_batch_size,
                )
                within_high_idx = within_high[within_sel]
                within_low_idx = within_low[within_sel]
                within_high_score, _ = model(
                    train_x[within_high_idx],
                    train_t[within_high_idx],
                )
                within_low_score, _ = model(
                    train_x[within_low_idx],
                    train_t[within_low_idx],
                )
                within_unit_loss = _direct_pair_loss(
                    within_high_score,
                    within_low_score,
                    within_weight[within_sel],
                )
            else:
                within_unit_loss = rank_loss.new_zeros(())
            causal_cpu_rng_state = None
            causal_cuda_rng_state = None
            if causal_contrastive_enabled and causal_contrastive_index is not None:
                # Keep paired ablations stochasticity-matched: auxiliary
                # dropout masks must not advance the RNG used by subsequent
                # primary-ranking updates.
                causal_cpu_rng_state = torch.random.get_rng_state()
                if torch.cuda.is_available():
                    causal_cuda_rng_state = torch.cuda.get_rng_state_all()
                start = (step * causal_anchor_batch_size) % max(
                    len(causal_anchor_order), 1
                )
                causal_anchor_slice = causal_anchor_order[
                    start : start + causal_anchor_batch_size
                ]
                if not len(causal_anchor_slice):
                    causal_anchor_slice = causal_anchor_order[:causal_anchor_batch_size]

                if causal_mode == "uncertainty_aware_ordinal_triplet":
                    (
                        uact_anchor_np,
                        uact_positive_np,
                        uact_negative_np,
                        uact_weight_np,
                        uact_diagnostic,
                    ) = _sample_uact_triplets(
                        causal_contrastive_index,
                        causal_anchor_slice,
                        same_patient_negative_probability=(
                            causal_same_patient_probability
                        ),
                        uncertainty_weighting=causal_uncertainty_weighting,
                        rng=causal_rng,
                    )
                    if len(uact_anchor_np):
                        uact_anchor = torch.from_numpy(uact_anchor_np)
                        uact_positive = torch.from_numpy(uact_positive_np)
                        uact_negative = torch.from_numpy(uact_negative_np)
                        uact_weight = torch.from_numpy(uact_weight_np)
                        _, uact_anchor_projection = model(
                            train_x[uact_anchor], train_t[uact_anchor]
                        )
                        _, uact_positive_projection = model(
                            train_x[uact_positive], train_t[uact_positive]
                        )
                        _, uact_negative_projection = model(
                            train_x[uact_negative], train_t[uact_negative]
                        )
                        causal_contrastive_loss = _weighted_cosine_triplet_loss(
                            uact_anchor_projection,
                            uact_positive_projection,
                            uact_negative_projection,
                            uact_weight,
                            margin=causal_margin,
                        )
                    else:
                        causal_contrastive_loss = rank_loss.new_zeros(())
                    causal_info_nce_loss = rank_loss.new_zeros(())
                    causal_hard_negative_loss = rank_loss.new_zeros(())
                    epoch_sampled_triplets += int(
                        uact_diagnostic["sampled_triplets"]
                    )
                    epoch_same_patient_negatives += int(
                        uact_diagnostic["same_patient_negative_count"]
                    )
                    epoch_global_negatives += int(
                        uact_diagnostic["global_negative_count"]
                    )
                    if int(uact_diagnostic["sampled_triplets"]) > 0:
                        epoch_positive_gaps.append(
                            float(uact_diagnostic["mean_positive_effect_gap"])
                        )
                        epoch_negative_gaps.append(
                            float(uact_diagnostic["mean_negative_effect_gap"])
                        )
                        epoch_triplet_reliability.append(
                            float(uact_diagnostic["mean_triplet_reliability"])
                        )
                else:
                    (
                        causal_indices_np,
                        hard_anchor_np,
                        hard_positive_np,
                        hard_negative_np,
                    ) = _sample_causal_contrastive_batch(
                        causal_contrastive_index,
                        causal_anchor_slice,
                        batch_size=causal_batch_size,
                        hard_negative_fraction=causal_hard_negative_fraction,
                        rng=causal_rng,
                    )
                    if len(causal_indices_np):
                        causal_indices = torch.from_numpy(causal_indices_np)
                        _, causal_projection = model(
                            train_x[causal_indices],
                            train_t[causal_indices],
                        )
                        causal_effect = torch.from_numpy(
                            causal_contrastive_index.standardized_effect[
                                causal_indices_np
                            ]
                        )
                        causal_class = torch.from_numpy(
                            causal_contrastive_index.effect_class[
                                causal_indices_np
                            ].astype(np.int64)
                        )
                        causal_reliability_np = (
                            causal_contrastive_index.reliability[
                                causal_indices_np
                            ]
                            if causal_uncertainty_weighting
                            else np.ones(
                                len(causal_indices_np), dtype=np.float32
                            )
                        )
                        causal_reliability = torch.from_numpy(
                            causal_reliability_np.astype(np.float32)
                        )
                        causal_info_nce_loss = (
                            causal_supervised_contrastive_loss(
                                causal_projection,
                                causal_effect,
                                causal_class,
                                causal_reliability,
                                positive_radius=(
                                    causal_contrastive_index.positive_radius
                                ),
                                temperature=causal_temperature,
                                distance_scale=float(
                                    causal_settings.get(
                                        "distance_scale",
                                        causal_contrastive_index.positive_radius,
                                    )
                                ),
                            )
                        )
                        hard_anchor = torch.from_numpy(hard_anchor_np)
                        hard_positive = torch.from_numpy(hard_positive_np)
                        hard_negative = torch.from_numpy(hard_negative_np)
                        causal_hard_negative_loss = (
                            _cosine_hard_negative_loss(
                                causal_projection,
                                hard_anchor,
                                hard_positive,
                                hard_negative,
                                causal_hard_negative_margin,
                            )
                        )
                        causal_contrastive_loss = (
                            causal_info_nce_loss
                            + causal_hard_negative_weight
                            * causal_hard_negative_loss
                        )
                    else:
                        causal_info_nce_loss = rank_loss.new_zeros(())
                        causal_hard_negative_loss = rank_loss.new_zeros(())
                        causal_contrastive_loss = rank_loss.new_zeros(())
            else:
                causal_info_nce_loss = rank_loss.new_zeros(())
                causal_hard_negative_loss = rank_loss.new_zeros(())
                causal_contrastive_loss = rank_loss.new_zeros(())
            if causal_cpu_rng_state is not None:
                torch.random.set_rng_state(causal_cpu_rng_state)
                if causal_cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(causal_cuda_rng_state)
            active_weight = (
                lambda_triplet if triplet_enabled and epoch >= warmup else 0.0
            )
            active_calibration_weight = (
                lambda_calibration
                if calibration_enabled and epoch >= calibration_warmup
                else 0.0
            )
            active_within_unit_weight = (
                lambda_within_unit
                if within_unit_enabled and epoch >= within_unit_warmup
                else 0.0
            )
            active_causal_contrastive_weight = (
                _ramped_auxiliary_weight(
                    epoch,
                    lambda_causal_contrastive,
                    causal_warmup,
                    causal_ramp,
                )
                if causal_contrastive_enabled else 0.0
            )
            if (
                causal_contrastive_enabled
                and active_causal_contrastive_weight > 0.0
                and not gradient_norm_measured
                and causal_contrastive_loss.requires_grad
            ):
                contrastive_parameters = tuple(
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("score_head")
                )
                gradients = torch.autograd.grad(
                    active_causal_contrastive_weight
                    * causal_contrastive_loss,
                    contrastive_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                squared_norm = sum(
                    float(torch.sum(gradient.detach() ** 2))
                    for gradient in gradients
                    if gradient is not None
                )
                epoch_contrastive_gradient_norm = squared_norm ** 0.5
                gradient_norm_measured = True
            total = (
                rank_loss
                + active_weight * triplet_loss
                + active_causal_contrastive_weight * causal_contrastive_loss
                + active_calibration_weight * calibration_loss
                + active_within_unit_weight * within_unit_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if causal_contrastive_enabled:
                epoch_causal_losses.append(
                    float(causal_contrastive_loss.detach())
                )
                epoch_causal_info_nce_losses.append(
                    float(causal_info_nce_loss.detach())
                )
                epoch_causal_hard_losses.append(
                    float(causal_hard_negative_loss.detach())
                )

        model.eval()
        with torch.no_grad():
            train_scores, train_projection = model(train_x, train_t)
            val_scores, _ = model(validation_x, validation_t)
            train_loss = float(_pair_loss(
                train_scores, train_high, train_low, train_weight
            ))
            val_loss = float(_pair_loss(val_scores, val_high, val_low, val_weight))
            full_triplet = (
                float(_triplet_loss(
                    train_projection,
                    triplet_anchor,
                    triplet_positive,
                    triplet_negative,
                    margin,
                ))
                if triplet_enabled else 0.0
            )
            validation_metrics = _validation_metrics(
                val_scores.cpu().numpy(),
                validation_signal,
                validation_pairs,
                settings,
            )
            val_calibration_rmse = float(np.sqrt(np.mean(
                (
                    val_scores.cpu().numpy()
                    - validation_calibration_target
                ) ** 2
            )))
            train_calibration_loss = (
                float(_weighted_smooth_l1_loss(
                    train_scores,
                    calibration_target,
                    calibration_weight,
                    calibration_beta,
                ))
                if calibration_enabled else 0.0
            )
            full_within_unit_loss = (
                float(_pair_loss(
                    train_scores,
                    within_high,
                    within_low,
                    within_weight,
                ))
                if within_unit_enabled else 0.0
            )
        history_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_pairwise_loss": train_loss,
            "validation_pairwise_loss": val_loss,
            "triplet_loss": full_triplet,
            "effective_triplet_weight": (
                0.0 if (not triplet_enabled or epoch < warmup) else lambda_triplet
            ),
            "causal_contrastive_mode": causal_mode,
            "causal_contrastive_loss": (
                float(np.mean(epoch_causal_losses))
                if epoch_causal_losses
                else 0.0
            ),
            "causal_info_nce_loss": (
                float(np.mean(epoch_causal_info_nce_losses))
                if epoch_causal_info_nce_losses
                else 0.0
            ),
            "causal_hard_negative_loss": (
                float(np.mean(epoch_causal_hard_losses))
                if epoch_causal_hard_losses
                else 0.0
            ),
            "effective_causal_contrastive_weight": (
                _ramped_auxiliary_weight(
                    epoch,
                    lambda_causal_contrastive,
                    causal_warmup,
                    causal_ramp,
                )
                if causal_contrastive_enabled
                else 0.0
            ),
            "valid_anchor_count": (
                int(len(causal_contrastive_index.eligible_anchors))
                if causal_contrastive_index is not None
                else 0
            ),
            "positive_pair_count": (
                int(causal_contrastive_index.audit["positive_pair_count"])
                if causal_contrastive_index is not None
                else 0
            ),
            "clear_negative_count": (
                int(causal_contrastive_index.audit["clear_negative_pair_count"])
                if causal_contrastive_index is not None
                else 0
            ),
            "ambiguous_pair_count": (
                int(causal_contrastive_index.audit["ambiguous_pair_count"])
                if causal_contrastive_index is not None
                else 0
            ),
            "sampled_uact_triplets": int(epoch_sampled_triplets),
            "same_patient_negative_count": int(
                epoch_same_patient_negatives
            ),
            "global_negative_count": int(epoch_global_negatives),
            "mean_positive_effect_gap": (
                float(np.mean(epoch_positive_gaps))
                if epoch_positive_gaps
                else 0.0
            ),
            "mean_negative_effect_gap": (
                float(np.mean(epoch_negative_gaps))
                if epoch_negative_gaps
                else 0.0
            ),
            "mean_triplet_reliability": (
                float(np.mean(epoch_triplet_reliability))
                if epoch_triplet_reliability
                else 0.0
            ),
            "contrastive_gradient_norm": float(
                epoch_contrastive_gradient_norm
            ),
            "calibration_loss": train_calibration_loss,
            "validation_calibration_rmse": val_calibration_rmse,
            "effective_calibration_weight": (
                0.0 if epoch < calibration_warmup else lambda_calibration
            ),
            "within_unit_auxiliary_loss": full_within_unit_loss,
            "effective_within_unit_weight": (
                0.0 if epoch < within_unit_warmup else lambda_within_unit
            ),
            **validation_metrics,
        }
        history.append(history_row)
        if epoch >= earliest_checkpoint_epoch:
            metric = validation_metrics["selection_metric"]
            if metric > best_metric + 1e-7:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                remaining = patience
            else:
                remaining -= 1
                if remaining <= 0:
                    break

    if best_epoch < earliest_checkpoint_epoch:
        best_epoch = int(history[-1]["epoch"])
        best_metric = float(history[-1]["selection_metric"])
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    active_history = [
        row
        for row in history
        if float(row.get("effective_causal_contrastive_weight", 0.0)) > 0.0
    ]
    total_sampled_uact_triplets = int(
        sum(int(row.get("sampled_uact_triplets", 0)) for row in history)
    )
    total_same_patient_negatives = int(
        sum(int(row.get("same_patient_negative_count", 0)) for row in history)
    )
    maximum_contrastive_gradient_norm = float(
        max(
            (float(row.get("contrastive_gradient_norm", 0.0)) for row in history),
            default=0.0,
        )
    )
    if causal_strict_diagnostics and causal_contrastive_enabled:
        if not active_history:
            raise RuntimeError(
                "Contrastive training was enabled but never received a "
                "positive optimization weight."
            )
        if (
            causal_mode == "uncertainty_aware_ordinal_triplet"
            and total_sampled_uact_triplets == 0
        ):
            raise RuntimeError(
                "UACT was enabled but no valid triplets were sampled."
            )
        if (
            causal_mode == "uncertainty_aware_ordinal_triplet"
            and causal_require_same_patient
            and total_same_patient_negatives == 0
        ):
            raise RuntimeError(
                "Same-patient hard negatives were required but never sampled."
            )
        if maximum_contrastive_gradient_norm <= 0.0:
            raise RuntimeError(
                "The contrastive objective produced no measurable encoder "
                "gradient."
            )

    pair_counts = (
        train_pairs.pair_type.value_counts().to_dict()
        if "pair_type" in train_pairs else {"unspecified": int(len(train_pairs))}
    )
    return ContrastiveCausalRanker(
        processor=processor,
        model=model,
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        treatment_mapping=treatment_mapping,
        audit={
            "architecture_family": (
                "contrastively_pretrained_film_treatment_conditioned_siamese_ranker"
                if contrastive_pretraining_enabled
                else "film_treatment_conditioned_siamese_contrastive_ranker"
            ),
            "contrastive_pretraining_enabled": contrastive_pretraining_enabled,
            "contrastive_pretraining": pretraining_audit,
            "ranking_finetuning_freeze_encoder_epochs": freeze_encoder_epochs,
            "ranking_finetuning_freeze_encoder_entire": freeze_encoder_entire_finetuning,
            "ranking_finetuning_encoder_learning_rate": encoder_learning_rate,
            "ranking_finetuning_head_learning_rate": head_learning_rate,
            "ranking_finetuning_stochastic_seed": finetuning_stochastic_seed,
            "contrastive_objective": causal_mode,
            "primary_objective": "direct_global_ordinal_causal_ranking",
            "contrastive_role": (
                "uncertainty_aware_responder_guided_ordinal_triplets"
                if causal_mode == "uncertainty_aware_ordinal_triplet"
                else "responder_aware_info_nce_ablation"
            ),
            "score_semantics": "global_ordinal_priority_not_individual_cate",
            "oracle_inputs_used": False,
            "individual_cate_estimated_then_sorted": False,
            "early_stopping_target": "heldout_composite_ranking_and_topk_dr_value",
            "model_seed": seed,
            "encoded_feature_count": int(train_x.shape[1]),
            "treatments": treatment_mapping,
            "train_pairs": int(len(train_pairs)),
            "train_pair_types": {str(k): int(v) for k, v in pair_counts.items()},
            "validation_pairs": int(len(validation_pairs)),
            "triplets": int(len(triplet_anchor)),
            "triplet_enabled": triplet_enabled,
            "triplet_weight": lambda_triplet,
            "triplet_margin": margin,
            "causal_contrastive_enabled": causal_contrastive_enabled,
            "causal_contrastive_mode": causal_mode,
            "causal_contrastive_weight": lambda_causal_contrastive,
            "causal_contrastive_temperature": causal_temperature,
            "causal_contrastive_margin": causal_margin,
            "causal_contrastive_warmup_epochs": causal_warmup,
            "causal_contrastive_ramp_epochs": causal_ramp,
            "causal_contrastive_minimum_post_ramp_epochs": (
                causal_minimum_post_ramp
            ),
            "causal_contrastive_uncertainty_weighting": (
                causal_uncertainty_weighting
            ),
            "causal_contrastive_same_patient_hard_negatives": (
                causal_same_patient_enabled
            ),
            "causal_contrastive_same_patient_probability": (
                causal_same_patient_probability
            ),
            "causal_contrastive_require_same_patient": (
                causal_require_same_patient
            ),
            "causal_contrastive_hard_negative_fraction": (
                causal_hard_negative_fraction
            ),
            "causal_contrastive_hard_negative_margin": (
                causal_hard_negative_margin
            ),
            "causal_contrastive_hard_negative_weight": (
                causal_hard_negative_weight
            ),
            "causal_contrastive_active_epochs": int(len(active_history)),
            "causal_contrastive_total_sampled_triplets": (
                total_sampled_uact_triplets
            ),
            "causal_contrastive_total_same_patient_negatives": (
                total_same_patient_negatives
            ),
            "causal_contrastive_max_gradient_norm": (
                maximum_contrastive_gradient_norm
            ),
            "causal_contrastive_index": (
                causal_contrastive_index.audit
                if causal_contrastive_index is not None else {}
            ),
            "calibration_enabled": calibration_enabled,
            "calibration_weight": lambda_calibration,
            "calibration_huber_beta": calibration_beta,
            "calibration_target_mean": calibration_mean,
            "calibration_target_scale": calibration_scale,
            "pair_type_loss_multipliers": pair_type_loss_multipliers,
            "within_unit_auxiliary_enabled": within_unit_enabled,
            "within_unit_auxiliary_weight": lambda_within_unit,
            "within_unit_auxiliary_pairs": int(len(within_unit_pairs)),
            "positive_repeat_distance_quantile": (
                positive_repeat_distance_quantile
            ),
            "contrastive_warmup_epochs": warmup,
            "minimum_contrastive_epochs": minimum_active,
            "minimum_training_epochs": minimum_training_epochs,
            "earliest_checkpoint_epoch": earliest_checkpoint_epoch,
            "checkpoint_received_contrastive_updates": bool(
                (triplet_enabled or causal_contrastive_enabled)
                and best_epoch >= earliest_checkpoint_epoch
            ),
            "best_epoch": best_epoch,
            "best_validation_selection_metric": best_metric,
            "best_validation_metrics": history[best_epoch] if best_epoch < len(history) else history[-1],
            "history": history,
        },
    )
