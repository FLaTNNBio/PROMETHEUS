"""Direct causal ranking over supported patient-care-profile opportunities."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F


PROFILE_RANKING_VARIANTS = (
    "independent_profile_rankers",
    "global_rank_only",
    "global_rank_plus_contrastive",
    "direct_pairwise_gbdt",
    "linear_pairwise_additive",
    "linear_pairwise_interactions",
    "profile_mean_dr",
    "prespecified_stratum_dr",
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
    sampling_backend: str = "exhaustive"
    sampling_attempts: int = 0
    requested_pairs: int = 0
    quota_satisfied: bool = True
    estimated_peak_candidate_bytes: int = 0

    def __len__(self) -> int:
        return len(self.left)


@dataclass(frozen=True)
class SimilarityBatch:
    left: np.ndarray
    right: np.ndarray
    similar: np.ndarray
    weight: np.ndarray
    is_cross_profile: np.ndarray
    sampling_backend: str = "exhaustive"
    sampling_attempts: int = 0
    requested_pairs: int = 0
    quota_satisfied: bool = True
    estimated_peak_candidate_bytes: int = 0
    robust_scale: float = float("nan")

    def __len__(self) -> int:
        return len(self.left)


@dataclass(frozen=True)
class OrderedTripletBatch:
    """Anchor/near/far constraints derived from repeated learner-safe DR signals."""

    anchor: np.ndarray
    near: np.ndarray
    far: np.ndarray
    weight: np.ndarray
    is_cross_profile: np.ndarray
    distance_gap: np.ndarray
    order_agreement: np.ndarray
    sampling_backend: str = "bounded_ordered_triplet"
    sampling_attempts: int = 0
    requested_pairs: int = 0
    quota_satisfied: bool = True
    estimated_peak_candidate_bytes: int = 0
    robust_scale: float = float("nan")

    def __len__(self) -> int:
        return len(self.anchor)


ContrastiveBatch = SimilarityBatch | OrderedTripletBatch


def _contrastive_batch_hash(batch: ContrastiveBatch) -> str:
    """Stable audit hash for the sampled learner-safe contrastive supervision."""

    digest = hashlib.sha256(type(batch).__name__.encode("utf-8"))
    fields = (
        ("left", "right", "similar", "weight", "is_cross_profile")
        if isinstance(batch, SimilarityBatch)
        else (
            "anchor", "near", "far", "weight", "is_cross_profile",
            "distance_gap", "order_agreement",
        )
    )
    for field in fields:
        values = np.ascontiguousarray(getattr(batch, field))
        digest.update(field.encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


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


def _empty_ordered_triplets() -> OrderedTripletBatch:
    return OrderedTripletBatch(
        anchor=np.array([], dtype=np.int64),
        near=np.array([], dtype=np.int64),
        far=np.array([], dtype=np.int64),
        weight=np.array([], dtype=float),
        is_cross_profile=np.array([], dtype=bool),
        distance_gap=np.array([], dtype=float),
        order_agreement=np.array([], dtype=float),
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


def _validate_pair_sampling_backend(backend: str) -> str:
    value = str(backend)
    if value not in {"exhaustive", "stratified_streaming"}:
        raise ValueError(
            "pair_sampling_backend must be 'exhaustive' or "
            "'stratified_streaming'"
        )
    return value


def _stream_profile_pairs(
    frame: pd.DataFrame,
    *,
    pairs: int,
    seed: int,
    minimum_signal_gap: float,
    minimum_direction_agreement: float,
    within_profile_fraction: float,
    allow_cross_profile: bool,
    allow_same_patient_cross_profile: bool,
    maximum_pair_weight: float,
    maximum_attempts: int,
) -> PairBatch:
    """Draw rank pairs without materializing the quadratic pair universe."""

    signal = frame.dr_pseudo_outcome.to_numpy(float)
    repeated = frame.loc[:, _repeat_columns(frame)].to_numpy(float)
    reliability = frame.dr_reliability_weight.to_numpy(float)
    profiles = frame.care_profile_id.astype(str).to_numpy()
    patients = frame.patient_id.astype(str).to_numpy()
    domains = (
        frame.ranking_domain.astype(str).to_numpy()
        if "ranking_domain" in frame
        else np.repeat("__configured_common_domain__", len(frame))
    )
    requested = int(pairs)
    targets = {
        False: (
            requested
            if not allow_cross_profile
            else int(round(requested * float(within_profile_fraction)))
        ),
        True: (
            0
            if not allow_cross_profile
            else requested - int(round(requested * float(within_profile_fraction)))
        ),
    }
    selected: dict[bool, list[tuple[int, int, float, float]]] = {
        False: [], True: [],
    }
    seen: set[tuple[int, int]] = set()
    rng = np.random.default_rng(int(seed))
    attempts = 0
    peak_batch = 0
    while sum(map(len, selected.values())) < requested and attempts < maximum_attempts:
        remaining = requested - sum(map(len, selected.values()))
        batch_size = min(max(256, 8 * remaining), 4096, maximum_attempts - attempts)
        if batch_size <= 0:
            break
        peak_batch = max(peak_batch, batch_size)
        raw_left = rng.integers(0, len(frame), size=batch_size)
        raw_right = rng.integers(0, len(frame), size=batch_size)
        attempts += batch_size
        nonself = raw_left != raw_right
        left = np.minimum(raw_left[nonself], raw_right[nonself])
        right = np.maximum(raw_left[nonself], raw_right[nonself])
        if not len(left):
            continue
        signal_difference = signal[left] - signal[right]
        direction = np.sign(signal_difference)
        same_profile = profiles[left] == profiles[right]
        cross = ~same_profile
        valid = np.abs(signal_difference) > float(minimum_signal_gap)
        valid &= domains[left] == domains[right]
        if not allow_cross_profile:
            valid &= same_profile
        elif not allow_same_patient_cross_profile:
            valid &= ~(cross & (patients[left] == patients[right]))
        repeat_difference = repeated[left] - repeated[right]
        agreement = np.mean(
            repeat_difference * direction[:, None] > 0.0, axis=1
        )
        valid &= agreement >= float(minimum_direction_agreement)
        for position in np.flatnonzero(valid):
            is_cross = bool(cross[position])
            if len(selected[is_cross]) >= targets[is_cross]:
                continue
            key = (int(left[position]), int(right[position]))
            if key in seen:
                continue
            seen.add(key)
            selected[is_cross].append((
                key[0], key[1], float(direction[position]),
                float(agreement[position]),
            ))
            if sum(map(len, selected.values())) >= requested:
                break
    counts = {key: len(value) for key, value in selected.items()}
    if counts != targets:
        raise ValueError(
            "stratified_streaming ranking sampler could not satisfy quotas "
            f"within={counts[False]}/{targets[False]}, "
            f"cross={counts[True]}/{targets[True]} after {attempts} attempts"
        )
    rows = selected[False] + selected[True]
    order = rng.permutation(len(rows))
    left = np.asarray([rows[index][0] for index in order], dtype=np.int64)
    right = np.asarray([rows[index][1] for index in order], dtype=np.int64)
    direction = np.asarray([rows[index][2] for index in order], dtype=float)
    agreement = np.asarray([rows[index][3] for index in order], dtype=float)
    cross = profiles[left] != profiles[right]
    weight = np.sqrt(reliability[left] * reliability[right]) * agreement
    if not np.isfinite(weight).all() or (weight < 0.0).any():
        raise ValueError("Streaming ranking pair weights must be finite and non-negative")
    weight = np.clip(weight, 1e-8, float(maximum_pair_weight))
    mean_weight = float(weight.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 1e-12:
        raise ValueError("Streaming ranking pair weights have a degenerate mean")
    weight /= mean_weight
    return PairBatch(
        left=left, right=right, direction=direction, weight=weight.astype(float),
        is_cross_profile=cross.astype(bool), direction_agreement=agreement,
        sampling_backend="stratified_streaming", sampling_attempts=int(attempts),
        requested_pairs=requested, quota_satisfied=True,
        estimated_peak_candidate_bytes=int(peak_batch * 14 * 8),
    )


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
    pair_sampling_backend: str = "exhaustive",
    pair_sampling_maximum_attempts: int | None = None,
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
    if (reliability < 0.0).any():
        raise ValueError("Ranking-pair reliability must be non-negative")
    backend = _validate_pair_sampling_backend(pair_sampling_backend)
    if backend == "stratified_streaming":
        maximum_attempts = int(
            pair_sampling_maximum_attempts
            if pair_sampling_maximum_attempts is not None
            else max(10_000, int(pairs) * 200)
        )
        if maximum_attempts < int(pairs):
            raise ValueError("pair_sampling_maximum_attempts must cover requested pairs")
        return _stream_profile_pairs(
            frame, pairs=int(pairs), seed=int(seed),
            minimum_signal_gap=float(minimum_signal_gap),
            minimum_direction_agreement=float(minimum_direction_agreement),
            within_profile_fraction=float(within_profile_fraction),
            allow_cross_profile=bool(allow_cross_profile),
            allow_same_patient_cross_profile=bool(
                allow_same_patient_cross_profile
            ),
            maximum_pair_weight=float(maximum_pair_weight),
            maximum_attempts=maximum_attempts,
        )
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
        sampling_backend="exhaustive",
        sampling_attempts=int(len(left)),
        requested_pairs=int(pairs),
        quota_satisfied=bool(len(selected) == min(int(pairs), len(candidates))),
        estimated_peak_candidate_bytes=int(len(left) * 14 * 8),
    )


def _robust_scale(values: np.ndarray) -> float:
    """Return a finite positive MAD-first scale or reject degenerate input."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Robust scale requires at least two one-dimensional values")
    if not np.isfinite(values).all():
        raise ValueError("Robust scale requires finite values")
    median = float(np.median(values))
    mad_scale = float(1.4826 * np.median(np.abs(values - median)))
    if mad_scale > 1e-8:
        return mad_scale
    standard_scale = float(np.std(values))
    if not np.isfinite(standard_scale) or standard_scale <= 1e-8:
        raise ValueError("Robust scale is degenerate (MAD and SD are near zero)")
    return standard_scale


def _sample_ordered_contrastive_triplets(
    frame: pd.DataFrame,
    *,
    triplets: int,
    seed: int,
    minimum_order_agreement: float,
    minimum_distance_gap: float,
    within_profile_fraction: float,
    allow_same_patient_cross_profile: bool,
    use_reliability_weight: bool,
    maximum_pair_weight: float,
    maximum_attempts: int | None = None,
    max_reuse_per_opportunity: int | None = None,
) -> OrderedTripletBatch:
    """Sample stable response-distance orderings without binning DR signals.

    The anchor, near and far roles are determined only from repeated cross-fitted DR
    supervision on the rank-training split. Synthetic truth is neither accepted nor
    inspected. Within-profile and cross-profile triplets use separate robust scales.
    """

    repeat_columns = _repeat_columns(frame)
    required = {
        "patient_id", "care_profile_id", "dr_pseudo_outcome",
        "dr_reliability_weight",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or not repeat_columns:
        raise ValueError(
            f"Ordered contrastive triplets require repeated DR supervision: {missing}"
        )
    if int(triplets) < 1:
        raise ValueError("Ordered contrastive triplet count must be positive")
    if not 0.0 <= float(minimum_order_agreement) <= 1.0:
        raise ValueError("minimum_order_agreement must be in [0, 1]")
    if float(minimum_distance_gap) < 0.0:
        raise ValueError("minimum_distance_gap must be non-negative")
    if not 0.0 <= float(within_profile_fraction) <= 1.0:
        raise ValueError("ordered within-profile fraction must be in [0, 1]")
    if float(maximum_pair_weight) <= 0.0:
        raise ValueError("ordered contrastive maximum weight must be positive")
    if maximum_attempts is not None and int(maximum_attempts) < int(triplets):
        raise ValueError("ordered triplet attempt budget must cover requested triplets")
    if (
        max_reuse_per_opportunity is not None
        and int(max_reuse_per_opportunity) < 1
    ):
        raise ValueError("ordered triplet maximum reuse must be positive")
    if len(frame) < 3:
        return _empty_ordered_triplets()

    repeated = frame.loc[:, repeat_columns].to_numpy(float)
    reliability = frame.dr_reliability_weight.to_numpy(float)
    if not np.isfinite(repeated).all() or not np.isfinite(reliability).all():
        raise ValueError("Ordered contrastive supervision must be finite")
    if (reliability < 0.0).any():
        raise ValueError("Ordered contrastive reliability must be non-negative")
    profiles = frame.care_profile_id.astype(str).to_numpy()
    patients = frame.patient_id.astype(str).to_numpy()
    domains = (
        frame.ranking_domain.astype(str).to_numpy()
        if "ranking_domain" in frame.columns
        else np.full(len(frame), "configured", dtype=object)
    )
    centers = np.median(repeated, axis=1)
    global_scale = _robust_scale(centers)
    profile_scales = {
        profile: _robust_scale(centers[profiles == profile])
        for profile in np.unique(profiles)
    }
    rng = np.random.default_rng(int(seed))
    requested = int(triplets)
    within_target = int(round(requested * float(within_profile_fraction)))
    targets = ((False, within_target), (True, requested - within_target))
    selected: list[tuple[int, int, int, bool, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    reuse = np.zeros(len(frame), dtype=np.int64)
    total_attempts = 0

    for cross_context, target in targets:
        accepted = 0
        attempts = 0
        category_attempt_budget = (
            max(2_000, target * 200)
            if maximum_attempts is None
            else max(
                int(target),
                int(np.ceil(int(maximum_attempts) * target / max(requested, 1))),
            )
        )
        while accepted < target and attempts < category_attempt_budget:
            attempts += 1
            total_attempts += 1
            anchor = int(rng.integers(0, len(frame)))
            if cross_context:
                eligible = (
                    (profiles != profiles[anchor])
                    & (domains == domains[anchor])
                )
                if not allow_same_patient_cross_profile:
                    eligible &= patients != patients[anchor]
                scale = global_scale
            else:
                eligible = (
                    (profiles == profiles[anchor])
                    & (domains == domains[anchor])
                )
                eligible[anchor] = False
                scale = profile_scales[str(profiles[anchor])]
            candidates = np.flatnonzero(eligible)
            if len(candidates) < 2:
                continue
            first, second = map(
                int, rng.choice(candidates, size=2, replace=False).tolist()
            )
            first_distance = np.abs(repeated[anchor] - repeated[first]) / scale
            second_distance = np.abs(repeated[anchor] - repeated[second]) / scale
            first_center = float(np.median(first_distance))
            second_center = float(np.median(second_distance))
            if first_center <= second_center:
                near, far = first, second
                near_distance, far_distance = first_distance, second_distance
                near_center, far_center = first_center, second_center
            else:
                near, far = second, first
                near_distance, far_distance = second_distance, first_distance
                near_center, far_center = second_center, first_center
            gap = float(far_center - near_center)
            agreement = float(np.mean(near_distance < far_distance))
            key = (anchor, near, far)
            if (
                gap < float(minimum_distance_gap)
                or agreement < float(minimum_order_agreement)
                or key in seen
                or (
                    max_reuse_per_opportunity is not None
                    and (
                        reuse[anchor] >= int(max_reuse_per_opportunity)
                        or reuse[near] >= int(max_reuse_per_opportunity)
                        or reuse[far] >= int(max_reuse_per_opportunity)
                    )
                )
            ):
                continue
            seen.add(key)
            reuse[anchor] += 1
            reuse[near] += 1
            reuse[far] += 1
            selected.append((anchor, near, far, cross_context, gap, agreement))
            accepted += 1

        if accepted != int(target):
            context = "cross-profile" if cross_context else "within-profile"
            raise ValueError(
                "bounded ordered triplet sampler could not satisfy the "
                f"{context} quota: observed={accepted}, requested={target}, "
                f"attempts={attempts}"
            )

    if not selected:
        return _empty_ordered_triplets()
    rng.shuffle(selected)
    anchor = np.asarray([item[0] for item in selected], dtype=np.int64)
    near = np.asarray([item[1] for item in selected], dtype=np.int64)
    far = np.asarray([item[2] for item in selected], dtype=np.int64)
    cross = np.asarray([item[3] for item in selected], dtype=bool)
    gap = np.asarray([item[4] for item in selected], dtype=float)
    agreement = np.asarray([item[5] for item in selected], dtype=float)
    weight = agreement.copy()
    if use_reliability_weight:
        weight *= np.cbrt(
            reliability[anchor] * reliability[near] * reliability[far]
        )
    weight = np.clip(weight, 1e-8, float(maximum_pair_weight))
    weight /= max(float(weight.mean()), 1e-8)
    return OrderedTripletBatch(
        anchor=anchor,
        near=near,
        far=far,
        weight=weight.astype(float),
        is_cross_profile=cross,
        distance_gap=gap,
        order_agreement=agreement,
        sampling_backend="bounded_ordered_triplet",
        sampling_attempts=int(total_attempts),
        requested_pairs=int(requested),
        quota_satisfied=bool(len(selected) == requested),
        estimated_peak_candidate_bytes=int(len(frame) * 8 * 12),
        robust_scale=float(global_scale),
    )


def _sample_contrastive_pairs(
    frame: pd.DataFrame,
    *,
    pairs: int,
    bins: int,
    seed: int,
    minimum_label_agreement: float,
    allow_same_patient_cross_profile: bool,
    within_profile_fraction: float | None = None,
    positive_fraction: float = 0.50,
    use_reliability_weight: bool = False,
    maximum_pair_weight: float = 5.0,
    target_mode: str = "hard_bins",
    soft_bandwidth: float = 1.0,
    soft_positive_threshold: float = 0.70,
    soft_negative_threshold: float = 0.30,
    uncertainty_penalty: float = 1.0,
    opposite_treatment_arms_only: bool = False,
    pair_sampling_backend: str = "exhaustive",
    pair_sampling_maximum_attempts: int | None = None,
    max_reuse_per_opportunity: int | None = None,
    minimum_profile_samples: int = 2,
) -> SimilarityBatch:
    if int(pairs) < 1 or int(bins) < 3 or len(frame) < 2:
        return _empty_similarity()
    if within_profile_fraction is not None and not 0.0 <= float(
        within_profile_fraction
    ) <= 1.0:
        raise ValueError("contrastive_within_profile_fraction must be in [0, 1]")
    if not 0.0 <= float(positive_fraction) <= 1.0:
        raise ValueError("contrastive_positive_fraction must be in [0, 1]")
    if float(maximum_pair_weight) <= 0.0:
        raise ValueError("contrastive maximum pair weight must be positive")
    if str(target_mode) not in {
        "hard_bins",
        "soft_repeat_distance",
        "profile_residual_soft_distance",
    }:
        raise ValueError(f"Unknown contrastive_target_mode: {target_mode}")
    if float(soft_bandwidth) <= 0.0 or float(uncertainty_penalty) < 0.0:
        raise ValueError("Soft contrastive bandwidth must be positive and penalty non-negative")
    if not 0.0 <= float(soft_negative_threshold) < float(
        soft_positive_threshold
    ) <= 1.0:
        raise ValueError("Soft contrastive thresholds must satisfy 0 <= negative < positive <= 1")
    required = {
        "patient_id", "care_profile_id", "dr_pseudo_outcome",
        "dr_reliability_weight",
    }
    repeat_columns = _repeat_columns(frame)
    missing = sorted(required.difference(frame.columns))
    if missing or not repeat_columns:
        raise ValueError(
            f"Contrastive pairs require repeated DR supervision: missing={missing}"
        )
    if int(minimum_profile_samples) < 2:
        raise ValueError("contrastive_minimum_profile_samples must be at least 2")
    if max_reuse_per_opportunity is not None and int(max_reuse_per_opportunity) < 1:
        raise ValueError("contrastive_max_reuse_per_opportunity must be positive")
    backend = _validate_pair_sampling_backend(pair_sampling_backend)
    if max_reuse_per_opportunity is not None and backend != "stratified_streaming":
        raise ValueError(
            "contrastive_max_reuse_per_opportunity requires stratified_streaming"
        )
    signal = frame.dr_pseudo_outcome.to_numpy(float)
    repeated = frame.loc[:, repeat_columns].to_numpy(float)
    reliability = frame.dr_reliability_weight.to_numpy(float)
    patients = frame.patient_id.astype(str).to_numpy()
    profiles = frame.care_profile_id.astype(str).to_numpy()
    domains = (
        frame.ranking_domain.astype(str).to_numpy()
        if "ranking_domain" in frame
        else np.repeat("__configured_common_domain__", len(frame))
    )
    if not np.isfinite(signal).all():
        raise ValueError("Contrastive dr_pseudo_outcome must be finite")
    if not np.isfinite(repeated).all():
        raise ValueError("Repeated DR pseudo-outcomes must be finite")
    if not np.isfinite(reliability).all() or (reliability < 0.0).any():
        raise ValueError(
            "Contrastive reliability weights must be finite and non-negative"
        )
    treatment = None
    if opposite_treatment_arms_only:
        if "profile_treatment" not in frame:
            raise ValueError(
                "Opposite-arm contrastive pairing requires profile_treatment metadata"
            )
        treatment = pd.to_numeric(
            frame.profile_treatment, errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(treatment).all() or not set(np.unique(treatment)).issubset(
            {0.0, 1.0}
        ):
            raise ValueError("profile_treatment must be finite and binary")
    response_repetitions = repeated.copy()
    robust_scale = 0.0
    boundaries = np.array([], dtype=float)
    labels = np.array([], dtype=int)
    repeated_labels = np.empty((len(frame), 0), dtype=int)
    if str(target_mode) == "hard_bins":
        boundaries = np.unique(
            np.quantile(signal, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1])
        )
        if len(boundaries) < 2:
            return _empty_similarity()
        labels = np.digitize(signal, boundaries)
        repeated_labels = np.stack(
            [
                np.digitize(repeated[:, repeat], boundaries)
                for repeat in range(repeated.shape[1])
            ],
            axis=1,
        )
    else:
        if str(target_mode) == "profile_residual_soft_distance":
            # Remove profile-level location before defining response neighborhoods.
            # The ranking target remains absolute and globally comparable.
            for profile in np.unique(profiles):
                mask = profiles == profile
                if int(mask.sum()) < int(minimum_profile_samples):
                    raise ValueError(
                        "Profile-residual contrastive targets require at least "
                        f"{minimum_profile_samples} rows per profile; {profile!r} "
                        f"has {int(mask.sum())}"
                    )
                response_repetitions[mask] -= np.median(
                    response_repetitions[mask], axis=0, keepdims=True
                )
        repeated_center = np.median(response_repetitions, axis=1)
        robust_scale = _robust_scale(repeated_center)

    def evaluate_candidates(left: np.ndarray, right: np.ndarray):
        cross = profiles[left] != profiles[right]
        valid = domains[left] == domains[right]
        if not allow_same_patient_cross_profile:
            valid &= ~(cross & (patients[left] == patients[right]))
        if opposite_treatment_arms_only:
            # Treatment assignment is eligibility metadata, never an encoder input.
            valid &= ~cross & (treatment[left] != treatment[right])
        if str(target_mode) == "hard_bins":
            label_gap = np.abs(labels[left] - labels[right])
            positive = label_gap == 0
            negative = label_gap >= 2
            similarity_target = positive.astype(float)
            positive_agreement = np.mean(
                repeated_labels[left] == repeated_labels[right], axis=1
            )
            negative_agreement = np.mean(
                np.abs(repeated_labels[left] - repeated_labels[right]) >= 2,
                axis=1,
            )
        else:
            absolute_repeat_difference = np.abs(
                response_repetitions[left] - response_repetitions[right]
            )
            median_difference = np.median(absolute_repeat_difference, axis=1)
            normalized_distance = median_difference / robust_scale
            if not np.isfinite(normalized_distance).all():
                raise ValueError("Normalized contrastive distances must be finite")
            similarity_target = np.exp(
                -0.5 * np.square(normalized_distance / float(soft_bandwidth))
            )
            repeat_uncertainty = np.median(
                np.abs(
                    absolute_repeat_difference - median_difference[:, None]
                ),
                axis=1,
            ) / robust_scale
            if not np.isfinite(repeat_uncertainty).all():
                raise ValueError("Contrastive repeated-signal uncertainty must be finite")
            confidence = np.exp(
                -float(uncertainty_penalty) * repeat_uncertainty
            )
            positive = similarity_target >= float(soft_positive_threshold)
            negative = similarity_target <= float(soft_negative_threshold)
            positive_agreement = confidence
            negative_agreement = confidence
        if not all(
            np.isfinite(values).all()
            for values in (similarity_target, positive_agreement, negative_agreement)
        ):
            raise ValueError("Contrastive targets and agreement must be finite")
        return (
            cross, valid, positive, negative, similarity_target,
            positive_agreement, negative_agreement,
        )

    requested = int(pairs)
    positive_target_count = int(requested * float(positive_fraction))
    negative_target_count = requested - positive_target_count
    if backend == "stratified_streaming":
        if within_profile_fraction is None:
            targets = {
                (True, None): positive_target_count,
                (False, None): negative_target_count,
            }
        else:
            positive_within = int(round(
                positive_target_count * float(within_profile_fraction)
            ))
            negative_within = int(round(
                negative_target_count * float(within_profile_fraction)
            ))
            targets = {
                (True, False): positive_within,
                (True, True): positive_target_count - positive_within,
                (False, False): negative_within,
                (False, True): negative_target_count - negative_within,
            }
        selected: dict[tuple[bool, bool | None], list[tuple[int, int, float, float]]] = {
            key: [] for key in targets
        }
        seen: set[tuple[int, int]] = set()
        reuse = np.zeros(len(frame), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        attempts = 0
        peak_batch = 0
        maximum_attempts = int(
            pair_sampling_maximum_attempts
            if pair_sampling_maximum_attempts is not None
            else max(20_000, requested * 400)
        )
        if maximum_attempts < requested:
            raise ValueError(
                "pair_sampling_maximum_attempts must cover contrastive pairs"
            )
        while sum(map(len, selected.values())) < requested and attempts < maximum_attempts:
            remaining = requested - sum(map(len, selected.values()))
            batch_size = min(
                max(512, 12 * remaining), 8192, maximum_attempts - attempts
            )
            if batch_size <= 0:
                break
            peak_batch = max(peak_batch, batch_size)
            raw_left = rng.integers(0, len(frame), size=batch_size)
            raw_right = rng.integers(0, len(frame), size=batch_size)
            attempts += batch_size
            nonself = raw_left != raw_right
            left = np.minimum(raw_left[nonself], raw_right[nonself])
            right = np.maximum(raw_left[nonself], raw_right[nonself])
            if not len(left):
                continue
            (
                cross, valid, positive, negative, similarity_target,
                positive_agreement, negative_agreement,
            ) = evaluate_candidates(left, right)
            agreement = np.where(positive, positive_agreement, negative_agreement)
            valid &= (positive | negative) & (
                agreement >= float(minimum_label_agreement)
            )
            for position in np.flatnonzero(valid):
                is_positive = bool(positive[position])
                category = (
                    is_positive,
                    bool(cross[position])
                    if within_profile_fraction is not None else None,
                )
                if category not in targets or len(selected[category]) >= targets[category]:
                    continue
                pair_key = (int(left[position]), int(right[position]))
                if pair_key in seen:
                    continue
                if max_reuse_per_opportunity is not None and (
                    reuse[pair_key[0]] >= int(max_reuse_per_opportunity)
                    or reuse[pair_key[1]] >= int(max_reuse_per_opportunity)
                ):
                    continue
                seen.add(pair_key)
                reuse[pair_key[0]] += 1
                reuse[pair_key[1]] += 1
                selected[category].append((
                    pair_key[0], pair_key[1],
                    float(similarity_target[position]), float(agreement[position]),
                ))
                if sum(map(len, selected.values())) >= requested:
                    break
        observed = {key: len(value) for key, value in selected.items()}
        if observed != targets:
            raise ValueError(
                "stratified_streaming contrastive sampler could not satisfy "
                f"quotas observed={observed}, requested={targets} after "
                f"{attempts} attempts"
            )
        rows = [row for category in targets for row in selected[category]]
        order = rng.permutation(len(rows))
        left = np.asarray([rows[index][0] for index in order], dtype=np.int64)
        right = np.asarray([rows[index][1] for index in order], dtype=np.int64)
        similar = np.asarray([rows[index][2] for index in order], dtype=float)
        agreement = np.asarray([rows[index][3] for index in order], dtype=float)
        cross = profiles[left] != profiles[right]
        estimated_peak_bytes = int(peak_batch * (16 + 5 * repeated.shape[1]) * 8)
    else:
        left_universe, right_universe = np.triu_indices(len(frame), k=1)
        (
            cross_universe, valid, positive, negative, similarity_target,
            positive_agreement, negative_agreement,
        ) = evaluate_candidates(left_universe, right_universe)
        valid_positive = np.flatnonzero(
            valid & positive & (
                positive_agreement >= float(minimum_label_agreement)
            )
        )
        valid_negative = np.flatnonzero(
            valid & negative & (
                negative_agreement >= float(minimum_label_agreement)
            )
        )
        if not len(valid_positive) and not len(valid_negative):
            return _empty_similarity()
        rng = np.random.default_rng(int(seed))
        if within_profile_fraction is None:
            chosen_positive = (
                rng.choice(
                    valid_positive,
                    min(positive_target_count, len(valid_positive)),
                    replace=False,
                )
                if len(valid_positive) else np.array([], dtype=int)
            )
            chosen_negative = (
                rng.choice(
                    valid_negative,
                    min(requested - len(chosen_positive), len(valid_negative)),
                    replace=False,
                )
                if len(valid_negative) else np.array([], dtype=int)
            )
        else:
            chosen_positive = _sample_balanced_indices(
                valid_positive[~cross_universe[valid_positive]],
                valid_positive[cross_universe[valid_positive]],
                min(positive_target_count, len(valid_positive)),
                float(within_profile_fraction), rng,
            )
            chosen_negative = _sample_balanced_indices(
                valid_negative[~cross_universe[valid_negative]],
                valid_negative[cross_universe[valid_negative]],
                min(requested - len(chosen_positive), len(valid_negative)),
                float(within_profile_fraction), rng,
            )
        chosen = np.concatenate((chosen_positive, chosen_negative)).astype(np.int64)
        rng.shuffle(chosen)
        left = left_universe[chosen].astype(np.int64)
        right = right_universe[chosen].astype(np.int64)
        similar = similarity_target[chosen].astype(float)
        agreement = np.where(
            positive[chosen], positive_agreement[chosen], negative_agreement[chosen]
        )
        cross = cross_universe[chosen].astype(bool)
        attempts = int(len(left_universe))
        estimated_peak_bytes = int(
            len(left_universe) * (16 + 5 * repeated.shape[1]) * 8
        )
    weight = agreement.copy()
    if use_reliability_weight:
        weight *= np.sqrt(reliability[left] * reliability[right])
    if not np.isfinite(weight).all() or (weight < 0.0).any():
        raise ValueError("Final contrastive pair weights must be finite and non-negative")
    weight = np.clip(weight, 1e-8, float(maximum_pair_weight))
    mean_weight = float(weight.mean()) if len(weight) else 0.0
    if not np.isfinite(mean_weight) or mean_weight <= 1e-12:
        raise ValueError("Final contrastive pair weights have a degenerate mean")
    weight /= mean_weight
    if not np.isfinite(similar).all() or not np.isfinite(weight).all():
        raise ValueError("Final contrastive targets and weights must be finite")
    return SimilarityBatch(
        left=left.astype(np.int64),
        right=right.astype(np.int64),
        similar=similar,
        weight=weight.astype(float),
        is_cross_profile=cross.astype(bool),
        sampling_backend=backend,
        sampling_attempts=int(attempts),
        requested_pairs=requested,
        quota_satisfied=bool(len(left) == requested),
        estimated_peak_candidate_bytes=estimated_peak_bytes,
        robust_scale=float(robust_scale),
    )


def _mine_dynamic_contrastive_pairs(
    candidate_pairs: SimilarityBatch,
    representation: np.ndarray,
    *,
    pairs: int,
    seed: int,
    positive_fraction: float,
    within_profile_fraction: float,
) -> SimilarityBatch:
    """Mine hard DR-labeled pairs from the model's current representation.

    Pair labels and weights are fixed learner-safe inputs derived from repeated
    cross-fitted DR signals.  The current representation is used only to choose
    hard positives (still far apart) and hard negatives (still close together);
    it never supplies a treatment-effect label or a score-sorting target.
    """

    values = np.asarray(representation, dtype=float)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Dynamic contrastive mining requires a finite 2D representation")
    if int(pairs) < 1:
        raise ValueError("Dynamic contrastive pair count must be positive")
    if not 0.0 <= float(positive_fraction) <= 1.0:
        raise ValueError("Dynamic contrastive positive fraction must be in [0, 1]")
    if not 0.0 <= float(within_profile_fraction) <= 1.0:
        raise ValueError("Dynamic contrastive within-profile fraction must be in [0, 1]")
    if not len(candidate_pairs):
        return _empty_similarity()
    maximum_index = int(max(candidate_pairs.left.max(), candidate_pairs.right.max()))
    if maximum_index >= len(values):
        raise ValueError("Dynamic contrastive candidates exceed representation rows")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-8)
    cosine = np.sum(
        normalized[candidate_pairs.left] * normalized[candidate_pairs.right],
        axis=1,
    )
    distance = np.clip(1.0 - cosine, 0.0, 2.0)
    positive = candidate_pairs.similar >= 0.5
    within = ~candidate_pairs.is_cross_profile
    requested = min(int(pairs), len(candidate_pairs))
    positive_total = int(round(requested * float(positive_fraction)))
    negative_total = requested - positive_total
    targets = (
        (True, True, int(round(positive_total * float(within_profile_fraction)))),
        (True, False, positive_total - int(round(
            positive_total * float(within_profile_fraction)
        ))),
        (False, True, int(round(negative_total * float(within_profile_fraction)))),
        (False, False, negative_total - int(round(
            negative_total * float(within_profile_fraction)
        ))),
    )
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    used = np.zeros(len(candidate_pairs), dtype=bool)
    for is_positive, is_within, target in targets:
        available = np.flatnonzero(
            (positive == is_positive) & (within == is_within) & ~used
        )
        if not len(available) or target <= 0:
            continue
        # Far positives and close negatives are the most informative violations.
        hardness = distance[available] if is_positive else -distance[available]
        jitter = rng.uniform(0.0, 1e-12, size=len(available))
        order = np.argsort(-(hardness + jitter), kind="stable")
        chosen = available[order[: min(int(target), len(available))]]
        used[chosen] = True
        selected.extend(map(int, chosen))

    if len(selected) < requested:
        remaining = np.flatnonzero(~used)
        if len(remaining):
            hardness = np.where(positive[remaining], distance[remaining], -distance[remaining])
            jitter = rng.uniform(0.0, 1e-12, size=len(remaining))
            order = np.argsort(-(hardness + jitter), kind="stable")
            selected.extend(
                map(int, remaining[order[: requested - len(selected)]])
            )
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    weight = candidate_pairs.weight[selected_array].astype(float)
    weight /= max(float(weight.mean()), 1e-8)
    return SimilarityBatch(
        left=candidate_pairs.left[selected_array].astype(np.int64),
        right=candidate_pairs.right[selected_array].astype(np.int64),
        similar=positive[selected_array].astype(float),
        weight=weight,
        is_cross_profile=candidate_pairs.is_cross_profile[selected_array].astype(bool),
    )


def _similarity_pair_overlap(
    previous: SimilarityBatch, current: SimilarityBatch
) -> float:
    if not len(previous) or not len(current):
        return 0.0
    before = set(zip(previous.left.tolist(), previous.right.tolist()))
    after = set(zip(current.left.tolist(), current.right.tolist()))
    return float(len(before.intersection(after)) / max(len(after), 1))


def _dynamic_refresh_seed(base_seed: int, model_seed: int, epoch: int) -> int:
    state = np.random.SeedSequence(
        int(base_seed), spawn_key=(int(model_seed), int(epoch), 911)
    ).generate_state(1, dtype=np.uint32)
    return 1 + int(state[0]) % 2_000_000_000


def _similarity_arm_diagnostics(
    frame: pd.DataFrame, pairs: SimilarityBatch
) -> dict[str, Any]:
    """Aggregate pair composition without persisting patient-level rows."""

    if "profile_treatment" not in frame or not len(pairs):
        return {
            "arm_metadata_available": "profile_treatment" in frame,
            "opposite_arm_pairs": 0,
            "same_arm_pairs": 0,
            "opposite_arm_fraction": 0.0,
        }
    treatment = pd.to_numeric(frame.profile_treatment, errors="coerce").to_numpy(float)
    if not np.isfinite(treatment).all() or not set(np.unique(treatment)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("profile_treatment must be finite and binary")
    opposite = treatment[pairs.left] != treatment[pairs.right]
    return {
        "arm_metadata_available": True,
        "opposite_arm_pairs": int(opposite.sum()),
        "same_arm_pairs": int((~opposite).sum()),
        "opposite_arm_fraction": float(opposite.mean()),
    }


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
    pairs: ContrastiveBatch,
    margin: float,
    *,
    loss_type: str = "legacy_margin",
    temperature: float = 0.10,
) -> torch.Tensor:
    if not len(pairs):
        return representation.sum() * 0.0
    weight = torch.as_tensor(
        pairs.weight, dtype=representation.dtype, device=representation.device
    )
    if str(loss_type) in {
        "ordered_cosine_triplet", "ordered_cosine_triplet_margin",
    }:
        if not isinstance(pairs, OrderedTripletBatch):
            raise TypeError(f"{loss_type} requires OrderedTripletBatch")
        normalized = F.normalize(representation, p=2.0, dim=1, eps=1e-8)
        anchor = torch.as_tensor(
            pairs.anchor, dtype=torch.long, device=representation.device
        )
        near = torch.as_tensor(
            pairs.near, dtype=torch.long, device=representation.device
        )
        far = torch.as_tensor(
            pairs.far, dtype=torch.long, device=representation.device
        )
        near_similarity = torch.sum(normalized[anchor] * normalized[near], dim=1)
        far_similarity = torch.sum(normalized[anchor] * normalized[far], dim=1)
        cosine_gap = near_similarity - far_similarity
        if str(loss_type) == "ordered_cosine_triplet_margin":
            if not 0.0 < float(margin) <= 2.0:
                raise ValueError("Ordered cosine triplet margin must be in (0, 2]")
            loss = F.relu(float(margin) - cosine_gap)
        else:
            if not 0.0 < float(temperature) <= 1.0:
                raise ValueError("contrastive_temperature must be in (0, 1]")
            loss = F.softplus(-cosine_gap / float(temperature))
        return torch.mean(weight * loss)
    if not isinstance(pairs, SimilarityBatch):
        raise TypeError(f"{loss_type} requires SimilarityBatch")
    left = torch.as_tensor(pairs.left, dtype=torch.long, device=representation.device)
    right = torch.as_tensor(pairs.right, dtype=torch.long, device=representation.device)
    similar = torch.as_tensor(
        pairs.similar, dtype=representation.dtype, device=representation.device
    )
    if str(loss_type) == "legacy_margin":
        distance = torch.linalg.vector_norm(
            representation[left] - representation[right], dim=1
        )
        loss = similar * distance.square() + (1.0 - similar) * F.relu(
            float(margin) - distance
        ).square()
    elif str(loss_type) == "normalized_euclidean_margin":
        if not 0.0 < float(margin) <= 2.0:
            raise ValueError("Normalized Euclidean margin must be in (0, 2]")
        normalized = F.normalize(representation, p=2.0, dim=1, eps=1e-8)
        distance = torch.linalg.vector_norm(
            normalized[left] - normalized[right], dim=1
        )
        loss = similar * distance.square() + (1.0 - similar) * F.relu(
            float(margin) - distance
        ).square()
    elif str(loss_type) == "normalized_cosine_binary":
        if not 0.0 < float(temperature) <= 1.0:
            raise ValueError("contrastive_temperature must be in (0, 1]")
        normalized = F.normalize(representation, p=2.0, dim=1, eps=1e-8)
        cosine = torch.sum(normalized[left] * normalized[right], dim=1)
        loss = F.binary_cross_entropy_with_logits(
            cosine / float(temperature), similar, reduction="none"
        )
    else:
        raise ValueError(f"Unknown contrastive_loss_type: {loss_type}")
    return torch.mean(weight * loss)


def _scheduled_contrastive_weight(
    base_weight: float,
    epoch: int,
    total_epochs: int,
    *,
    warmup_epochs: int,
    ramp_epochs: int,
    final_rank_only_epochs: int,
) -> float:
    """Deterministic auxiliary-loss schedule with ranking-only bookends."""

    values = (warmup_epochs, ramp_epochs, final_rank_only_epochs)
    if any(int(value) < 0 for value in values):
        raise ValueError("Contrastive schedule lengths cannot be negative")
    if sum(map(int, values)) > int(total_epochs):
        raise ValueError("Contrastive schedule exceeds configured epochs")
    if float(base_weight) <= 0.0:
        return 0.0
    if int(epoch) < int(warmup_epochs):
        return 0.0
    if int(final_rank_only_epochs) and int(epoch) >= (
        int(total_epochs) - int(final_rank_only_epochs)
    ):
        return 0.0
    ramp_position = int(epoch) - int(warmup_epochs)
    if int(ramp_epochs) and ramp_position < int(ramp_epochs):
        return float(base_weight) * float(ramp_position + 1) / float(ramp_epochs)
    return float(base_weight)


def _cosine_optimal_similarity(target: float, temperature: float) -> float:
    if not 0.0 < float(target) < 1.0:
        raise ValueError("Soft cosine targets must be strictly between zero and one")
    if not 0.0 < float(temperature) <= 1.0:
        raise ValueError("contrastive_temperature must be in (0, 1]")
    return float(temperature) * float(
        np.log(float(target) / (1.0 - float(target)))
    )


def _validate_cosine_geometry(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate threshold optima against the configured cosine geometry."""

    source = str(
        settings.get("contrastive_representation_source", "projection_head")
    )
    geometry = str(settings.get("contrastive_geometry_source", "legacy"))
    if geometry not in {"legacy", "signed_joint", "linear_adapter"}:
        raise ValueError(
            "contrastive_geometry_source must be 'legacy', 'signed_joint', or "
            "'linear_adapter'"
        )
    explicit = "contrastive_geometry_source" in settings
    if geometry != "legacy" and source != "shared_joint_encoder":
        raise ValueError(
            "Signed contrastive geometry requires "
            "contrastive_representation_source=shared_joint_encoder"
        )
    if geometry != "legacy" and not bool(
        settings.get("use_shared_joint_encoder", False)
    ):
        raise ValueError("Signed contrastive geometry requires the shared joint encoder")
    minimum_cosine = (
        -1.0
        if geometry in {"signed_joint", "linear_adapter"}
        or source == "projection_head"
        else 0.0
    )
    maximum_cosine = 1.0
    temperature = float(settings.get("contrastive_temperature", 0.10))
    negative_target = float(
        settings.get("contrastive_soft_negative_threshold", 0.30)
    )
    positive_target = float(
        settings.get("contrastive_soft_positive_threshold", 0.70)
    )
    negative_optimum = _cosine_optimal_similarity(
        negative_target, temperature
    )
    positive_optimum = _cosine_optimal_similarity(
        positive_target, temperature
    )
    feasible = bool(
        minimum_cosine <= negative_optimum <= maximum_cosine
        and minimum_cosine <= positive_optimum <= maximum_cosine
    )
    strongly_saturated = bool(
        abs(negative_optimum) > 0.95 or abs(positive_optimum) > 0.95
    )
    if explicit and str(settings.get("contrastive_loss_type")) == (
        "normalized_cosine_binary"
    ):
        if not negative_target < 0.5 < positive_target:
            raise ValueError(
                "Cosine BCE thresholds must straddle target probability 0.5"
            )
        if not feasible:
            raise ValueError(
                "Cosine BCE target optima are outside the configured geometry: "
                f"negative={negative_optimum:.6f}, positive={positive_optimum:.6f}, "
                f"geometry=[{minimum_cosine:.1f}, {maximum_cosine:.1f}]"
            )
        if strongly_saturated:
            raise ValueError(
                "Cosine BCE threshold optima are strongly saturated near the "
                "geometry boundary"
            )
    return {
        "geometry_source": geometry,
        "geometry_explicitly_configured": bool(explicit),
        "minimum_realizable_cosine": minimum_cosine,
        "maximum_realizable_cosine": maximum_cosine,
        "negative_threshold_optimal_cosine": negative_optimum,
        "positive_threshold_optimal_cosine": positive_optimum,
        "threshold_optima_feasible": feasible,
        "threshold_optima_strongly_saturated": strongly_saturated,
        "realizable_probability_min": float(
            1.0 / (1.0 + np.exp(-minimum_cosine / temperature))
        ),
        "realizable_probability_max": float(
            1.0 / (1.0 + np.exp(-maximum_cosine / temperature))
        ),
    }


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
    """Profile-conditioned ranker with an optional shared joint bottleneck."""

    def __init__(
        self,
        input_dim: int,
        profile_count: int,
        hidden_dim: int,
        profile_embedding_dim: int,
        projection_dim: int,
        *,
        use_shared_joint_encoder: bool = False,
        joint_representation_dim: int | None = None,
        profile_conditioning: str = "legacy_concat",
        contrastive_geometry_source: str = "legacy",
        use_projection_head: bool = True,
    ):
        super().__init__()
        if profile_conditioning not in {"legacy_concat", "film"}:
            raise ValueError(
                "profile_conditioning must be 'legacy_concat' or 'film'"
            )
        if profile_conditioning == "film" and not use_shared_joint_encoder:
            raise ValueError(
                "FiLM profile conditioning requires use_shared_joint_encoder=true"
            )
        self.use_shared_joint_encoder = bool(use_shared_joint_encoder)
        self.profile_conditioning = str(profile_conditioning)
        self.contrastive_geometry_source = str(contrastive_geometry_source)
        if self.contrastive_geometry_source not in {
            "legacy", "signed_joint", "linear_adapter",
        }:
            raise ValueError("Unknown contrastive_geometry_source")
        if (
            self.contrastive_geometry_source != "legacy"
            and not self.use_shared_joint_encoder
        ):
            raise ValueError("Signed contrastive geometry requires a shared joint encoder")
        if self.use_shared_joint_encoder:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        else:
            # Preserve the frozen legacy architecture exactly for R0 and old bundles.
            self.clinical_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        self.profile_embedding = nn.Embedding(profile_count, profile_embedding_dim)
        self.profile_film = (
            nn.Linear(profile_embedding_dim, 2 * hidden_dim)
            if self.profile_conditioning == "film"
            else None
        )
        joint_input_dim = hidden_dim + profile_embedding_dim
        shared_dim = int(joint_representation_dim or hidden_dim)
        if shared_dim < 1:
            raise ValueError("joint_representation_dim must be positive")
        if self.use_shared_joint_encoder:
            joint_layers: list[nn.Module] = [
                nn.Linear(joint_input_dim, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, shared_dim),
                nn.LayerNorm(shared_dim),
            ]
            if self.contrastive_geometry_source != "signed_joint":
                joint_layers.append(nn.ReLU())
            self.joint_encoder = nn.Sequential(*joint_layers)
            head_input_dim = shared_dim
        else:
            self.joint_encoder = nn.Identity()
            head_input_dim = joint_input_dim
        self.scoring_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contrastive_adapter = (
            nn.Sequential(
                nn.Linear(head_input_dim, head_input_dim),
                nn.LayerNorm(head_input_dim),
            )
            if self.contrastive_geometry_source == "linear_adapter"
            else None
        )
        self.projection_head = (
            (
                nn.Linear(head_input_dim, projection_dim)
                if self.use_shared_joint_encoder
                else nn.Sequential(
                    nn.Linear(head_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, projection_dim),
                )
            )
            if bool(use_projection_head) else None
        )

    def _components(
        self, features: torch.Tensor, profile_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clinical = self.clinical_encoder(features)
        profile = self.profile_embedding(profile_index)
        conditioned = clinical
        if self.profile_film is not None:
            gamma, beta = self.profile_film(profile).chunk(2, dim=1)
            conditioned = clinical * (1.0 + torch.tanh(gamma)) + beta
        joint_input = torch.cat((conditioned, profile), dim=1)
        shared_joint = self.joint_encoder(joint_input)
        return clinical, profile, shared_joint

    def joint_representation(
        self, features: torch.Tensor, profile_index: torch.Tensor
    ) -> torch.Tensor:
        return self._components(features, profile_index)[2]

    def contrastive_representation(
        self, shared_joint: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.contrastive_adapter(shared_joint)
            if self.contrastive_adapter is not None
            else shared_joint
        )

    def forward_components(
        self, features: torch.Tensor, profile_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clinical, _, shared_joint = self._components(features, profile_index)
        projection = (
            self.projection_head(shared_joint)
            if self.projection_head is not None
            else shared_joint.new_empty((len(shared_joint), 0))
        )
        return (
            self.scoring_head(shared_joint).squeeze(1),
            projection,
            clinical,
            shared_joint,
        )

    def forward(
        self, features: torch.Tensor, profile_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score, projection, _, _ = self.forward_components(features, profile_index)
        return score, projection


def _shared_representation_parameters(
    model: _ProfileRankNetwork,
) -> tuple[nn.Parameter, ...]:
    modules: list[nn.Module] = [
        model.clinical_encoder,
        model.profile_embedding,
        model.joint_encoder,
    ]
    if model.profile_film is not None:
        modules.append(model.profile_film)
    return tuple(
        parameter
        for module in modules
        for parameter in module.parameters()
    )


def _gradient_cosine(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> float:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is not None:
            left_value = left_gradient.detach()
            left_sq += float(left_value.square().sum().cpu())
        if right_gradient is not None:
            right_value = right_gradient.detach()
            right_sq += float(right_value.square().sum().cpu())
        if left_gradient is not None and right_gradient is not None:
            dot += float(
                (left_gradient.detach() * right_gradient.detach()).sum().cpu()
            )
    denominator = float(np.sqrt(left_sq * right_sq))
    return 0.0 if denominator <= 1e-12 else float(dot / denominator)


def _gradient_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum().cpu())
    return float(np.sqrt(squared))


def _tensor_gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared += float(gradient.detach().square().sum().cpu())
    return float(np.sqrt(squared))


def _contrastive_geometry_diagnostics(
    representation: torch.Tensor,
    pairs: ContrastiveBatch,
    *,
    collapsed_variance_threshold: float = 1e-8,
    ordered_margin: float = 0.0,
) -> dict[str, float]:
    """Aggregate geometry diagnostics without exposing opportunity-level values."""

    values = representation.detach()
    if values.ndim != 2 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("Geometry diagnostics require a finite 2D representation")
    normalized = F.normalize(values, p=2.0, dim=1, eps=1e-8)
    ordered_gap = values.new_empty(0)
    if isinstance(pairs, SimilarityBatch) and len(pairs):
        left = torch.as_tensor(pairs.left, dtype=torch.long, device=values.device)
        right = torch.as_tensor(pairs.right, dtype=torch.long, device=values.device)
        cosine = torch.sum(normalized[left] * normalized[right], dim=1)
        positive_mask = torch.as_tensor(
            pairs.similar >= 0.5, dtype=torch.bool, device=values.device
        )
        positive_cosine = cosine[positive_mask]
        negative_cosine = cosine[~positive_mask]
    elif isinstance(pairs, OrderedTripletBatch) and len(pairs):
        anchor = torch.as_tensor(pairs.anchor, dtype=torch.long, device=values.device)
        near = torch.as_tensor(pairs.near, dtype=torch.long, device=values.device)
        far = torch.as_tensor(pairs.far, dtype=torch.long, device=values.device)
        positive_cosine = torch.sum(normalized[anchor] * normalized[near], dim=1)
        negative_cosine = torch.sum(normalized[anchor] * normalized[far], dim=1)
        ordered_gap = positive_cosine - negative_cosine
    else:
        positive_cosine = values.new_empty(0)
        negative_cosine = values.new_empty(0)
    norms = torch.linalg.vector_norm(values, dim=1)
    dimension_variance = torch.var(values, dim=0, unbiased=False)
    centered = values - torch.mean(values, dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square() / max(len(values) - 1, 1)
    eigen_sum = float(eigenvalues.sum().cpu())
    effective_rank = (
        float((eigenvalues.sum().square() / eigenvalues.square().sum()).cpu())
        if float(eigenvalues.square().sum().cpu()) > 1e-20 else 0.0
    )
    positive_mean = (
        float(positive_cosine.mean().cpu()) if len(positive_cosine) else 0.0
    )
    negative_mean = (
        float(negative_cosine.mean().cpu()) if len(negative_cosine) else 0.0
    )
    return {
        "mean_positive_cosine": positive_mean,
        "mean_negative_cosine": negative_mean,
        "positive_negative_cosine_gap": positive_mean - negative_mean,
        "ordered_margin": float(ordered_margin),
        "ordered_margin_satisfaction_fraction": (
            float((ordered_gap >= float(ordered_margin)).float().mean().cpu())
            if len(ordered_gap) else 0.0
        ),
        "ordered_margin_shortfall_mean": (
            float(F.relu(float(ordered_margin) - ordered_gap).mean().cpu())
            if len(ordered_gap) else 0.0
        ),
        "mean_representation_norm": float(norms.mean().cpu()),
        "minimum_representation_norm": float(norms.min().cpu()),
        "maximum_representation_norm": float(norms.max().cpu()),
        "representation_dimension_variance_min": float(
            dimension_variance.min().cpu()
        ),
        "representation_dimension_variance_median": float(
            torch.median(dimension_variance).cpu()
        ),
        "effective_representation_rank": effective_rank,
        "collapsed_dimension_fraction": float(
            (dimension_variance <= float(collapsed_variance_threshold))
            .to(torch.float32).mean().cpu()
        ),
        "representation_total_variance": eigen_sum,
    }


def _contrastive_reuse_diagnostics(
    pairs: ContrastiveBatch, opportunity_count: int
) -> dict[str, float | int]:
    if int(opportunity_count) < 1 or not len(pairs):
        return {
            "contrastive_mean_reuse": 0.0,
            "contrastive_median_reuse": 0.0,
            "contrastive_max_reuse": 0,
            "contrastive_reuse_p90": 0.0,
            "contrastive_reuse_p99": 0.0,
            "contrastive_unique_opportunity_fraction": 0.0,
        }
    rows = (
        np.concatenate((pairs.left, pairs.right))
        if isinstance(pairs, SimilarityBatch)
        else np.concatenate((pairs.anchor, pairs.near, pairs.far))
    )
    counts = np.bincount(rows.astype(np.int64), minlength=int(opportunity_count))
    used = counts[counts > 0]
    return {
        "contrastive_mean_reuse": float(used.mean()),
        "contrastive_median_reuse": float(np.median(used)),
        "contrastive_max_reuse": int(used.max()),
        "contrastive_reuse_p90": float(np.quantile(used, 0.90)),
        "contrastive_reuse_p99": float(np.quantile(used, 0.99)),
        "contrastive_unique_opportunity_fraction": float(
            len(used) / int(opportunity_count)
        ),
    }


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
    return result.copy()


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
    contrastive_pairs: ContrastiveBatch,
    profile_mapping: Mapping[str, int],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    settings: Mapping[str, Any],
    *,
    model_seed: int,
    contrastive_seed: int,
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
    representation_source = str(
        settings.get("contrastive_representation_source", "projection_head")
    )
    if representation_source not in {
        "projection_head", "shared_clinical_encoder", "shared_joint_encoder",
    }:
        raise ValueError(
            f"Unknown contrastive_representation_source: {representation_source}"
        )
    cosine_geometry_audit = _validate_cosine_geometry(settings)
    retain_inactive_projection = bool(
        settings.get("contrastive_retain_inactive_projection_head", True)
    )
    model = _ProfileRankNetwork(
        input_dim=train_x.shape[1],
        profile_count=len(profile_mapping),
        hidden_dim=int(settings["hidden_dim"]),
        profile_embedding_dim=int(settings["profile_embedding_dim"]),
        projection_dim=int(settings["projection_dim"]),
        use_shared_joint_encoder=bool(
            settings.get("use_shared_joint_encoder", False)
        ),
        joint_representation_dim=int(
            settings.get("joint_representation_dim", settings["hidden_dim"])
        ),
        profile_conditioning=str(
            settings.get("profile_conditioning", "legacy_concat")
        ),
        contrastive_geometry_source=str(
            settings.get("contrastive_geometry_source", "legacy")
        ),
        use_projection_head=bool(
            (
                float(contrastive_weight) > 0.0
                and representation_source == "projection_head"
            )
            or retain_inactive_projection
        ),
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
    total_epochs = int(settings["epochs"])
    loss_type = str(settings.get("contrastive_loss_type", "legacy_margin"))
    temperature = float(settings.get("contrastive_temperature", 0.10))
    warmup_epochs = int(settings.get("contrastive_warmup_epochs", 0))
    ramp_epochs = int(settings.get("contrastive_ramp_epochs", 0))
    final_rank_only_epochs = int(
        settings.get("contrastive_final_rank_only_epochs", 0)
    )
    pretrain_epochs = int(settings.get("contrastive_pretrain_epochs", 0))
    pretrain_only = bool(settings.get("contrastive_pretrain_only", False))
    minimum_active_epochs = int(
        settings.get("contrastive_minimum_active_epochs", 0)
    )
    if minimum_active_epochs < 0:
        raise ValueError("contrastive_minimum_active_epochs cannot be negative")
    active_contrastive_epochs_seen = 0
    contrastive_started = False
    best_checkpoint_phase = "not_selected"
    geometry_interval = int(
        settings.get("contrastive_geometry_diagnostic_interval", 1)
    )
    if geometry_interval < 1:
        raise ValueError("contrastive_geometry_diagnostic_interval must be positive")
    weight_mode = str(settings.get("contrastive_weight_mode", "fixed"))
    if weight_mode not in {
        "fixed",
        "shared_gradient_ratio",
        "aligned_gradient_ratio",
    }:
        raise ValueError(f"Unknown contrastive_weight_mode: {weight_mode}")
    gradient_target_ratio = float(
        settings.get("contrastive_gradient_target_ratio", 0.20)
    )
    adaptive_weight_min = float(settings.get("contrastive_weight_min", 0.0))
    adaptive_weight_max = float(
        settings.get("contrastive_weight_max", max(float(contrastive_weight), 0.20))
    )
    gradient_alignment_floor = float(
        settings.get("contrastive_gradient_alignment_floor", 0.0)
    )
    if gradient_target_ratio <= 0.0:
        raise ValueError("contrastive_gradient_target_ratio must be positive")
    if not 0.0 <= adaptive_weight_min <= adaptive_weight_max:
        raise ValueError("Adaptive contrastive weight bounds are invalid")
    if not -1.0 <= gradient_alignment_floor < 1.0:
        raise ValueError(
            "contrastive_gradient_alignment_floor must be in [-1, 1)"
        )
    if pretrain_epochs < 0:
        raise ValueError("contrastive_pretrain_epochs cannot be negative")
    target_mode = str(settings.get("contrastive_target_mode", "hard_bins"))
    dynamic_requested = target_mode in {
        "dynamic_dr_margin",
        "opposite_arm_dynamic_dr_margin",
    }
    opposite_arm_requested = target_mode == "opposite_arm_dynamic_dr_margin"
    dynamic_active = dynamic_requested and float(contrastive_weight) > 0.0
    refresh_epochs = int(settings.get("contrastive_refresh_epochs", 1))
    dynamic_pair_count = int(settings.get("contrastive_pairs", len(contrastive_pairs)))
    dynamic_positive_fraction = float(
        settings.get("contrastive_positive_fraction", 0.50)
    )
    dynamic_within_fraction = (
        1.0
        if opposite_arm_requested
        else float(settings.get("contrastive_within_profile_fraction", 0.50))
    )
    if refresh_epochs < 1:
        raise ValueError("contrastive_refresh_epochs must be positive")
    if dynamic_requested and not isinstance(contrastive_pairs, SimilarityBatch):
        raise TypeError("Dynamic DR margin requires a SimilarityBatch candidate pool")
    if dynamic_requested and pretrain_epochs:
        raise ValueError(
            "Dynamic DR margin uses ranking warm-up and does not allow "
            "contrastive pretraining"
        )
    if dynamic_active and warmup_epochs >= total_epochs - final_rank_only_epochs:
        raise ValueError("Dynamic contrastive schedule leaves no active refresh epoch")
    active_contrastive_pairs: ContrastiveBatch = (
        _empty_similarity() if dynamic_active else contrastive_pairs
    )
    refresh_records: list[dict[str, Any]] = []
    candidate_pair_count = int(len(contrastive_pairs))
    candidate_supervision_sha256 = _contrastive_batch_hash(contrastive_pairs)
    candidate_arm_diagnostics = (
        _similarity_arm_diagnostics(train, contrastive_pairs)
        if isinstance(contrastive_pairs, SimilarityBatch)
        else {
            "arm_metadata_available": "profile_treatment" in train,
            "opposite_arm_pairs": 0,
            "same_arm_pairs": 0,
            "opposite_arm_fraction": 0.0,
        }
    )
    if float(contrastive_weight) > 0.0:
        for pretrain_epoch in range(pretrain_epochs):
            model.train()
            optimizer.zero_grad()
            train_score, projection, clinical, shared_joint = model.forward_components(
                train_x, train_profile
            )
            contrastive_representation = (
                clinical
                if representation_source == "shared_clinical_encoder"
                else model.contrastive_representation(shared_joint)
                if representation_source == "shared_joint_encoder"
                else projection
            )
            ranking_loss = pairwise_ranking_loss(train_score, train_pairs)
            contrastive_loss = _contrastive_loss(
                contrastive_representation,
                contrastive_pairs,
                float(settings["contrastive_margin"]),
                loss_type=loss_type,
                temperature=temperature,
            )
            contrastive_loss.backward()
            gradients = {
                "gradient_norm_clinical_encoder": _gradient_norm(
                    model.clinical_encoder
                ),
                "gradient_norm_profile_embedding": _gradient_norm(
                    model.profile_embedding
                ),
                "gradient_norm_joint_encoder": _gradient_norm(model.joint_encoder),
                "gradient_norm_profile_film": (
                    _gradient_norm(model.profile_film)
                    if model.profile_film is not None else 0.0
                ),
                "gradient_norm_scoring_head": _gradient_norm(model.scoring_head),
                "gradient_norm_projection_head": _gradient_norm(
                    model.projection_head
                ),
                "gradient_norm_contrastive_adapter": _gradient_norm(
                    model.contrastive_adapter
                ),
            }
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_score, _ = model(validation_x, validation_profile)
                validation_loss = float(
                    pairwise_ranking_loss(
                        validation_score, validation_pairs
                    ).cpu()
                )
            history.append({
                "training_phase": "contrastive_pretrain",
                "schedule_phase": "contrastive_pretrain",
                "epoch": pretrain_epoch,
                "total_loss": float(contrastive_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "effective_contrastive_weight": 1.0,
                "ranking_shared_gradient_norm": 0.0,
                "contrastive_shared_gradient_norm": float(
                    gradients["gradient_norm_clinical_encoder"] ** 2
                    + gradients["gradient_norm_profile_embedding"] ** 2
                    + gradients["gradient_norm_joint_encoder"] ** 2
                    + gradients["gradient_norm_profile_film"] ** 2
                ) ** 0.5,
                "shared_gradient_cosine": 0.0,
                "effective_shared_gradient_ratio": 0.0,
                "contrastive_joint_encoder_gradient_norm": 0.0,
                "contrastive_adapter_gradient_norm": float(
                    gradients["gradient_norm_contrastive_adapter"]
                ),
                "validation_pairwise_loss": validation_loss,
                **{
                    key: np.nan for key in (
                        "mean_positive_cosine", "mean_negative_cosine",
                        "positive_negative_cosine_gap", "mean_representation_norm",
                        "minimum_representation_norm", "maximum_representation_norm",
                        "representation_dimension_variance_min",
                        "representation_dimension_variance_median",
                        "effective_representation_rank",
                        "collapsed_dimension_fraction", "representation_total_variance",
                    )
                },
                **gradients,
            })
    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()
        train_score, projection, clinical, shared_joint = model.forward_components(
            train_x, train_profile
        )
        contrastive_representation = (
            clinical
            if representation_source == "shared_clinical_encoder"
            else model.contrastive_representation(shared_joint)
            if representation_source == "shared_joint_encoder"
            else projection
        )
        ranking_loss = pairwise_ranking_loss(train_score, train_pairs)
        refreshed = False
        pair_retention = 0.0
        refresh_seed = 0
        contrastive_end = total_epochs - final_rank_only_epochs
        if (
            dynamic_active
            and epoch >= warmup_epochs
            and epoch < contrastive_end
            and (epoch - warmup_epochs) % refresh_epochs == 0
        ):
            refresh_seed = _dynamic_refresh_seed(
                int(contrastive_seed), int(model_seed), int(epoch)
            )
            previous_pairs = active_contrastive_pairs
            active_contrastive_pairs = _mine_dynamic_contrastive_pairs(
                contrastive_pairs,
                contrastive_representation.detach().cpu().numpy(),
                pairs=dynamic_pair_count,
                seed=refresh_seed,
                positive_fraction=dynamic_positive_fraction,
                within_profile_fraction=dynamic_within_fraction,
            )
            if not len(active_contrastive_pairs):
                raise ValueError("Dynamic DR contrastive mining produced no pairs")
            pair_retention = (
                _similarity_pair_overlap(previous_pairs, active_contrastive_pairs)
                if len(previous_pairs) else 0.0
            )
            refreshed = True
            refresh_records.append({
                "epoch": int(epoch),
                "seed": int(refresh_seed),
                "pairs": int(len(active_contrastive_pairs)),
                "retention": float(pair_retention),
                "sha256": _contrastive_batch_hash(active_contrastive_pairs),
            })
        contrastive_loss = _contrastive_loss(
            contrastive_representation,
            active_contrastive_pairs,
            float(settings["contrastive_margin"]),
            loss_type=loss_type,
            temperature=temperature,
        )
        schedule_multiplier = _scheduled_contrastive_weight(
            1.0 if float(contrastive_weight) > 0.0 else 0.0,
            epoch,
            total_epochs,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            final_rank_only_epochs=final_rank_only_epochs,
        )
        ranking_shared_gradient_norm = 0.0
        contrastive_shared_gradient_norm = 0.0
        shared_gradient_cosine = 0.0
        contrastive_joint_encoder_gradient_norm = 0.0
        contrastive_adapter_gradient_norm = 0.0
        if float(contrastive_weight) > 0.0 and len(active_contrastive_pairs):
            shared_parameters = _shared_representation_parameters(model)
            ranking_gradients = torch.autograd.grad(
                ranking_loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            contrastive_gradients = torch.autograd.grad(
                contrastive_loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            ranking_shared_gradient_norm = _tensor_gradient_norm(ranking_gradients)
            contrastive_shared_gradient_norm = _tensor_gradient_norm(
                contrastive_gradients
            )
            shared_gradient_cosine = _gradient_cosine(
                ranking_gradients, contrastive_gradients
            )
            joint_parameters = tuple(model.joint_encoder.parameters())
            if joint_parameters:
                contrastive_joint_encoder_gradient_norm = _tensor_gradient_norm(
                    torch.autograd.grad(
                        contrastive_loss, joint_parameters, retain_graph=True,
                        allow_unused=True,
                    )
                )
            if model.contrastive_adapter is not None:
                contrastive_adapter_gradient_norm = _tensor_gradient_norm(
                    torch.autograd.grad(
                        contrastive_loss,
                        tuple(model.contrastive_adapter.parameters()),
                        retain_graph=True, allow_unused=True,
                    )
                )
        if weight_mode in {
            "shared_gradient_ratio",
            "aligned_gradient_ratio",
        } and schedule_multiplier > 0.0:
            unconstrained_weight = (
                gradient_target_ratio
                * ranking_shared_gradient_norm
                / max(contrastive_shared_gradient_norm, 1e-12)
            )
            effective_contrastive_weight = schedule_multiplier * float(
                np.clip(
                    unconstrained_weight,
                    adaptive_weight_min,
                    adaptive_weight_max,
                )
            )
            if weight_mode == "aligned_gradient_ratio":
                # Map the configured cosine floor to zero and cosine 1 to one.
                # This preserves mildly conflicting updates when requested without
                # ever turning the auxiliary loss coefficient negative.
                alignment_factor = max(
                    0.0,
                    (shared_gradient_cosine - gradient_alignment_floor)
                    / (1.0 - gradient_alignment_floor),
                )
                effective_contrastive_weight *= alignment_factor
        else:
            effective_contrastive_weight = (
                schedule_multiplier * float(contrastive_weight)
            )
        if pretrain_only:
            effective_contrastive_weight = 0.0
        active_now = effective_contrastive_weight > 0.0
        if active_now:
            active_contrastive_epochs_seen += 1
            if not contrastive_started:
                # Require the selected checkpoint to have seen the auxiliary loss
                # when a minimum active duration was explicitly requested.
                if minimum_active_epochs > 0:
                    best_loss = float("inf")
                    best_epoch = -1
                    best_state = None
                patience_used = 0
                contrastive_started = True
        effective_shared_gradient_ratio = (
            effective_contrastive_weight
            * contrastive_shared_gradient_norm
            / max(ranking_shared_gradient_norm, 1e-12)
            if ranking_shared_gradient_norm > 0.0
            else 0.0
        )
        total_loss = (
            ranking_loss + effective_contrastive_weight * contrastive_loss
        )
        final_phase_start = total_epochs - final_rank_only_epochs
        schedule_phase = (
            "rank_only"
            if float(contrastive_weight) <= 0.0
            else "warmup_rank_only"
            if epoch < warmup_epochs
            else "final_rank_only"
            if final_rank_only_epochs > 0 and epoch >= final_phase_start
            else "joint_contrastive"
        )
        geometry_diagnostics = {
            key: np.nan for key in (
                "mean_positive_cosine", "mean_negative_cosine",
                "positive_negative_cosine_gap", "mean_representation_norm",
                "minimum_representation_norm", "maximum_representation_norm",
                "representation_dimension_variance_min",
                "representation_dimension_variance_median",
                "effective_representation_rank", "collapsed_dimension_fraction",
                "representation_total_variance", "ordered_margin",
                "ordered_margin_satisfaction_fraction",
                "ordered_margin_shortfall_mean",
            )
        }
        if (
            float(contrastive_weight) > 0.0
            and len(active_contrastive_pairs)
            and epoch >= warmup_epochs
            and epoch < final_phase_start
            and (epoch - warmup_epochs) % geometry_interval == 0
        ):
            geometry_diagnostics = _contrastive_geometry_diagnostics(
                contrastive_representation, active_contrastive_pairs,
                collapsed_variance_threshold=float(
                    settings.get("contrastive_collapsed_variance_threshold", 1e-8)
                ),
                ordered_margin=(
                    float(settings["contrastive_margin"])
                    if loss_type == "ordered_cosine_triplet_margin" else 0.0
                ),
            )
        total_loss.backward()
        gradients = {
            "gradient_norm_clinical_encoder": _gradient_norm(model.clinical_encoder),
            "gradient_norm_profile_embedding": _gradient_norm(model.profile_embedding),
            "gradient_norm_joint_encoder": _gradient_norm(model.joint_encoder),
            "gradient_norm_profile_film": (
                _gradient_norm(model.profile_film)
                if model.profile_film is not None else 0.0
            ),
            "gradient_norm_scoring_head": _gradient_norm(model.scoring_head),
            "gradient_norm_projection_head": _gradient_norm(model.projection_head),
            "gradient_norm_contrastive_adapter": _gradient_norm(
                model.contrastive_adapter
            ),
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
                "training_phase": "ranking",
                "schedule_phase": schedule_phase,
                "epoch": epoch,
                "total_loss": float(total_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "effective_contrastive_weight": float(
                    effective_contrastive_weight
                ),
                "ranking_shared_gradient_norm": ranking_shared_gradient_norm,
                "contrastive_shared_gradient_norm": contrastive_shared_gradient_norm,
                "shared_gradient_cosine": shared_gradient_cosine,
                "effective_shared_gradient_ratio": effective_shared_gradient_ratio,
                "contrastive_joint_encoder_gradient_norm": (
                    contrastive_joint_encoder_gradient_norm
                ),
                "contrastive_adapter_gradient_norm": (
                    contrastive_adapter_gradient_norm
                ),
                "validation_pairwise_loss": validation_loss,
                "contrastive_pairs_refreshed": bool(refreshed),
                "contrastive_refresh_seed": int(refresh_seed),
                "contrastive_active_pairs": int(len(active_contrastive_pairs)),
                "contrastive_pair_retention": float(pair_retention),
                **geometry_diagnostics,
                **gradients,
            }
        )
        if validation_loss < best_loss - float(settings["minimum_improvement"]):
            best_loss = validation_loss
            best_epoch = epoch
            best_checkpoint_phase = schedule_phase
            best_state = copy.deepcopy(model.state_dict())
            patience_used = 0
        else:
            patience_used += 1
            contrastive_budget_satisfied = (
                float(contrastive_weight) <= 0.0
                or active_contrastive_epochs_seen >= minimum_active_epochs
            )
            final_rank_only_budget_satisfied = (
                float(contrastive_weight) <= 0.0
                or final_rank_only_epochs <= 0
                or epoch + 1 >= total_epochs
            )
            if (
                patience_used >= int(settings["patience"])
                and contrastive_budget_satisfied
                and final_rank_only_budget_satisfied
            ):
                break
    if best_state is None:
        raise RuntimeError("Ranker did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(score_x, score_profile)[0].cpu().numpy().astype(float)
    history_frame = pd.DataFrame(history)
    active_contrastive_history = history_frame.loc[
        history_frame.training_phase.eq("ranking")
        & history_frame.effective_contrastive_weight.gt(0.0)
    ]
    geometry_history = history_frame.loc[
        history_frame.training_phase.eq("ranking")
        & history_frame.mean_positive_cosine.notna()
    ]
    geometry_summary = {
        key: float(geometry_history[key].mean()) if len(geometry_history) else 0.0
        for key in (
            "mean_positive_cosine", "mean_negative_cosine",
            "positive_negative_cosine_gap", "mean_representation_norm",
            "minimum_representation_norm", "maximum_representation_norm",
            "representation_dimension_variance_min",
            "representation_dimension_variance_median",
            "effective_representation_rank", "collapsed_dimension_fraction",
            "representation_total_variance", "ordered_margin",
            "ordered_margin_satisfaction_fraction",
            "ordered_margin_shortfall_mean",
        )
    }
    final_history = history_frame.loc[
        history_frame.training_phase.eq("ranking")
        & history_frame.schedule_phase.eq("final_rank_only")
    ]
    if dynamic_active:
        if not refresh_records or not len(active_contrastive_pairs):
            raise RuntimeError("Dynamic DR contrastive training never refreshed pairs")
        contrastive_pairs = active_contrastive_pairs
    active_arm_diagnostics = (
        _similarity_arm_diagnostics(train, contrastive_pairs)
        if isinstance(contrastive_pairs, SimilarityBatch)
        else {
            "arm_metadata_available": "profile_treatment" in train,
            "opposite_arm_pairs": 0,
            "same_arm_pairs": 0,
            "opposite_arm_fraction": 0.0,
        }
    )
    if opposite_arm_requested and (
        active_arm_diagnostics["opposite_arm_pairs"] != len(contrastive_pairs)
        or active_arm_diagnostics["same_arm_pairs"] != 0
    ):
        raise RuntimeError("Opposite-arm contrastive mining selected an invalid pair")
    ordered_triplets = isinstance(contrastive_pairs, OrderedTripletBatch)
    if ordered_triplets:
        contrastive_rows = np.concatenate(
            (
                contrastive_pairs.anchor,
                contrastive_pairs.near,
                contrastive_pairs.far,
            )
        )
        similarity_targets = np.array([], dtype=float)
        target_gap = contrastive_pairs.distance_gap
        order_agreement = contrastive_pairs.order_agreement
    else:
        contrastive_rows = np.concatenate(
            (contrastive_pairs.left, contrastive_pairs.right)
        )
        similarity_targets = contrastive_pairs.similar
        target_gap = np.array([], dtype=float)
        order_agreement = np.array([], dtype=float)
    reuse_diagnostics = _contrastive_reuse_diagnostics(
        contrastive_pairs, len(train)
    )
    maximum_reuse = settings.get("contrastive_max_reuse_per_opportunity")
    reuse_limit_satisfied = bool(
        maximum_reuse is None
        or reuse_diagnostics["contrastive_max_reuse"] <= int(maximum_reuse)
    )
    joint_contrastive_gradient_max = float(
        history_frame.contrastive_joint_encoder_gradient_norm.max()
    )
    adapter_contrastive_gradient_max = float(
        history_frame.contrastive_adapter_gradient_norm.max()
    )
    adapter_joint_ratio = (
        joint_contrastive_gradient_max / adapter_contrastive_gradient_max
        if adapter_contrastive_gradient_max > 1e-12 else 0.0
    )
    geometry_was_evaluated = bool(len(geometry_history))
    geometry_diagnostic_gates = {
        "status": (
            "evaluated_diagnostic_only"
            if geometry_was_evaluated else "not_applicable_or_not_evaluated"
        ),
        "representation_not_collapsed": bool(
            geometry_was_evaluated
            and geometry_summary["effective_representation_rank"]
            >= float(settings.get("contrastive_minimum_effective_rank", 2.0))
            and geometry_summary["collapsed_dimension_fraction"]
            <= float(
                settings.get("contrastive_maximum_collapsed_dimension_fraction", 0.5)
            )
        ),
        "positive_negative_cosine_separated": bool(
            geometry_was_evaluated
            and geometry_summary["positive_negative_cosine_gap"]
            > float(settings.get("contrastive_minimum_cosine_gap", 0.0))
        ),
        "joint_encoder_receives_contrastive_gradient": bool(
            float(contrastive_weight) <= 0.0
            or joint_contrastive_gradient_max
            > float(settings.get("contrastive_minimum_gradient_norm", 1e-10))
        ),
        "adapter_receives_gradient_when_configured": bool(
            model.contrastive_adapter is None
            or adapter_contrastive_gradient_max
            > float(settings.get("contrastive_minimum_gradient_norm", 1e-10))
        ),
        "adapter_transmits_gradient_to_joint_encoder": bool(
            model.contrastive_adapter is None
            or adapter_joint_ratio >= float(
                settings.get(
                    "contrastive_minimum_joint_to_adapter_gradient_ratio", 0.01
                )
            )
        ),
        "negative_pairs_reach_signed_cosine": bool(
            geometry_was_evaluated
            and geometry_summary["mean_negative_cosine"] < 0.0
        ),
        "ordered_margin_satisfaction_above_half": bool(
            loss_type != "ordered_cosine_triplet_margin"
            or (
                geometry_was_evaluated
                and geometry_summary["ordered_margin_satisfaction_fraction"] > 0.5
            )
        ),
        "reuse_limit_satisfied": reuse_limit_satisfied,
        "eligible_for_model_selection": False,
    }
    audit = {
        "architecture_family": "siamese_patient_profile_causal_ranker",
        "primary_objective": "direct_pairwise_causal_ranking",
        "contrastive_role": (
            "auxiliary_causal_response_representation_regularizer"
            if float(contrastive_weight) > 0.0
            else "disabled_rank_only_ablation"
        ),
        "contrastive_active": bool(float(contrastive_weight) > 0.0),
        "contrastive_pair_source": (
            "rank_train_repeated_cross_fitted_dr_" + target_mode
        ),
        "dynamic_contrastive_mining": bool(dynamic_active),
        "opposite_arm_contrastive_matching": bool(opposite_arm_requested),
        "treatment_assignment_used_as_model_feature": False,
        "treatment_assignment_used_for_pair_eligibility_only": bool(
            opposite_arm_requested
        ),
        "dynamic_mining_uses_current_embedding": bool(dynamic_active),
        "dynamic_model_ite_or_cate_used": False,
        "dynamic_label_source": (
            "fixed_rank_train_repeated_cross_fitted_dr_similarity"
            if dynamic_requested else "not_applicable"
        ),
        "contrastive_refresh_epochs": int(refresh_epochs),
        "contrastive_refresh_count": int(len(refresh_records)),
        "contrastive_refresh_records": refresh_records,
        "contrastive_candidate_pairs": candidate_pair_count,
        "contrastive_candidate_opposite_arm_pairs": candidate_arm_diagnostics[
            "opposite_arm_pairs"
        ],
        "contrastive_candidate_same_arm_pairs": candidate_arm_diagnostics[
            "same_arm_pairs"
        ],
        "contrastive_candidate_opposite_arm_fraction": candidate_arm_diagnostics[
            "opposite_arm_fraction"
        ],
        "contrastive_candidate_supervision_sha256": candidate_supervision_sha256,
        "ranking_head_used_for_priority_score": True,
        "projection_head_used_for_priority_score": False,
        "contrastive_representation_source": representation_source,
        "contrastive_geometry_source": str(
            settings.get("contrastive_geometry_source", "legacy")
        ),
        "cosine_geometry_validation": cosine_geometry_audit,
        "projection_head_used_for_contrastive_loss": bool(
            float(contrastive_weight) > 0.0
            and representation_source == "projection_head"
            and model.projection_head is not None
        ),
        "shared_clinical_encoder_used_for_contrastive_loss": bool(
            float(contrastive_weight) > 0.0
            and representation_source == "shared_clinical_encoder"
        ),
        "shared_joint_encoder_used_for_contrastive_loss": bool(
            float(contrastive_weight) > 0.0
            and representation_source == "shared_joint_encoder"
        ),
        "contrastive_adapter_used_for_contrastive_loss": bool(
            model.contrastive_adapter is not None
            and representation_source == "shared_joint_encoder"
        ),
        "use_shared_joint_encoder": bool(model.use_shared_joint_encoder),
        "profile_conditioning": str(model.profile_conditioning),
        "best_epoch": int(best_epoch),
        "best_validation_pairwise_loss": float(best_loss),
        "epochs_completed": int(len(history)),
        "model_seed": int(model_seed),
        "contrastive_weight": float(contrastive_weight),
        "contrastive_weight_mode": weight_mode,
        "contrastive_gradient_target_ratio": gradient_target_ratio,
        "contrastive_weight_min": adaptive_weight_min,
        "contrastive_weight_max": adaptive_weight_max,
        "contrastive_loss_type": loss_type,
        "contrastive_target_mode": str(
            settings.get("contrastive_target_mode", "hard_bins")
        ),
        "contrastive_temperature": temperature,
        "contrastive_margin": float(settings["contrastive_margin"]),
        "contrastive_projection_l2_normalized": (
            loss_type in {
                "normalized_euclidean_margin", "normalized_cosine_binary",
                "ordered_cosine_triplet", "ordered_cosine_triplet_margin",
            }
        ),
        "contrastive_warmup_epochs": warmup_epochs,
        "contrastive_ramp_epochs": ramp_epochs,
        "contrastive_final_rank_only_epochs": final_rank_only_epochs,
        "contrastive_pretrain_epochs": pretrain_epochs,
        "contrastive_pretrain_only": pretrain_only,
        "contrastive_minimum_active_epochs": minimum_active_epochs,
        "contrastive_active_epochs_seen": int(active_contrastive_epochs_seen),
        "contrastive_gradient_alignment_floor": gradient_alignment_floor,
        "best_checkpoint_after_contrastive_start": bool(
            contrastive_started and best_epoch >= warmup_epochs
        ),
        "best_checkpoint_after_contrastive_activation": bool(
            contrastive_started
            and best_checkpoint_phase in {"joint_contrastive", "final_rank_only"}
        ),
        "best_checkpoint_phase": str(best_checkpoint_phase),
        "final_rank_only_phase_started": bool(len(final_history)),
        "final_rank_only_epochs_completed": int(len(final_history)),
        "best_checkpoint_policy": (
            "minimum_fixed_validation_pairwise_loss_after_contrastive_activation_"
            "across_joint_and_final_rank_only_phases"
        ),
        "ranking_epochs_completed": int(
            history_frame.training_phase.eq("ranking").sum()
        ),
        "mean_effective_contrastive_weight": float(
            history_frame.effective_contrastive_weight.mean()
        ),
        "mean_ranking_shared_gradient_norm": float(
            history_frame.ranking_shared_gradient_norm.mean()
        ),
        "mean_contrastive_shared_gradient_norm": float(
            history_frame.contrastive_shared_gradient_norm.mean()
        ),
        "mean_shared_gradient_cosine": float(
            history_frame.shared_gradient_cosine.fillna(0.0).mean()
        ),
        "mean_active_shared_gradient_cosine": float(
            active_contrastive_history.shared_gradient_cosine.mean()
            if len(active_contrastive_history) else 0.0
        ),
        "fraction_active_conflicting_gradients": float(
            active_contrastive_history.shared_gradient_cosine.lt(0.0).mean()
            if len(active_contrastive_history) else 0.0
        ),
        "mean_effective_shared_gradient_ratio": float(
            history_frame.effective_shared_gradient_ratio.mean()
        ),
        "mean_active_effective_shared_gradient_ratio": float(
            active_contrastive_history.effective_shared_gradient_ratio.mean()
            if len(active_contrastive_history) else 0.0
        ),
        "max_active_effective_shared_gradient_ratio": float(
            active_contrastive_history.effective_shared_gradient_ratio.max()
            if len(active_contrastive_history) else 0.0
        ),
        "projection_head_max_gradient_norm": float(
            history_frame.gradient_norm_projection_head.max()
        ),
        "joint_encoder_max_gradient_norm": float(
            history_frame.gradient_norm_joint_encoder.max()
        ),
        "profile_film_max_gradient_norm": float(
            history_frame.gradient_norm_profile_film.max()
        ),
        "contrastive_adapter_max_gradient_norm": float(
            history_frame.gradient_norm_contrastive_adapter.max()
        ),
        "contrastive_joint_encoder_gradient_max": joint_contrastive_gradient_max,
        "contrastive_adapter_gradient_max": adapter_contrastive_gradient_max,
        "contrastive_joint_to_adapter_gradient_ratio": adapter_joint_ratio,
        "contrastive_geometry_diagnostic_epochs": int(len(geometry_history)),
        **geometry_summary,
        "geometry_diagnostic_gates": geometry_diagnostic_gates,
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
        "contrastive_supervision_sha256": _contrastive_batch_hash(
            contrastive_pairs
        ),
        "pair_sampling_backend": str(
            getattr(contrastive_pairs, "sampling_backend", "bounded_triplet")
        ),
        "contrastive_sampling_attempts": int(
            getattr(contrastive_pairs, "sampling_attempts", 0)
        ),
        "contrastive_sampling_requested_pairs": int(
            getattr(contrastive_pairs, "requested_pairs", len(contrastive_pairs))
        ),
        "contrastive_sampling_quota_satisfied": bool(
            getattr(contrastive_pairs, "quota_satisfied", True)
        ),
        "contrastive_sampling_estimated_peak_candidate_bytes": int(
            getattr(contrastive_pairs, "estimated_peak_candidate_bytes", 0)
        ),
        "contrastive_robust_scale": float(
            getattr(contrastive_pairs, "robust_scale", 0.0)
        ),
        "contrastive_triplets": int(len(contrastive_pairs)) if ordered_triplets else 0,
        "contrastive_positive_pairs": int(
            (similarity_targets >= 0.5).sum()
        ),
        "contrastive_negative_pairs": int(
            len(similarity_targets) - (similarity_targets >= 0.5).sum()
        ),
        "contrastive_similarity_target_mean": (
            float(similarity_targets.mean()) if len(similarity_targets) else 0.0
        ),
        "contrastive_similarity_target_min": (
            float(similarity_targets.min()) if len(similarity_targets) else 0.0
        ),
        "contrastive_similarity_target_max": (
            float(similarity_targets.max()) if len(similarity_targets) else 0.0
        ),
        "contrastive_distance_gap_mean": (
            float(target_gap.mean()) if len(target_gap) else 0.0
        ),
        "contrastive_order_agreement_mean": (
            float(order_agreement.mean()) if len(order_agreement) else 0.0
        ),
        "contrastive_within_profile_pairs": int(
            (~contrastive_pairs.is_cross_profile).sum()
        ),
        "contrastive_cross_profile_pairs": int(
            contrastive_pairs.is_cross_profile.sum()
        ),
        "contrastive_opposite_arm_pairs": active_arm_diagnostics[
            "opposite_arm_pairs"
        ],
        "contrastive_same_arm_pairs": active_arm_diagnostics["same_arm_pairs"],
        "contrastive_opposite_arm_fraction": active_arm_diagnostics[
            "opposite_arm_fraction"
        ],
        "contrastive_unique_rows": int(
            len(np.unique(contrastive_rows))
            if len(contrastive_pairs) else 0
        ),
        **reuse_diagnostics,
        "contrastive_max_reuse_per_opportunity": (
            None if maximum_reuse is None else int(maximum_reuse)
        ),
        "contrastive_pair_weight_min": (
            float(contrastive_pairs.weight.min()) if len(contrastive_pairs) else 0.0
        ),
        "contrastive_pair_weight_max": (
            float(contrastive_pairs.weight.max()) if len(contrastive_pairs) else 0.0
        ),
        "projection_head_separate": bool(model.projection_head is not None),
        "projection_head_present": bool(model.projection_head is not None),
        "projection_head_status": (
            "active_contrastive_geometry"
            if (
                float(contrastive_weight) > 0.0
                and representation_source == "projection_head"
                and model.projection_head is not None
            )
            else "inactive_retained_for_legacy_compatibility"
            if model.projection_head is not None
            else "omitted_not_required"
        ),
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
            "use_shared_joint_encoder": bool(
                settings.get("use_shared_joint_encoder", False)
            ),
            "joint_representation_dim": int(
                settings.get("joint_representation_dim", settings["hidden_dim"])
            ),
            "profile_conditioning": str(
                settings.get("profile_conditioning", "legacy_concat")
            ),
            "contrastive_geometry_source": str(
                settings.get("contrastive_geometry_source", "legacy")
            ),
            "projection_head_present": bool(model.projection_head is not None),
            "contrastive_adapter_present": bool(
                model.contrastive_adapter is not None
            ),
        },
        "score_semantics": "globally_comparable_ordinal_priority_no_causal_zero",
        "architecture_family": "siamese_patient_profile_causal_ranker",
        "primary_objective": "direct_pairwise_causal_ranking",
        "contrastive_role": (
            "auxiliary_causal_response_representation_regularizer"
            if float(contrastive_weight) > 0.0
            else "disabled_rank_only_ablation"
        ),
        "inference_output": "ranking_head_raw_priority_score_only",
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


def _pair_batch_hash(pairs: PairBatch) -> str:
    """Hash sampled rank-pair supervision without persisting patient rows."""

    payload = {
        "left": np.asarray(pairs.left).tolist(),
        "right": np.asarray(pairs.right).tolist(),
        "direction": np.asarray(pairs.direction, dtype=float).tolist(),
        "weight": np.asarray(pairs.weight, dtype=float).tolist(),
        "is_cross_profile": np.asarray(
            pairs.is_cross_profile, dtype=bool
        ).tolist(),
        "direction_agreement": np.asarray(
            pairs.direction_agreement, dtype=float
        ).tolist(),
    }
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


def _ordinal_percentile(values: np.ndarray) -> np.ndarray:
    result = pd.Series(np.asarray(values, dtype=float)).rank(
        method="average", pct=True
    ).to_numpy(float)
    if not np.isfinite(result).all():
        raise RuntimeError("Baseline ordinalization produced non-finite scores")
    return result.copy()


def _linear_pairwise_design(
    features: np.ndarray,
    profile_index: np.ndarray,
    profile_count: int,
    *,
    interactions: bool,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    profiles = np.eye(int(profile_count), dtype=float)[profile_index]
    blocks = [values, profiles]
    if interactions:
        blocks.append(
            (values[:, :, None] * profiles[:, None, :]).reshape(len(values), -1)
        )
    return np.concatenate(blocks, axis=1)


def _fit_linear_pairwise_ranker(
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
    interactions: bool,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Fit a Bradley-Terry-style direct linear ranker on the frozen DR pairs."""

    encoder = _FeatureEncoder(numeric_features, categorical_features).fit(train)
    train_x = encoder.transform(train)
    validation_x = encoder.transform(validation)
    score_x = encoder.transform(score_frame)
    train_profile = _profile_indices(train, profile_mapping)
    validation_profile = _profile_indices(validation, profile_mapping)
    score_profile = _profile_indices(score_frame, profile_mapping)
    train_design = _linear_pairwise_design(
        train_x, train_profile, len(profile_mapping), interactions=interactions
    )
    validation_design = _linear_pairwise_design(
        validation_x,
        validation_profile,
        len(profile_mapping),
        interactions=interactions,
    )
    score_design = _linear_pairwise_design(
        score_x, score_profile, len(profile_mapping), interactions=interactions
    )
    pair_x = train_design[train_pairs.left] - train_design[train_pairs.right]
    label = (train_pairs.direction > 0).astype(int)
    augmented_x = np.concatenate((pair_x, -pair_x), axis=0)
    augmented_y = np.concatenate((label, 1 - label))
    augmented_weight = np.tile(train_pairs.weight, 2)
    model = LogisticRegression(
        C=float(settings.get("linear_pairwise_c", 1.0)),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=int(settings.get("linear_pairwise_max_iter", 500)),
        random_state=int(model_seed),
    ).fit(augmented_x, augmented_y, sample_weight=augmented_weight)
    validation_score = _ordinal_percentile(validation_design @ model.coef_[0])
    validation_loss = float(pairwise_ranking_loss(
        torch.as_tensor(validation_score, dtype=torch.float32), validation_pairs
    ))
    scores = _ordinal_percentile(score_design @ model.coef_[0])
    history = pd.DataFrame([{
        "epoch": 0,
        "total_loss": np.nan,
        "ranking_loss": np.nan,
        "contrastive_loss": 0.0,
        "validation_pairwise_loss": validation_loss,
    }])
    audit = {
        "model_seed": int(model_seed),
        "training_objective": "direct_pairwise_bradley_terry_logistic",
        "feature_form": (
            "preindex_features_profile_intercepts_and_interactions"
            if interactions
            else "preindex_features_and_profile_intercepts"
        ),
        "patient_profile_interactions": bool(interactions),
        "individual_effect_regression_used": False,
        "individual_cate_estimated_then_sorted": False,
        "train_pairs": int(len(train_pairs)),
        "fixed_validation_pairs": int(len(validation_pairs)),
        "validation_pairwise_loss": validation_loss,
        "design_columns": int(train_design.shape[1]),
        "regularization_c": float(settings.get("linear_pairwise_c", 1.0)),
        "max_iter": int(settings.get("linear_pairwise_max_iter", 500)),
        "score_ordinalized": True,
        "oracle_used": False,
    }
    return scores, history, audit


def _weighted_group_mean(
    signal: np.ndarray, weight: np.ndarray, indices: np.ndarray
) -> tuple[float, float]:
    selected_weight = weight[indices]
    total_weight = float(selected_weight.sum())
    if total_weight <= 0.0:
        raise ValueError("Group baseline requires positive reliability weight")
    return (
        float(np.sum(selected_weight * signal[indices]) / total_weight),
        total_weight,
    )


def _fit_group_dr_baseline(
    train: pd.DataFrame,
    score_frame: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    use_strata: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a coarse rank-train DR group comparator and return ordinal scores."""

    signal = train.dr_pseudo_outcome.to_numpy(float)
    weight = train.dr_reliability_weight.to_numpy(float)
    if not np.isfinite(signal).all() or not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Group DR baseline requires finite positive learner weights")
    all_indices = np.arange(len(train), dtype=int)
    global_mean, _ = _weighted_group_mean(signal, weight, all_indices)
    profile_strength = float(settings.get("profile_mean_dr_shrinkage_strength", 20.0))
    stratum_strength = float(settings.get("stratum_dr_shrinkage_strength", 30.0))
    if profile_strength < 0.0 or stratum_strength < 0.0:
        raise ValueError("Group DR shrinkage strengths must be non-negative")
    train_profile = train.care_profile_id.astype(str).to_numpy()
    score_profile = score_frame.care_profile_id.astype(str).to_numpy()
    profile_means: dict[str, float] = {}
    for profile in sorted(set(train_profile)):
        indices = np.flatnonzero(train_profile == profile)
        mean, mass = _weighted_group_mean(signal, weight, indices)
        profile_means[profile] = float(
            (mass * mean + profile_strength * global_mean)
            / (mass + profile_strength)
        )
    unknown_profiles = sorted(set(score_profile).difference(profile_means))
    if unknown_profiles:
        raise ValueError(f"Group DR baseline lacks profiles: {unknown_profiles}")
    if not use_strata:
        raw = np.asarray([profile_means[value] for value in score_profile], dtype=float)
        return _ordinal_percentile(raw), {
            "training_objective": "rank_train_profile_average_dr_group_ranking",
            "personalized": False,
            "profile_groups": int(len(profile_means)),
            "profile_shrinkage_strength": profile_strength,
            "individual_cate_estimated_then_sorted": False,
            "score_ordinalized": True,
            "oracle_used": False,
        }

    stratum_columns = tuple(map(str, settings.get(
        "stratum_dr_columns", ("baseline_need_level", "current_care_profile")
    )))
    if not stratum_columns:
        raise ValueError("Prespecified stratum baseline requires stratum columns")
    missing = sorted(
        set(stratum_columns).difference(train.columns)
        | set(stratum_columns).difference(score_frame.columns)
    )
    if missing:
        raise ValueError(f"Prespecified stratum columns are missing: {missing}")

    def keys(frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, list(stratum_columns)].fillna("__missing__").astype(
            str
        ).agg("|".join, axis=1).to_numpy()

    train_stratum = keys(train)
    score_stratum = keys(score_frame)
    cell_means: dict[tuple[str, str], float] = {}
    for profile in sorted(set(train_profile)):
        for stratum in sorted(set(train_stratum[train_profile == profile])):
            indices = np.flatnonzero(
                (train_profile == profile) & (train_stratum == stratum)
            )
            mean, mass = _weighted_group_mean(signal, weight, indices)
            fallback = profile_means[profile]
            cell_means[(profile, stratum)] = float(
                (mass * mean + stratum_strength * fallback)
                / (mass + stratum_strength)
            )
    raw = np.asarray([
        cell_means.get((profile, stratum), profile_means[profile])
        for profile, stratum in zip(score_profile, score_stratum)
    ], dtype=float)
    supported_cells = sum(
        (profile, stratum) in cell_means
        for profile, stratum in zip(score_profile, score_stratum)
    )
    return _ordinal_percentile(raw), {
        "training_objective": "prespecified_preindex_stratum_profile_dr_group_ranking",
        "personalized": "coarse_prespecified_groups_only",
        "stratum_columns": list(stratum_columns),
        "stratum_thresholds_fit_from_data": False,
        "cell_groups": int(len(cell_means)),
        "score_rows_with_seen_cell": int(supported_cells),
        "score_rows_with_profile_fallback": int(len(score_frame) - supported_cells),
        "profile_shrinkage_strength": profile_strength,
        "stratum_shrinkage_strength": stratum_strength,
        "individual_cate_estimated_then_sorted": False,
        "score_ordinalized": True,
        "oracle_used": False,
    }


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
    pair_metadata = [
        column for column in ("profile_treatment",) if column in supervision
    ]
    safe_supervision = supervision.loc[
        supervision.causal_supervision_status.eq("supported")
        & supervision.split.isin((TRAIN_SPLIT, VALIDATION_SPLIT)),
        [
            "patient_id",
            "care_profile_id",
            "split",
            "dr_pseudo_outcome",
            "dr_reliability_weight",
            *pair_metadata,
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


def _variant_seed(base_seed: int, variant: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}|{str(variant)}".encode("utf-8")
    ).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 2_000_000_000


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
    contrastive_loss_type = str(
        settings.get("contrastive_loss_type", "legacy_margin")
    )
    if contrastive_loss_type not in {
        "legacy_margin", "normalized_euclidean_margin",
        "normalized_cosine_binary", "ordered_cosine_triplet",
        "ordered_cosine_triplet_margin",
    }:
        raise ValueError(f"Unknown contrastive_loss_type: {contrastive_loss_type}")
    contrastive_representation_source = str(
        settings.get("contrastive_representation_source", "projection_head")
    )
    if contrastive_representation_source not in {
        "projection_head",
        "shared_clinical_encoder",
        "shared_joint_encoder",
    }:
        raise ValueError(
            "Unknown contrastive_representation_source: "
            f"{contrastive_representation_source}"
        )
    use_shared_joint_encoder = bool(
        settings.get("use_shared_joint_encoder", False)
    )
    profile_conditioning = str(
        settings.get("profile_conditioning", "legacy_concat")
    )
    if profile_conditioning not in {"legacy_concat", "film"}:
        raise ValueError(
            "profile_conditioning must be 'legacy_concat' or 'film'"
        )
    if profile_conditioning == "film" and not use_shared_joint_encoder:
        raise ValueError(
            "FiLM profile conditioning requires use_shared_joint_encoder=true"
        )
    if (
        contrastive_representation_source == "shared_joint_encoder"
        and not use_shared_joint_encoder
    ):
        raise ValueError(
            "shared_joint_encoder contrastive source requires "
            "use_shared_joint_encoder=true"
        )
    contrastive_temperature = float(
        settings.get("contrastive_temperature", 0.10)
    )
    if not 0.0 < contrastive_temperature <= 1.0:
        raise ValueError("contrastive_temperature must be in (0, 1]")
    _validate_cosine_geometry(settings)
    raw_contrastive_within = settings.get(
        "contrastive_within_profile_fraction"
    )
    contrastive_within_fraction = (
        None
        if raw_contrastive_within is None
        else float(raw_contrastive_within)
    )
    contrastive_target_mode = str(
        settings.get("contrastive_target_mode", "hard_bins")
    )
    if contrastive_target_mode not in {
        "hard_bins", "soft_repeat_distance",
        "profile_residual_soft_distance", "ordered_triplets",
        "dynamic_dr_margin", "opposite_arm_dynamic_dr_margin",
    }:
        raise ValueError(f"Unknown contrastive_target_mode: {contrastive_target_mode}")
    ordered_loss = contrastive_loss_type in {
        "ordered_cosine_triplet", "ordered_cosine_triplet_margin",
    }
    if (contrastive_target_mode == "ordered_triplets") != ordered_loss:
        raise ValueError(
            "ordered_triplets and an ordered cosine triplet loss must be configured "
            "together"
        )
    if contrastive_loss_type == "ordered_cosine_triplet_margin" and not (
        0.0 < float(settings.get("contrastive_margin", 0.0)) <= 2.0
    ):
        raise ValueError("ordered cosine triplet margin must be in (0, 2]")
    dynamic_target_mode = contrastive_target_mode in {
        "dynamic_dr_margin", "opposite_arm_dynamic_dr_margin",
    }
    if dynamic_target_mode != (
        contrastive_loss_type == "normalized_euclidean_margin"
    ):
        raise ValueError(
            "Dynamic DR margin targets and normalized_euclidean_margin must be "
            "configured together"
        )
    opposite_arm_target = (
        contrastive_target_mode == "opposite_arm_dynamic_dr_margin"
    )
    if opposite_arm_target and contrastive_within_fraction not in {None, 1.0}:
        raise ValueError(
            "opposite_arm_dynamic_dr_margin requires "
            "contrastive_within_profile_fraction=1.0"
        )

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
        "pair_sampling_backend": str(
            settings.get("pair_sampling_backend", "exhaustive")
        ),
        "pair_sampling_maximum_attempts": settings.get(
            "pair_sampling_maximum_attempts"
        ),
    }
    rank_pair_sampling_started = time.perf_counter()
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
    ranking_pair_sampling_seconds = float(
        time.perf_counter() - rank_pair_sampling_started
    )
    allow_same_patient_contrastive = bool(
        settings.get(
            "allow_same_patient_contrastive_pairs",
            settings["allow_same_patient_cross_profile_pairs"],
        )
    )
    contrastive_pair_sampling_started = time.perf_counter()
    if contrastive_target_mode == "ordered_triplets":
        contrastive_pairs: ContrastiveBatch = _sample_ordered_contrastive_triplets(
            train,
            triplets=int(settings["contrastive_pairs"]),
            seed=int(contrastive_seed),
            minimum_order_agreement=float(
                settings.get(
                    "contrastive_minimum_order_agreement",
                    settings["minimum_direction_agreement"],
                )
            ),
            minimum_distance_gap=float(
                settings.get("contrastive_minimum_distance_gap", 0.20)
            ),
            within_profile_fraction=float(
                0.50
                if contrastive_within_fraction is None
                else contrastive_within_fraction
            ),
            allow_same_patient_cross_profile=allow_same_patient_contrastive,
            use_reliability_weight=bool(
                settings.get("contrastive_use_reliability_weight", False)
            ),
            maximum_pair_weight=float(settings["maximum_pair_weight"]),
            maximum_attempts=settings.get("pair_sampling_maximum_attempts"),
            max_reuse_per_opportunity=settings.get(
                "contrastive_max_reuse_per_opportunity"
            ),
        )
    else:
        candidate_multiplier = (
            int(settings.get("contrastive_candidate_pool_multiplier", 4))
            if dynamic_target_mode else 1
        )
        if candidate_multiplier < 1:
            raise ValueError("contrastive_candidate_pool_multiplier must be positive")
        contrastive_pairs = _sample_contrastive_pairs(
            train,
            pairs=int(settings["contrastive_pairs"]) * candidate_multiplier,
            bins=int(settings["contrastive_response_bins"]),
            seed=int(contrastive_seed),
            minimum_label_agreement=float(
                settings.get(
                    "contrastive_minimum_label_agreement",
                    settings["minimum_direction_agreement"],
                )
            ),
            allow_same_patient_cross_profile=allow_same_patient_contrastive,
            within_profile_fraction=(
                1.0 if opposite_arm_target else contrastive_within_fraction
            ),
            positive_fraction=float(
                settings.get("contrastive_positive_fraction", 0.50)
            ),
            use_reliability_weight=bool(
                settings.get("contrastive_use_reliability_weight", False)
            ),
            maximum_pair_weight=float(settings["maximum_pair_weight"]),
            target_mode=(
                "soft_repeat_distance"
                if dynamic_target_mode
                else contrastive_target_mode
            ),
            soft_bandwidth=float(
                settings.get("contrastive_soft_bandwidth", 1.0)
            ),
            soft_positive_threshold=float(
                settings.get("contrastive_soft_positive_threshold", 0.70)
            ),
            soft_negative_threshold=float(
                settings.get("contrastive_soft_negative_threshold", 0.30)
            ),
            uncertainty_penalty=float(
                settings.get("contrastive_uncertainty_penalty", 1.0)
            ),
            opposite_treatment_arms_only=opposite_arm_target,
            pair_sampling_backend=str(
                settings.get("pair_sampling_backend", "exhaustive")
            ),
            pair_sampling_maximum_attempts=settings.get(
                "pair_sampling_maximum_attempts"
            ),
            max_reuse_per_opportunity=settings.get(
                "contrastive_max_reuse_per_opportunity"
            ),
            minimum_profile_samples=int(
                settings.get("contrastive_minimum_profile_samples", 2)
            ),
        )
    contrastive_pair_sampling_seconds = float(
        time.perf_counter() - contrastive_pair_sampling_started
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
            sampling_backend=train_pairs.sampling_backend,
            sampling_attempts=train_pairs.sampling_attempts,
            requested_pairs=train_pairs.requested_pairs,
            quota_satisfied=train_pairs.quota_satisfied,
            estimated_peak_candidate_bytes=(
                train_pairs.estimated_peak_candidate_bytes
            ),
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
                    contrastive_seed=int(contrastive_seed),
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
                "architecture_family": "siamese_patient_profile_causal_ranker",
                "primary_objective": "direct_pairwise_causal_ranking",
                "contrastive_role": (
                    "auxiliary_causal_response_representation_regularizer"
                    if float(contrastive_weight) > 0.0
                    else "disabled_rank_only_ablation"
                ),
                "inference_output": "ranking_head_raw_priority_score_only",
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
                    contrastive_seed=int(contrastive_seed),
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
        elif variant in {
            "linear_pairwise_additive", "linear_pairwise_interactions",
        }:
            linear_seed = _variant_seed(int(gbdt_seed), variant)
            scores, history, audit = _fit_linear_pairwise_ranker(
                train,
                validation,
                score_frame,
                train_pairs,
                validation_pairs,
                profile_mapping,
                numeric_features,
                categorical_features,
                settings,
                model_seed=linear_seed,
                interactions=variant == "linear_pairwise_interactions",
            )
            history.insert(0, "method", variant)
            histories.append(history)
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    scores,
                    primary,
                    "direct_pairwise_linear_ordinal_baseline",
                )
            )
            variant_audits[variant] = audit
        elif variant in {"profile_mean_dr", "prespecified_stratum_dr"}:
            scores, audit = _fit_group_dr_baseline(
                train,
                score_frame,
                settings,
                use_strata=variant == "prespecified_stratum_dr",
            )
            score_outputs.append(
                _method_score_frame(
                    score_frame,
                    variant,
                    scores,
                    primary,
                    (
                        "coarse_prespecified_stratum_causal_baseline"
                        if variant == "prespecified_stratum_dr"
                        else "nonpersonalized_profile_average_causal_baseline"
                    ),
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
    # The scientific identity of a validation-pair sample must not depend on the
    # display name of the primary arm that consumed it.
    validation_pair_hash = _pair_batch_hash(validation_pairs)
    primary_audit = variant_audits.get(primary, {})
    audit = {
        "status": "trained",
        "architecture_family": "siamese_patient_profile_causal_ranker",
        "primary_training_objective": "direct_pairwise_causal_ranking",
        "contrastive_component_role": "auxiliary_representation_regularizer",
        "contrastive_may_replace_primary_objective": False,
        "contrastive_retention_policy": (
            "retain_only_after_nonoracle_validation_selection_and_frozen_controls"
        ),
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
        "fixed_training_pair_hash": _pair_batch_hash(train_pairs),
        "pair_sampling_backend": str(
            settings.get("pair_sampling_backend", "exhaustive")
        ),
        "ranking_pair_sampling_seconds": ranking_pair_sampling_seconds,
        "contrastive_pair_sampling_seconds": contrastive_pair_sampling_seconds,
        "training_pair_sampling_attempts": int(train_pairs.sampling_attempts),
        "validation_pair_sampling_attempts": int(
            validation_pairs.sampling_attempts
        ),
        "training_pair_sampling_quota_satisfied": bool(
            train_pairs.quota_satisfied
        ),
        "validation_pair_sampling_quota_satisfied": bool(
            validation_pairs.quota_satisfied
        ),
        "training_pair_sampling_estimated_peak_candidate_bytes": int(
            train_pairs.estimated_peak_candidate_bytes
        ),
        "validation_pair_sampling_estimated_peak_candidate_bytes": int(
            validation_pairs.estimated_peak_candidate_bytes
        ),
        "allow_same_patient_cross_profile_pairs": bool(
            settings["allow_same_patient_cross_profile_pairs"]
        ),
        "allow_same_patient_contrastive_pairs": bool(
            settings.get(
                "allow_same_patient_contrastive_pairs",
                settings["allow_same_patient_cross_profile_pairs"],
            )
        ),
        "contrastive_loss_type": contrastive_loss_type,
        "contrastive_representation_source": contrastive_representation_source,
        "contrastive_target_mode": contrastive_target_mode,
        "contrastive_temperature": contrastive_temperature,
        "contrastive_within_profile_fraction": contrastive_within_fraction,
        "contrastive_positive_fraction": float(
            settings.get("contrastive_positive_fraction", 0.50)
        ),
        "contrastive_minimum_label_agreement": float(
            settings.get(
                "contrastive_minimum_label_agreement",
                settings["minimum_direction_agreement"],
            )
        ),
        "contrastive_use_reliability_weight": bool(
            settings.get("contrastive_use_reliability_weight", False)
        ),
        "contrastive_soft_bandwidth": float(
            settings.get("contrastive_soft_bandwidth", 1.0)
        ),
        "contrastive_soft_positive_threshold": float(
            settings.get("contrastive_soft_positive_threshold", 0.70)
        ),
        "contrastive_soft_negative_threshold": float(
            settings.get("contrastive_soft_negative_threshold", 0.30)
        ),
        "contrastive_uncertainty_penalty": float(
            settings.get("contrastive_uncertainty_penalty", 1.0)
        ),
        "contrastive_candidate_pool_multiplier": int(
            settings.get("contrastive_candidate_pool_multiplier", 4)
            if dynamic_target_mode else 1
        ),
        "contrastive_refresh_epochs": int(
            settings.get("contrastive_refresh_epochs", 1)
        ),
        "dynamic_model_ite_or_cate_used": False,
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
