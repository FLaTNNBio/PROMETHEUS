from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .causal_signals import aggregate_repeated_signals, repeated_doubly_robust_signals
from .losses import (
    apply_response_bin_boundaries,
    causal_contrastive_loss,
    fit_response_bin_boundaries,
    pairwise_causal_ranking_loss,
    permute_response_labels,
    prometheus_objective,
    sample_contrastive_pairs,
    sample_ranking_pairs,
)


TRAINING_MODES = (
    "independent_rankers",
    "unified_rank_only",
    "unified_random_contrastive",
    "unified_covariate_contrastive",
    "prometheus_causal_contrastive",
)

MODE_TO_CONTRASTIVE_SOURCE = {
    "independent_rankers": "none",
    "unified_rank_only": "none",
    "unified_random_contrastive": "random",
    "unified_covariate_contrastive": "covariate",
    "prometheus_causal_contrastive": "causal",
}


@dataclass(frozen=True)
class TransitionArrays:
    x: np.ndarray
    transition_treatment: np.ndarray
    outcome: np.ndarray
    nuisance: np.ndarray
    patient_ids: np.ndarray | None = None


class _TransitionConditionedRanker(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        transition_count: int,
        hidden_dim: int,
        latent_dim: int,
        separate_contrastive_head: bool = False,
    ):
        super().__init__()
        self.separate_contrastive_head = bool(separate_contrastive_head)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.transition_embedding = nn.Embedding(transition_count, hidden_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.scoring_head = nn.Linear(latent_dim, 1)
        self.contrastive_projection_head = (
            nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            if self.separate_contrastive_head
            else None
        )

    def _joint_features(
        self, x: torch.Tensor, transition_index: int | torch.Tensor
    ) -> torch.Tensor:
        h = self.clinical_encoder(x)
        if isinstance(transition_index, int):
            index = torch.full((len(x),), transition_index, dtype=torch.long, device=x.device)
        else:
            index = torch.as_tensor(transition_index, dtype=torch.long, device=x.device)
            if index.ndim == 0:
                index = index.expand(len(x))
            if index.shape != (len(x),):
                raise ValueError("transition_index tensor must contain one index per row")
        v = self.transition_embedding(index)
        return torch.cat((h, v, h * v), dim=1)

    def forward(self, x: torch.Tensor, transition_index: int | torch.Tensor):
        z = self.projection_head(self._joint_features(x, transition_index))
        return self.scoring_head(z).squeeze(-1), z

    def contrastive_embedding(
        self, x: torch.Tensor, transition_index: int | torch.Tensor
    ) -> torch.Tensor:
        """Return the auxiliary embedding without reusing the ranking projection in v2."""

        joint = self._joint_features(x, transition_index)
        if self.contrastive_projection_head is None:
            return self.projection_head(joint)
        return self.contrastive_projection_head(joint)


def deterministic_kmeans_groups(x, groups: int, seed: int, iterations: int = 25) -> np.ndarray:
    """Deterministic train-only covariate grouping for the covariate ablation."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < groups or groups < 2:
        raise ValueError("Need x[n,p] with at least one row per covariate group")
    if not np.isfinite(x).all():
        raise ValueError("Covariates must be finite")
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=groups, replace=False)].copy()
    labels = np.full(len(x), -1, dtype=np.int64)
    for _ in range(iterations):
        squared_distance = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        updated = np.argmin(squared_distance, axis=1).astype(np.int64)
        if np.array_equal(updated, labels):
            break
        labels = updated
        nearest_distance = squared_distance[np.arange(len(x)), labels]
        for group in range(groups):
            members = x[labels == group]
            if len(members):
                centers[group] = members.mean(axis=0)
            else:
                replacement = int(np.argmax(nearest_distance))
                centers[group] = x[replacement]
                labels[replacement] = group
                nearest_distance[replacement] = -np.inf
    return labels


def _as_transition_arrays(value) -> TransitionArrays:
    if isinstance(value, TransitionArrays):
        result = value
    elif len(value) == 4:
        result = TransitionArrays(*value)
    elif len(value) == 5:
        result = TransitionArrays(*value)
    else:
        raise ValueError("Transition data must contain x, D, outcome, nuisance, and optional patient_ids")
    x = np.asarray(result.x, dtype=float)
    d = np.asarray(result.transition_treatment, dtype=float)
    outcome = np.asarray(result.outcome, dtype=float)
    nuisance = np.asarray(result.nuisance, dtype=float)
    patient_ids = None if result.patient_ids is None else np.asarray(result.patient_ids).astype(str)
    if nuisance.ndim == 2 and nuisance.shape == (len(x), 3):
        nuisance = nuisance[:, None, :]
    if x.ndim != 2 or nuisance.ndim != 3 or nuisance.shape[0] != len(x) or nuisance.shape[2] != 3:
        raise ValueError("Expected x[n,p] and nuisance[n,repetitions,3]")
    if nuisance.shape[1] < 1:
        raise ValueError("At least one nuisance repetition is required")
    if d.shape != (len(x),) or outcome.shape != (len(x),):
        raise ValueError("Transition treatment and outcome must align with x")
    if not set(np.unique(d)).issubset({0.0, 1.0}):
        raise ValueError("Transition treatment D must be binary")
    if patient_ids is not None and patient_ids.shape != (len(x),):
        raise ValueError("Patient identifiers must align with x")
    if not all(np.isfinite(array).all() for array in (x, d, outcome, nuisance)):
        raise ValueError("Training arrays must be finite")
    return TransitionArrays(x, d, outcome, nuisance, patient_ids)


def _assert_train_validation_patient_disjoint(train: dict, validation: dict) -> None:
    train_ids = {
        patient
        for data in train.values()
        if data.patient_ids is not None
        for patient in data.patient_ids
    }
    validation_ids = {
        patient
        for data in validation.values()
        if data.patient_ids is not None
        for patient in data.patient_ids
    }
    overlap = train_ids.intersection(validation_ids)
    if overlap:
        example = sorted(overlap)[:5]
        raise ValueError(f"Patient-level train/validation leakage detected: {example}")


def _fixed_signals(
    data: TransitionArrays,
    propensity_clip_epsilon: float,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    repeated = repeated_doubly_robust_signals(
        data.transition_treatment,
        data.outcome,
        data.nuisance,
        propensity_clip_epsilon,
    )
    return aggregate_repeated_signals(repeated, aggregation), repeated


def _cyclic_slice(values: np.ndarray, start: int, size: int) -> np.ndarray:
    if not len(values):
        return values
    return values[np.arange(start, start + size) % len(values)]


def _module_gradient_norm(module: nn.Module) -> float:
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum().cpu())
    return float(math.sqrt(squared))


def _mean_or_nan(values) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _observed_autoc(score, signal) -> float:
    order = np.argsort(-np.asarray(score))
    values = np.asarray(signal)[order]
    curve = np.cumsum(values) / np.arange(1, len(values) + 1) - np.mean(values)
    return float(np.trapezoid(curve, x=np.arange(1, len(values) + 1) / len(values)))


class PrometheusRanker:
    """Unified transition-conditioned direct causal ranker.

    The network receives pre-index covariates and a transition identifier.  A
    Repeated cross-fitted doubly robust signals are aggregated without oracle
    information. Only sufficiently stable pair directions guide the standard
    pairwise causal ranking loss; causal contrastive regularization on the latent
    representation remains optional. Scores are ordinal and comparable only
    within the same transition.
    """

    def __init__(
        self,
        transition_names=("1_to_2", "2_to_3", "3_to_4", "4_to_5", "5_to_6"),
        training_mode="prometheus_causal_contrastive",
        pairs_per_transition=6000,
        contrastive_pairs_per_transition=3000,
        epochs=20,
        batch_size=512,
        hidden_dim=128,
        latent_dim=32,
        lr=3e-4,
        patience=4,
        seed=42,
        device="cpu",
        lambda_con=0.10,
        lambda_reg=1e-6,
        contrastive_margin=1.0,
        num_response_bins=5,
        negative_bin_separation=2,
        ranking_min_signal_gap=0.0,
        propensity_clip_epsilon=0.02,
        positive_negative_pair_ratio=1.0,
        normalize_contrastive_embeddings=False,
        pair_reliability_weighting=None,
        dr_signal_aggregation="median",
        min_pair_direction_agreement=0.80,
    ):
        self.transition_names = tuple(transition_names)
        self.training_mode = str(training_mode)
        self.pairs_per_transition = int(pairs_per_transition)
        self.contrastive_pairs_per_transition = int(contrastive_pairs_per_transition)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = str(device)
        self.lambda_con = float(lambda_con)
        self.lambda_reg = float(lambda_reg)
        self.contrastive_margin = float(contrastive_margin)
        self.num_response_bins = int(num_response_bins)
        self.negative_bin_separation = int(negative_bin_separation)
        self.ranking_min_signal_gap = ranking_min_signal_gap
        self.propensity_clip_epsilon = float(propensity_clip_epsilon)
        self.positive_negative_pair_ratio = float(positive_negative_pair_ratio)
        self.normalize_contrastive_embeddings = bool(normalize_contrastive_embeddings)
        self.dr_signal_aggregation = str(dr_signal_aggregation)
        self.min_pair_direction_agreement = float(min_pair_direction_agreement)
        reliability = pair_reliability_weighting or {}
        self.reliability_enabled = bool(reliability.get("enabled", False))
        self.reliability_max_weight = float(reliability.get("max_weight", 1.0))

        if self.training_mode not in TRAINING_MODES:
            raise ValueError(f"Unknown training mode {self.training_mode!r}")
        self.contrastive_source = MODE_TO_CONTRASTIVE_SOURCE[self.training_mode]
        if len(self.transition_names) < 1 or len(set(self.transition_names)) != len(self.transition_names):
            raise ValueError("Need at least one unique transition name")
        if self.training_mode == "independent_rankers" and len(self.transition_names) != 1:
            raise ValueError("independent_rankers must be fitted one transition at a time")
        if self.training_mode in {"independent_rankers", "unified_rank_only"} and self.lambda_con != 0:
            raise ValueError("Ranking-only modes require lambda_con = 0")
        if self.pairs_per_transition < 1 or self.batch_size < 2 or self.epochs < 1 or self.patience < 1:
            raise ValueError("Invalid training sizes")
        if self.contrastive_source != "none" and self.contrastive_pairs_per_transition < 1:
            raise ValueError("Contrastive training requires a positive pair budget")
        if self.lambda_con < 0 or self.lambda_reg < 0 or self.contrastive_margin <= 0:
            raise ValueError("Invalid objective configuration")
        if self.num_response_bins < 2 or self.negative_bin_separation < 1:
            raise ValueError("Invalid response-bin configuration")
        if not 0 < self.propensity_clip_epsilon < 0.5:
            raise ValueError("propensity_clip_epsilon must be in (0, 0.5)")
        if self.positive_negative_pair_ratio <= 0 or self.reliability_max_weight <= 0:
            raise ValueError("Invalid pair sampling configuration")
        if self.dr_signal_aggregation not in {"median", "mean"}:
            raise ValueError("dr_signal_aggregation must be 'median' or 'mean'")
        if not 0.0 <= self.min_pair_direction_agreement <= 1.0:
            raise ValueError("min_pair_direction_agreement must be in [0, 1]")

    def _gap(self, name: str) -> float:
        if isinstance(self.ranking_min_signal_gap, dict):
            if name not in self.ranking_min_signal_gap:
                raise ValueError(f"Missing ranking_min_signal_gap for {name}")
            result = float(self.ranking_min_signal_gap[name])
        else:
            result = float(self.ranking_min_signal_gap)
        if result < 0:
            raise ValueError("ranking_min_signal_gap must be non-negative")
        return result

    def _training_groups(self, name, scaled_x, signal, transition_index):
        causal_groups = apply_response_bin_boundaries(signal, self.response_bin_boundaries[name])
        if self.contrastive_source == "causal":
            return causal_groups, True
        if self.contrastive_source == "random":
            return permute_response_labels(causal_groups, self.seed + 1009 * transition_index), True
        if self.contrastive_source == "covariate":
            return deterministic_kmeans_groups(
                scaled_x, self.num_response_bins, self.seed + 1009 * transition_index
            ), False
        raise ValueError("Ranking-only modes do not construct contrastive groups")

    def fit(self, training_data: dict, validation_data: dict):
        if set(training_data) != set(self.transition_names) or set(validation_data) != set(self.transition_names):
            raise ValueError("Training and validation data must contain every configured transition exactly once")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train = {name: _as_transition_arrays(training_data[name]) for name in self.transition_names}
        validation = {name: _as_transition_arrays(validation_data[name]) for name in self.transition_names}
        _assert_train_validation_patient_disjoint(train, validation)
        feature_dims = {value.x.shape[1] for value in (*train.values(), *validation.values())}
        if len(feature_dims) != 1:
            raise ValueError("All transitions must share the same feature schema")

        self.scaler = StandardScaler().fit(np.concatenate([train[name].x for name in self.transition_names]))
        scaled_train = {name: self.scaler.transform(train[name].x) for name in self.transition_names}
        scaled_validation = {name: self.scaler.transform(validation[name].x) for name in self.transition_names}
        train_fixed = {
            name: _fixed_signals(
                train[name], self.propensity_clip_epsilon, self.dr_signal_aggregation
            )
            for name in self.transition_names
        }
        validation_fixed = {
            name: _fixed_signals(
                validation[name], self.propensity_clip_epsilon, self.dr_signal_aggregation
            )
            for name in self.transition_names
        }
        train_signal = {name: value[0] for name, value in train_fixed.items()}
        train_signal_repeated = {name: value[1] for name, value in train_fixed.items()}
        validation_signal = {name: value[0] for name, value in validation_fixed.items()}
        self.response_bin_boundaries = {
            name: fit_response_bin_boundaries(train_signal[name], self.num_response_bins)
            for name in self.transition_names
        }
        self.ranking_gaps = {name: self._gap(name) for name in self.transition_names}
        tensors = {
            name: torch.as_tensor(scaled_train[name], dtype=torch.float32, device=self.device)
            for name in self.transition_names
        }
        validation_x = {
            name: torch.as_tensor(scaled_validation[name], dtype=torch.float32, device=self.device)
            for name in self.transition_names
        }
        self.model = _TransitionConditionedRanker(
            next(iter(feature_dims)), len(self.transition_names), self.hidden_dim, self.latent_dim
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        fixed_groups = {}
        if self.contrastive_source != "none" and self.lambda_con > 0:
            for transition_index, name in enumerate(self.transition_names):
                fixed_groups[name] = self._training_groups(
                    name, scaled_train[name], train_signal[name], transition_index
                )

        best = -np.inf
        state = None
        wait = 0
        self.best_epoch = -1
        self.best_validation_by_transition = {}
        self.history = []
        for epoch in range(self.epochs):
            ranking_pairs = {}
            for transition_index, name in enumerate(self.transition_names):
                batch = sample_ranking_pairs(
                    train_signal[name], self.pairs_per_transition, self.ranking_gaps[name],
                    self.seed + 10000 * epoch + 101 * transition_index,
                    self.reliability_enabled, self.reliability_max_weight,
                    train_signal_repeated[name], self.min_pair_direction_agreement,
                )
                if not len(batch.left):
                    raise RuntimeError(f"No valid causal ranking pairs for transition {name}")
                ranking_pairs[name] = batch

            contrastive_pairs = {}
            if self.contrastive_source != "none" and self.lambda_con > 0:
                for transition_index, name in enumerate(self.transition_names):
                    groups, ordered = fixed_groups[name]
                    contrastive_pairs[name] = sample_contrastive_pairs(
                        groups,
                        self.contrastive_pairs_per_transition,
                        self.seed + 20000 * epoch + 131 * transition_index,
                        self.negative_bin_separation,
                        self.positive_negative_pair_ratio,
                        ordered,
                        train_signal[name],
                        self.reliability_enabled,
                        self.reliability_max_weight,
                    )

            steps = max(
                math.ceil(len(ranking_pairs[name].left) / self.batch_size)
                for name in self.transition_names
            )
            self.model.train()
            total_values, ranking_values, contrastive_values, regularization_values = [], [], [], []
            transition_rank_values = {name: [] for name in self.transition_names}
            transition_con_values = {name: [] for name in self.transition_names}
            distance_diagnostics = {name: [] for name in self.transition_names}
            gradient_values = {
                "shared_encoder": [], "transition_embeddings": [],
                "projection_head": [], "scoring_head": [],
            }
            for step in range(steps):
                per_transition_ranking = []
                per_transition_contrastive = []
                for transition_index, name in enumerate(self.transition_names):
                    pairs = ranking_pairs[name]
                    size = min(self.batch_size, len(pairs.left))
                    start = step * self.batch_size
                    i = torch.as_tensor(_cyclic_slice(pairs.left, start, size), dtype=torch.long, device=self.device)
                    j = torch.as_tensor(_cyclic_slice(pairs.right, start, size), dtype=torch.long, device=self.device)
                    direction = torch.as_tensor(_cyclic_slice(pairs.target, start, size), dtype=torch.float32, device=self.device)
                    weights = torch.as_tensor(_cyclic_slice(pairs.weights, start, size), dtype=torch.float32, device=self.device)
                    score_i, _ = self.model(tensors[name][i], transition_index)
                    score_j, _ = self.model(tensors[name][j], transition_index)
                    rank_loss = pairwise_causal_ranking_loss(score_i, score_j, direction, weights)
                    per_transition_ranking.append(rank_loss)
                    transition_rank_values[name].append(float(rank_loss.detach().cpu()))

                    con_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    if name in contrastive_pairs and len(contrastive_pairs[name].left):
                        pairs_con = contrastive_pairs[name]
                        con_size = min(self.batch_size, len(pairs_con.left))
                        ci = torch.as_tensor(_cyclic_slice(pairs_con.left, start, con_size), dtype=torch.long, device=self.device)
                        cj = torch.as_tensor(_cyclic_slice(pairs_con.right, start, con_size), dtype=torch.long, device=self.device)
                        similar = torch.as_tensor(_cyclic_slice(pairs_con.target, start, con_size), dtype=torch.float32, device=self.device)
                        con_weights = torch.as_tensor(_cyclic_slice(pairs_con.weights, start, con_size), dtype=torch.float32, device=self.device)
                        _, z_i = self.model(tensors[name][ci], transition_index)
                        _, z_j = self.model(tensors[name][cj], transition_index)
                        con_loss, diagnostics = causal_contrastive_loss(
                            z_i, z_j, similar, self.contrastive_margin,
                            self.normalize_contrastive_embeddings, con_weights, True,
                        )
                        distance_diagnostics[name].append(diagnostics)
                    per_transition_contrastive.append(con_loss)
                    transition_con_values[name].append(float(con_loss.detach().cpu()))

                ranking_loss = torch.stack(per_transition_ranking).mean()
                contrastive_loss = torch.stack(per_transition_contrastive).mean()
                l2_penalty = sum(parameter.square().sum() for parameter in self.model.parameters())
                total_loss = prometheus_objective(
                    ranking_loss, contrastive_loss, l2_penalty, self.lambda_con, self.lambda_reg
                )
                optimizer.zero_grad()
                total_loss.backward()
                gradient_values["shared_encoder"].append(_module_gradient_norm(self.model.clinical_encoder))
                gradient_values["transition_embeddings"].append(_module_gradient_norm(self.model.transition_embedding))
                gradient_values["projection_head"].append(_module_gradient_norm(self.model.projection_head))
                gradient_values["scoring_head"].append(_module_gradient_norm(self.model.scoring_head))
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_values.append(float(total_loss.detach().cpu()))
                ranking_values.append(float(ranking_loss.detach().cpu()))
                contrastive_values.append(float(contrastive_loss.detach().cpu()))
                regularization_values.append(float((self.lambda_reg * l2_penalty).detach().cpu()))

            self.model.eval()
            validation_metrics = {}
            with torch.no_grad():
                for transition_index, name in enumerate(self.transition_names):
                    score, _ = self.model(validation_x[name], transition_index)
                    validation_metrics[name] = _observed_autoc(score.cpu().numpy(), validation_signal[name])
            validation_metric = float(np.mean(list(validation_metrics.values())))
            row = {
                "epoch": epoch,
                "total_loss": float(np.mean(total_values)),
                "ranking_loss": float(np.mean(ranking_values)),
                "contrastive_loss": float(np.mean(contrastive_values)),
                "regularization_loss": float(np.mean(regularization_values)),
                "validation_mean_observed_autoc": validation_metric,
                "gradient_norm_shared_encoder": float(np.mean(gradient_values["shared_encoder"])),
                "gradient_norm_transition_embeddings": float(np.mean(gradient_values["transition_embeddings"])),
                "gradient_norm_projection_head": float(np.mean(gradient_values["projection_head"])),
                "gradient_norm_scoring_head": float(np.mean(gradient_values["scoring_head"])),
            }
            for name in self.transition_names:
                pair = ranking_pairs[name]
                con_pair = contrastive_pairs.get(name)
                diagnostics = distance_diagnostics[name]
                row.update({
                    f"loss_{name}": float(np.mean(transition_rank_values[name])) + self.lambda_con * float(np.mean(transition_con_values[name])),
                    f"ranking_loss_{name}": float(np.mean(transition_rank_values[name])),
                    f"contrastive_loss_{name}": float(np.mean(transition_con_values[name])),
                    f"valid_ranking_pairs_{name}": int(len(pair.left)),
                    f"valid_pair_fraction_{name}": float(1.0 - pair.discarded_fraction),
                    f"discarded_pair_fraction_{name}": pair.discarded_fraction,
                    f"discarded_ambiguous_pair_fraction_{name}": max(
                        0.0, pair.discarded_fraction - pair.stability_discarded_fraction
                    ),
                    f"discarded_unstable_pair_fraction_{name}": pair.stability_discarded_fraction,
                    f"mean_pair_direction_agreement_{name}": pair.mean_direction_agreement,
                    f"positive_contrastive_pairs_{name}": int(np.sum(con_pair.target == 1)) if con_pair is not None else 0,
                    f"negative_contrastive_pairs_{name}": int(np.sum(con_pair.target == 0)) if con_pair is not None else 0,
                    f"mean_positive_latent_distance_{name}": _mean_or_nan(d["mean_positive_distance"] for d in diagnostics),
                    f"mean_negative_latent_distance_{name}": _mean_or_nan(d["mean_negative_distance"] for d in diagnostics),
                    f"active_negative_margin_fraction_{name}": _mean_or_nan(d["active_negative_margin_fraction"] for d in diagnostics),
                    f"validation_observed_autoc_{name}": validation_metrics[name],
                })
            self.history.append(row)
            if validation_metric > best + 1e-6:
                best = validation_metric
                state = copy.deepcopy(self.model.state_dict())
                wait = 0
                self.best_epoch = epoch
                self.best_validation_by_transition = dict(validation_metrics)
            else:
                wait += 1
            if wait >= self.patience:
                break

        if state is None:
            raise RuntimeError("Ranker did not produce a valid checkpoint")
        self.model.load_state_dict(state)
        self.best_validation_autoc = float(best)
        self.training_diagnostics = {
            "model": "prometheus_unified_transition_conditioned_ranker",
            "training_mode": self.training_mode,
            "transition_names": list(self.transition_names),
            "transition_weights": {name: 1.0 / len(self.transition_names) for name in self.transition_names},
            "contrastive_source": self.contrastive_source,
            "cross_fitted_doubly_robust_signal_fixed": True,
            "dr_signal_aggregation": self.dr_signal_aggregation,
            "nuisance_repetitions": {
                name: int(train[name].nuisance.shape[1]) for name in self.transition_names
            },
            "min_pair_direction_agreement": self.min_pair_direction_agreement,
            "pairwise_causal_ranking_loss": True,
            "score_generated_pair_labels": False,
            "cross_transition_pairs": False,
            "lambda_con": self.lambda_con,
            "lambda_reg": self.lambda_reg,
            "contrastive_margin": self.contrastive_margin,
            "num_response_bins": self.num_response_bins,
            "negative_bin_separation": self.negative_bin_separation,
            "ranking_min_signal_gap": self.ranking_gaps,
            "propensity_clip_epsilon": self.propensity_clip_epsilon,
            "pairs_per_transition": self.pairs_per_transition,
            "positive_negative_pair_ratio": self.positive_negative_pair_ratio,
            "normalize_contrastive_embeddings": self.normalize_contrastive_embeddings,
            "pair_reliability_weighting": {
                "enabled": self.reliability_enabled,
                "max_weight": self.reliability_max_weight,
            },
            "response_bin_boundaries": {
                name: boundaries.tolist() for name, boundaries in self.response_bin_boundaries.items()
            },
            "response_bins_fitted_on": (
                "rank_train_aggregated_repeated_cross_fitted_doubly_robust_signals_only"
            ),
            "best_epoch": self.best_epoch,
            "best_validation_mean_observed_autoc": self.best_validation_autoc,
            "best_validation_by_transition": self.best_validation_by_transition,
        }
        return self

    def response_bins(self, signal, transition_name):
        if transition_name not in self.response_bin_boundaries:
            raise ValueError(f"Unknown transition {transition_name!r}")
        return apply_response_bin_boundaries(signal, self.response_bin_boundaries[transition_name])

    def _predict(self, x, transition_name):
        if transition_name not in self.transition_names:
            raise ValueError(f"Unknown transition {transition_name!r}")
        transformed = torch.as_tensor(
            self.scaler.transform(np.asarray(x, dtype=float)), dtype=torch.float32, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(transformed, self.transition_names.index(transition_name))

    def predict_score(self, x, transition_name):
        score, _ = self._predict(x, transition_name)
        return score.cpu().numpy()

    def predict_representation(self, x, transition_name):
        _, representation = self._predict(x, transition_name)
        return representation.cpu().numpy()

    def rank(self, x, transition_name):
        return np.argsort(-self.predict_score(x, transition_name))
