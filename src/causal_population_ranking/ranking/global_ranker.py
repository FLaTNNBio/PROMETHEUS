from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from .global_pairs import GlobalPairBatch, sample_global_contrastive_pairs, sample_global_ranking_pairs
from .losses import causal_contrastive_loss, pairwise_causal_ranking_loss
from .opportunities import OpportunityArrays, assert_patient_split_disjoint
from .prometheus_ranker import _TransitionConditionedRanker
from ..version import PROMETHEUS_SYNTHETIC_LEGACY_GLOBAL_METHOD_VERSION


GLOBAL_TRAINING_MODES = (
    "independent_local_rankers",
    "unified_local_ranker",
    "unified_global_ranker",
    "global_random_contrastive",
    "global_covariate_contrastive",
    "prometheus_global_causal_contrastive",
    "prometheus_global_causal_contrastive_v2",
)

TRAINING_MODE_ALIASES = {
    "independent_rankers": "independent_local_rankers",
    "unified_rank_only": "unified_local_ranker",
    "prometheus_causal_contrastive": "prometheus_global_causal_contrastive",
}

MODE_TO_CONTRASTIVE_SOURCE = {
    "independent_local_rankers": "none",
    "unified_local_ranker": "none",
    "unified_global_ranker": "none",
    "global_random_contrastive": "random",
    "global_covariate_contrastive": "covariate",
    "prometheus_global_causal_contrastive": "causal",
    "prometheus_global_causal_contrastive_v2": "causal",
}


def observed_opportunity_concordance(
    score,
    signal,
    transition_index,
    pair_kind: str = "global",
    max_pairs: int = 100_000,
    seed: int = 0,
) -> float:
    """Concordance on held-out DR signals for within, cross, or pooled pairs."""

    score = np.asarray(score, dtype=float)
    signal = np.asarray(signal, dtype=float)
    transition = np.asarray(transition_index, dtype=np.int64)
    if score.shape != signal.shape or score.shape != transition.shape or score.ndim != 1:
        raise ValueError("Opportunity concordance arrays must align")
    if pair_kind not in {"within", "cross", "global"}:
        raise ValueError("pair_kind must be within, cross, or global")
    if len(score) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    draw = min(max_pairs, max(2_000, len(score) * 30))
    left = rng.integers(0, len(score), draw)
    right = rng.integers(0, len(score), draw)
    valid = (left != right) & (signal[left] != signal[right])
    if pair_kind == "within":
        valid &= transition[left] == transition[right]
    elif pair_kind == "cross":
        valid &= transition[left] != transition[right]
    if not valid.any():
        return float("nan")
    product = (score[left[valid]] - score[right[valid]]) * (signal[left[valid]] - signal[right[valid]])
    return float(np.mean((product > 0) + 0.5 * (product == 0)))


def fixed_pair_concordance(score, pairs: GlobalPairBatch) -> dict[str, float]:
    """Evaluate one immutable validation pair set."""

    score = np.asarray(score, dtype=float)
    product = (score[pairs.left] - score[pairs.right]) * pairs.target

    def value(mask) -> float:
        selected = product[np.asarray(mask, dtype=bool)]
        return float(np.mean((selected > 0) + 0.5 * (selected == 0))) if len(selected) else float("nan")

    return {
        "within": value(~pairs.is_cross),
        "cross": value(pairs.is_cross),
        "global": value(np.ones(len(pairs), dtype=bool)),
    }


def validation_dr_policy_value(score, opportunities: OpportunityArrays, fraction: float) -> float:
    """Mean held-out DR signal for a top-score, at-most-one-action policy."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("validation policy fraction must be in (0,1]")
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    target = max(1, int(np.ceil(float(fraction) * len(np.unique(opportunities.patient_ids)))))
    selected, patients = [], set()
    for index in order:
        patient = str(opportunities.patient_ids[index])
        if patient in patients:
            continue
        selected.append(int(index))
        patients.add(patient)
        if len(selected) >= target:
            break
    return float(np.mean(opportunities.signal[selected])) if selected else float("nan")


def _cyclic(values: np.ndarray, start: int, size: int) -> np.ndarray:
    if not len(values):
        return values
    return values[np.arange(start, start + size) % len(values)]


def _gradient_norm(module: nn.Module) -> float:
    return float(math.sqrt(sum(
        float(parameter.grad.detach().square().sum().cpu())
        for parameter in module.parameters() if parameter.grad is not None
    )))


class GlobalPrometheusRanker:
    """Direct global causal ranker over patient-transition opportunities.

    The output is a globally comparable ordinal priority. It is not a calibrated
    individual treatment-effect estimate; cardinal allocation values are produced
    later by validation-only monotone calibration.
    """

    def __init__(
        self,
        transition_names=("1_to_2", "2_to_3", "3_to_4", "4_to_5", "5_to_6"),
        training_mode="prometheus_global_causal_contrastive",
        within_pairs_per_epoch=15_000,
        cross_pairs_per_epoch=15_000,
        beta_cross=0.50,
        min_signal_gap_days=2.0,
        min_pair_direction_agreement=0.80,
        balance_transition_pairs=True,
        allow_same_patient_cross_pairs=False,
        contrastive_pairs_per_epoch=8_000,
        positive_max_gap_days=3.0,
        negative_min_gap_days=15.0,
        lambda_con=0.10,
        lambda_reg=1e-6,
        contrastive_margin=1.0,
        reliability_weighting=False,
        reliability_max_weight=25.0,
        validation_beta_cross=0.50,
        checkpoint_metric="global_concordance",
        validation_policy_fraction=0.20,
        contrastive_pair_scope="pooled",
        contrastive_require_stable_direction=False,
        contrastive_reliability_weighting=False,
        epochs=20,
        batch_size=512,
        hidden_dim=128,
        latent_dim=32,
        learning_rate=3e-4,
        patience=4,
        permute_training_pair_labels=False,
        seed=42,
        device="cpu",
    ):
        self.transition_names = tuple(transition_names)
        self.training_mode = TRAINING_MODE_ALIASES.get(str(training_mode), str(training_mode))
        self.within_pairs_per_epoch = int(within_pairs_per_epoch)
        self.cross_pairs_per_epoch = int(cross_pairs_per_epoch)
        self.beta_cross = float(beta_cross)
        self.min_signal_gap_days = float(min_signal_gap_days)
        self.min_pair_direction_agreement = float(min_pair_direction_agreement)
        self.balance_transition_pairs = bool(balance_transition_pairs)
        self.allow_same_patient_cross_pairs = bool(allow_same_patient_cross_pairs)
        self.contrastive_pairs_per_epoch = int(contrastive_pairs_per_epoch)
        self.positive_max_gap_days = float(positive_max_gap_days)
        self.negative_min_gap_days = float(negative_min_gap_days)
        self.lambda_con = float(lambda_con)
        self.lambda_reg = float(lambda_reg)
        self.contrastive_margin = float(contrastive_margin)
        self.reliability_weighting = bool(reliability_weighting)
        self.reliability_max_weight = float(reliability_max_weight)
        self.validation_beta_cross = float(validation_beta_cross)
        self.checkpoint_metric = str(checkpoint_metric)
        self.validation_policy_fraction = float(validation_policy_fraction)
        self.contrastive_pair_scope = str(contrastive_pair_scope)
        self.contrastive_require_stable_direction = bool(
            contrastive_require_stable_direction
        )
        self.contrastive_reliability_weighting = bool(
            contrastive_reliability_weighting
        )
        self.separate_contrastive_head = (
            self.training_mode == "prometheus_global_causal_contrastive_v2"
        )
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.permute_training_pair_labels = bool(permute_training_pair_labels)
        self.seed = int(seed)
        self.device = str(device)
        if self.training_mode not in GLOBAL_TRAINING_MODES:
            raise ValueError(f"Unknown global training mode {self.training_mode!r}")
        self.contrastive_source = MODE_TO_CONTRASTIVE_SOURCE[self.training_mode]
        self.uses_cross_transition_pairs = self.training_mode not in {
            "independent_local_rankers", "unified_local_ranker"
        }
        if not self.uses_cross_transition_pairs:
            self.beta_cross = 0.0
            self.cross_pairs_per_epoch = 0
        if not 0.0 <= self.beta_cross <= 1.0 or not 0.0 <= self.validation_beta_cross <= 1.0:
            raise ValueError("Cross-transition loss and validation weights must be in [0, 1]")
        if self.checkpoint_metric not in {"global_concordance", "dr_policy_value"}:
            raise ValueError("checkpoint metric must be global_concordance or dr_policy_value")
        if not 0.0 < self.validation_policy_fraction <= 1.0:
            raise ValueError("validation_policy_fraction must be in (0,1]")
        if self.within_pairs_per_epoch < 1 or self.cross_pairs_per_epoch < 0:
            raise ValueError("Global pair budgets are invalid")
        if self.uses_cross_transition_pairs and self.cross_pairs_per_epoch < 1:
            raise ValueError("A global mode requires cross_pairs_per_epoch > 0")
        if not 0.0 <= self.min_pair_direction_agreement <= 1.0 or self.min_signal_gap_days < 0:
            raise ValueError("Pair direction reliability settings are invalid")
        if not 0 <= self.positive_max_gap_days < self.negative_min_gap_days:
            raise ValueError("Pooled contrastive thresholds must satisfy positive < negative")
        if self.contrastive_source == "none" and self.lambda_con != 0:
            raise ValueError("Ranking-only modes require lambda_con = 0")
        if self.contrastive_source != "none" and self.contrastive_pairs_per_epoch < 1:
            raise ValueError("Contrastive modes require a positive pair budget")
        if self.contrastive_pair_scope not in {
            "pooled", "within_action", "balanced_mixed"
        }:
            raise ValueError("Unknown contrastive pair scope")
        if self.lambda_con < 0 or self.lambda_reg < 0 or self.contrastive_margin <= 0:
            raise ValueError("Objective coefficients are invalid")
        if self.epochs < 1 or self.batch_size < 2 or self.patience < 1:
            raise ValueError("Training sizes are invalid")

    def _pair_loss(
        self,
        x: torch.Tensor,
        transition: torch.Tensor,
        batch: GlobalPairBatch,
        start: int,
    ) -> torch.Tensor:
        if not len(batch):
            return x.sum() * 0.0
        size = min(self.batch_size, len(batch))
        left = torch.as_tensor(_cyclic(batch.left, start, size), dtype=torch.long, device=self.device)
        right = torch.as_tensor(_cyclic(batch.right, start, size), dtype=torch.long, device=self.device)
        target = torch.as_tensor(_cyclic(batch.target, start, size), dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(_cyclic(batch.weights, start, size), dtype=torch.float32, device=self.device)
        score_left, _ = self.model(x[left], transition[left])
        score_right, _ = self.model(x[right], transition[right])
        return pairwise_causal_ranking_loss(score_left, score_right, target, weights)

    def fit(self, training: OpportunityArrays, validation: OpportunityArrays) -> "GlobalPrometheusRanker":
        assert_patient_split_disjoint(training.patient_ids, validation.patient_ids)
        if training.x.shape[1] != validation.x.shape[1]:
            raise ValueError("Training and validation feature schemas differ")
        if training.repeated_signal.shape[1] != validation.repeated_signal.shape[1]:
            raise ValueError("Training and validation nuisance repetition counts differ")
        if np.max(training.transition_index) >= len(self.transition_names):
            raise ValueError("Training contains an unknown transition index")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler = StandardScaler().fit(training.x)
        scaled_training = self.scaler.transform(training.x)
        scaled_validation = self.scaler.transform(validation.x)
        x = torch.as_tensor(scaled_training, dtype=torch.float32, device=self.device)
        transition = torch.as_tensor(training.transition_index, dtype=torch.long, device=self.device)
        self.model = _TransitionConditionedRanker(
            training.x.shape[1], len(self.transition_names), self.hidden_dim,
            self.latent_dim, separate_contrastive_head=self.separate_contrastive_head,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        validation_pairs = sample_global_ranking_pairs(
            validation,
            self.within_pairs_per_epoch,
            self.cross_pairs_per_epoch,
            self.min_signal_gap_days,
            self.min_pair_direction_agreement,
            self.seed + 700_001,
            self.balance_transition_pairs,
            self.allow_same_patient_cross_pairs,
            self.reliability_weighting,
            self.reliability_max_weight,
        )
        if not len(validation_pairs) or not (~validation_pairs.is_cross).any():
            raise RuntimeError("No fixed within-action validation pairs")
        if self.uses_cross_transition_pairs and not validation_pairs.is_cross.any():
            raise RuntimeError("No fixed cross-action validation pairs")
        initial_score = self._predict_scaled(scaled_validation, validation.transition_index)[0]
        initial_metrics = fixed_pair_concordance(initial_score, validation_pairs)
        initial_policy_value = validation_dr_policy_value(
            initial_score, validation, self.validation_policy_fraction
        )
        self.initial_validation_metric = (
            initial_policy_value
            if self.checkpoint_metric == "dr_policy_value"
            else initial_metrics["global"]
        )
        best, best_state, wait = -np.inf, None, 0
        self.best_epoch = -1
        self.history = []
        best_validation = {}
        for epoch in range(self.epochs):
            pairs = sample_global_ranking_pairs(
                training, self.within_pairs_per_epoch, self.cross_pairs_per_epoch,
                self.min_signal_gap_days, self.min_pair_direction_agreement,
                self.seed + epoch * 10_007, self.balance_transition_pairs,
                self.allow_same_patient_cross_pairs, self.reliability_weighting,
                self.reliability_max_weight,
            )
            if self.permute_training_pair_labels:
                permutation = np.random.default_rng(
                    self.seed + 900_001 + epoch * 10_007
                ).permutation(len(pairs.target))
                pairs = GlobalPairBatch(
                    pairs.left, pairs.right, pairs.target[permutation],
                    pairs.weights, pairs.is_cross, pairs.direction_agreement,
                    {**pairs.diagnostics, "training_pair_labels_permuted": True},
                )
            within = pairs.subset(~pairs.is_cross)
            cross = pairs.subset(pairs.is_cross)
            if not len(within) or (self.uses_cross_transition_pairs and not len(cross)):
                raise RuntimeError("No valid stable within/cross ranking pairs for this epoch")
            contrastive = None
            if self.contrastive_source != "none" and self.lambda_con > 0:
                contrastive = sample_global_contrastive_pairs(
                    training, self.contrastive_pairs_per_epoch, self.seed + 500_009 + epoch,
                    self.positive_max_gap_days, self.negative_min_gap_days,
                    self.contrastive_source, scaled_training,
                    self.contrastive_pair_scope,
                    self.contrastive_require_stable_direction,
                    self.min_pair_direction_agreement,
                    self.contrastive_reliability_weighting,
                )
                if not len(contrastive):
                    raise RuntimeError("No valid pooled contrastive pairs")
            steps = max(
                math.ceil(len(within) / self.batch_size),
                math.ceil(len(cross) / self.batch_size) if len(cross) else 1,
                math.ceil(len(contrastive) / self.batch_size) if contrastive is not None else 1,
            )
            epoch_values = {key: [] for key in (
                "total", "within", "cross", "contrastive", "positive_within_distance",
                "negative_within_distance", "positive_cross_distance", "negative_cross_distance",
                "positive_total_distance", "negative_total_distance",
                "encoder_gradient", "embedding_gradient", "contrastive_head_gradient",
            )}
            self.model.train()
            for step in range(steps):
                start = step * self.batch_size
                within_loss = self._pair_loss(x, transition, within, start)
                cross_loss = self._pair_loss(x, transition, cross, start)
                ranking_loss = (1.0 - self.beta_cross) * within_loss + self.beta_cross * cross_loss
                contrastive_loss = x.sum() * 0.0
                if contrastive is not None:
                    size = min(self.batch_size, len(contrastive))
                    ci = torch.as_tensor(_cyclic(contrastive.left, start, size), dtype=torch.long, device=self.device)
                    cj = torch.as_tensor(_cyclic(contrastive.right, start, size), dtype=torch.long, device=self.device)
                    similar = torch.as_tensor(_cyclic(contrastive.target, start, size), dtype=torch.float32, device=self.device)
                    con_weights = torch.as_tensor(
                        _cyclic(contrastive.weights, start, size),
                        dtype=torch.float32, device=self.device,
                    )
                    pair_cross = torch.as_tensor(_cyclic(contrastive.is_cross, start, size), dtype=torch.bool, device=self.device)
                    zi = self.model.contrastive_embedding(x[ci], transition[ci])
                    zj = self.model.contrastive_embedding(x[cj], transition[cj])
                    contrastive_loss = causal_contrastive_loss(
                        zi, zj, similar, margin=self.contrastive_margin,
                        weights=con_weights,
                    )
                    distance = torch.linalg.vector_norm(zi - zj, dim=1).detach().cpu().numpy()
                    similar_numpy = similar.detach().cpu().numpy() > 0.5
                    cross_numpy = pair_cross.detach().cpu().numpy()
                    for kind, kind_mask in (("within", ~cross_numpy), ("cross", cross_numpy)):
                        for label, label_mask in (("positive", similar_numpy), ("negative", ~similar_numpy)):
                            values = distance[kind_mask & label_mask]
                            if len(values):
                                epoch_values[f"{label}_{kind}_distance"].append(float(values.mean()))
                    for label, label_mask in (("positive", similar_numpy), ("negative", ~similar_numpy)):
                        values = distance[label_mask]
                        if len(values):
                            epoch_values[f"{label}_total_distance"].append(float(values.mean()))
                l2_penalty = sum(parameter.square().sum() for parameter in self.model.parameters())
                total_loss = ranking_loss + self.lambda_con * contrastive_loss + self.lambda_reg * l2_penalty
                optimizer.zero_grad()
                total_loss.backward()
                epoch_values["encoder_gradient"].append(_gradient_norm(self.model.clinical_encoder))
                epoch_values["embedding_gradient"].append(_gradient_norm(self.model.transition_embedding))
                if self.model.contrastive_projection_head is not None:
                    epoch_values["contrastive_head_gradient"].append(
                        _gradient_norm(self.model.contrastive_projection_head)
                    )
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                epoch_values["total"].append(float(total_loss.detach().cpu()))
                epoch_values["within"].append(float(within_loss.detach().cpu()))
                epoch_values["cross"].append(float(cross_loss.detach().cpu()))
                epoch_values["contrastive"].append(float(contrastive_loss.detach().cpu()))
            validation_score = self._predict_scaled(scaled_validation, validation.transition_index)[0]
            validation_metrics = fixed_pair_concordance(validation_score, validation_pairs)
            policy_value = validation_dr_policy_value(
                validation_score, validation, self.validation_policy_fraction
            )
            primary = (
                policy_value
                if self.checkpoint_metric == "dr_policy_value"
                else (
                    validation_metrics["global"]
                    if self.uses_cross_transition_pairs else validation_metrics["within"]
                )
            )
            def mean_or_nan(key: str) -> float:
                return float(np.mean(epoch_values[key])) if epoch_values[key] else float("nan")
            row = {
                "epoch": epoch, "total_loss": mean_or_nan("total"),
                "within_ranking_loss": mean_or_nan("within"), "cross_ranking_loss": mean_or_nan("cross"),
                "contrastive_loss": mean_or_nan("contrastive"),
                "valid_within_pairs": pairs.diagnostics["valid_within_pairs"],
                "valid_cross_pairs": pairs.diagnostics["valid_cross_pairs"],
                "discarded_ambiguous_fraction": pairs.diagnostics["discarded_ambiguous_fraction"],
                "discarded_unstable_fraction": pairs.diagnostics["discarded_unstable_fraction"],
                "direction_agreement": pairs.diagnostics["mean_direction_agreement"],
                "same_patient_cross_pair_fraction": pairs.diagnostics["same_patient_cross_pair_fraction"],
                "mean_positive_within_distance": mean_or_nan("positive_within_distance"),
                "mean_negative_within_distance": mean_or_nan("negative_within_distance"),
                "mean_positive_cross_distance": mean_or_nan("positive_cross_distance"),
                "mean_negative_cross_distance": mean_or_nan("negative_cross_distance"),
                "mean_positive_pooled_distance": mean_or_nan("positive_total_distance"),
                "mean_negative_pooled_distance": mean_or_nan("negative_total_distance"),
                "pair_counts_per_transition_pair": json.dumps(
                    pairs.diagnostics["pair_counts_per_transition_pair"], sort_keys=True
                ),
                "gradient_norm_shared_encoder": mean_or_nan("encoder_gradient"),
                "gradient_norm_transition_embeddings": mean_or_nan("embedding_gradient"),
                "gradient_norm_contrastive_head": mean_or_nan("contrastive_head_gradient"),
                "validation_within_observed_concordance": validation_metrics["within"],
                "validation_cross_observed_concordance": validation_metrics["cross"],
                "validation_global_observed_concordance": validation_metrics["global"],
                "validation_policy_value_dr": policy_value,
                "validation_checkpoint_metric": primary,
            }
            self.history.append(row)
            if np.isfinite(primary) and primary > best + 1e-7:
                best, wait = float(primary), 0
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                best_validation = {**validation_metrics, "policy_value_dr": policy_value}
                self.best_pair_diagnostics = pairs.diagnostics
                self.best_contrastive_diagnostics = contrastive.diagnostics if contrastive is not None else {}
            else:
                wait += 1
            if wait >= self.patience:
                break
        if best_state is None:
            raise RuntimeError("Global ranker did not produce a finite validation checkpoint")
        self.model.load_state_dict(best_state)
        self.training_diagnostics = {
            "model": "prometheus_global_transition_conditioned_ranker",
            "training_mode": self.training_mode,
            "ranking_unit": "patient_transition_opportunity",
            "score_semantics": "globally_comparable_ordinal_priority" if self.uses_cross_transition_pairs else "ordinal_within_transition",
            "cross_transition_pairs": self.uses_cross_transition_pairs,
            "separate_contrastive_head": self.separate_contrastive_head,
            "contrastive_pair_scope": self.contrastive_pair_scope,
            "contrastive_reliability_weighting": self.contrastive_reliability_weighting,
            "pairwise_causal_ranking_loss": True,
            "score_generated_pair_labels": False,
            "training_pair_labels_permuted": self.permute_training_pair_labels,
            "oracle_supervision": False,
            "signals_standardized_by_transition": False,
            "nuisance_repetitions": int(training.repeated_signal.shape[1]),
            "beta_cross": self.beta_cross,
            "validation_beta_cross": self.validation_beta_cross,
            "early_stopping_metric": self.checkpoint_metric,
            "checkpoint_metric": self.checkpoint_metric,
            "fixed_validation_pairs": True,
            "fixed_validation_pair_diagnostics": validation_pairs.diagnostics,
            "initial_validation_metric": self.initial_validation_metric,
            "best_epoch": self.best_epoch,
            "best_validation_metric": best,
            "best_validation": best_validation,
            "best_validation_within": best_validation.get("within"),
            "best_validation_cross": best_validation.get("cross"),
            "best_validation_global": best_validation.get("global"),
            "validation_policy_value_dr": best_validation.get("policy_value_dr"),
            "best_pair_diagnostics": self.best_pair_diagnostics,
            "best_contrastive_diagnostics": self.best_contrastive_diagnostics,
        }
        return self

    def _predict_scaled(self, scaled_x, transition_indices) -> tuple[np.ndarray, np.ndarray]:
        x = torch.as_tensor(np.asarray(scaled_x), dtype=torch.float32, device=self.device)
        transition = torch.as_tensor(np.asarray(transition_indices), dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            score, representation = self.model(x, transition)
        return score.cpu().numpy(), representation.cpu().numpy()

    def predict_opportunity_scores(self, x, transition_indices) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        transition_indices = np.asarray(transition_indices, dtype=np.int64)
        if x.ndim != 2 or transition_indices.shape != (len(x),):
            raise ValueError("x and mixed transition indices must align")
        return self._predict_scaled(self.scaler.transform(x), transition_indices)[0]

    def predict_opportunity_representations(self, x, transition_indices) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        transition_indices = np.asarray(transition_indices, dtype=np.int64)
        if x.ndim != 2 or transition_indices.shape != (len(x),):
            raise ValueError("x and mixed transition indices must align")
        return self._predict_scaled(self.scaler.transform(x), transition_indices)[1]

    def predict_score(self, x, transition_name: str) -> np.ndarray:
        if transition_name not in self.transition_names:
            raise ValueError(f"Unknown transition {transition_name!r}")
        x = np.asarray(x, dtype=float)
        return self.predict_opportunity_scores(
            x, np.full(len(x), self.transition_names.index(transition_name), dtype=np.int64)
        )

    def save(
        self,
        path: str | Path,
        method_version: str = PROMETHEUS_SYNTHETIC_LEGACY_GLOBAL_METHOD_VERSION,
    ) -> None:
        torch.save({
            "method_version": str(method_version),
            "state_dict": self.model.state_dict(),
            "transition_names": self.transition_names,
            "training_mode": self.training_mode,
            "feature_mean": self.scaler.mean_, "feature_scale": self.scaler.scale_,
            "hidden_dim": self.hidden_dim, "latent_dim": self.latent_dim,
            "training_diagnostics": self.training_diagnostics,
        }, Path(path))
