"""Strong non-oracle tree baselines for global patient-action prioritization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler

from .global_pairs import sample_global_ranking_pairs
from .opportunities import OpportunityArrays


def _reliability_weight(repeated_signal: np.ndarray) -> np.ndarray:
    repeated = np.asarray(repeated_signal, dtype=float)
    median = np.median(repeated, axis=1, keepdims=True)
    weight = 1.0 / (1.0 + np.median(np.abs(repeated - median), axis=1))
    return weight / weight.mean() if weight.mean() > 0 else np.ones(len(weight))


def _action_augmented(x: np.ndarray, action: np.ndarray, action_count: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    action = np.asarray(action, dtype=np.int64)
    if x.ndim != 2 or action.shape != (len(x),):
        raise ValueError("Patient features and action indices must align")
    if np.any((action < 0) | (action >= int(action_count))):
        raise ValueError("Unknown action index in tree baseline")
    one_hot = np.eye(int(action_count), dtype=float)[action]
    return np.column_stack((x, one_hot))


class DirectPairwiseGBDTRanker:
    """Direct pairwise causal ranker using stable DR comparisons and GBDT.

    The model never regresses an individual CATE. It learns whether one observed
    patient-action opportunity should outrank another, then scores new opportunities
    by their average win probability against a fixed, train-only anchor panel.
    """

    def __init__(
        self,
        action_count: int,
        within_pairs: int = 15_000,
        cross_pairs: int = 15_000,
        min_signal_gap_days: float = 2.0,
        min_direction_agreement: float = 0.80,
        anchor_count: int = 256,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        learning_rate: float = 0.05,
        seed: int = 0,
    ):
        self.action_count = int(action_count)
        self.within_pairs = int(within_pairs)
        self.cross_pairs = int(cross_pairs)
        self.min_signal_gap_days = float(min_signal_gap_days)
        self.min_direction_agreement = float(min_direction_agreement)
        self.anchor_count = int(anchor_count)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        if self.action_count < 2 or min(self.within_pairs, self.cross_pairs) < 1:
            raise ValueError("Direct pairwise GBDT requires actions and pair budgets")
        if self.anchor_count < self.action_count or self.max_iter < 1:
            raise ValueError("Invalid direct pairwise GBDT settings")

    def fit(self, training: OpportunityArrays) -> "DirectPairwiseGBDTRanker":
        self.scaler = StandardScaler().fit(training.x)
        phi = _action_augmented(
            self.scaler.transform(training.x), training.transition_index,
            self.action_count,
        )
        pairs = sample_global_ranking_pairs(
            training,
            within_pairs=self.within_pairs,
            cross_pairs=self.cross_pairs,
            min_signal_gap_days=self.min_signal_gap_days,
            min_direction_agreement=self.min_direction_agreement,
            seed=self.seed + 101,
            balance_transition_pairs=True,
            allow_same_patient_cross_pairs=False,
            reliability_weighting=True,
        )
        difference = phi[pairs.left] - phi[pairs.right]
        label = (pairs.target > 0).astype(int)
        # Antisymmetric augmentation prevents arbitrary left/right conventions.
        design = np.vstack((difference, -difference))
        target = np.concatenate((label, 1 - label))
        weight = np.concatenate((pairs.weights, pairs.weights))
        self.model = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=1e-4,
            random_state=self.seed,
        ).fit(design, target, sample_weight=weight)

        rng = np.random.default_rng(self.seed + 211)
        per_action = max(1, self.anchor_count // self.action_count)
        anchors = []
        for action in range(self.action_count):
            members = np.flatnonzero(training.transition_index == action)
            if not len(members):
                raise ValueError(f"No training anchors for action {action}")
            anchors.extend(rng.choice(
                members, size=min(per_action, len(members)), replace=False
            ).tolist())
        remaining = self.anchor_count - len(anchors)
        if remaining > 0:
            pool = np.setdiff1d(np.arange(len(training.x)), np.asarray(anchors))
            anchors.extend(rng.choice(
                pool, size=min(remaining, len(pool)), replace=False
            ).tolist())
        self.anchor_phi = phi[np.asarray(anchors, dtype=int)]
        self.diagnostics = {
            "method": "direct_pairwise_gbdt_ranker",
            "oracle_supervision": False,
            "target": "stable_pairwise_repeated_dr_order",
            "training_pairs": int(len(pairs)),
            "within_pairs": int((~pairs.is_cross).sum()),
            "cross_pairs": int(pairs.is_cross.sum()),
            "anchor_count": int(len(self.anchor_phi)),
            "seed": self.seed,
        }
        return self

    def predict_opportunity_scores(self, x, action_index) -> np.ndarray:
        phi = _action_augmented(
            self.scaler.transform(np.asarray(x, dtype=float)),
            np.asarray(action_index, dtype=np.int64), self.action_count,
        )
        score = np.empty(len(phi), dtype=float)
        chunk_size = max(1, 65_536 // max(1, len(self.anchor_phi)))
        for start in range(0, len(phi), chunk_size):
            current = phi[start:start + chunk_size]
            difference = current[:, None, :] - self.anchor_phi[None, :, :]
            probability = self.model.predict_proba(
                difference.reshape(-1, difference.shape[-1])
            )[:, 1]
            score[start:start + len(current)] = probability.reshape(
                len(current), len(self.anchor_phi)
            ).mean(axis=1)
        return score


class DRGBDTPriority:
    """Pointwise DR pseudo-outcome regression baseline, pooled or per action."""

    def __init__(
        self,
        action_count: int,
        pooled: bool,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        learning_rate: float = 0.05,
        seed: int = 0,
    ):
        self.action_count = int(action_count)
        self.pooled = bool(pooled)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

    def _model(self, offset: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=1e-4,
            random_state=self.seed + int(offset),
        )

    def fit(self, training: OpportunityArrays) -> "DRGBDTPriority":
        weight = _reliability_weight(training.repeated_signal)
        if self.pooled:
            design = _action_augmented(
                training.x, training.transition_index, self.action_count
            )
            self.models = {"pooled": self._model(0).fit(
                design, training.signal, sample_weight=weight
            )}
        else:
            self.models = {}
            for action in range(self.action_count):
                mask = training.transition_index == action
                if not mask.any():
                    raise ValueError(f"No pointwise DR rows for action {action}")
                self.models[action] = self._model(1009 * action).fit(
                    training.x[mask], training.signal[mask],
                    sample_weight=weight[mask],
                )
        self.diagnostics = {
            "method": "pooled_dr_gbdt" if self.pooled else "independent_dr_gbdt",
            "oracle_supervision": False,
            "target": "robust_repeated_dr_signal",
            "training_rows": int(len(training.x)),
            "action_count": self.action_count,
            "seed": self.seed,
        }
        return self

    def predict_opportunity_scores(self, x, action_index) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        action = np.asarray(action_index, dtype=np.int64)
        if self.pooled:
            return self.models["pooled"].predict(
                _action_augmented(x, action, self.action_count)
            ).astype(float)
        score = np.empty(len(x), dtype=float)
        for value, model in self.models.items():
            mask = action == int(value)
            if mask.any():
                score[mask] = model.predict(x[mask])
        return score


class DRRandomForestPriority:
    """Action-conditioned ExtraTrees regression on cross-fitted DR supervision."""

    def __init__(
        self,
        action_count: int,
        estimators: int = 160,
        min_samples_leaf: int = 20,
        seed: int = 0,
    ):
        self.action_count = int(action_count)
        self.estimators = int(estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)

    def fit(self, training: OpportunityArrays) -> "DRRandomForestPriority":
        design = _action_augmented(
            training.x, training.transition_index, self.action_count
        )
        self.model = ExtraTreesRegressor(
            n_estimators=self.estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt",
            n_jobs=1,
            random_state=self.seed,
        ).fit(
            design, training.signal,
            sample_weight=_reliability_weight(training.repeated_signal),
        )
        self.diagnostics = {
            "method": "dr_random_forest_priority",
            "oracle_supervision": False,
            "target": "robust_repeated_dr_signal",
            "training_rows": int(len(training.x)),
            "estimators": self.estimators,
            "seed": self.seed,
        }
        return self

    def predict_opportunity_scores(self, x, action_index) -> np.ndarray:
        return self.model.predict(_action_augmented(
            np.asarray(x, dtype=float), np.asarray(action_index, dtype=np.int64),
            self.action_count,
        )).astype(float)


@dataclass
class _PolicyNode:
    action_values: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: "_PolicyNode | None" = None
    right: "_PolicyNode | None" = None


class DRPolicyTreePriority:
    """Greedy shallow multi-action policy tree fitted to train-only DR rewards."""

    def __init__(
        self,
        action_count: int,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
        thresholds_per_feature: int = 7,
        seed: int = 0,
    ):
        self.action_count = int(action_count)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.thresholds_per_feature = int(thresholds_per_feature)
        self.seed = int(seed)
        if self.action_count < 2 or self.max_depth < 1 or self.min_samples_leaf < 2:
            raise ValueError("Invalid DR policy-tree settings")

    def _leaf_values(self, indices: np.ndarray) -> tuple[np.ndarray, float]:
        reward = self.reward[indices]
        values = np.nan_to_num(reward, nan=0.0).sum(axis=0) / len(indices)
        welfare = float(max(0.0, values.max()) * len(indices))
        return values, welfare

    def _grow(self, indices: np.ndarray, depth: int) -> _PolicyNode:
        values, parent_welfare = self._leaf_values(indices)
        node = _PolicyNode(values)
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node
        best = None
        quantiles = np.linspace(0.10, 0.90, self.thresholds_per_feature)
        for feature in range(self.patient_x.shape[1]):
            candidates = np.unique(np.quantile(self.patient_x[indices, feature], quantiles))
            for threshold in candidates:
                left = indices[self.patient_x[indices, feature] <= threshold]
                right = indices[self.patient_x[indices, feature] > threshold]
                if min(len(left), len(right)) < self.min_samples_leaf:
                    continue
                _, left_welfare = self._leaf_values(left)
                _, right_welfare = self._leaf_values(right)
                gain = left_welfare + right_welfare - parent_welfare
                proposal = (float(gain), -feature, -float(threshold), feature, float(threshold), left, right)
                if best is None or proposal[:3] > best[:3]:
                    best = proposal
        if best is None or best[0] <= 1e-8:
            return node
        _, _, _, node.feature, node.threshold, left, right = best
        node.left = self._grow(left, depth + 1)
        node.right = self._grow(right, depth + 1)
        return node

    def fit(self, training: OpportunityArrays) -> "DRPolicyTreePriority":
        patients, inverse = np.unique(training.patient_ids, return_inverse=True)
        patient_x = np.empty((len(patients), training.x.shape[1]), dtype=float)
        reward = np.full((len(patients), self.action_count), np.nan, dtype=float)
        for patient_index in range(len(patients)):
            rows = np.flatnonzero(inverse == patient_index)
            patient_x[patient_index] = training.x[rows[0]]
            if not np.allclose(training.x[rows], patient_x[patient_index], atol=1e-10):
                raise ValueError("Policy-tree patient features differ across actions")
            for row in rows:
                reward[patient_index, training.transition_index[row]] = training.signal[row]
        self.patient_x = patient_x
        self.reward = reward
        self.root = self._grow(np.arange(len(patients)), 0)

        def count(node: _PolicyNode) -> tuple[int, int]:
            if node.left is None or node.right is None:
                return 1, 1
            left_nodes, left_leaves = count(node.left)
            right_nodes, right_leaves = count(node.right)
            return 1 + left_nodes + right_nodes, left_leaves + right_leaves

        nodes, leaves = count(self.root)
        self.diagnostics = {
            "method": "dr_policy_tree",
            "oracle_supervision": False,
            "target": "patient_action_robust_dr_reward_matrix",
            "training_patients": int(len(patients)),
            "nodes": int(nodes),
            "leaves": int(leaves),
            "max_depth": self.max_depth,
            "seed": self.seed,
        }
        return self

    def _values(self, row: np.ndarray) -> np.ndarray:
        node = self.root
        while node.left is not None and node.right is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.action_values

    def predict_opportunity_scores(self, x, action_index) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        action = np.asarray(action_index, dtype=np.int64)
        if x.ndim != 2 or action.shape != (len(x),):
            raise ValueError("Policy-tree prediction inputs must align")
        return np.asarray([
            self._values(row)[int(current_action)]
            for row, current_action in zip(x, action)
        ], dtype=float)
