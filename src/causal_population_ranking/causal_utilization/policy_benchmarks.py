from __future__ import annotations

import numpy as np
import pandas as pd

from .stratifier import PACKAGES


SPECS = (
    ("1_to_2", 1, 2, "true_benefit_1_to_2"),
    ("2_to_3", 2, 3, "true_benefit_2_to_3"),
    ("3_to_4", 3, 4, "true_benefit_3_to_4"),
    ("4_to_5", 4, 5, "true_benefit_4_to_5"),
    ("5_to_6", 5, 6, "true_benefit_5_to_6"),
)


def _aligned_truth(clinical: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    ids = clinical.patient_id.astype(str)
    aligned = truth.assign(patient_id=truth.patient_id.astype(str)).set_index("patient_id").loc[ids].reset_index()
    if aligned.patient_id.tolist() != ids.tolist():
        raise RuntimeError("Patient alignment failed in policy benchmark")
    return aligned


def _finish(clinical: pd.DataFrame, levels: np.ndarray, reason: str) -> pd.DataFrame:
    out = clinical[["patient_id", "baseline_care_level"]].copy()
    out["current_care_level"] = out["baseline_care_level"]
    out["assigned_causal_level"] = levels
    out["recommended_package"] = pd.Series(levels).map(PACKAGES).to_numpy()
    out["assignment_reason"] = reason
    return out


def learned_transition_budgets(learned: pd.DataFrame) -> dict[str, int]:
    return {name: int(learned[f"passed_{name}"].astype(bool).sum()) for name, _, _, _ in SPECS}


def random_equal_resource_policy(clinical: pd.DataFrame, learned: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Random feasible cascade using exactly the learned transition counts."""
    rng = np.random.default_rng(seed)
    levels = clinical.baseline_care_level.to_numpy(int).copy()
    budgets = learned_transition_budgets(learned)
    realized = {}
    for name, lower, upper, _ in SPECS:
        candidates = np.where((levels == lower) & clinical[f"eligible_{name}"].to_numpy(bool))[0]
        k = budgets[name]
        if k > len(candidates):
            raise RuntimeError(f"Random benchmark infeasible for {name}: {k}>{len(candidates)}")
        chosen = rng.choice(candidates, size=k, replace=False) if k else np.array([], dtype=int)
        levels[chosen] = upper
        realized[name] = len(chosen)
    return _finish(clinical, levels, "random feasible cascade with learned resource counts"), {"budgets": budgets, "realized": realized}


def burden_only_score(cohort: pd.DataFrame) -> np.ndarray:
    """Transparent pre-index morbidity burden comparator; ordinal only."""
    return (
        np.log1p(cohort.condition_distinct.to_numpy(float))
        + .35*np.log1p(cohort.medication_distinct.to_numpy(float))
        + .50*cohort.multimorbidity.to_numpy(float)
        + .25*cohort.polypharmacy.to_numpy(float)
    )


def deterministic_score_equal_resource_policy(
    clinical: pd.DataFrame, learned: pd.DataFrame, score, label: str
) -> tuple[pd.DataFrame, dict]:
    """Feasible sequential policy using a fixed non-causal priority score."""
    score=np.asarray(score,float)
    if len(score)!=len(clinical) or not np.isfinite(score).all():
        raise ValueError(f"Invalid {label} comparator score")
    ids=clinical.patient_id.astype(str).to_numpy();levels=clinical.baseline_care_level.to_numpy(int).copy()
    budgets=learned_transition_budgets(learned);realized={}
    for name,lower,upper,_ in SPECS:
        candidates=np.where((levels==lower)&clinical[f"eligible_{name}"].to_numpy(bool))[0];k=budgets[name]
        if k>len(candidates):raise RuntimeError(f"{label} comparator infeasible for {name}: {k}>{len(candidates)}")
        order=np.lexsort((ids[candidates],-score[candidates]));chosen=candidates[order[:k]]
        levels[chosen]=upper;realized[name]=len(chosen)
    reason=f"{label} feasible cascade with learned resource counts"
    return _finish(clinical,levels,reason),{"budgets":budgets,"realized":realized,"score_definition":label}


def transition_score_equal_resource_policy(
    clinical: pd.DataFrame,
    learned: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """One-step policy with a distinct ordinal score for every transition."""
    identifiers = clinical.patient_id.astype(str).to_numpy()
    levels = clinical.baseline_care_level.to_numpy(int).copy()
    budgets = learned_transition_budgets(learned)
    realized = {}
    for name, lower, upper, _ in SPECS:
        frame = scores[name].assign(patient_id=scores[name].patient_id.astype(str)).set_index("patient_id")
        aligned = frame.reindex(identifiers).score.to_numpy(float)
        candidates = np.where(
            (levels == lower)
            & clinical[f"eligible_{name}"].to_numpy(bool)
            & np.isfinite(aligned)
        )[0]
        count = budgets[name]
        if count > len(candidates):
            raise RuntimeError(f"{label} comparator infeasible for {name}: {count}>{len(candidates)}")
        order = np.lexsort((identifiers[candidates], -aligned[candidates]))
        chosen = candidates[order[:count]]
        levels[chosen] = upper
        realized[name] = len(chosen)
    return _finish(clinical, levels, f"{label} one-step policy"), {
        "budgets": budgets,
        "realized": realized,
        "score_definition": label,
    }


def global_oracle_equal_resource_bound(
    clinical: pd.DataFrame, learned: pd.DataFrame, truth: pd.DataFrame, time_limit: float = 120.0
) -> tuple[float, dict]:
    """Componentwise top-k upper bound at learned resource counts.

    Each feasible cascade selects exactly k people from the eligible pool at a
    transition. Its benefit cannot exceed the sum of that pool's k largest true
    benefits. Adding these five independent maxima relaxes hierarchy and is
    therefore a rigorous, conservative upper bound for the full policy.
    """
    aligned = _aligned_truth(clinical, truth)
    budgets = learned_transition_budgets(learned)
    contributions = {}
    for name, _, _, benefit_column in SPECS:
        eligible = np.where(clinical[f"eligible_{name}"].to_numpy(bool))[0]
        values = aligned.loc[eligible, benefit_column].to_numpy(float)
        k = budgets[name]
        if k > len(values):
            raise RuntimeError(f"Oracle bound infeasible for {name}: {k}>{len(values)}")
        contribution = float(np.sort(values)[-k:].sum()) if k else 0.0
        contributions[name] = contribution
    objective = float(sum(contributions.values()))
    diagnostics = {
        "budgets": budgets,
        "solver_mode": "componentwise_topk_relaxation_upper_bound",
        "transition_upper_bound_contributions": contributions,
        "objective_increment": objective,
    }
    return objective, diagnostics
