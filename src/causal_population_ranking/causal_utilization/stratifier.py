from __future__ import annotations

import numpy as np
import pandas as pd


PACKAGES = {
    1: "ordinary prevention",
    2: "light monitoring",
    3: "structured disease management",
    4: "multidisciplinary integrated care",
    5: "intensive case management / home care",
    6: "level-six pathway",
}
TRANSITIONS = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6))


class HierarchicalCausalStratifier:
    """Apply capacity-qualified one-step transition decisions."""

    def __init__(self, thresholds: dict[str, float]):
        self.thresholds = thresholds

    def assign(
        self,
        clinical: pd.DataFrame,
        transition_predictions: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        out = clinical.copy()
        level = out.baseline_care_level.to_numpy(int).copy()
        identifiers = out.patient_id.astype(str)
        for lower, upper in TRANSITIONS:
            name = f"{lower}_to_{upper}"
            prediction = transition_predictions[name].set_index("patient_id").reindex(identifiers)
            score = prediction.score.to_numpy(float)
            support = np.array(
                [bool(value) if pd.notna(value) else False for value in prediction.supported],
                dtype=bool,
            )
            eligible = out[f"eligible_{name}"].to_numpy(bool)
            if "selected" in prediction:
                capacity_selected = prediction.selected.fillna(False).to_numpy(bool)
            else:
                capacity_selected = score >= self.thresholds[name]
            passed = (
                (out.baseline_care_level.to_numpy(int) == lower)
                & eligible
                & support
                & np.isfinite(score)
                & capacity_selected
            )
            level[passed] = upper
            out[f"score_{name}"] = score
            out[f"support_{name}"] = prediction.support_label.fillna("not_eligible").to_numpy()
            out[f"passed_{name}"] = passed
        out["assigned_causal_level"] = level
        out["recommended_package"] = pd.Series(level).map(PACKAGES).to_numpy()
        out["assignment_reason"] = [_reason(row) for _, row in out.iterrows()]
        return out


def _reason(row) -> str:
    passed = [
        column.removeprefix("passed_").replace("_", "→")
        for column in row.index
        if column.startswith("passed_") and bool(row[column])
    ]
    if passed:
        return "baseline care plus one-step causal escalation " + ", ".join(passed)
    return "baseline care retained; no supported capacity-qualified escalation"
