from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpportunityArrays:
    """Arrays whose rows are patient-transition opportunities, not patients."""

    x: np.ndarray
    transition_index: np.ndarray
    patient_ids: np.ndarray
    signal: np.ndarray
    repeated_signal: np.ndarray
    source_row: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        transition = np.asarray(self.transition_index, dtype=np.int64)
        patients = np.asarray(self.patient_ids).astype(str)
        signal = np.asarray(self.signal, dtype=float)
        repeated = np.asarray(self.repeated_signal, dtype=float)
        n = len(x)
        if x.ndim != 2 or transition.shape != (n,) or patients.shape != (n,) or signal.shape != (n,):
            raise ValueError("Opportunity x, transition, patient, and signal arrays must align")
        if repeated.ndim != 2 or repeated.shape[0] != n or repeated.shape[1] < 1:
            raise ValueError("repeated_signal must have shape [opportunities, repetitions]")
        if np.any(transition < 0):
            raise ValueError("Transition indices must be non-negative")
        if not all(np.isfinite(value).all() for value in (x, signal, repeated)):
            raise ValueError("Opportunity arrays must be finite")
        if self.source_row is not None and np.asarray(self.source_row).shape != (n,):
            raise ValueError("source_row must align with opportunities")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "transition_index", transition)
        object.__setattr__(self, "patient_ids", patients)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "repeated_signal", repeated)
        if self.source_row is not None:
            object.__setattr__(self, "source_row", np.asarray(self.source_row, dtype=np.int64))

    def subset(self, index) -> "OpportunityArrays":
        index = np.asarray(index)
        return OpportunityArrays(
            self.x[index], self.transition_index[index], self.patient_ids[index],
            self.signal[index], self.repeated_signal[index],
            None if self.source_row is None else self.source_row[index],
        )

    def to_frame(self, transition_names: tuple[str, ...]) -> pd.DataFrame:
        if np.any(self.transition_index >= len(transition_names)):
            raise ValueError("An opportunity transition index has no configured name")
        frame = pd.DataFrame({
            "patient_id": self.patient_ids,
            "transition_index": self.transition_index,
            "transition": [transition_names[index] for index in self.transition_index],
            "dr_signal_days": self.signal,
        })
        if self.source_row is not None:
            frame["source_row"] = self.source_row
        for repeat in range(self.repeated_signal.shape[1]):
            frame[f"dr_signal_repeat_{repeat}"] = self.repeated_signal[:, repeat]
        return frame


def concatenate_opportunities(values: list[OpportunityArrays]) -> OpportunityArrays:
    if not values:
        raise ValueError("Need at least one opportunity block")
    if len({value.x.shape[1] for value in values}) != 1:
        raise ValueError("All opportunity blocks must share a feature schema")
    if len({value.repeated_signal.shape[1] for value in values}) != 1:
        raise ValueError("All transitions must use the same nuisance repetition count")
    return OpportunityArrays(
        np.concatenate([value.x for value in values], axis=0),
        np.concatenate([value.transition_index for value in values]),
        np.concatenate([value.patient_ids for value in values]),
        np.concatenate([value.signal for value in values]),
        np.concatenate([value.repeated_signal for value in values], axis=0),
        np.concatenate([value.source_row for value in values])
        if all(value.source_row is not None for value in values) else None,
    )


def assert_patient_split_disjoint(*patient_groups) -> None:
    sets = [set(np.asarray(group).astype(str)) for group in patient_groups]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = sets[left].intersection(sets[right])
            if overlap:
                raise ValueError(f"Patient-level opportunity split leakage: {sorted(overlap)[:5]}")
