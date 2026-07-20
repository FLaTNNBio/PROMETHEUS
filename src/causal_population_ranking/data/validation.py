from __future__ import annotations
import pandas as pd

ORACLE_PREFIXES = ("true_", "oracle_", "potential_", "latent_rank", "benefit_true")


def assert_no_oracle_columns(frame: pd.DataFrame) -> None:
    bad = [c for c in frame.columns if c.startswith(ORACLE_PREFIXES)]
    if bad: raise ValueError(f"Oracle leakage detected: {bad}")


def assert_patient_level_split_integrity(transition_frames: dict[str, pd.DataFrame]) -> None:
    """Assert one immutable split per patient across every transition."""
    patient_split: dict[str, str] = {}
    for transition, frame in transition_frames.items():
        missing = {"patient_id", "split"}.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing split columns for {transition}: {sorted(missing)}")
        for patient, split in zip(frame.patient_id.astype(str), frame.split.astype(str)):
            previous = patient_split.setdefault(patient, split)
            if previous != split:
                raise ValueError(
                    f"Patient-level train/validation/test leakage detected for {patient}: "
                    f"{previous} versus {split}"
                )


def validate_cohort(frame: pd.DataFrame) -> dict:
    if frame["patient_id"].duplicated().any(): raise ValueError("Duplicate patients")
    numeric = frame.select_dtypes("number")
    if not numeric.empty and not numeric.map(lambda x: pd.notna(x)).all().all():
        raise ValueError("Non-finite/missing numeric feature")
    return {"patients": len(frame), "constant_features": [c for c in numeric if numeric[c].nunique() <= 1],
            "missing_cells": int(frame.isna().sum().sum())}
