from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {table_name}: {sorted(missing)}"
        )


def read_synthea_tables(csv_dir: Path) -> dict[str, pd.DataFrame]:
    filenames = {
        "patients": "patients.csv",
        "conditions": "conditions.csv",
        "encounters": "encounters.csv",
        "medications": "medications.csv",
    }

    tables: dict[str, pd.DataFrame] = {}

    for name, filename in filenames.items():
        path = csv_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

        tables[name] = pd.read_csv(path, low_memory=False)

    return tables


def build_patient_features(
    tables: dict[str, pd.DataFrame],
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    patients = tables["patients"].copy()
    conditions = tables["conditions"].copy()
    encounters = tables["encounters"].copy()
    medications = tables["medications"].copy()

    require_columns(
        patients,
        {"Id", "BIRTHDATE", "GENDER"},
        "patients.csv",
    )
    require_columns(
        conditions,
        {"PATIENT", "START", "CODE"},
        "conditions.csv",
    )
    require_columns(
        encounters,
        {"PATIENT", "START", "ENCOUNTERCLASS"},
        "encounters.csv",
    )
    require_columns(
        medications,
        {"PATIENT", "START", "CODE"},
        "medications.csv",
    )

    patients["BIRTHDATE"] = pd.to_datetime(
        patients["BIRTHDATE"],
        errors="coerce",
    )

    patients["age"] = (
        reference_date - patients["BIRTHDATE"]
    ).dt.days.div(365.25)

    patients = patients.loc[
        patients["BIRTHDATE"].notna()
        & patients["age"].between(50, 100)
    ].copy()

    lookback_start = reference_date - pd.Timedelta(days=365)

    # Conditions observed before the index date.
    conditions["START"] = pd.to_datetime(
        conditions["START"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    prior_conditions = conditions.loc[
        conditions["START"].between(
            lookback_start,
            reference_date,
            inclusive="left",
        )
    ]

    condition_features = (
        prior_conditions.groupby("PATIENT")
        .agg(
            condition_count=("CODE", "nunique"),
        )
        .reset_index()
        .rename(columns={"PATIENT": "Id"})
    )

    # Encounters occurring in the one-year lookback window.
    encounters["START"] = pd.to_datetime(
        encounters["START"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    prior_encounters = encounters.loc[
        encounters["START"].between(
            lookback_start,
            reference_date,
            inclusive="left",
        )
    ].copy()

    prior_encounters["is_inpatient"] = (
        prior_encounters["ENCOUNTERCLASS"]
        .astype(str)
        .str.lower()
        .eq("inpatient")
        .astype(int)
    )

    prior_encounters["is_emergency"] = (
        prior_encounters["ENCOUNTERCLASS"]
        .astype(str)
        .str.lower()
        .isin({"emergency", "urgentcare"})
        .astype(int)
    )

    encounter_features = (
        prior_encounters.groupby("PATIENT")
        .agg(
            encounter_count=("ENCOUNTERCLASS", "size"),
            prior_inpatient=("is_inpatient", "sum"),
            prior_emergency=("is_emergency", "sum"),
        )
        .reset_index()
        .rename(columns={"PATIENT": "Id"})
    )

    # Medications observed in the one-year lookback.
    medications["START"] = pd.to_datetime(
        medications["START"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    prior_medications = medications.loc[
        medications["START"].between(
            lookback_start,
            reference_date,
            inclusive="left",
        )
    ]

    medication_features = (
        prior_medications.groupby("PATIENT")
        .agg(
            medication_count=("CODE", "nunique"),
        )
        .reset_index()
        .rename(columns={"PATIENT": "Id"})
    )

    features = patients[
        ["Id", "age", "GENDER"]
    ].rename(
        columns={
            "Id": "patient_id",
            "GENDER": "gender",
        }
    )

    for feature_table in (
        condition_features,
        encounter_features,
        medication_features,
    ):
        feature_table = feature_table.rename(
            columns={"Id": "patient_id"}
        )

        features = features.merge(
            feature_table,
            on="patient_id",
            how="left",
        )

    count_columns = [
        "condition_count",
        "encounter_count",
        "prior_inpatient",
        "prior_emergency",
        "medication_count",
    ]

    features[count_columns] = (
        features[count_columns]
        .fillna(0)
        .astype(float)
    )

    features["female"] = (
        features["gender"]
        .astype(str)
        .str.upper()
        .eq("F")
        .astype(int)
    )

    features["multimorbidity"] = (
        features["condition_count"] >= 2
    ).astype(int)

    features["polypharmacy"] = (
        features["medication_count"] >= 5
    ).astype(int)

    return features


def simulate_causal_problem(
    features: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = features.copy()

    # Standardised covariates used by the data-generating process.
    age_z = (data["age"] - data["age"].mean()) / (
        data["age"].std() + 1e-8
    )

    condition_z = np.log1p(data["condition_count"])
    inpatient_z = np.log1p(data["prior_inpatient"])
    emergency_z = np.log1p(data["prior_emergency"])
    medication_z = np.log1p(data["medication_count"])

    # Risk under usual care.
    risk_logit = (
        -2.8
        + 0.35 * age_z
        + 0.42 * condition_z
        + 0.70 * inpatient_z
        + 0.38 * emergency_z
        + 0.20 * medication_z
        + 0.35 * data["multimorbidity"]
        + 0.20 * data["polypharmacy"]
        + 0.35
        * data["multimorbidity"]
        * (data["prior_inpatient"] > 0)
    )

    p_y0 = sigmoid(risk_logit)

    # True heterogeneous benefit of intensive care management.
    # Positive tau means reduction in adverse-event probability.
    tau_true = (
        0.015
        + 0.025 * data["multimorbidity"]
        + 0.040 * (data["prior_inpatient"] >= 1)
        + 0.025 * (data["prior_emergency"] >= 2)
        + 0.020 * data["polypharmacy"]
        - 0.025 * (data["age"] >= 90)
        - 0.020
        * (
            (data["condition_count"] >= 8)
            & (data["prior_inpatient"] >= 3)
        )
        + 0.030
        * (
            data["multimorbidity"]
            & (data["prior_emergency"] >= 1)
        )
    )

    tau_true = np.clip(tau_true, -0.03, 0.25)

    p_y1 = np.clip(
        p_y0 - tau_true,
        0.001,
        0.999,
    )

    # Confounded observational treatment allocation.
    propensity_logit = (
        -1.2
        + 0.30 * age_z
        + 0.45 * condition_z
        + 0.65 * inpatient_z
        + 0.30 * emergency_z
        + 0.25 * data["multimorbidity"]
        - 0.20 * (data["age"] >= 90)
    )

    propensity_true = np.clip(
        sigmoid(propensity_logit),
        0.03,
        0.97,
    )

    treatment = rng.binomial(1, propensity_true)

    # Potential outcomes are generated independently conditional on X.
    y0 = rng.binomial(1, p_y0)
    y1 = rng.binomial(1, p_y1)

    observed_outcome = np.where(treatment == 1, y1, y0)

    data["index_date"] = "2025-01-01"
    data["treatment"] = treatment
    data["outcome"] = observed_outcome

    # These columns are hidden from the learner and retained for evaluation.
    data["true_propensity"] = propensity_true
    data["true_risk_usual_care"] = p_y0
    data["true_risk_treated"] = p_y1
    data["true_cate_benefit"] = tau_true
    data["potential_outcome_0"] = y0
    data["potential_outcome_1"] = y1

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a semi-synthetic causal dataset from Synthea CSV files."
    )

    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/causal_population.csv"),
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2025-01-01",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    reference_date = pd.Timestamp(args.reference_date)

    tables = read_synthea_tables(args.csv_dir.resolve())

    features = build_patient_features(
        tables=tables,
        reference_date=reference_date,
    )

    if features.empty:
        raise RuntimeError(
            "No eligible patients were found. "
            "Check the reference date and generated population."
        )

    dataset = simulate_causal_problem(
        features=features,
        seed=args.seed,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        args.output,
        index=False,
    )

    print(f"Patients: {len(dataset):,}")
    print(f"Treated rate: {dataset['treatment'].mean():.3f}")
    print(f"Outcome rate: {dataset['outcome'].mean():.3f}")
    print(
        "Mean true benefit: "
        f"{dataset['true_cate_benefit'].mean():.4f}"
    )
    print(f"Dataset written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
