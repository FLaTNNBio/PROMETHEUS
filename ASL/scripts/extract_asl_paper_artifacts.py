#!/usr/bin/env python
"""Build reproducible ASL dataset-description and evaluation artifacts for the paper.

The script combines:
1. source-data counts and missingness from output_acg_2025.zip;
2. exact cohort reconstruction from the locked ASL configuration;
3. run-level and paired evaluation summaries from a completed campaign;
4. opportunity/profile diagnostics when the campaign preserved score artifacts.

It never uses oracle quantities for model selection. Oracle fields are reported only
in explicitly labelled evaluation tables.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from causal_population_ranking.applications.asl_semisynthetic_case import (
    ASL_FEATURE_COLUMNS,
    generate_asl_semisynthetic_cohort,
)

SCENARIOS = ("partially_aligned", "risk_benefit_misaligned", "mixed_response")
DEFAULT_VARIANT = "contrastive_pretrained_finetuned"


def _bootstrap_ci(values: Iterable[float], seed: int = 20260725) -> tuple[float, float, float, int]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(x.mean())
    if x.size == 1:
        return mean, mean, mean, 1
    rng = np.random.default_rng(seed)
    means = np.empty(10000, dtype=float)
    for i in range(len(means)):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi), int(x.size)


def _summary_rows(df: pd.DataFrame, metrics: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for metric in metrics:
            if metric not in group:
                continue
            mean, lo, hi, n = _bootstrap_ci(pd.to_numeric(group[metric], errors="coerce"))
            rows.append({**base, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n_runs": n})
    return pd.DataFrame(rows)


def _read_zip_csv(zf: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    names = [n for n in zf.namelist() if n.replace("\\", "/").endswith(suffix)]
    if len(names) != 1:
        raise FileNotFoundError(f"Expected one {suffix} in archive; found {len(names)}")
    with zf.open(names[0]) as fh:
        return pd.read_csv(fh, compression="gzip" if names[0].endswith(".gz") else None)


def source_dataset_tables(input_zip: Path, out: Path) -> dict[str, int]:
    with zipfile.ZipFile(input_zip) as zf:
        files = {
            "patients": "curated/patients.csv.gz",
            "patient_features": "curated/patient_features_2025.csv.gz",
            "hospitalizations": "curated/hospitalizations.csv.gz",
            "diagnoses": "curated/diagnoses_long.csv.gz",
            "exemptions": "curated/exemptions_long.csv.gz",
            "followup_deaths": "curated/followup_deaths.csv.gz",
        }
        frames = {name: _read_zip_csv(zf, suffix) for name, suffix in files.items()}
        dq_names = [n for n in zf.namelist() if n.endswith("reports/data_quality_summary.json")]
        dq = {}
        if len(dq_names) == 1:
            with zf.open(dq_names[0]) as fh:
                dq = json.load(fh)

    counts = {name: int(len(frame)) for name, frame in frames.items()}
    pd.DataFrame([
        {"source_table": name, "records": len(frame), "columns": len(frame.columns)}
        for name, frame in frames.items()
    ]).to_csv(out / "asl_source_table_counts.csv", index=False)

    feature = frames["patient_features"]
    miss = pd.DataFrame({
        "variable": feature.columns,
        "missing_n": [int(feature[c].isna().sum()) for c in feature.columns],
        "missing_rate": [float(feature[c].isna().mean()) for c in feature.columns],
        "unique_values": [int(feature[c].nunique(dropna=True)) for c in feature.columns],
        "dtype": [str(feature[c].dtype) for c in feature.columns],
    }).sort_values(["missing_rate", "variable"], ascending=[False, True])
    miss.to_csv(out / "asl_patient_feature_missingness.csv", index=False)

    selected_rows = []
    for c in [
        "age_at_index", "sex", "deceased_by_index", "hospitalization_count",
        "total_reported_los_days", "total_hospital_reimbursement",
        "exemption_record_count", "distinct_exemption_group_count",
        "distinct_diagnosis_count", "enrollment_district",
    ]:
        if c not in feature:
            continue
        s = feature[c]
        row: dict[str, Any] = {"variable": c, "n": int(s.notna().sum()), "missing_rate": float(s.isna().mean())}
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() >= max(10, int(0.5 * s.notna().sum())):
            row.update({
                "mean": float(numeric.mean()), "sd": float(numeric.std(ddof=0)),
                "median": float(numeric.median()), "q1": float(numeric.quantile(0.25)),
                "q3": float(numeric.quantile(0.75)), "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
            })
        else:
            vc = s.astype("string").fillna("<missing>").value_counts().head(10)
            row["top_categories"] = "; ".join(f"{k}:{v}" for k, v in vc.items())
        selected_rows.append(row)
    pd.DataFrame(selected_rows).to_csv(out / "asl_source_descriptive_statistics.csv", index=False)

    (out / "asl_data_quality_summary.json").write_text(json.dumps(dq, indent=2), encoding="utf-8")
    return counts


def reconstruct_cohorts(config: dict[str, Any], input_path: Path, out: Path, runs: int) -> pd.DataFrame:
    base = config["asl_semisynthetic"]
    rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    profile_rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for run_id in range(runs):
            offset = 10_000 * (scenario_index * runs + run_id + 1)
            settings = copy.deepcopy(base)
            settings["response_scenario"] = scenario
            settings["seed"] = int(settings["seed"]) + offset
            settings["causal_supervision"]["nuisance_seed"] = int(settings["causal_supervision"]["nuisance_seed"]) + offset
            n = int(settings["patients"])
            cohort = generate_asl_semisynthetic_cohort(
                input_path, n, seed=int(settings["seed"]), response_scenario=scenario,
                minimum_age=int(settings.get("minimum_age", 18)),
            )
            learner = cohort.learner
            # Opportunity counts are deterministic from eligibility and the patient split;
            # no nuisance model refit is needed for a descriptive extraction.
            opp_blocks = []
            for profile in (1, 2, 3):
                mask = learner[f"eligible_{profile}"].astype(bool)
                block = learner.loc[mask, ["patient_id", "split"]].copy()
                block["profile_id"] = profile
                opp_blocks.append(block)
            opp = pd.concat(opp_blocks, ignore_index=True)
            any_eligible = learner[["eligible_1", "eligible_2", "eligible_3"]].any(axis=1)
            test_patients = learner.loc[learner.split.eq("test"), "patient_id"].nunique()
            test_opp = opp.loc[opp.split.eq("test")]
            rows.append({
                "scenario": scenario, "run_id": run_id,
                "source_rows": cohort.audit["source_rows"],
                "active_adult_rows_before_sampling": cohort.audit["active_adult_rows_before_sampling"],
                "patients_retained": cohort.audit["patients_used"],
                "patients_with_admissible_profile": int(any_eligible.sum()),
                "admissible_opportunities": int(len(opp)),
                "patients_with_supported_profile": int(opp.patient_id.nunique()),
                "supported_opportunities": int(len(opp)),
                "opportunity_support_coverage": 1.0,
                "test_patients": int(test_patients),
                "test_opportunities": int(len(test_opp)),
                "nuisance_rows": int(learner.split.eq("nuisance_train").sum()),
                "minimum_stabilized_propensity": float(settings["causal_supervision"].get("propensity_clip", math.nan)),
            })
            temp = cohort.baseline_summary.copy()
            temp.insert(0, "run_id", run_id)
            temp.insert(0, "scenario", scenario)
            feature_rows.append(temp)
            for profile in (1, 2, 3):
                eligible = learner[f"eligible_{profile}"].astype(bool)
                opportunities = opp.profile_id.eq(profile)
                profile_rows.append({
                    "scenario": scenario, "run_id": run_id, "profile_id": profile,
                    "eligible_patients": int(eligible.sum()),
                    "eligibility_rate": float(eligible.mean()),
                    "opportunities": int(opportunities.sum()),
                    "assigned_patients": int(learner.assigned_profile.eq(profile).sum()),
                    "assignment_rate": float(learner.assigned_profile.eq(profile).mean()),
                    "mean_true_effect_evaluation_only": float(cohort.oracle[f"tau_{profile}"].mean()),
                    "negative_effect_rate_evaluation_only": float((cohort.oracle[f"tau_{profile}"] < 0).mean()),
                })

    cohort_df = pd.DataFrame(rows)
    cohort_df.to_csv(out / "asl_cohort_flow_by_run.csv", index=False)
    pd.concat(feature_rows, ignore_index=True).to_csv(out / "asl_baseline_summary_by_run.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(out / "asl_profile_population_diagnostics.csv", index=False)
    _summary_rows(
        cohort_df,
        [c for c in cohort_df.columns if c not in {"scenario", "run_id"}],
        ["scenario"],
    ).to_csv(out / "asl_cohort_flow_summary.csv", index=False)
    return cohort_df


def find_results(root: Path) -> pd.DataFrame:
    candidates = sorted(root.rglob("asl_ablation_results*.csv"))
    if not candidates:
        policy_files = sorted(root.rglob("asl_ablation_policy_result.csv"))
        rows = []
        rx = re.compile(r"(?P<scenario>.+)__run_(?P<run>\d+)__(?P<variant>.+)$")
        for f in policy_files:
            m = rx.match(f.parent.name)
            if not m:
                continue
            d = pd.read_csv(f)
            d.insert(0, "variant", m.group("variant")) if "variant" not in d else None
            d.insert(0, "run_id", int(m.group("run")))
            d.insert(0, "scenario", m.group("scenario"))
            rows.append(d)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)
    return pd.read_csv(candidates[-1])


def evaluation_tables(report_root: Path, out: Path, variant: str) -> pd.DataFrame:
    results = find_results(report_root)
    if results.empty:
        (out / "WARNING_no_evaluation_results.txt").write_text(
            f"No asl_ablation_results CSV or run policy files found under {report_root}\n",
            encoding="utf-8",
        )
        return results
    results.to_csv(out / "asl_all_run_results.csv", index=False)
    selected = results.loc[results.variant.astype(str).eq(variant)].copy()
    metrics = [
        "normalized_value", "regret", "ranking_concordance",
        "cross_profile_concordance", "within_patient_accuracy",
        "within_patient_recommendation_accuracy", "ndcg_at_25pct",
        "harm_at_25pct", "selected_harm_rate", "served",
        "validation_selection_metric", "validation_global_concordance",
        "validation_cross_profile_concordance", "validation_within_patient_concordance",
        "validation_ndcg", "validation_topk_dr_value",
    ]
    metrics = [m for m in metrics if m in selected]
    summary = _summary_rows(selected, metrics, ["scenario", "variant"])
    summary.to_csv(out / "asl_primary_model_evaluation_summary.csv", index=False)
    _summary_rows(results, metrics, ["scenario", "variant"]).to_csv(
        out / "asl_all_variants_evaluation_summary.csv", index=False
    )

    # Opportunity-level files are available in patched/new runs.
    score_files = sorted(report_root.rglob("asl_opportunity_scores.csv"))
    rec_rows: list[dict[str, Any]] = []
    for f in score_files:
        m = re.match(r"(?P<scenario>.+)__run_(?P<run>\d+)__(?P<variant>.+)$", f.parent.name)
        if not m or m.group("variant") != variant:
            continue
        scores = pd.read_csv(f)
        method = "PROMETHEUS-Contrastive"
        if "method" in scores and method in set(scores.method.astype(str)):
            s = scores.loc[scores.method.astype(str).eq(method)].copy()
        else:
            s = scores.copy()
        selected_rows = s.loc[s.selected.astype(bool)] if "selected" in s else s.iloc[0:0]
        patient_best = s.sort_values("allocator_value", ascending=False).drop_duplicates("patient_id")
        positive = patient_best.loc[patient_best.allocator_value > 0]
        rec_rows.append({
            "scenario": m.group("scenario"), "run_id": int(m.group("run")), "variant": variant,
            "evaluated_patients": int(s.patient_id.nunique()),
            "opportunities_scored": int(len(s)),
            "patients_receiving_positive_recommendation": int(positive.patient_id.nunique()),
            "recommendation_rate": float(positive.patient_id.nunique() / s.patient_id.nunique()) if s.patient_id.nunique() else math.nan,
            "patients_allocated": int(selected_rows.patient_id.nunique()),
            "selected_opportunities": int(len(selected_rows)),
            "selected_harm_rate_recomputed": float((selected_rows.true_effect_evaluation_only < 0).mean()) if len(selected_rows) else 0.0,
            "oracle_profile_agreement_recomputed": float(
                (patient_best.set_index("patient_id").profile_id ==
                 s.loc[s.groupby("patient_id").true_effect_evaluation_only.idxmax()].set_index("patient_id").profile_id).mean()
            ),
        })
    if rec_rows:
        rec = pd.DataFrame(rec_rows)
        rec.to_csv(out / "asl_recommendation_allocation_by_run.csv", index=False)
        _summary_rows(rec, [
            "evaluated_patients", "opportunities_scored",
            "patients_receiving_positive_recommendation", "recommendation_rate",
            "patients_allocated", "selected_opportunities",
            "selected_harm_rate_recomputed", "oracle_profile_agreement_recomputed",
        ], ["scenario", "variant"]).to_csv(
            out / "asl_recommendation_allocation_summary.csv", index=False
        )
    return results


def write_latex(out: Path, cohort: pd.DataFrame, variant: str) -> None:
    lines = [
        "% Auto-generated by extract_asl_paper_artifacts.py",
        "% Counts are scenario-specific because each locked run samples the real ASL covariate population.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{ASL semi-synthetic cohort construction across locked runs.}",
        "\\label{tab:asl-cohort-flow-generated}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Measure & Partially aligned & Risk--benefit misaligned & Mixed response \\\\",
        "\\midrule",
    ]
    labels = {
        "patients_retained": "Patients retained",
        "patients_with_admissible_profile": "Patients with $\\geq1$ admissible profile",
        "admissible_opportunities": "Admissible patient--profile opportunities",
        "test_patients": "Test patients",
        "test_opportunities": "Test opportunities",
    }
    for col, label in labels.items():
        vals = []
        for sc in SCENARIOS:
            x = cohort.loc[cohort.scenario.eq(sc), col]
            vals.append(f"{x.mean():,.1f}" if len(x) else "--")
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "", f"% Primary model: {variant}"]
    (out / "asl_paper_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(out: Path, report_root: Path, variant: str) -> None:
    text = f"""# ASL paper artifacts

Primary model: `{variant}`
Evaluation root: `{report_root}`

## Files
- `asl_source_table_counts.csv`: source administrative table dimensions.
- `asl_patient_feature_missingness.csv`: complete feature missingness and cardinality audit.
- `asl_source_descriptive_statistics.csv`: selected source-variable summaries.
- `asl_cohort_flow_by_run.csv`: exact reconstructed cohort and opportunity counts.
- `asl_cohort_flow_summary.csv`: mean and 95% bootstrap CI across runs.
- `asl_baseline_summary_by_run.csv`: baseline covariate summaries.
- `asl_profile_population_diagnostics.csv`: eligibility, assignment and oracle-only effect diagnostics.
- `asl_primary_model_evaluation_summary.csv`: paper metrics for the selected model.
- `asl_all_variants_evaluation_summary.csv`: complete ablation summary.
- `asl_recommendation_allocation_*`: generated only when patched runs contain `asl_opportunity_scores.csv`.
- `asl_paper_tables.tex`: initial LaTeX cohort table.

## Important interpretation
The current ASL benchmark implements profile eligibility but not a separate empirical-support exclusion layer at opportunity level. Therefore `supported_opportunities` equals the admissible opportunities in the reconstructed cohort. Propensity support is instead documented through stabilized propensity minima and arm-level effective sample sizes. Do not describe this as patient-level overlap filtering unless that mechanism is added to the code.

Oracle outcomes/effects are semi-synthetic and evaluation-only. They must not be presented as estimated real clinical effects in ASL Benevento.
"""
    (out / "README_ASL_PAPER_ARTIFACTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="output_acg_2025.zip")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--reports", required=True, type=Path, help="Completed confirmation/campaign directory")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_dataset_tables(args.input, args.output)
    cohort = reconstruct_cohorts(config, args.input, args.output, args.runs)
    evaluation_tables(args.reports, args.output, args.variant)
    write_latex(args.output, cohort, args.variant)
    write_readme(args.output, args.reports, args.variant)
    print(f"ASL paper artifacts written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
