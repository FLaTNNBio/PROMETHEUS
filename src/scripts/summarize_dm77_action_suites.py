from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_population_ranking.version import PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION
from causal_population_ranking.evaluation.action_policy_metrics import (
    ranking_stability_metrics,
)


METRICS = (
    "within_action_concordance",
    "cross_action_concordance",
    "global_concordance",
    "hard_cross_action_concordance",
    "global_allocation_value",
    "global_regret",
    "normalized_oracle_efficiency",
    "benefit_at_5pct",
    "oracle_top_5pct_recovery",
    "value_at_5pct",
    "regret_at_5pct",
    "ndcg_at_5pct",
    "benefit_at_10pct",
    "oracle_top_10pct_recovery",
    "value_at_10pct",
    "regret_at_10pct",
    "ndcg_at_10pct",
    "benefit_at_20pct",
    "oracle_top_20pct_recovery",
    "value_at_20pct",
    "regret_at_20pct",
    "ndcg_at_20pct",
    "benefit_at_budget",
    "value_at_budget",
    "regret_at_budget",
)
STABILITY_METRICS = (
    "ranking_spearman",
    "ranking_kendall",
    "top_5pct_jaccard",
    "top_10pct_jaccard",
    "top_20pct_jaccard",
    "allocation_identical_fraction",
    "allocation_jaccard",
    "policy_value_absolute_difference",
    "near_boundary_allocation_instability",
)
CALIBRATION_METRICS = (
    "validation_dr_mae",
    "validation_dr_mse",
    "validation_rank_correlation",
    "validation_dr_calibration_slope",
    "validation_dr_calibration_intercept",
    "synthetic_oracle_mae",
    "synthetic_oracle_bias",
    "selected_benefit_bias",
    "synthetic_oracle_calibration_slope",
    "synthetic_oracle_calibration_intercept",
)
COMPARISONS = (
    ("prometheus_global_causal_contrastive_v2", "unified_global_ranker"),
    ("prometheus_global_causal_contrastive_v2", "unified_local_ranker"),
    ("prometheus_global_causal_contrastive_v2", "action_mean_priority"),
    ("prometheus_global_causal_contrastive", "unified_global_ranker"),
    ("prometheus_global_causal_contrastive", "unified_local_ranker"),
    ("prometheus_global_causal_contrastive", "action_mean_priority"),
    ("unified_global_ranker", "unified_local_ranker"),
    ("unified_global_ranker", "action_mean_priority"),
    ("unified_global_ranker", "direct_pairwise_gbdt_ranker"),
    ("unified_global_ranker", "pooled_dr_gbdt"),
    ("unified_global_ranker", "dr_random_forest_priority"),
    ("unified_global_ranker", "dr_policy_tree"),
    ("prometheus_global_causal_contrastive_v2", "direct_pairwise_gbdt_ranker"),
    ("prometheus_global_causal_contrastive_v2", "pooled_dr_gbdt"),
    ("prometheus_global_causal_contrastive_v2", "dr_policy_tree"),
)


def _experiment_condition(manifest: dict) -> tuple[str, dict]:
    signal = manifest.get("causal_signal_configuration", {})
    condition = {
        "causal_signal_estimator": manifest.get(
            "causal_signal_estimator", signal.get("estimator", "dr")
        ),
        "causal_signal_aggregation": signal.get("aggregation", "median"),
        "causal_signal_winsorize": bool(signal.get("winsorize", True)),
        "causal_signal_reliability": bool(
            signal.get("reliability_weighting", True)
        ),
        "min_direction_agreement": signal.get("min_direction_agreement"),
        "beta_cross": manifest.get("ranking_beta_cross", 0.5),
        "permuted_training_pair_labels": bool(
            manifest.get("training_pair_labels_permuted_negative_control", False)
        ),
        "current_care_features_excluded": bool(
            manifest.get("current_care_state_features_excluded_ablation", False)
        ),
    }
    return json.dumps(condition, sort_keys=True, separators=(",", ":")), condition


def _condition_columns(manifest: dict) -> dict:
    identifier, condition = _experiment_condition(manifest)
    return {"experiment_condition": identifier, **condition}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in root.rglob("run_manifest.json"):
        manifest = _json(manifest_path)
        if manifest.get("method_version") != PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION:
            continue
        comparison_path = manifest_path.parent / "baseline_comparison.csv"
        if not comparison_path.exists():
            continue
        frame = pd.read_csv(comparison_path)
        primary_ranking_method = str(
            manifest.get("primary_ranking_method", "unified_global_ranker")
        )
        for row in frame.to_dict("records"):
            rows.append({
                "run": manifest["run_id"],
                "seed": int(manifest["seed"]),
                "dataset_seed": int(manifest.get("dataset_seed", manifest["seed"])),
                "dgp_scenario": manifest["action_dgp_scenario"],
                "primary_ranking_method": primary_ranking_method,
                **_condition_columns(manifest),
                **row,
            })
    if not rows:
        raise FileNotFoundError(f"No integrated DM77 suite runs found under {root}")
    return pd.DataFrame(rows)


def collect_stability(root: Path) -> pd.DataFrame:
    runs = []
    for manifest_path in root.rglob("run_manifest.json"):
        manifest = _json(manifest_path)
        if manifest.get("method_version") != PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION:
            continue
        evaluation_path = manifest_path.parent / "evaluation_ranked_actions.csv"
        assignment_path = manifest_path.parent / "baseline_policy_assignments.csv"
        if not evaluation_path.exists() or not assignment_path.exists():
            continue
        evaluation = pd.read_csv(evaluation_path)
        assignments = pd.read_csv(assignment_path)
        primary_ranking_method = str(
            manifest.get("primary_ranking_method", "unified_global_ranker")
        )
        selected_column = f"selected__{primary_ranking_method}"
        if selected_column not in assignments:
            raise ValueError(
                f"Missing primary allocation column {selected_column!r} in {assignment_path}"
            )
        frame = evaluation.loc[:, [
            "patient_id", "action_id", "raw_priority_score", "true_cate",
        ]].merge(
            assignments.loc[:, ["patient_id", "action_id", selected_column]].rename(
                columns={selected_column: "selected_primary"}
            ),
            on=["patient_id", "action_id"], validate="one_to_one",
        )
        if frame[["patient_id", "action_id"]].duplicated().any():
            raise ValueError("Stability evaluation requires unique patient-action keys")
        runs.append({
            "run": manifest["run_id"],
            "seed": int(manifest["seed"]),
            "dataset_seed": int(manifest.get("dataset_seed", manifest["seed"])),
            "dgp_scenario": manifest["action_dgp_scenario"],
            "population_scenario": manifest.get("population_scenario", "unknown"),
            "sample_size": int(manifest["sample_size"]),
            "primary_ranking_method": primary_ranking_method,
            **_condition_columns(manifest),
            "frame": frame,
        })
    rows = []
    group_keys = (
        "dgp_scenario", "population_scenario", "sample_size", "dataset_seed",
        "primary_ranking_method",
        "experiment_condition",
    )
    runs.sort(key=lambda item: tuple(item[key] for key in group_keys) + (item["seed"], item["run"]))
    for _, grouped in itertools.groupby(runs, key=lambda item: tuple(item[key] for key in group_keys)):
        group = list(grouped)
        for left, right in itertools.combinations(group, 2):
            if left["seed"] == right["seed"]:
                continue
            merged = left["frame"].merge(
                right["frame"],
                on=["patient_id", "action_id"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            if len(merged) != len(left["frame"]) or len(merged) != len(right["frame"]):
                raise ValueError(
                    "Seed-stability runs with a shared dataset_seed must have identical "
                    "patient-action opportunity sets"
                )
            if not np.allclose(merged.true_cate_left, merged.true_cate_right):
                raise ValueError("Shared-dataset stability runs have different oracle CATEs")
            metrics = ranking_stability_metrics(
                merged.raw_priority_score_left,
                merged.raw_priority_score_right,
                merged.selected_primary_left,
                merged.selected_primary_right,
                merged.true_cate_left,
            )
            rows.append({
                "dgp_scenario": left["dgp_scenario"],
                "population_scenario": left["population_scenario"],
                "sample_size": left["sample_size"],
                "dataset_seed": left["dataset_seed"],
                "primary_ranking_method": left["primary_ranking_method"],
                "experiment_condition": left["experiment_condition"],
                "run_left": left["run"], "run_right": right["run"],
                "seed_left": left["seed"], "seed_right": right["seed"],
                **metrics,
            })
    return pd.DataFrame(rows)


def collect_calibrations(root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in root.rglob("run_manifest.json"):
        manifest = _json(manifest_path)
        if manifest.get("method_version") != PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION:
            continue
        diagnostics_path = manifest_path.parent / "calibration_diagnostics.csv"
        if not diagnostics_path.exists():
            continue
        primary_ranking_method = str(
            manifest.get("primary_ranking_method", "unified_global_ranker")
        )
        for row in pd.read_csv(diagnostics_path).to_dict("records"):
            rows.append({
                "run": manifest["run_id"],
                "seed": int(manifest["seed"]),
                "dataset_seed": int(manifest.get("dataset_seed", manifest["seed"])),
                "dgp_scenario": manifest["action_dgp_scenario"],
                "primary_ranking_method": primary_ranking_method,
                **_condition_columns(manifest),
                **row,
            })
    return pd.DataFrame(rows)


def collect_run_table(root: Path, filename: str) -> pd.DataFrame:
    """Collect a per-run diagnostic table without inventing missing results."""

    rows = []
    for manifest_path in root.rglob("run_manifest.json"):
        manifest = _json(manifest_path)
        if manifest.get("method_version") != PROMETHEUS_DM77_INTEGRATED_METHOD_VERSION:
            continue
        path = manifest_path.parent / filename
        if not path.exists():
            continue
        for row in pd.read_csv(path).to_dict("records"):
            rows.append({
                "run": manifest["run_id"],
                "seed": int(manifest["seed"]),
                "dataset_seed": int(manifest.get("dataset_seed", manifest["seed"])),
                "dgp_scenario": manifest["action_dgp_scenario"],
                "primary_ranking_method": manifest.get(
                    "primary_ranking_method", "unified_global_ranker"
                ),
                **_condition_columns(manifest),
                **row,
            })
    return pd.DataFrame(rows)


def summarize_run_table(frame: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    requested_keys = ("experiment_condition", *group_columns)
    keys = tuple(dict.fromkeys(
        column for column in requested_keys if column in frame
    ))
    excluded = {
        "seed", "dataset_seed", "run", *keys,
    }
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    rows = []
    grouped = frame.groupby(list(keys), dropna=False, sort=True) if keys else [((), frame)]
    for group_key, group in grouped:
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        labels = dict(zip(keys, group_key))
        for metric in numeric:
            per_seed = group.groupby("seed", sort=True)[metric].mean()
            summary = _summary(per_seed.to_numpy(float))
            if summary:
                rows.append({**labels, "metric": metric, **summary})
    return pd.DataFrame(rows)


def summarize_calibrations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "dgp_scenario", "primary_ranking_method", "experiment_condition",
            "method", "metric",
            "n_seeds", "mean",
            "standard_deviation", "median",
            "95_percent_confidence_interval_lower",
            "95_percent_confidence_interval_upper",
        ])
    rows = []
    for (scenario, primary_ranking_method, condition, method), group in frame.groupby(
        [
            "dgp_scenario", "primary_ranking_method",
            "experiment_condition", "method",
        ], sort=True
    ):
        for metric in CALIBRATION_METRICS:
            if metric not in group:
                continue
            per_seed = group.groupby("seed", sort=True)[metric].mean()
            summary = _summary(per_seed.to_numpy(float))
            if summary:
                rows.append({
                    "dgp_scenario": scenario,
                    "primary_ranking_method": primary_ranking_method,
                    "experiment_condition": condition,
                    "method": method,
                    "metric": metric,
                    **summary,
                })
    return pd.DataFrame(rows)


def summarize_stability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "dgp_scenario", "dataset_seed", "primary_ranking_method",
            "experiment_condition", "metric",
            "n_seed_pairs", "mean",
            "standard_deviation", "median",
            "95_percent_confidence_interval_lower",
            "95_percent_confidence_interval_upper",
        ])
    rows = []
    for (scenario, dataset_seed, primary_ranking_method, condition), group in frame.groupby(
        [
            "dgp_scenario", "dataset_seed", "primary_ranking_method",
            "experiment_condition",
        ], sort=True
    ):
        for metric in STABILITY_METRICS:
            summary = _summary(group[metric].to_numpy(float))
            if summary:
                summary["n_seed_pairs"] = summary.pop("n_seeds")
                rows.append({
                    "dgp_scenario": scenario,
                    "dataset_seed": int(dataset_seed),
                    "primary_ranking_method": primary_ranking_method,
                    "experiment_condition": condition,
                    "metric": metric,
                    **summary,
                })
    return pd.DataFrame(rows)


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    half_width = 1.96 * standard_deviation / np.sqrt(len(values)) if len(values) > 1 else float("nan")
    mean = float(values.mean())
    return {
        "n_seeds": int(len(values)),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "median": float(np.median(values)),
        "95_percent_confidence_interval_lower": mean - half_width,
        "95_percent_confidence_interval_upper": mean + half_width,
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, primary_ranking_method, condition, method), group in frame.groupby(
        [
            "dgp_scenario", "primary_ranking_method",
            "experiment_condition", "method",
        ], sort=True
    ):
        for metric in METRICS:
            per_seed = group.groupby("seed", sort=True)[metric].mean()
            summary = _summary(per_seed.to_numpy(float))
            if summary:
                rows.append({
                    "dgp_scenario": scenario,
                    "primary_ranking_method": primary_ranking_method,
                    "experiment_condition": condition,
                    "method": method, "metric": metric,
                    **summary,
                })
    return pd.DataFrame(rows)


def paired_differences(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows, summary_rows = [], []
    for (scenario, primary_ranking_method, condition), scenario_frame in frame.groupby(
        ["dgp_scenario", "primary_ranking_method", "experiment_condition"],
        sort=True,
    ):
        for candidate_name, baseline_name in COMPARISONS:
            candidate = scenario_frame.loc[
                scenario_frame.method == candidate_name
            ].set_index(["run", "seed"])
            baseline = scenario_frame.loc[
                scenario_frame.method == baseline_name
            ].set_index(["run", "seed"])
            shared = candidate.index.intersection(baseline.index)
            for metric in METRICS:
                differences_by_seed: dict[int, list[float]] = {}
                for run, seed in shared:
                    difference = (
                        float(candidate.loc[(run, seed), metric])
                        - float(baseline.loc[(run, seed), metric])
                    )
                    raw_rows.append({
                        "run": run, "dgp_scenario": scenario, "seed": int(seed),
                        "primary_ranking_method": primary_ranking_method,
                        "experiment_condition": condition,
                        "method": candidate_name, "baseline": baseline_name,
                        "metric": metric, "paired_difference": difference,
                    })
                    differences_by_seed.setdefault(int(seed), []).append(difference)
                differences_array = np.asarray([
                    np.mean(values) for values in differences_by_seed.values()
                ], dtype=float)
                summary = _summary(differences_array)
                if summary:
                    wins = (
                        differences_array < 0
                        if "regret" in metric
                        else differences_array > 0
                    )
                    summary_rows.append({
                        "dgp_scenario": scenario,
                        "primary_ranking_method": primary_ranking_method,
                        "experiment_condition": condition,
                        "method": candidate_name, "baseline": baseline_name,
                        "metric": metric, "paired_difference": summary["mean"],
                        "win_rate": float(np.mean(wins)),
                        **summary,
                    })
    return pd.DataFrame(raw_rows), pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize integrated DM77 action suites")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = collect_runs(Path(args.runs_dir))
    raw.to_csv(output / "dm77_action_results_raw.csv", index=False)
    summarize(raw).to_csv(output / "dm77_action_results_summary.csv", index=False)
    paired_raw, paired_summary = paired_differences(raw)
    paired_raw.to_csv(output / "dm77_action_paired_differences_raw.csv", index=False)
    paired_summary.to_csv(output / "dm77_action_paired_differences_summary.csv", index=False)
    stability = collect_stability(Path(args.runs_dir))
    stability.to_csv(output / "dm77_action_seed_stability_raw.csv", index=False)
    summarize_stability(stability).to_csv(
        output / "dm77_action_seed_stability_summary.csv", index=False
    )
    calibrations = collect_calibrations(Path(args.runs_dir))
    calibrations.to_csv(output / "dm77_calibration_results_raw.csv", index=False)
    summarize_calibrations(calibrations).to_csv(
        output / "dm77_calibration_results_summary.csv", index=False
    )
    auxiliary = {
        "observed_test_rate_metrics.csv": (
            "dm77_rate", ("dgp_scenario", "primary_ranking_method", "method")
        ),
        "actionwise_ope_metrics.csv": (
            "dm77_actionwise_ope",
            ("dgp_scenario", "primary_ranking_method", "method", "action_id"),
        ),
        "allocator_ablation.csv": (
            "dm77_allocator_ablation",
            ("dgp_scenario", "primary_ranking_method", "allocator"),
        ),
        "budget_value_curve.csv": (
            "dm77_budget_curve",
            ("dgp_scenario", "primary_ranking_method", "method", "budget_fraction"),
        ),
        "subgroup_policy_metrics.csv": (
            "dm77_subgroup_policy",
            ("dgp_scenario", "primary_ranking_method", "attribute", "group", "method"),
        ),
    }
    for filename, (stem, groups) in auxiliary.items():
        table = collect_run_table(Path(args.runs_dir), filename)
        table.to_csv(output / f"{stem}_raw.csv", index=False)
        summarize_run_table(table, groups).to_csv(
            output / f"{stem}_summary.csv", index=False
        )
    print(output)


if __name__ == "__main__":
    main()
