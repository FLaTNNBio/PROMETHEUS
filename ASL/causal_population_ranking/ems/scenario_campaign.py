"""Multi-seed campaign for EMS risk-benefit response scenarios.

Each run refits the complete learner using a distinct seed bundle. Oracle truth is
opened only inside the standard evaluation stage. The campaign then aggregates
run-level outputs and computes paired PROMETHEUS-minus-baseline intervals.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from .case_study import run_ems_case_study
from .synthetic import RESPONSE_SCENARIO_PRESETS


def _offset_explicit_seeds(value: Any, offset: int) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                int(child) + int(offset)
                if str(key).endswith("_seed")
                else _offset_explicit_seeds(child, offset)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_offset_explicit_seeds(child, offset) for child in value]
    return deepcopy(value)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(repetitions), len(values)), replace=True)
    boot = samples.mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, float(low), float(high)


def _aggregate_policy_results(
    results: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, policy, optimizer), group in results.groupby(
        ["response_scenario", "Policy", "Optimizer"], sort=False
    ):
        for metric in ("True value", "Normalized value", "Regret", "Benefit/cost"):
            mean, low, high = _bootstrap_mean_interval(
                group[metric].to_numpy(float),
                seed=bootstrap_seed + len(rows) * 37,
                repetitions=bootstrap_repetitions,
            )
            rows.append({
                "response_scenario": scenario,
                "Policy": policy,
                "Optimizer": optimizer,
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "runs": int(group.run_id.nunique()),
            })
    return pd.DataFrame(rows)


def _paired_differences(
    results: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    milp = results.loc[results.Optimizer.eq("MILP")].copy()
    wide = milp.pivot_table(
        index=["response_scenario", "run_id"],
        columns="Policy",
        values="Normalized value",
        aggfunc="first",
    ).reset_index()
    comparators = ["Risk-first", "Need-first", "Outcome-first", "Profile-mean"]
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scenario, group in wide.groupby("response_scenario", sort=False):
        if "PROMETHEUS" not in group:
            continue
        for comparator in comparators:
            if comparator not in group:
                continue
            difference = (
                group["PROMETHEUS"].to_numpy(float)
                - group[comparator].to_numpy(float)
            )
            for run_id, value in zip(group.run_id, difference):
                run_rows.append({
                    "response_scenario": scenario,
                    "run_id": int(run_id),
                    "comparison": f"PROMETHEUS - {comparator}",
                    "normalized_value_difference": float(value),
                })
            mean, low, high = _bootstrap_mean_interval(
                difference,
                seed=bootstrap_seed + len(summary_rows) * 101,
                repetitions=bootstrap_repetitions,
            )
            summary_rows.append({
                "response_scenario": scenario,
                "comparison": f"PROMETHEUS - {comparator}",
                "mean_difference": mean,
                "ci_low": low,
                "ci_high": high,
                "wins": int(np.sum(difference > 0.0)),
                "ties": int(np.sum(difference == 0.0)),
                "losses": int(np.sum(difference < 0.0)),
                "runs": int(len(difference)),
            })
    return pd.DataFrame(run_rows), pd.DataFrame(summary_rows)


def run_ems_scenario_campaign(
    base_config_path: str | Path,
    output_root: str | Path,
    *,
    scenarios: Iterable[str] | None = None,
    runs_per_scenario: int = 10,
    seed_stride: int = 10_000,
    bootstrap_seed: int = 91_001,
    bootstrap_repetitions: int = 5_000,
    smoke: bool = False,
) -> Path:
    """Run a full multi-seed scenario campaign and aggregate paired results."""

    base_path = Path(base_config_path).resolve()
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    # Generated campaign configurations are written below ``output_root``.  A
    # relative data.directory would otherwise be resolved against that temporary
    # location rather than against the repository containing the base config.
    # Freeze an absolute source-data path before writing per-run configurations.
    project_root = next(
        (parent for parent in base_path.parents if (parent / "pyproject.toml").is_file()),
        base_path.parent,
    )
    data_directory = Path(base_config["data"]["directory"])
    if not data_directory.is_absolute():
        data_directory = (project_root / data_directory).resolve()
    base_config["data"]["directory"] = str(data_directory)

    selected = list(scenarios or RESPONSE_SCENARIO_PRESETS.keys())
    unknown = sorted(set(selected) - set(RESPONSE_SCENARIO_PRESETS))
    if unknown:
        raise ValueError(f"Unknown response scenarios: {unknown}")
    if runs_per_scenario < 2:
        raise ValueError("Use at least two runs per scenario for uncertainty intervals")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config_dir = root / "configs"
    runs_dir = root / "runs"
    config_dir.mkdir()
    runs_dir.mkdir()

    policy_frames: list[pd.DataFrame] = []
    score_diagnostic_frames: list[pd.DataFrame] = []
    dgp_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(selected):
        for run_id in range(runs_per_scenario):
            offset = (scenario_index * runs_per_scenario + run_id + 1) * seed_stride
            config = _offset_explicit_seeds(base_config, offset)
            config["simulation"]["response_scenario"] = scenario
            config["config_version"] = (
                f"{base_config['config_version']}|{scenario}|run-{run_id:03d}"
            )
            run_name = f"{scenario}__run_{run_id:03d}"
            config_path = config_dir / f"{run_name}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            run_output = runs_dir / run_name
            run_ems_case_study(
                config_path,
                run_output,
                smoke=smoke,
                include_constraint_sensitivity=False,
            )
            policy = pd.read_csv(
                run_output / "evaluation_only" / "policy_benchmark.csv"
            )
            policy.insert(0, "run_id", run_id)
            policy.insert(0, "response_scenario", scenario)
            policy_frames.append(policy)
            score_diagnostics = pd.read_csv(
                run_output / "evaluation_only" / "policy_score_diagnostics.csv"
            )
            score_diagnostics.insert(0, "run_id", run_id)
            score_diagnostics.insert(0, "response_scenario", scenario)
            score_diagnostic_frames.append(score_diagnostics)
            dgp = json.loads(
                (run_output / "dgp_audit.json").read_text(encoding="utf-8")
            )
            dgp_rows.append({
                "response_scenario": scenario,
                "run_id": run_id,
                "nurse_risk_benefit_spearman": dgp["risk_benefit_spearman"][
                    "nurse_supported"
                ],
                "medicalized_risk_benefit_spearman": dgp[
                    "risk_benefit_spearman"
                ]["medicalized_total"],
                "nurse_negative_fraction": dgp["effect_distribution"][
                    "nurse_supported"
                ]["negative_fraction"],
                "medicalized_negative_fraction": dgp["effect_distribution"][
                    "medicalized_total"
                ]["negative_fraction"],
                "nurse_near_zero_fraction": dgp["effect_distribution"][
                    "nurse_supported"
                ]["near_zero_fraction"],
                "medicalized_near_zero_fraction": dgp["effect_distribution"][
                    "medicalized_total"
                ]["near_zero_fraction"],
            })

    all_policy = pd.concat(policy_frames, ignore_index=True)
    all_score_diagnostics = pd.concat(score_diagnostic_frames, ignore_index=True)
    dgp_diagnostics = pd.DataFrame(dgp_rows)
    aggregate = _aggregate_policy_results(
        all_policy,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    paired_runs, paired_summary = _paired_differences(
        all_policy,
        bootstrap_seed=bootstrap_seed + 1_000_000,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    score_summary = (
        all_score_diagnostics.groupby(
            ["response_scenario", "policy", "scope"], dropna=False
        )
        .agg(
            runs=("run_id", "nunique"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            kendall_mean=("kendall", "mean"),
            pairwise_concordance_mean=("pairwise_concordance", "mean"),
            ndcg_at_25pct_mean=("ndcg_at_25pct", "mean"),
            top25pct_overlap_mean=("top25pct_overlap", "mean"),
        )
        .reset_index()
    )
    all_policy.to_csv(root / "all_policy_results.csv", index=False)
    all_score_diagnostics.to_csv(root / "all_score_diagnostics.csv", index=False)
    score_summary.to_csv(root / "score_diagnostics_summary.csv", index=False)
    dgp_diagnostics.to_csv(root / "dgp_diagnostics.csv", index=False)
    aggregate.to_csv(root / "policy_summary_bootstrap.csv", index=False)
    paired_runs.to_csv(root / "paired_differences_by_run.csv", index=False)
    paired_summary.to_csv(root / "paired_differences_summary.csv", index=False)
    (root / "campaign_manifest.json").write_text(
        json.dumps({
            "status": "completed",
            "base_config": str(base_path),
            "response_scenarios": selected,
            "runs_per_scenario": runs_per_scenario,
            "smoke": bool(smoke),
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_repetitions": bootstrap_repetitions,
            "oracle_policy_construction_only_for_oracle_row": True,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root
