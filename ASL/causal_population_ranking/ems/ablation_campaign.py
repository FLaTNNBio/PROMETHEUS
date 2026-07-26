"""Paired multi-seed ablations for the EMS direct causal ranker."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from .case_study import run_ems_case_study
from .scenario_campaign import _offset_explicit_seeds
from .synthetic import RESPONSE_SCENARIO_PRESETS


ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_contrastive": {
        "ranking": {
            "triplet": {"enabled": False, "weight": 0.0, "maximum_triplets": 0},
            "contrastive_weight": 0.0,
        }
    },
    "no_global_cross_treatment_pairs": {
        "ranking": {
            "pair_type_fractions": {
                "within_unit": 0.30,
                "within_treatment": 0.70,
                "global_cross_treatment": 0.00,
            }
        }
    },
    "no_within_unit_pairs": {
        "ranking": {
            "pair_type_fractions": {
                "within_unit": 0.00,
                "within_treatment": 0.50,
                "global_cross_treatment": 0.50,
            }
        }
    },
}


def _deep_update(target: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 3000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(int(repetitions), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def run_ems_ablation_campaign(
    base_config_path: str | Path,
    output_root: str | Path,
    *,
    variants: Iterable[str] = ("full", "no_contrastive"),
    scenarios: Iterable[str] = ("partially_aligned", "risk_benefit_misaligned"),
    runs_per_scenario: int = 3,
    seed_stride: int = 10_000,
    smoke: bool = False,
) -> Path:
    """Run paired ablations using identical DGP seeds within each run."""

    base_path = Path(base_config_path).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    selected_variants = list(variants)
    selected_scenarios = list(scenarios)
    unknown_variants = sorted(set(selected_variants) - set(ABLATION_VARIANTS))
    unknown_scenarios = sorted(set(selected_scenarios) - set(RESPONSE_SCENARIO_PRESETS))
    if unknown_variants:
        raise ValueError(f"Unknown ablation variants: {unknown_variants}")
    if unknown_scenarios:
        raise ValueError(f"Unknown response scenarios: {unknown_scenarios}")
    if "full" not in selected_variants:
        raise ValueError("The full variant is required as the paired reference")
    if runs_per_scenario < 2:
        raise ValueError("Use at least two runs per scenario")

    project_root = next(
        (parent for parent in base_path.parents if (parent / "pyproject.toml").is_file()),
        base_path.parent,
    )
    data_directory = Path(base["data"]["directory"])
    if not data_directory.is_absolute():
        data_directory = (project_root / data_directory).resolve()
    base["data"]["directory"] = str(data_directory)

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs_dir = root / "configs"
    runs_dir = root / "runs"
    configs_dir.mkdir()
    runs_dir.mkdir()

    policy_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    nonoracle_frames: list[pd.DataFrame] = []

    for scenario_index, scenario in enumerate(selected_scenarios):
        for run_id in range(runs_per_scenario):
            # All variants for the same scenario/run share exactly the same seed
            # bundle, so differences are paired and attributable to the ablation.
            offset = (scenario_index * runs_per_scenario + run_id + 1) * seed_stride
            paired_base = _offset_explicit_seeds(base, offset)
            paired_base["simulation"]["response_scenario"] = scenario
            for variant in selected_variants:
                config = deepcopy(paired_base)
                _deep_update(config, ABLATION_VARIANTS[variant])
                config["config_version"] = (
                    f"{base['config_version']}|ablation={variant}|"
                    f"{scenario}|run-{run_id:03d}"
                )
                name = f"{scenario}__run_{run_id:03d}__{variant}"
                config_path = configs_dir / f"{name}.yaml"
                config_path.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                run_output = runs_dir / name
                run_ems_case_study(
                    config_path,
                    run_output,
                    smoke=smoke,
                    include_constraint_sensitivity=False,
                )

                policy = pd.read_csv(run_output / "evaluation_only" / "policy_benchmark.csv")
                row = policy.loc[
                    policy.Policy.eq("PROMETHEUS") & policy.Optimizer.eq("MILP")
                ].iloc[0]
                policy_rows.append({
                    "response_scenario": scenario,
                    "run_id": run_id,
                    "variant": variant,
                    "normalized_value": float(row["Normalized value"]),
                    "true_value": float(row["True value"]),
                    "regret": float(row["Regret"]),
                })

                score = pd.read_csv(
                    run_output / "evaluation_only" / "policy_score_diagnostics.csv"
                )
                score = score.loc[score.policy.eq("PROMETHEUS")].copy()
                score.insert(0, "variant", variant)
                score.insert(0, "run_id", run_id)
                score.insert(0, "response_scenario", scenario)
                score_frames.append(score)

                nonoracle = pd.read_csv(run_output / "nonoracle_metrics.csv")
                nonoracle.insert(0, "variant", variant)
                nonoracle.insert(0, "run_id", run_id)
                nonoracle.insert(0, "response_scenario", scenario)
                nonoracle_frames.append(nonoracle)

                audit = json.loads((run_output / "ranking_audit.json").read_text(encoding="utf-8"))
                ranking_rows.append({
                    "response_scenario": scenario,
                    "run_id": run_id,
                    "variant": variant,
                    "best_epoch": int(audit["best_epoch"]),
                    "validation_selection_metric": float(audit["best_validation_selection_metric"]),
                    "triplet_enabled": bool(audit.get("triplet_enabled", True)),
                    "triplets": int(audit.get("triplets", 0)),
                    "triplet_weight": float(audit.get("triplet_weight", 0.0)),
                })

    policy_results = pd.DataFrame(policy_rows)
    score_results = pd.concat(score_frames, ignore_index=True)
    nonoracle_results = pd.concat(nonoracle_frames, ignore_index=True)
    ranking_audit = pd.DataFrame(ranking_rows)

    summary_rows: list[dict[str, Any]] = []
    for (scenario, variant), group in policy_results.groupby(
        ["response_scenario", "variant"], sort=False
    ):
        values = group.normalized_value.to_numpy(float)
        low, high = _bootstrap_interval(values, seed=91_001 + len(summary_rows))
        summary_rows.append({
            "response_scenario": scenario,
            "variant": variant,
            "runs": int(group.run_id.nunique()),
            "normalized_value_mean": float(values.mean()),
            "normalized_value_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci_low": low,
            "ci_high": high,
            "regret_mean": float(group.regret.mean()),
        })
    summary = pd.DataFrame(summary_rows)

    wide = policy_results.pivot_table(
        index=["response_scenario", "run_id"],
        columns="variant",
        values="normalized_value",
        aggfunc="first",
    ).reset_index()
    paired_rows: list[dict[str, Any]] = []
    for scenario, group in wide.groupby("response_scenario", sort=False):
        for variant in selected_variants:
            if variant == "full":
                continue
            diff = group["full"].to_numpy(float) - group[variant].to_numpy(float)
            low, high = _bootstrap_interval(diff, seed=92_001 + len(paired_rows))
            paired_rows.append({
                "response_scenario": scenario,
                "comparison": f"full minus {variant}",
                "runs": int(len(diff)),
                "mean_difference": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "full_wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "full_losses": int((diff < 0).sum()),
            })
    paired = pd.DataFrame(paired_rows)

    policy_results.to_csv(root / "ablation_policy_results.csv", index=False)
    summary.to_csv(root / "ablation_policy_summary.csv", index=False)
    paired.to_csv(root / "ablation_paired_differences.csv", index=False)
    score_results.to_csv(root / "ablation_score_diagnostics.csv", index=False)
    nonoracle_results.to_csv(root / "ablation_nonoracle_metrics.csv", index=False)
    ranking_audit.to_csv(root / "ablation_ranking_audit.csv", index=False)
    (root / "ablation_manifest.json").write_text(
        json.dumps({
            "status": "completed",
            "base_config": str(base_path),
            "variants": selected_variants,
            "response_scenarios": selected_scenarios,
            "runs_per_scenario": runs_per_scenario,
            "paired_seed_bundle_across_variants": True,
            "smoke": bool(smoke),
        }, indent=2),
        encoding="utf-8",
    )
    return root
