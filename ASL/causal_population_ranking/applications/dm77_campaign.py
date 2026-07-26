"""Multi-seed campaign for the DM77 application."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .dm77_case import run_dm77_case


DM77_RESPONSE_SCENARIOS = (
    "risk_benefit_aligned",
    "partially_aligned",
    "risk_benefit_misaligned",
    "mixed_response",
)


def _interval(values: np.ndarray, seed: int, repetitions: int = 3000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_dm77_campaign(
    base_config: Mapping[str, Any],
    output_root: str | Path,
    *,
    scenarios: Iterable[str] = DM77_RESPONSE_SCENARIOS,
    runs_per_scenario: int = 10,
    smoke: bool = False,
) -> Path:
    if runs_per_scenario < 2:
        raise ValueError("At least two runs are required")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    frames = []
    for scenario_index, scenario in enumerate(scenarios):
        if scenario not in DM77_RESPONSE_SCENARIOS:
            raise ValueError(f"Unknown DM77 scenario: {scenario}")
        for run_id in range(runs_per_scenario):
            config = copy.deepcopy(base_config)
            offset = 10_000 * (scenario_index * runs_per_scenario + run_id + 1)
            dm = config["dm77"]
            dm["response_scenario"] = scenario
            dm["seed"] += offset
            dm["causal_supervision"]["nuisance_seed"] += offset
            dm["ranking"]["model_seed"] += offset
            dm["pairs"]["train_seed"] += offset
            dm["pairs"]["validation_seed"] += offset
            run_dir = root / "runs" / f"{scenario}__run_{run_id:03d}"
            result = run_dm77_case(config, run_dir, smoke=smoke)["results"].copy()
            result["scenario"] = scenario
            result["run_id"] = run_id
            frames.append(result)
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(root / "all_policy_results.csv", index=False)

    summary = []
    for (scenario, method), group in all_results.groupby(["scenario", "method"]):
        values = group.normalized_value.to_numpy(float)
        low, high = _interval(values, seed=91_001 + len(summary))
        summary.append({
            "scenario": scenario,
            "method": method,
            "runs": len(values),
            "mean_normalized_value": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "mean_regret": float(group.regret.mean()),
        })
    pd.DataFrame(summary).to_csv(root / "policy_summary_bootstrap.csv", index=False)

    paired = []
    pivot = all_results.pivot_table(
        index=["scenario", "run_id"], columns="method", values="normalized_value"
    )
    target = "PROMETHEUS-Contrastive"
    for scenario in pivot.index.get_level_values("scenario").unique():
        frame = pivot.xs(scenario, level="scenario")
        for method in frame.columns:
            if method in {target, "Oracle"}:
                continue
            diff = (frame[target] - frame[method]).dropna().to_numpy(float)
            low, high = _interval(diff, seed=92_001 + len(paired))
            paired.append({
                "scenario": scenario,
                "comparison": f"{target} minus {method}",
                "runs": len(diff),
                "mean_difference": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
            })
    pd.DataFrame(paired).to_csv(root / "paired_differences_summary.csv", index=False)
    return root
