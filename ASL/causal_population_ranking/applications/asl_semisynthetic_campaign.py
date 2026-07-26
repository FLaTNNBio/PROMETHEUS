"""Multi-seed campaign using real ASL covariates and semi-synthetic T/Y."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .asl_semisynthetic_case import run_asl_semisynthetic_case


ASL_RESPONSE_SCENARIOS = (
    "risk_benefit_aligned",
    "partially_aligned",
    "risk_benefit_misaligned",
    "mixed_response",
)


def _interval(values: np.ndarray, seed: int, repetitions: int = 3000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def run_asl_semisynthetic_campaign(
    base_config: Mapping[str, Any],
    input_path: str | Path,
    output_root: str | Path,
    *,
    scenarios: Iterable[str] = ASL_RESPONSE_SCENARIOS,
    runs_per_scenario: int = 5,
    smoke: bool = False,
) -> Path:
    if runs_per_scenario < 2:
        raise ValueError("At least two runs are required")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    frames: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        if scenario not in ASL_RESPONSE_SCENARIOS:
            raise ValueError(f"Unknown ASL scenario: {scenario}")
        for run_id in range(runs_per_scenario):
            config = copy.deepcopy(base_config)
            offset = 10_000 * (scenario_index * runs_per_scenario + run_id + 1)
            settings = config["asl_semisynthetic"]
            settings["response_scenario"] = scenario
            settings["seed"] += offset
            settings["causal_supervision"]["nuisance_seed"] += offset
            if "baselines" in settings and "seed" in settings["baselines"]:
                settings["baselines"]["seed"] += offset
            settings["ranking"]["model_seed"] += offset
            settings["pairs"]["train_seed"] += offset
            settings["pairs"]["validation_seed"] += offset
            run_dir = root / "runs" / f"{scenario}__run_{run_id:03d}"
            result = run_asl_semisynthetic_case(
                config,
                input_path,
                run_dir,
                smoke=smoke,
            )["results"].copy()
            result["scenario"] = scenario
            result["run_id"] = run_id
            frames.append(result)
    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(root / "all_policy_results.csv", index=False)

    summary = []
    for (scenario, method), group in all_results.groupby(["scenario", "method"]):
        values = group.normalized_value.to_numpy(float)
        low, high = _interval(values, seed=101_001 + len(summary))
        summary.append({
            "scenario": scenario,
            "method": method,
            "runs": len(values),
            "mean_normalized_value": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "mean_regret": float(group.regret.mean()),
            "mean_ranking_concordance": float(group.ranking_concordance.mean()),
            "mean_cross_profile_concordance": float(group.cross_profile_concordance.mean()),
            "mean_within_patient_accuracy": float(group.within_patient_recommendation_accuracy.mean()),
            "mean_ndcg_at_25pct": float(group.ndcg_at_25pct.mean()),
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
            low, high = _interval(diff, seed=102_001 + len(paired))
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

    lines = [
        "# ASL real-X semi-synthetic campaign",
        "",
        "The population and baseline covariates are from the pseudonymized ASL extract.",
        "Treatment assignment and outcomes are semi-synthetic and oracle quantities are evaluation-only.",
        "",
    ]
    summary_frame = pd.DataFrame(summary)
    for scenario in summary_frame.scenario.unique():
        best = summary_frame.loc[
            (summary_frame.scenario == scenario)
            & (summary_frame.method != "Oracle")
        ].sort_values("mean_normalized_value", ascending=False).iloc[0]
        lines.append(
            f"- **{scenario}**: {best.method}, mean normalized value "
            f"{best.mean_normalized_value:.4f} [{best.ci_low:.4f}, {best.ci_high:.4f}]"
        )
    (root / "scientific_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root
