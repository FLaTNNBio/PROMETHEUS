from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from causal_population_ranking.version import (
    PROMETHEUS_SYNTHETIC_LEGACY_GLOBAL_METHOD_VERSION,
)


METHOD_VERSION = PROMETHEUS_SYNTHETIC_LEGACY_GLOBAL_METHOD_VERSION
METRICS = (
    "cross_transition_concordance",
    "within_transition_concordance",
    "global_concordance",
    "global_pooled_autoc",
    "global_allocation_value",
    "benefit_per_capacity_unit",
    "global_regret",
    "fraction_of_oracle_benefit",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover(root: Path) -> list[Path]:
    runs = [path.parent for path in root.rglob("manifest.json")
            if _read_json(path).get("method_version") == METHOD_VERSION]
    if not runs:
        raise FileNotFoundError(f"No {METHOD_VERSION} runs found under {root}")
    return sorted(runs)


def _collect(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        manifest = _read_json(run / "manifest.json")
        metrics = _read_json(run / "global_metrics.json")
        rows.append({
            "run": run.name,
            "seed": int(manifest["seed"]),
            "scenario": manifest["scenario"],
            "method": manifest["model_variant"],
            **{metric: metrics.get(metric, np.nan) for metric in METRICS},
        })
    return pd.DataFrame(rows)


def _mean_se(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(["scenario", "method"], sort=True):
        for metric in METRICS:
            values = subset[metric].astype(float).dropna().to_numpy()
            if len(values):
                rows.append({
                    "scenario": keys[0], "method": keys[1], "metric": metric,
                    "n_seeds": int(len(values)), "mean": float(values.mean()),
                    "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan"),
                })
    return pd.DataFrame(rows)


def _paired(frame: pd.DataFrame, baseline: str = "unified_local_ranker") -> pd.DataFrame:
    reference = frame[frame.method == baseline].set_index(["scenario", "seed"])
    rows = []
    for method in sorted(set(frame.method).difference({baseline})):
        candidate = frame[frame.method == method].set_index(["scenario", "seed"])
        for key in candidate.index.intersection(reference.index):
            for metric in METRICS:
                left, right = float(candidate.loc[key, metric]), float(reference.loc[key, metric])
                if np.isfinite(left) and np.isfinite(right):
                    rows.append({
                        "scenario": key[0], "seed": int(key[1]), "method": method,
                        "baseline": baseline, "metric": metric, "difference": left - right,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize global PROMETHEUS repeated-seed suites")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = _collect(_discover(Path(args.runs_dir)))
    frame.to_csv(output / "global_by_seed.csv", index=False)
    _mean_se(frame).to_csv(output / "global_mean_se.csv", index=False)
    paired = _paired(frame)
    paired.to_csv(output / "global_paired_differences.csv", index=False)
    if not paired.empty:
        paired.groupby(["scenario", "method", "baseline", "metric"], as_index=False).difference.agg(
            ["count", "mean", "sem"]
        ).reset_index().to_csv(output / "global_paired_mean_se.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
