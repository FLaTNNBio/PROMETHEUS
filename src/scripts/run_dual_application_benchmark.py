from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from causal_population_ranking.applications.dm77_case import run_dm77_case
from causal_population_ranking.ems.case_study import run_ems_case_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PROMETHEUS in DM77 and EMS application scenarios."
    )
    parser.add_argument("--config", default="configs/applications/dual_application.yaml")
    parser.add_argument("--output", default="reports/dual_application")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (root / args.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    dm77_output = output / "dm77"
    dm77 = run_dm77_case(config, dm77_output, smoke=args.smoke)

    ems_config = (root / config["ems"]["config_path"]).resolve()
    ems_output = output / "ems"
    run_ems_case_study(ems_config, ems_output, smoke=args.smoke)

    dm77_results = dm77["results"].copy()
    ems_raw = pd.read_csv(ems_output / "evaluation_only" / "policy_benchmark.csv")
    ems_results = pd.DataFrame({
        "application": "EMS",
        "method": ems_raw["Policy"],
        "optimizer": ems_raw["Optimizer"],
        "true_value": ems_raw["True value"],
        "oracle_value": ems_raw["True value"] + ems_raw["Regret"],
        "normalized_value": ems_raw["Normalized value"],
        "regret": ems_raw["Regret"],
        "served": ems_raw["Served"],
        "ranking_concordance": float("nan"),
        "oracle_access_for_policy_construction": ems_raw["Policy"].eq("Oracle"),
        "oracle_access_for_evaluation": True,
    })
    combined = pd.concat([dm77_results, ems_results], ignore_index=True)
    combined.to_csv(output / "combined_policy_results.csv", index=False)

    summary = []
    for application, frame in combined.groupby("application"):
        non_oracle = frame.loc[~frame.oracle_access_for_policy_construction]
        best = non_oracle.sort_values("normalized_value", ascending=False).iloc[0]
        summary.append({
            "application": application,
            "best_nonoracle_method": best.method,
            "normalized_value": float(best.normalized_value),
            "regret": float(best.regret),
        })
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Dual-application benchmark", "",
        "The same treatment-conditioned Siamese contrastive ranker is used in both domains.",
        "Oracle quantities are evaluation-only.", "",
    ]
    for item in summary:
        lines.append(
            f"- **{item['application']}**: best non-oracle = "
            f"{item['best_nonoracle_method']}, normalized value = "
            f"{item['normalized_value']:.3f}."
        )
    (output / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
