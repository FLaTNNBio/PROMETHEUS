from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from causal_population_ranking.applications.asl_semisynthetic_case import (
    run_asl_semisynthetic_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PROMETHEUS with real ASL covariates and semi-synthetic treatments/outcomes."
    )
    parser.add_argument("--input", required=True, help="output_acg_2025.zip or patient_features_2025.csv.gz")
    parser.add_argument("--config", default="configs/applications/asl_semisynthetic.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scenario",
        choices=("risk_benefit_aligned", "partially_aligned", "risk_benefit_misaligned", "mixed_response"),
        default=None,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.scenario:
        config["asl_semisynthetic"]["response_scenario"] = args.scenario
    result = run_asl_semisynthetic_case(
        config,
        args.input,
        args.output,
        smoke=args.smoke,
    )
    print(Path(args.output).resolve())
    print(result["results"].sort_values("normalized_value", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
