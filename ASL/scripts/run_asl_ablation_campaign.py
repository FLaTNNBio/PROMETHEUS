from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from causal_population_ranking.applications.asl_ablation_campaign import (
    ASL_ABLATION_VARIANTS,
    run_asl_ablation_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired PROMETHEUS ablations on real ASL covariates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--config",
        default="configs/applications/asl_semisynthetic.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=list(ASL_ABLATION_VARIANTS))
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[
            "partially_aligned",
            "risk_benefit_misaligned",
            "mixed_response",
        ],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = run_asl_ablation_campaign(
        config,
        args.input,
        args.output,
        variants=args.variants,
        scenarios=args.scenarios,
        runs_per_scenario=args.runs,
        smoke=args.smoke,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
