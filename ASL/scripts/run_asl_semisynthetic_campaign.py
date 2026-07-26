from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from causal_population_ranking.applications.asl_semisynthetic_campaign import (
    run_asl_semisynthetic_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed PROMETHEUS campaign on real ASL covariates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/applications/asl_semisynthetic.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = run_asl_semisynthetic_campaign(
        config,
        args.input,
        args.output,
        scenarios=args.scenarios or (
            "risk_benefit_aligned",
            "partially_aligned",
            "risk_benefit_misaligned",
            "mixed_response",
        ),
        runs_per_scenario=args.runs,
        smoke=args.smoke,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
