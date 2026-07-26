from __future__ import annotations

import argparse
from pathlib import Path

from causal_population_ranking.ems.scenario_campaign import run_ems_scenario_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/applications/ems_case_study.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help=(
            "Subset of: risk_benefit_aligned partially_aligned "
            "risk_benefit_misaligned mixed_response"
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = run_ems_scenario_campaign(
        Path(args.config),
        Path(args.output),
        scenarios=args.scenarios,
        runs_per_scenario=args.runs,
        smoke=args.smoke,
    )
    print(output)


if __name__ == "__main__":
    main()
