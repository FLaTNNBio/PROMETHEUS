from __future__ import annotations

import argparse
from pathlib import Path

from causal_population_ranking.ems.ablation_campaign import (
    ABLATION_VARIANTS,
    run_ems_ablation_campaign,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired EMS PROMETHEUS ranking ablations."
    )
    parser.add_argument(
        "--config", default="configs/applications/ems_case_study.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--variants", nargs="*", default=["full", "no_contrastive"],
        choices=sorted(ABLATION_VARIANTS),
    )
    parser.add_argument(
        "--scenarios", nargs="*",
        default=["partially_aligned", "risk_benefit_misaligned"],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_ems_ablation_campaign(
        Path(args.config),
        Path(args.output),
        variants=args.variants,
        scenarios=args.scenarios,
        runs_per_scenario=args.runs,
        smoke=args.smoke,
    )
    print(result)


if __name__ == "__main__":
    main()
