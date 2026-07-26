from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from causal_population_ranking.applications.dm77_campaign import run_dm77_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/applications/dual_application.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run_dm77_campaign(
        config,
        args.output,
        scenarios=args.scenarios or (
            "risk_benefit_aligned", "partially_aligned",
            "risk_benefit_misaligned", "mixed_response",
        ),
        runs_per_scenario=args.runs,
        smoke=args.smoke,
    )
    print(result)


if __name__ == "__main__":
    main()
