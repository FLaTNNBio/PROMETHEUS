from __future__ import annotations

import argparse
from pathlib import Path

from causal_population_ranking.ems.case_study import run_ems_case_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/applications/ems_case_study.yaml",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-constraint-sensitivity",
        action="store_true",
        help="Run only the central benchmark and score diagnostics.",
    )
    args = parser.parse_args()
    output = run_ems_case_study(
        Path(args.config),
        Path(args.output) if args.output else None,
        smoke=args.smoke,
        include_constraint_sensitivity=not args.skip_constraint_sensitivity,
    )
    print(output)


if __name__ == "__main__":
    main()
