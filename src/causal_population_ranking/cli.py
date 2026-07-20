from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from scripts.run_pipeline import run_pipeline
from .data.synthea_runner import SyntheaRunner


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="causal-ranking",
        description="PROMETHEUS care-profile causal-ranking pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    setup = subparsers.add_parser("setup-synthea")
    setup.add_argument("--synthea-dir", default="synthea")
    run = subparsers.add_parser("run")
    run.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        synthea_dir = "synthea" if Path("synthea").exists() else "vendor/synthea"
        status = {
            "python": sys.version,
            "platform": platform.platform(),
            **SyntheaRunner(synthea_dir).doctor(),
        }
        print(json.dumps(status, indent=2))
        return 0
    if args.command == "setup-synthea":
        print(json.dumps(SyntheaRunner(args.synthea_dir).setup(), indent=2))
        return 0

    output = run_pipeline(args.config)
    print(f"Run completed: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
