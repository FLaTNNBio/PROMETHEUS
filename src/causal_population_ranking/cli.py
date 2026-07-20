from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .causal_utilization.global_prometheus_runner import run_global_prometheus
from .causal_utilization.prometheus_runner import run_prometheus as run_legacy_prometheus
from .config import load_prometheus_config
from .data.synthea_runner import SyntheaRunner


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="causal-ranking",
        description="PROMETHEUS global patient-action causal ranking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    setup = subparsers.add_parser("setup-synthea")
    setup.add_argument("--synthea-dir", default="synthea")
    run = subparsers.add_parser("run")
    run.add_argument(
        "--config", default="configs/prometheus/dm77_integrated_default.yaml"
    )
    legacy = subparsers.add_parser("run-legacy-local")
    legacy.add_argument("--config", required=True)
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

    runner = run_legacy_prometheus if args.command == "run-legacy-local" else run_global_prometheus
    output = runner(load_prometheus_config(args.config))
    print(f"Run completed: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
