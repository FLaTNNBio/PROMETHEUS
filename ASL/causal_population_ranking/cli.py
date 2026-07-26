"""Thin command-line router for the repository's supported workflows.

Scientific behavior belongs to the versioned configs and their runners; this module
only validates command arguments and dispatches them.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from scripts.run_pipeline import run_pipeline
from .data.synthea_runner import SyntheaRunner
from .evaluation.external import run_external_validation
from .ems import run_ems_case_study
from .research import (
    run_balanced_encoder_benchmark,
    run_baseline_benchmark,
    run_contrastive_benchmark,
    run_contrastive_v2_benchmark,
    run_contrastive_v2_1_benchmark,
    run_contrastive_v2_2_benchmark,
    run_dynamic_pairs_benchmark,
    run_opposite_arm_benchmark,
    run_shared_encoder_benchmark,
)
from .reporting import generate_paper_report


BENCHMARKS = {
    "baselines": (run_baseline_benchmark, "configs/research/baselines.yaml"),
    "contrastive": (run_contrastive_benchmark, "configs/research/contrastive.yaml"),
    "contrastive-v2": (
        run_contrastive_v2_benchmark,
        "configs/research/contrastive_v2.yaml",
    ),
    "contrastive-v2.1": (
        run_contrastive_v2_1_benchmark,
        "configs/research/contrastive_v2_1.yaml",
    ),
    "contrastive-v2.2": (
        run_contrastive_v2_2_benchmark,
        "configs/research/contrastive_v2_2.yaml",
    ),
    "dynamic-pairs": (
        run_dynamic_pairs_benchmark,
        "configs/research/dynamic_pairs.yaml",
    ),
    "opposite-arm": (
        run_opposite_arm_benchmark,
        "configs/research/opposite_arm.yaml",
    ),
    "shared-encoder": (
        run_shared_encoder_benchmark,
        "configs/research/shared_encoder.yaml",
    ),
    "balanced-encoder": (
        run_balanced_encoder_benchmark,
        "configs/research/balanced_encoder.yaml",
    ),
}


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
    external = subparsers.add_parser("validate-external")
    external.add_argument("--config", default="configs/external_validation.yaml")
    external.add_argument("--predictions", required=True)
    external.add_argument("--clinical-reference", required=True)
    external.add_argument("--feature-provenance", required=True)
    external.add_argument("--governance-manifest", required=True)
    external.add_argument("--subgroups", required=True)
    external.add_argument("--acg")
    external.add_argument("--output-dir", required=True)
    paper = subparsers.add_parser("report-paper")
    paper.add_argument("--artifact-dir", required=True)
    paper.add_argument("--output-dir", required=True)
    ems = subparsers.add_parser(
        "ems-case-study",
        help="Run the aggregate-informed semi-synthetic EMS/118 case study.",
    )
    ems.add_argument(
        "--config",
        default="configs/applications/ems_case_study.yaml",
    )
    ems.add_argument("--output-dir")
    ems.add_argument(
        "--smoke",
        action="store_true",
        help="Run the bounded end-to-end verification protocol.",
    )
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run one reproducible research benchmark.",
    )
    benchmark.add_argument("study", choices=tuple(BENCHMARKS))
    benchmark.add_argument(
        "--config",
        help="Optional protocol override; the selected study has a clean default.",
    )
    benchmark.add_argument("--output-dir")
    benchmark.add_argument(
        "--smoke",
        action="store_true",
        help="Use the bounded smoke protocol (contrastive studies only).",
    )
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
    if args.command == "validate-external":
        output = run_external_validation(
            config_path=args.config,
            predictions_path=args.predictions,
            clinical_reference_path=args.clinical_reference,
            feature_provenance_path=args.feature_provenance,
            governance_manifest_path=args.governance_manifest,
            subgroup_path=args.subgroups,
            acg_path=args.acg,
            output_dir=args.output_dir,
        )
        print(f"External validation completed: {Path(output).resolve()}")
        return 0
    if args.command == "report-paper":
        output = generate_paper_report(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
        )
        print(f"Paper report completed: {Path(output).resolve()}")
        return 0
    if args.command == "ems-case-study":
        output = run_ems_case_study(
            config_path=args.config,
            output_dir=args.output_dir,
            smoke=bool(args.smoke),
        )
        print(f"EMS case study completed: {Path(output).resolve()}")
        return 0
    if args.command == "benchmark":
        contrastive_studies = {
            "contrastive", "contrastive-v2", "contrastive-v2.1",
            "contrastive-v2.2",
        }
        if args.smoke and args.study not in contrastive_studies:
            parser.error("--smoke is available only for contrastive benchmarks")
        runner, default_config = BENCHMARKS[args.study]
        kwargs = {
            "config_path": args.config or default_config,
            "output_dir": args.output_dir,
        }
        if args.study in contrastive_studies:
            kwargs["smoke"] = bool(args.smoke)
        output = runner(**kwargs)
        print(f"Benchmark '{args.study}' completed: {Path(output).resolve()}")
        return 0

    output = run_pipeline(args.config)
    print(f"Run completed: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
