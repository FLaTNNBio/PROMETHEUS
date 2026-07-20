from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:  # Package import in tests/editable installs.
    from .run_prometheus_suites import build_suite_plan, run_suite
except ImportError:  # Direct execution: py src\scripts\run_all_dm77_experiments.py
    from run_prometheus_suites import build_suite_plan, run_suite


METHOD_CONFIG = "configs/prometheus/dm77_method_ablation_suites.yaml"
METHOD_SUITES = (
    "causal_signal_estimator_ablation",
    "causal_signal_robustness_ablation",
    "pair_construction_ablation",
    "permuted_pair_label_control",
    "current_care_state_ablation",
    "dgp_negative_controls",
    "dgp_identification_stress",
)
PUBLICATION_SUITES = (
    (
        "configs/prometheus/dm77_integrated_experiment_suites.yaml",
        "dm77_primary_multi_seed",
    ),
    (
        "configs/prometheus/dm77_integrated_experiment_suites.yaml",
        "dm77_initial_stress_ablation",
    ),
    (
        "configs/prometheus/dm77_integrated_experiment_suites.yaml",
        "dm77_seed_stability",
    ),
    (
        "configs/prometheus/dm77_integrated_contrastive_v2_experiment_suites.yaml",
        "dm77_contrastive_v2_primary_multi_seed",
    ),
    (
        "configs/prometheus/dm77_integrated_contrastive_v2_experiment_suites.yaml",
        "dm77_contrastive_v2_initial_stress_ablation",
    ),
    (
        "configs/prometheus/dm77_integrated_contrastive_v2_experiment_suites.yaml",
        "dm77_contrastive_v2_seed_stability",
    ),
)


def experiment_groups(profile: str) -> list[tuple[str, str, list[int] | None]]:
    seed_override = [17] if profile == "quick" else None
    groups = [(METHOD_CONFIG, suite, seed_override) for suite in METHOD_SUITES]
    if profile == "publication":
        groups.extend((config, suite, None) for config, suite in PUBLICATION_SUITES)
    return groups


def _write_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all prespecified PROMETHEUS DM77 experiments sequentially"
    )
    parser.add_argument(
        "--profile", choices=("quick", "full", "publication"), default="quick",
        help=(
            "quick: every method condition with seed 17; full: all method seeds; "
            "publication: full plus the large primary/contrastive suites"
        ),
    )
    parser.add_argument(
        "--output-root", default="artifacts/dm77_method_ablations"
    )
    parser.add_argument(
        "--summary-dir", default="artifacts/dm77_method_ablations_summary"
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Skip completed plan entries (enabled by default)",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Do not run the repository test suite before experiments",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned/completed/remaining counts without running anything",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    groups = experiment_groups(args.profile)
    plan_rows = []
    for config, suite, seeds in groups:
        _, plan = build_suite_plan(suite, config, seeds=seeds)
        completed = 0
        for index, (seed, _) in enumerate(plan):
            planned_root = output_root / suite / f"s{seed}_r{index:03d}"
            completed += int(
                planned_root.exists()
                and any(planned_root.rglob("run_manifest.json"))
            )
        plan_rows.append({
            "suite": suite,
            "suite_config": config,
            "planned_runs": len(plan),
            "completed_runs_found": completed,
            "remaining_runs": len(plan) - completed if args.resume else len(plan),
            "seed_override": seeds,
        })
    total_runs = sum(row["planned_runs"] for row in plan_rows)
    status_path = output_root / "master_status.json"
    status = {
        "profile": args.profile,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "summary_dir": str(args.summary_dir),
        "resume": bool(args.resume),
        "planned_runs": total_runs,
        "state": "running",
        "suites": plan_rows,
        "completed_suites": [],
    }
    _write_status(status_path, status)
    print(f"Profile: {args.profile} | suites: {len(groups)} | planned runs: {total_runs}")
    print(f"Resume: {args.resume} | output: {output_root}")
    for row in plan_rows:
        print(
            f"  {row['suite']}: planned={row['planned_runs']} "
            f"completed={row['completed_runs_found']} remaining={row['remaining_runs']}"
        )
    if args.dry_run:
        status["state"] = "planned"
        status["remaining_runs"] = sum(row["remaining_runs"] for row in plan_rows)
        _write_status(status_path, status)
        print(f"DRY RUN: {status_path}")
        return
    started = time.time()
    try:
        if not args.skip_tests:
            print("[preflight] py -m pytest -q")
            subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
        for position, (config, suite, seeds) in enumerate(groups, start=1):
            print(f"[{position}/{len(groups)}] {suite}")
            result = run_suite(
                suite=suite,
                suite_config=config,
                output_root=output_root,
                seeds=seeds,
                resume=bool(args.resume),
            )
            status["completed_suites"].append(result)
            status["last_completed_suite"] = suite
            status["duration_seconds"] = time.time() - started
            _write_status(status_path, status)

        summary_script = Path(__file__).with_name("summarize_dm77_action_suites.py")
        print("[summary] aggregating completed runs")
        subprocess.run([
            sys.executable, str(summary_script),
            "--runs-dir", str(output_root),
            "--output-dir", str(args.summary_dir),
        ], check=True)
        status["state"] = "complete"
        status["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["duration_seconds"] = time.time() - started
        _write_status(status_path, status)
        print(f"COMPLETE: {status_path}")
        print(f"SUMMARY: {args.summary_dir}")
    except BaseException as error:
        status["state"] = "failed"
        status["failure"] = f"{type(error).__name__}: {error}"
        status["duration_seconds"] = time.time() - started
        _write_status(status_path, status)
        print(f"FAILED: {status_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
