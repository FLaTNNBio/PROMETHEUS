from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml

from causal_population_ranking.causal_utilization.global_prometheus_runner import run_global_prometheus
from causal_population_ranking.config import load_prometheus_config


AXES = {
    "scenario",
    "shared_effect_correlation",
    "transition_sample_imbalance",
    "overlap_strength",
    "risk_benefit_alignment",
    "training_mode",
    "synthetic_scenario",
    "dgp_scenario",
    "dataset_seed",
    "causal_signal_estimator",
    "causal_signal_aggregation",
    "causal_signal_winsorize",
    "causal_signal_reliability",
    "min_direction_agreement",
    "beta_cross",
    "checkpoint_metric",
    "lambda_con",
    "contrastive_pair_scope",
    "permute_training_pair_labels",
    "exclude_current_care_state_features",
}


def _apply(base: dict, values: dict, seed: int, output_root: Path, run_index: int) -> dict:
    config = copy.deepcopy(base)
    config["run"]["seed"] = int(seed)
    if "dataset_seed" in values:
        config["run"]["dataset_seed"] = int(values["dataset_seed"])
    if "scenario" in values:
        config["run"]["scenario"] = values["scenario"]
    for name in (
        "shared_effect_correlation", "transition_sample_imbalance",
        "overlap_strength", "risk_benefit_alignment",
    ):
        if name in values:
            config["simulation"][name] = float(values[name])
    if "training_mode" in values:
        mode = values["training_mode"]
        config["prometheus"]["training_mode"] = mode
        config["prometheus"]["lambda_con"] = (
            0.0 if mode in {"independent_local_rankers", "unified_local_ranker", "unified_global_ranker"}
            else float(base["prometheus"]["lambda_con"])
        )
    if "synthetic_scenario" in values:
        config["data"]["synthetic"]["scenario"] = values["synthetic_scenario"]
    if "dgp_scenario" in values:
        config["simulation"]["scenario"] = values["dgp_scenario"]
    if "causal_signal_estimator" in values:
        config["causal_signal"]["estimator"] = str(values["causal_signal_estimator"])
    if "causal_signal_aggregation" in values:
        config["causal_signal"]["aggregation"] = str(values["causal_signal_aggregation"])
    if "causal_signal_winsorize" in values:
        config["causal_signal"]["winsorize"] = bool(values["causal_signal_winsorize"])
    if "causal_signal_reliability" in values:
        config["causal_signal"]["reliability_weighting"] = bool(
            values["causal_signal_reliability"]
        )
    if "min_direction_agreement" in values:
        config["causal_signal"]["min_direction_agreement"] = float(
            values["min_direction_agreement"]
        )
    if "beta_cross" in values:
        config["prometheus"]["beta_cross"] = float(values["beta_cross"])
    if "checkpoint_metric" in values:
        config.setdefault("checkpoint", {})["metric"] = str(
            values["checkpoint_metric"]
        )
    if "lambda_con" in values:
        config["prometheus"]["lambda_con"] = float(values["lambda_con"])
    if "contrastive_pair_scope" in values:
        config.setdefault("contrastive", {})["pair_scope"] = str(
            values["contrastive_pair_scope"]
        )
    if "permute_training_pair_labels" in values:
        config.setdefault("negative_control", {})[
            "permute_training_pair_labels"
        ] = bool(values["permute_training_pair_labels"])
    if "exclude_current_care_state_features" in values:
        config.setdefault("ablation", {})[
            "exclude_current_care_state_features"
        ] = bool(values["exclude_current_care_state_features"])
    config["run"]["output_root"] = str(output_root / f"s{seed}_r{run_index:03d}")
    return config


def build_suite_plan(
    suite: str,
    suite_config: str | Path,
    seeds: list[int] | tuple[int, ...] | None = None,
    max_runs: int | None = None,
) -> tuple[dict, list[tuple[int, dict]]]:
    """Validate a suite document and return its deterministic execution plan."""

    document = yaml.safe_load(Path(suite_config).read_text(encoding="utf-8"))
    if suite not in document["suites"]:
        raise ValueError(f"Unknown suite {suite!r}")
    definition = document["suites"][suite]
    unknown = set(definition).difference(AXES | {"seed"})
    if unknown:
        raise ValueError(f"Unknown global suite axes: {sorted(unknown)}")
    if not set(definition).intersection(AXES - {"dataset_seed", "scenario"}):
        raise ValueError("A suite must vary at least one scientific method or DGP axis")
    selected_seeds = seeds or definition.get("seed") or document.get(
        "default_seeds", [11, 37, 71]
    )
    names = [name for name in definition if name != "seed"]
    conditions = [dict(zip(names, values)) for values in itertools.product(
        *(definition[name] for name in names)
    )]
    plan = [(int(seed), condition) for seed in selected_seeds for condition in conditions]
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("--max-runs must be positive")
        plan = plan[:max_runs]
    return document, plan


def run_suite(
    suite: str,
    suite_config: str | Path,
    output_root: str | Path,
    seeds: list[int] | tuple[int, ...] | None = None,
    max_runs: int | None = None,
    resume: bool = False,
) -> dict:
    """Execute one suite, optionally skipping completed plan entries."""

    document, plan = build_suite_plan(suite, suite_config, seeds, max_runs)
    suite_root = Path(output_root) / suite
    suite_root.mkdir(parents=True, exist_ok=True)
    records = [{
        "run_index": index,
        "seed": seed,
        "condition": condition,
        "output_root": str(suite_root / f"s{seed}_r{index:03d}"),
    } for index, (seed, condition) in enumerate(plan)]
    (suite_root / "suite_plan.json").write_text(
        json.dumps({"suite": suite, "runs": records}, indent=2), encoding="utf-8"
    )
    base = load_prometheus_config(document["base_config"])
    executed, skipped, outputs = 0, 0, []
    for index, (seed, condition) in enumerate(plan):
        planned_root = suite_root / f"s{seed}_r{index:03d}"
        completed = sorted(planned_root.rglob("run_manifest.json")) if planned_root.exists() else []
        if resume and completed:
            output = completed[-1].parent
            skipped += 1
            outputs.append(str(output))
            print(f"SKIP completed: {output}")
            continue
        output = run_global_prometheus(
            _apply(base, condition, seed, suite_root, index)
        )
        executed += 1
        outputs.append(str(output))
        print(output)
    result = {
        "suite": suite,
        "suite_config": str(suite_config),
        "planned_runs": len(plan),
        "executed_runs": executed,
        "skipped_completed_runs": skipped,
        "outputs": outputs,
    }
    (suite_root / "suite_status.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded global PROMETHEUS ablation suite")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--suite-config", default="configs/prometheus/global_experiment_suites.yaml")
    parser.add_argument("--output-root", default="artifacts/prometheus_global_suites")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help="Skip plan entries that already contain a run_manifest.json",
    )
    args = parser.parse_args()
    run_suite(
        args.suite, args.suite_config, args.output_root,
        seeds=args.seeds, max_runs=args.max_runs, resume=args.resume,
    )


if __name__ == "__main__":
    main()
