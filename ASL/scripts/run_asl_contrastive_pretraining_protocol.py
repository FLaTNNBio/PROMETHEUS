from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from causal_population_ranking.applications.asl_ablation_campaign import (
    run_asl_ablation_campaign,
)


PHASES = {
    "discovery": {
        "runs": 5,
        "variants": (
            "full",
            "contrastive_pretrained_finetuned",
            "contrastive_pretrained_frozen",
            "random_init",
            "joint_uact",
        ),
    },
    "ablation": {
        "runs": 5,
        "variants": (
            "full",
            "contrastive_pretrained_finetuned",
            "contrastive_pretrained_frozen",
            "random_init",
            "joint_uact",
            "legacy_triplet",
            "uact_no_uncertainty",
            "rank_only",
        ),
    },
    "confirmation": {
        "runs": 10,
        "variants": (
            "full",
            "contrastive_pretrained_finetuned",
            "random_init",
            "joint_uact",
            "legacy_triplet",
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PROMETHEUS contrastive-pretraining discovery, ablation, "
            "or frozen confirmation protocol."
        )
    )
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--config",
        default="configs/applications/asl_semisynthetic.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override the prespecified run count for engineering diagnostics.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[
            "partially_aligned",
            "risk_benefit_misaligned",
            "mixed_response",
        ],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    phase = PHASES[args.phase]
    runs = int(args.runs if args.runs is not None else phase["runs"])
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = run_asl_ablation_campaign(
        config,
        args.input,
        args.output,
        variants=phase["variants"],
        scenarios=args.scenarios,
        runs_per_scenario=runs,
        smoke=args.smoke,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
