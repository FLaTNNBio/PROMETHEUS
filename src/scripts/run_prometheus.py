import argparse

from causal_population_ranking.causal_utilization.global_prometheus_runner import run_global_prometheus
from causal_population_ranking.causal_utilization.prometheus_runner import run_prometheus as run_legacy_prometheus
from causal_population_ranking.config import load_prometheus_config


def main():
    parser = argparse.ArgumentParser(description="Run PROMETHEUS global patient-action causal ranking")
    parser.add_argument(
        "--config", default="configs/prometheus/dm77_integrated_default.yaml"
    )
    parser.add_argument("--legacy-local", action="store_true", help="Run the preserved local-transition baseline")
    parser.add_argument(
        "--synthetic-scenario",
        choices=(
            "baseline", "older_high_need", "social_fragility_shift",
            "utilization_surge", "combined_shift",
        ),
        help="Override data.synthetic.scenario for a fully synthetic global run",
    )
    args = parser.parse_args()
    runner = run_legacy_prometheus if args.legacy_local else run_global_prometheus
    config = load_prometheus_config(args.config)
    if args.synthetic_scenario:
        if args.legacy_local:
            parser.error("--synthetic-scenario is available only for the global runner")
        config["data"]["synthetic"]["scenario"] = args.synthetic_scenario
    print(runner(config))


if __name__ == "__main__":
    main()
