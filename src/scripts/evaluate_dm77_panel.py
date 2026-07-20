from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_population_ranking.evaluation.dm77_clinical_validation import (
    evaluate_dm77_against_adjudicated_panel,
    weighted_fleiss_kappa,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DM77 computable rules against an adjudicated panel"
    )
    parser.add_argument("--rule-assessments", required=True)
    parser.add_argument("--adjudicated-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--panel-rater-columns", nargs="*")
    args = parser.parse_args()
    rules = pd.read_csv(args.rule_assessments)
    reference = pd.read_csv(args.adjudicated_reference)
    result = evaluate_dm77_against_adjudicated_panel(rules, reference)
    summary = dict(result.summary)
    if args.panel_rater_columns:
        missing = set(args.panel_rater_columns).difference(reference)
        if missing:
            raise ValueError(f"Missing panel rater columns: {sorted(missing)}")
        summary["quadratic_weighted_fleiss_kappa"] = weighted_fleiss_kappa(
            reference.loc[:, args.panel_rater_columns].to_numpy(float)
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dm77_panel_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    result.confusion_matrix.to_csv(output / "dm77_level_confusion_matrix.csv")
    result.critical_level_metrics.to_csv(
        output / "dm77_critical_level_metrics.csv", index=False
    )
    print(output)


if __name__ == "__main__":
    main()
