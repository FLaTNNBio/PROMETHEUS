from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    if frame.empty:
        return "_Nessun dato disponibile._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    return "\n".join([
        line(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
        *(line(row) for row in rows),
    ])


def _single_case(root: Path) -> str:
    benchmark = pd.read_csv(root / "evaluation_only" / "policy_benchmark.csv")
    central = benchmark[[
        "Policy", "Optimizer", "Normalized value", "Regret", "Served", "Medicalized"
    ]].sort_values(["Optimizer", "Normalized value"], ascending=[True, False])
    lines = [
        "# Riepilogo scientifico EMS PROMETHEUS",
        "",
        "## Benchmark centrale",
        "",
        "Questo confronto usa un solo DGP e vincoli nominali comuni. La sensitivity dei vincoli, se presente, è riportata separatamente e non costituisce una campagna multi-DGP.",
        "",
        _markdown_table(central),
        "",
    ]
    diagnostics_path = root / "evaluation_only" / "policy_score_diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        prometheus = diagnostics.loc[
            diagnostics.policy.eq("PROMETHEUS"),
            ["scope", "spearman", "kendall", "pairwise_concordance", "ndcg_at_25pct", "top25pct_overlap"],
        ]
        lines += [
            "## Diagnostica diretta dello score PROMETHEUS",
            "",
            _markdown_table(prometheus),
            "",
        ]
    constraint_path = root / "evaluation_only" / "constraint_sensitivity" / "metrics.csv"
    if constraint_path.exists():
        constraints = pd.read_csv(constraint_path)
        prometheus = constraints.loc[
            constraints.policy.eq("PROMETHEUS") & constraints.optimizer.eq("MILP")
        ]
        lines += [
            "## Sensitivity dei vincoli",
            "",
            f"Regimi operativi valutati: **{prometheus.scenario_id.nunique()}**. Queste righe riutilizzano lo stesso ranking e cambiano soltanto budget/capacità.",
            "",
        ]
    return "\n".join(lines)


def _scenario_campaign(root: Path) -> str:
    summary = pd.read_csv(root / "policy_summary_bootstrap.csv")
    normalized = summary.loc[
        summary.metric.eq("Normalized value")
        & summary.Policy.isin([
            "PROMETHEUS", "Risk-first", "Need-first", "Outcome-first",
            "Profile-mean", "T-learner GBDT", "X-learner GBDT", "DR-learner GBDT",
        ])
        & summary.Optimizer.eq("MILP"),
        ["response_scenario", "Policy", "mean", "ci_low", "ci_high", "runs"],
    ].sort_values(["response_scenario", "mean"], ascending=[True, False])
    paired = pd.read_csv(root / "paired_differences_summary.csv")
    dgp = pd.read_csv(root / "dgp_diagnostics.csv")
    dgp_summary = dgp.groupby("response_scenario", as_index=False).agg(
        nurse_risk_benefit_spearman=("nurse_risk_benefit_spearman", "mean"),
        medicalized_risk_benefit_spearman=("medicalized_risk_benefit_spearman", "mean"),
        nurse_negative_fraction=("nurse_negative_fraction", "mean"),
        medicalized_negative_fraction=("medicalized_negative_fraction", "mean"),
    )
    lines = [
        "# Campagna multi-DGP EMS PROMETHEUS",
        "",
        "## Diagnostica dei DGP",
        "",
        _markdown_table(dgp_summary),
        "",
        "## Valore allocativo per scenario causale",
        "",
        _markdown_table(normalized),
        "",
        "## Differenze appaiate PROMETHEUS meno baseline",
        "",
        _markdown_table(paired),
        "",
    ]
    score_path = root / "score_diagnostics_summary.csv"
    if score_path.exists():
        score = pd.read_csv(score_path)
        score = score.loc[
            score.policy.eq("PROMETHEUS") & score.scope.eq("global_opportunities"),
            ["response_scenario", "runs", "spearman_mean", "pairwise_concordance_mean", "ndcg_at_25pct_mean", "top25pct_overlap_mean"],
        ]
        lines += [
            "## Qualità globale dello score PROMETHEUS",
            "",
            _markdown_table(score),
            "",
        ]
    return "\n".join(lines)


def _ablation_campaign(root: Path) -> str:
    summary = pd.read_csv(root / "ablation_policy_summary.csv")
    paired = pd.read_csv(root / "ablation_paired_differences.csv")
    audit = pd.read_csv(root / "ablation_ranking_audit.csv")
    audit_summary = audit.groupby(
        ["response_scenario", "variant"], as_index=False
    ).agg(
        runs=("run_id", "nunique"),
        best_epoch_mean=("best_epoch", "mean"),
        validation_selection_metric_mean=("validation_selection_metric", "mean"),
        triplet_enabled=("triplet_enabled", "first"),
        triplets_mean=("triplets", "mean"),
    )
    return "\n".join([
        "# Ablation EMS PROMETHEUS",
        "",
        "## Valore allocativo",
        "",
        _markdown_table(summary),
        "",
        "## Differenze appaiate",
        "",
        _markdown_table(paired),
        "",
        "## Audit del training",
        "",
        _markdown_table(audit_summary),
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a scientifically separated summary of EMS outputs."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    if (root / "ablation_policy_results.csv").exists():
        report = _ablation_campaign(root)
    elif (root / "all_policy_results.csv").exists():
        report = _scenario_campaign(root)
    elif (root / "evaluation_only" / "policy_benchmark.csv").exists():
        report = _single_case(root)
    else:
        raise FileNotFoundError(
            "Cartella non riconosciuta: attesi output single-case, scenario campaign o ablation campaign."
        )
    output = Path(args.output).resolve() if args.output else root / "scientific_summary.md"
    output.write_text(report + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
