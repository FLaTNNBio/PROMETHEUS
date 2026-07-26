from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _single_run(root: Path) -> None:
    source = root / "asl_policy_results.csv"
    frame = pd.read_csv(source)
    plot = frame.loc[~frame.method.eq("Oracle")].sort_values(
        "normalized_value", ascending=True
    )
    plt.figure(figsize=(11, max(5, 0.38 * len(plot))))
    plt.barh(plot.method, plot.normalized_value)
    plt.xlabel("Normalized policy value (1 = oracle)")
    plt.ylabel("Method")
    plt.title("ASL real-X semi-synthetic policy comparison")
    plt.tight_layout()
    plt.savefig(root / "asl_policy_comparison.png", dpi=180)
    plt.close()

    ranking = frame.loc[~frame.method.eq("Oracle")].sort_values(
        "ranking_concordance", ascending=True
    )
    plt.figure(figsize=(11, max(5, 0.38 * len(ranking))))
    plt.barh(ranking.method, ranking.ranking_concordance)
    plt.xlabel("Ranking concordance")
    plt.ylabel("Method")
    plt.title("ASL opportunity-ranking quality")
    plt.tight_layout()
    plt.savefig(root / "asl_ranking_concordance.png", dpi=180)
    plt.close()

    best = plot.sort_values("normalized_value", ascending=False).iloc[0]
    prometheus = frame.loc[frame.method.eq("PROMETHEUS-Contrastive")].iloc[0]
    lines = [
        "# ASL real-X semi-synthetic comparison",
        "",
        "The covariate distribution is real and comes from the pseudonymized ASL baseline.",
        "Treatment assignment and follow-up outcomes are simulated.",
        "",
        f"- Best non-oracle method: **{best.method}**",
        f"- Best normalized value: **{best.normalized_value:.4f}**",
        f"- PROMETHEUS normalized value: **{prometheus.normalized_value:.4f}**",
        f"- PROMETHEUS ranking concordance: **{prometheus.ranking_concordance:.4f}**",
        f"- PROMETHEUS cross-profile concordance: **{prometheus.cross_profile_concordance:.4f}**",
        f"- PROMETHEUS within-patient profile accuracy: **{prometheus.within_patient_recommendation_accuracy:.4f}**",
        f"- PROMETHEUS Harm@25%: **{prometheus.harm_at_25pct:.4f}**",
        f"- PROMETHEUS selected harm rate: **{prometheus.selected_harm_rate:.4f}**",
        "",
        "## Complete comparison",
        "",
        frame.sort_values("normalized_value", ascending=False)[[
            "method", "normalized_value", "regret", "ranking_concordance",
            "cross_profile_concordance", "within_patient_recommendation_accuracy",
            "ndcg_at_25pct", "harm_at_25pct", "selected_harm_rate", "served",
        ]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "These values do not estimate real ASL service effectiveness because T and Y are semi-synthetic.",
    ]
    (root / "asl_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _campaign(root: Path) -> None:
    summary = pd.read_csv(root / "policy_summary_bootstrap.csv")
    non_oracle = summary.loc[~summary.method.eq("Oracle")].copy()
    for scenario, group in non_oracle.groupby("scenario"):
        plot = group.sort_values("mean_normalized_value", ascending=True)
        plt.figure(figsize=(11, max(5, 0.38 * len(plot))))
        plt.barh(plot.method, plot.mean_normalized_value)
        plt.xlabel("Mean normalized policy value")
        plt.ylabel("Method")
        plt.title(f"ASL real-X campaign: {scenario}")
        plt.tight_layout()
        safe = scenario.replace("/", "_")
        plt.savefig(root / f"asl_campaign_{safe}.png", dpi=180)
        plt.close()

    paired_path = root / "paired_differences_summary.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    lines = [
        "# ASL real-X semi-synthetic campaign comparison",
        "",
        "Covariates are real ASL baseline variables; T and Y are semi-synthetic.",
        "",
    ]
    for scenario, group in summary.groupby("scenario"):
        best = group.loc[~group.method.eq("Oracle")].sort_values(
            "mean_normalized_value", ascending=False
        ).iloc[0]
        lines += [
            f"## {scenario}",
            "",
            f"Best non-oracle: **{best.method}**, normalized value "
            f"**{best.mean_normalized_value:.4f}** "
            f"[{best.ci_low:.4f}, {best.ci_high:.4f}].",
            "",
            group.sort_values("mean_normalized_value", ascending=False)[[
                "method", "runs", "mean_normalized_value", "ci_low", "ci_high",
                "mean_regret", "mean_ranking_concordance",
                "mean_cross_profile_concordance", "mean_within_patient_accuracy",
            ]].to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    if not paired.empty:
        lines += [
            "## Paired comparisons: PROMETHEUS minus baseline",
            "",
            paired.to_markdown(index=False, floatfmt=".4f"),
            "",
            "A positive mean difference favors PROMETHEUS.",
        ]
    (root / "asl_campaign_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Single-run or campaign output directory")
    args = parser.parse_args()
    root = Path(args.input).expanduser().resolve()
    if (root / "asl_policy_results.csv").exists():
        _single_run(root)
        print(root / "asl_comparison_report.md")
        return
    if (root / "policy_summary_bootstrap.csv").exists():
        _campaign(root)
        print(root / "asl_campaign_comparison_report.md")
        return
    raise FileNotFoundError(
        "No asl_policy_results.csv or policy_summary_bootstrap.csv found in "
        f"{root}"
    )


if __name__ == "__main__":
    main()
