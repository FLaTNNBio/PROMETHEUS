from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#475569")
TRANSITIONS = ("1_to_2", "2_to_3", "3_to_4", "4_to_5", "5_to_6")


def _svg_line_plot(
    frame: pd.DataFrame,
    x_column: str,
    title: str,
    y_label: str,
    output: Path,
    categorical: bool = False,
) -> None:
    if frame.empty:
        return
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 55, 80
    methods = sorted(frame.method.unique())
    if categorical:
        categories = [value for value in TRANSITIONS if value in set(frame[x_column])]
        x_numeric = {value: index for index, value in enumerate(categories)}
    else:
        categories = sorted(frame[x_column].astype(float).unique())
        x_numeric = {value: float(value) for value in categories}
    values = frame["mean"].astype(float).to_numpy()
    errors = frame["standard_error"].fillna(0.0).astype(float).to_numpy()
    y_min = float(np.min(values - errors))
    y_max = float(np.max(values + errors))
    if np.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    padding = 0.08 * (y_max - y_min)
    y_min -= padding
    y_max += padding
    x_values = list(x_numeric.values())
    x_min, x_max = min(x_values), max(x_values)
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5

    def px(value):
        numeric = x_numeric[value] if categorical else float(value)
        return left + (numeric - x_min) / (x_max - x_min) * (width - left - right)

    def py(value):
        return top + (y_max - float(value)) / (y_max - y_min) * (height - top - bottom)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = py(tick)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick:.3g}</text>',
        ])
    for category in categories:
        x = px(category)
        label = str(category).replace("_to_", "->")
        elements.append(
            f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-size="11" font-family="sans-serif">{escape(label)}</text>'
        )
    elements.append(
        f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-size="12" font-family="sans-serif">{escape(y_label)}</text>'
    )

    for method_index, method in enumerate(methods):
        color = COLORS[method_index % len(COLORS)]
        subset = frame[frame.method == method].copy()
        subset["_x"] = subset[x_column].map(x_numeric) if categorical else subset[x_column].astype(float)
        subset = subset.sort_values("_x")
        points = " ".join(f"{px(row[x_column]):.2f},{py(row['mean']):.2f}" for _, row in subset.iterrows())
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for _, row in subset.iterrows():
            x, y = px(row[x_column]), py(row["mean"])
            error = 0.0 if pd.isna(row["standard_error"]) else float(row["standard_error"])
            elements.extend([
                f'<line x1="{x:.2f}" y1="{py(row["mean"]-error):.2f}" x2="{x:.2f}" y2="{py(row["mean"]+error):.2f}" stroke="{color}"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>',
            ])
        legend_y = top + 18 * method_index
        elements.extend([
            f'<line x1="{width-265}" y1="{legend_y}" x2="{width-245}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{width-238}" y="{legend_y+4}" font-size="11" font-family="sans-serif">{escape(method)}</text>',
        ])
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements), encoding="utf-8")


def _summary(root: Path, suite: str, filename: str) -> pd.DataFrame:
    path = root / suite / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create diagnostic SVGs from actual suite summaries")
    parser.add_argument("--analysis-root", default="artifacts/prometheus_suite_analysis")
    parser.add_argument("--output-dir", default="reports/prometheus_diagnostics")
    args = parser.parse_args()
    root = Path(args.analysis_root)
    output = Path(args.output_dir)

    specifications = (
        ("shared_learning", "macro_mean_se.csv", "shared_effect_correlation", "autoc", "Performance versus shared-effect correlation", "AUTOC", "performance_vs_shared_effect.svg", False),
        ("imbalance", "macro_mean_se.csv", "transition_imbalance_rho", "autoc", "Performance versus transition imbalance", "AUTOC", "performance_vs_imbalance.svg", False),
        ("contrastive_ablation", "per_transition_mean_se.csv", "transition", "benefit_at_operational_capacity", "Benefit at operational capacity by transition", "Benefit@B", "benefit_at_capacity_by_transition.svg", True),
        ("overlap_stress", "macro_mean_se.csv", "overlap_strength", "policy_regret_at_operational_capacity", "Policy regret versus overlap strength", "Policy regret", "policy_regret_vs_overlap.svg", False),
        ("risk_benefit_misalignment", "macro_mean_se.csv", "risk_benefit_alignment", "benefit_at_operational_capacity", "Causal and risk ranking under risk-benefit alignment", "Benefit@B", "causal_vs_risk_alignment.svg", False),
    )
    generated = []
    for suite, filename, x_column, metric, title, y_label, output_name, categorical in specifications:
        frame = _summary(root, suite, filename)
        if frame.empty:
            continue
        frame = frame[frame.metric == metric]
        if suite == "risk_benefit_misalignment":
            frame = frame[frame.method.isin({
                "prognostic_risk_ranking", "unified_rank_only",
                "prometheus_causal_contrastive", "independent_rankers",
            })]
        target = output / output_name
        _svg_line_plot(frame, x_column, title, y_label, target, categorical)
        if target.exists():
            generated.append(str(target))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
