from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#475569")


def _point_plot(frame: pd.DataFrame, metric: str, output: Path) -> None:
    subset = frame[frame.metric == metric].copy()
    if subset.empty:
        return
    scenarios = sorted(subset.scenario.unique())
    methods = sorted(subset.method.unique())
    width, height = 1000, 520
    left, right, top, bottom = 90, 30, 55, 100
    values = subset["mean"].to_numpy(float)
    errors = subset.standard_error.fillna(0).to_numpy(float)
    low, high = float(np.min(values - errors)), float(np.max(values + errors))
    if np.isclose(low, high):
        low, high = low - 0.5, high + 0.5
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad

    def px(scenario_index, method_index):
        center = left + (scenario_index + 0.5) * (width - left - right) / len(scenarios)
        return center + (method_index - (len(methods) - 1) / 2) * 11

    def py(value):
        return top + (high - value) / (high - low) * (height - top - bottom)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="sans-serif">{escape(metric.replace("_", " ").title())}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    for tick in np.linspace(low, high, 6):
        y = py(tick)
        elements.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11" font-family="sans-serif">{tick:.3g}</text>',
        ])
    for scenario_index, scenario in enumerate(scenarios):
        center = left + (scenario_index + 0.5) * (width - left - right) / len(scenarios)
        elements.append(
            f'<text x="{center:.2f}" y="{height-bottom+25}" text-anchor="middle" font-size="11" font-family="sans-serif">{escape(str(scenario))}</text>'
        )
    for method_index, method in enumerate(methods):
        color = COLORS[method_index % len(COLORS)]
        for scenario_index, scenario in enumerate(scenarios):
            row = subset[(subset.method == method) & (subset.scenario == scenario)]
            if row.empty:
                continue
            mean = float(row.iloc[0]["mean"])
            error = float(row.iloc[0]["standard_error"]) if pd.notna(row.iloc[0]["standard_error"]) else 0.0
            x = px(scenario_index, method_index)
            elements.extend([
                f'<line x1="{x:.2f}" y1="{py(mean-error):.2f}" x2="{x:.2f}" y2="{py(mean+error):.2f}" stroke="{color}"/>',
                f'<circle cx="{x:.2f}" cy="{py(mean):.2f}" r="4" fill="{color}"/>',
            ])
        legend_y = top + 18 * method_index
        elements.append(f'<circle cx="{width-300}" cy="{legend_y}" r="4" fill="{color}"/>')
        elements.append(f'<text x="{width-288}" y="{legend_y+4}" font-size="11" font-family="sans-serif">{escape(method)}</text>')
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot global PROMETHEUS suite summaries from actual runs")
    parser.add_argument("--summary", required=True, help="Path to global_mean_se.csv")
    parser.add_argument("--output-dir", default="reports/prometheus_global_diagnostics")
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    output = Path(args.output_dir)
    for metric in (
        "cross_transition_concordance", "global_allocation_value",
        "global_regret", "fraction_of_oracle_benefit",
    ):
        _point_plot(frame, metric, output / f"{metric}.svg")
    print(output)


if __name__ == "__main__":
    main()
