
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "experiment",
    "scenario",
    "policy",
    "optimizer",
    "true_value",
    "oracle_value",
    "normalized_value",
    "regret",
}


def discover_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Il file deve essere CSV: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Percorso non trovato: {input_path}")

    csvs = sorted(input_path.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nessun CSV trovato in: {input_path}")
    return csvs


def load_results(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            skipped.append(f"{path}: errore lettura ({exc})")
            continue

        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            skipped.append(f"{path}: colonne mancanti {sorted(missing)}")
            continue

        frame = frame.copy()
        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        details = "\n".join(skipped[:20])
        raise RuntimeError(
            "Nessun CSV compatibile con lo schema benchmark PROMETHEUS.\n" + details
        )

    out = pd.concat(frames, ignore_index=True)

    numeric_columns = [
        "true_value",
        "oracle_value",
        "normalized_value",
        "regret",
        "benefit_per_cost",
        "deferral_rate",
        "served",
        "cost_used",
        "budget",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    key_columns = ["experiment", "scenario", "policy", "optimizer"]
    if "scenario_id" in out.columns:
        key_columns.insert(2, "scenario_id")
    duplicate_count = int(out.duplicated(key_columns, keep=False).sum())
    out.attrs["duplicate_key_rows"] = duplicate_count
    scenario_parts = [out["experiment"].astype(str), out["scenario"].astype(str)]
    if "scenario_id" in out.columns:
        scenario_parts.append(out["scenario_id"].astype(str))
    out["scenario_key"] = scenario_parts[0]
    for part in scenario_parts[1:]:
        out["scenario_key"] = out["scenario_key"] + "|" + part

    return out


def add_method_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = (
        df["policy"].astype(str).str.strip()
        + " | "
        + df["optimizer"].astype(str).str.strip()
    )
    return df


def aggregate_methods(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["policy", "optimizer", "method"]

    aggregations: dict[str, tuple[str, str]] = {
        "n_rows": ("normalized_value", "size"),
        "n_scenarios": ("scenario_key", "nunique"),
        "normalized_value_mean": ("normalized_value", "mean"),
        "normalized_value_std": ("normalized_value", "std"),
        "normalized_value_median": ("normalized_value", "median"),
        "normalized_value_min": ("normalized_value", "min"),
        "normalized_value_max": ("normalized_value", "max"),
        "regret_mean": ("regret", "mean"),
        "regret_std": ("regret", "std"),
        "regret_median": ("regret", "median"),
        "regret_max": ("regret", "max"),
        "true_value_mean": ("true_value", "mean"),
    }

    if "benefit_per_cost" in df.columns:
        aggregations["benefit_per_cost_mean"] = ("benefit_per_cost", "mean")
    if "deferral_rate" in df.columns:
        aggregations["deferral_rate_mean"] = ("deferral_rate", "mean")
    if "served" in df.columns:
        aggregations["served_mean"] = ("served", "mean")

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    scenario_keys = ["experiment", "scenario"]
    if "scenario_id" in df.columns:
        scenario_keys.append("scenario_id")

    best_by_scenario = (
        df.loc[~df["policy"].eq("Oracle")]
        .groupby(scenario_keys, dropna=False)["normalized_value"]
        .transform("max")
    )
    non_oracle = df.loc[~df["policy"].eq("Oracle")].copy()
    non_oracle["is_best_non_oracle"] = np.isclose(
        non_oracle["normalized_value"].to_numpy(),
        best_by_scenario.to_numpy(),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=False,
    )
    win_rates = (
        non_oracle.groupby(group_cols, dropna=False)["is_best_non_oracle"]
        .mean()
        .rename("win_rate_non_oracle")
        .reset_index()
    )

    summary = summary.merge(win_rates, on=group_cols, how="left")
    summary = summary.sort_values(
        ["normalized_value_mean", "regret_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return summary


def paired_comparison(
    df: pd.DataFrame,
    reference_policy: str = "PROMETHEUS",
    reference_optimizer: str = "MILP",
) -> pd.DataFrame:
    scenario_keys = ["experiment", "scenario"]
    if "scenario_id" in df.columns:
        scenario_keys.append("scenario_id")

    ref = df[
        df["policy"].eq(reference_policy)
        & df["optimizer"].eq(reference_optimizer)
    ][scenario_keys + ["normalized_value", "regret"]].rename(
        columns={
            "normalized_value": "reference_normalized_value",
            "regret": "reference_regret",
        }
    )

    if ref.empty:
        return pd.DataFrame()

    candidates = df[
        ~(
            df["policy"].eq(reference_policy)
            & df["optimizer"].eq(reference_optimizer)
        )
    ].copy()

    paired = candidates.merge(ref, on=scenario_keys, how="inner")
    paired["delta_normalized_vs_reference"] = (
        paired["normalized_value"] - paired["reference_normalized_value"]
    )
    paired["delta_regret_vs_reference"] = (
        paired["regret"] - paired["reference_regret"]
    )
    paired["beats_reference"] = paired["delta_normalized_vs_reference"] > 0
    paired["ties_reference"] = np.isclose(
        paired["delta_normalized_vs_reference"], 0.0, atol=1e-12
    )

    out = (
        paired.groupby(["policy", "optimizer", "method"], dropna=False)
        .agg(
            paired_scenarios=("delta_normalized_vs_reference", "size"),
            mean_delta_normalized=("delta_normalized_vs_reference", "mean"),
            median_delta_normalized=("delta_normalized_vs_reference", "median"),
            worst_delta_normalized=("delta_normalized_vs_reference", "min"),
            best_delta_normalized=("delta_normalized_vs_reference", "max"),
            mean_delta_regret=("delta_regret_vs_reference", "mean"),
            beat_rate=("beats_reference", "mean"),
            tie_rate=("ties_reference", "mean"),
        )
        .reset_index()
        .sort_values("mean_delta_normalized", ascending=False)
    )
    return out


def summarize_by_experiment(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["experiment", "policy", "optimizer", "method"], dropna=False)
        .agg(
            n_scenarios=("scenario_key", "nunique"),
            normalized_value_mean=("normalized_value", "mean"),
            normalized_value_std=("normalized_value", "std"),
            normalized_value_min=("normalized_value", "min"),
            regret_mean=("regret", "mean"),
            regret_max=("regret", "max"),
        )
        .reset_index()
        .sort_values(
            ["experiment", "normalized_value_mean"],
            ascending=[True, False],
        )
    )


def plot_method_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary.loc[~summary["policy"].eq("Oracle")].copy()
    plot_df = plot_df.sort_values("normalized_value_mean", ascending=True)

    plt.figure(figsize=(11, max(5, 0.38 * len(plot_df))))
    plt.barh(plot_df["method"], plot_df["normalized_value_mean"])
    plt.xlabel("Valore normalizzato medio (1 = oracle)")
    plt.ylabel("Metodo")
    plt.title("Confronto medio dei metodi")
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_value_by_method.png", dpi=180)
    plt.close()

    plot_df = plot_df.sort_values("regret_mean", ascending=False)
    plt.figure(figsize=(11, max(5, 0.38 * len(plot_df))))
    plt.barh(plot_df["method"], plot_df["regret_mean"])
    plt.xlabel("Regret medio (più basso è meglio)")
    plt.ylabel("Metodo")
    plt.title("Regret medio dei metodi")
    plt.tight_layout()
    plt.savefig(output_dir / "regret_by_method.png", dpi=180)
    plt.close()


def plot_budget_curves(df: pd.DataFrame, output_dir: Path) -> None:
    if "budget" not in df.columns or df["budget"].notna().sum() == 0:
        return

    budget_df = df[
        df["experiment"].astype(str).str.contains("budget", case=False, na=False)
    ].copy()
    if budget_df.empty:
        return

    selected_methods = [
        ("Oracle", "MILP"),
        ("PROMETHEUS", "MILP"),
        ("Risk-first", "MILP"),
        ("Need-first", "MILP"),
        ("Profile-mean", "MILP"),
        ("Random", "Greedy"),
    ]

    plt.figure(figsize=(10, 6))
    plotted = 0
    for policy, optimizer in selected_methods:
        subset = budget_df[
            budget_df["policy"].eq(policy)
            & budget_df["optimizer"].eq(optimizer)
        ]
        if subset.empty:
            continue
        curve = (
            subset.groupby("budget", as_index=False)["normalized_value"]
            .mean()
            .sort_values("budget")
        )
        plt.plot(
            curve["budget"],
            curve["normalized_value"],
            marker="o",
            label=f"{policy} | {optimizer}",
        )
        plotted += 1

    if plotted:
        plt.xlabel("Budget")
        plt.ylabel("Valore normalizzato")
        plt.title("Curva valore-budget")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "budget_curve_normalized_value.png", dpi=180)
    plt.close()


def write_markdown_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    best_non_oracle = summary.loc[~summary["policy"].eq("Oracle")].head(5)
    prometheus = summary[
        summary["policy"].eq("PROMETHEUS")
        & summary["optimizer"].eq("MILP")
    ]

    constraint_like = bool(
        df["experiment"].astype(str).str.contains(
            "budget|capacity|constraint|nursing|physician",
            case=False,
            regex=True,
        ).any()
    )
    lines = [
        "# Report confronto risultati PROMETHEUS",
        "",
        f"- Righe analizzate: **{len(df):,}**",
        f"- Esperimenti: **{df['experiment'].nunique():,}**",
        f"- Scenari distinti: **{df['scenario_key'].nunique():,}**",
        f"- Righe con chiave potenzialmente duplicata: **{df.attrs.get('duplicate_key_rows', 0):,}**",
        "",
    ]
    if constraint_like:
        lines.extend([
            "**Avvertenza:** questi esperimenti sembrano regimi di budget/capacità. "
            "Riutilizzano lo stesso DGP e lo stesso ranking; non devono essere "
            "interpretati come scenari causali indipendenti né usati da soli per "
            "valutare la superiorità del modello.",
            "",
        ])
    lines.extend([
        "## Migliori metodi non-oracle",
        "",
    ])

    if best_non_oracle.empty:
        lines.append("Nessun metodo non-oracle trovato.")
    else:
        lines.append(
            best_non_oracle[
                [
                    "method",
                    "normalized_value_mean",
                    "normalized_value_std",
                    "normalized_value_min",
                    "regret_mean",
                    "win_rate_non_oracle",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )

    lines += ["", "## PROMETHEUS | MILP", ""]
    if prometheus.empty:
        lines.append("La combinazione PROMETHEUS | MILP non è presente.")
    else:
        lines.append(
            prometheus[
                [
                    "n_scenarios",
                    "normalized_value_mean",
                    "normalized_value_std",
                    "normalized_value_min",
                    "regret_mean",
                    "regret_max",
                    "win_rate_non_oracle",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )

    lines += [
        "",
        "## Come leggere le metriche",
        "",
        "- `normalized_value`: valore ottenuto diviso valore oracle; più vicino a 1 è meglio.",
        "- `regret`: differenza tra oracle e metodo; più vicino a 0 è meglio.",
        "- `win_rate_non_oracle`: quota di scenari in cui il metodo è il migliore fra i metodi utilizzabili.",
        "- `normalized_value_min`: prestazione nel peggior scenario osservato.",
        "- Il confronto principale deve essere fatto a parità di scenario e, quando possibile, di ottimizzatore.",
        "",
    ]

    if not paired.empty:
        lines += [
            "## Confronto appaiato rispetto a PROMETHEUS | MILP",
            "",
            paired.head(10)[
                [
                    "method",
                    "paired_scenarios",
                    "mean_delta_normalized",
                    "worst_delta_normalized",
                    "beat_rate",
                ]
            ].to_markdown(index=False, floatfmt=".4f"),
            "",
            "Un delta positivo indica che la baseline supera PROMETHEUS nello stesso scenario; un delta negativo favorisce PROMETHEUS.",
            "",
        ]

    lines += [
        "## File generati",
        "",
        "- `method_summary.csv`: confronto aggregato generale.",
        "- `experiment_summary.csv`: confronto separato per famiglia di esperimenti.",
        "- `paired_vs_prometheus_milp.csv`: confronti appaiati scenario per scenario.",
        "- `scenario_level_results.csv`: dati consolidati.",
        "- PNG: grafici comparativi.",
    ]

    (output_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida e confronta i risultati dei benchmark PROMETHEUS."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV dei risultati oppure cartella contenente CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Cartella in cui salvare tabelle, grafici e report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_csvs(input_path)
    df = add_method_name(load_results(paths))

    summary = aggregate_methods(df)
    experiment_summary = summarize_by_experiment(df)
    paired = paired_comparison(df)

    df.to_csv(output_dir / "scenario_level_results.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    experiment_summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    if not paired.empty:
        paired.to_csv(output_dir / "paired_vs_prometheus_milp.csv", index=False)

    plot_method_summary(summary, output_dir)
    plot_budget_curves(df, output_dir)
    write_markdown_report(df, summary, paired, output_dir)

    metadata = {
        "input": str(input_path),
        "csv_files_considered": len(paths),
        "rows": int(len(df)),
        "experiments": int(df["experiment"].nunique()),
        "scenarios": int(
            df[["experiment", "scenario"]].drop_duplicates().shape[0]
        ),
        "duplicate_key_rows": int(df.attrs.get("duplicate_key_rows", 0)),
    }
    (output_dir / "comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Confronto completato.")
    print(f"Output: {output_dir}")
    print(f"Righe analizzate: {len(df):,}")
    print(f"Metodi: {summary.shape[0]:,}")
    print(f"Report: {output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
