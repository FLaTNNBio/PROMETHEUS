"""Generate paper tables from a completed, frozen PROMETHEUS artifact."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_INPUTS = (
    "run_manifest.json",
    "profile_candidate_declaration.json",
    "profile_experiment_discovery.csv",
    "profile_experiment_metrics.csv",
    "profile_diagnostic_campaign.csv",
)

LEGACY_PHASE9_REPORTING_BOOTSTRAP_SEED = 4703

MAIN_METRICS = (
    ("Baseline need", "rules", "quadratic_weighted_kappa", True,
     "Weighted Cohen's $\\kappa$", "Synthetic reference after freeze"),
    ("Baseline need", "rules", "ordinal_mae", True,
     "Ordinal MAE", "Synthetic reference after freeze"),
    ("Baseline need", "rules", "within_one_level_accuracy", True,
     "Within-one-level accuracy", "Synthetic reference after freeze"),
    ("Causal ranking", "global_rank_only", "global_pairwise_concordance", False,
     "Held-out DR global concordance", "Non-oracle"),
    ("Causal ranking", "global_rank_only", "global_pairwise_concordance", True,
     "Oracle global concordance", "Synthetic oracle after freeze"),
    ("Causal ranking", "global_rank_only", "within_profile_pairwise_concordance", False,
     "Held-out DR within-profile concordance", "Non-oracle"),
    ("Causal ranking", "global_rank_only", "within_profile_pairwise_concordance", True,
     "Oracle within-profile concordance", "Synthetic oracle after freeze"),
    ("Causal ranking", "global_rank_only", "cross_profile_pairwise_concordance", False,
     "Held-out DR cross-profile concordance", "Non-oracle"),
    ("Causal ranking", "global_rank_only", "cross_profile_pairwise_concordance", True,
     "Oracle cross-profile concordance", "Synthetic oracle after freeze"),
    ("Recommendation", "global_rank_only", "recommendation_rate", False,
     "Recommendation rate", "Non-oracle decision output"),
    ("Recommendation", "global_rank_only", "oracle_profile_exact_agreement", True,
     "Exact oracle-profile agreement", "Synthetic oracle after freeze"),
    ("Recommendation", "global_rank_only",
     "expected_recommended_true_benefit_days", True,
     "Expected recommended benefit (days)", "Synthetic oracle after freeze"),
    ("Recommendation", "global_rank_only", "mean_recommendation_regret_days", True,
     "Recommendation regret (days)", "Synthetic oracle after freeze"),
    ("Allocation", "global_rank_only",
     "conditional_deferral_rate_among_recommended", False,
     "Conditional deferral rate", "Non-oracle decision output"),
    ("Allocation", "global_rank_only", "allocated_true_value_days", True,
     "Allocated true value (aggregate days)", "Synthetic oracle after freeze"),
    ("Allocation", "global_rank_only", "normalized_allocation_value", True,
     "Normalized allocated value", "Synthetic oracle after freeze"),
    ("Allocation", "global_rank_only", "allocation_regret_days", True,
     "Allocation regret (aggregate days)", "Synthetic oracle after freeze"),
)

STABILITY_LABELS = {
    "ranking_score_spearman": "Rank-score Spearman",
    "ranking_top_10pct_jaccard": "Top-10\\% Jaccard",
    "ranking_top_20pct_jaccard": "Top-20\\% Jaccard",
    "recommended_patient_jaccard": "Recommended-patient Jaccard",
    "recommended_profile_exact_agreement": "Exact recommended-profile agreement",
    "allocated_patient_jaccard": "Allocated-patient Jaccard",
}

DIAGNOSTIC_LABELS = {
    "sharp_null_behavior": "Sharp-null behavior",
    "placebo_outcome_behavior": "Placebo-outcome behavior",
    "training_pair_label_permutation": "Permuted pair labels",
    "seeded_reproducibility": "Seeded exact reproducibility",
    "oracle_isolation": "Oracle isolation",
    "zero_allocation_violations": "Zero allocation violations",
    "hidden_confounding_declared_failure": "Hidden confounding declared failure",
    "threshold_sensitivity_monotone": "Monotone threshold sensitivity",
}

FIGURE_STEM_TO_CAPTION = {
    "contrastive_ablation": (
        "Prespecified rank-only versus causal-response-guided contrastive "
        "ablation. The left panel reports non-oracle discovery validation loss "
        "(mean and standard deviation over five seeds). The right panel reports "
        "paired rank-only-minus-contrastive concordance differences with 95\\% "
        "bootstrap intervals over ten untouched confirmation seeds."
    ),
    "stability": (
        "Fixed-dataset stability of the locked rank-only model. Points are mean "
        "pairwise agreement statistics and bars are 95\\% bootstrap intervals "
        "over the ten comparisons induced by five learner seeds."
    ),
    "decision_flow": (
        "Mean non-oracle decision counts across ten synthetic confirmation runs. "
        "Allocated and deferred recommendations partition the actionable "
        "recommendations, up to averaging precision."
    ),
    "budget_value": (
        "Synthetic post-freeze allocation value across prespecified shared-budget "
        "multipliers. Points are mean aggregate synthetic true-value days and the "
        "band is the 95\\% run-level bootstrap interval over ten confirmation seeds."
    ),
}

FIGURE_DESCRIPTIONS = {
    "contrastive_ablation": (
        "Two panels compare rank-only and rank-plus-contrastive models. Their "
        "discovery losses are almost equal, and both paired concordance intervals "
        "cross zero."
    ),
    "stability": (
        "Horizontal interval plot of six rank, recommendation, and allocation "
        "stability measures on a zero-to-one scale."
    ),
    "decision_flow": (
        "Horizontal bars show 5000 patients, about 2580 actionable recommendations, "
        "1642 allocations, and 938 deferred recommendations."
    ),
    "budget_value": (
        "An increasing line with confidence band shows aggregate synthetic value "
        "as the shared-budget multiplier rises from 0.25 to 1.25."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bool_mask(series: pd.Series, expected: bool) -> pd.Series:
    values = series.astype(str).str.strip().str.lower().eq("true")
    return values.eq(bool(expected))


def _number(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Paper result must be finite, received {value!r}")
    return result


def _one(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, expected in filters.items():
        if column not in frame:
            raise ValueError(f"Missing required result column: {column}")
        if isinstance(expected, bool):
            mask &= _bool_mask(frame[column], expected)
        else:
            mask &= frame[column].astype(str).eq(str(expected))
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one result for {filters}, found {len(selected)}"
        )
    return selected.iloc[0]


def _summary_row(
    metrics: pd.DataFrame,
    *,
    layer: str,
    variant: str,
    metric: str,
    uses_oracle: bool,
) -> pd.Series:
    return _one(
        metrics,
        record_type="uncertainty_summary",
        phase="confirmation_summary",
        layer=layer,
        variant=variant,
        metric=metric,
        uses_oracle=uses_oracle,
    )


def _with_normalized_allocation_summary(
    metrics: pd.DataFrame,
    *,
    selected_variant: str,
    bootstrap_samples: int,
    bootstrap_seed: int = LEGACY_PHASE9_REPORTING_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Backfill the frozen run-level allocation ratio for legacy artifacts.

    The canonical Phase-9 artifact predates persistence of this derived summary,
    but contains both run-level inputs. Newer and portable artifacts that already
    contain the summary are returned unchanged.
    """

    existing = metrics.loc[
        metrics["record_type"].astype(str).eq("uncertainty_summary")
        & metrics["phase"].astype(str).eq("confirmation_summary")
        & metrics["layer"].astype(str).eq("allocation")
        & metrics["variant"].astype(str).eq(selected_variant)
        & metrics["metric"].astype(str).eq("normalized_allocation_value")
        & _bool_mask(metrics["uses_oracle"], True)
    ]
    if len(existing) == 1:
        return metrics
    if len(existing) > 1:
        raise ValueError(
            "Expected at most one normalized allocation summary, "
            f"found {len(existing)}"
        )

    source = metrics.loc[
        metrics["record_type"].astype(str).eq("run_metric")
        & metrics["phase"].astype(str).eq("confirmation")
        & metrics["layer"].astype(str).eq("allocation")
        & metrics["variant"].astype(str).eq(selected_variant)
        & metrics["metric"].astype(str).isin(
            ("allocated_true_value_days", "oracle_allocation_true_value_days")
        )
        & _bool_mask(metrics["uses_oracle"], True)
    ].copy()
    if source.empty:
        raise ValueError(
            "Normalized allocation value is absent and its run-level inputs "
            "are unavailable"
        )
    source["value"] = pd.to_numeric(source["value"], errors="raise")
    paired = source.pivot(
        index="run_seed",
        columns="metric",
        values="value",
    )
    expected_columns = {
        "allocated_true_value_days",
        "oracle_allocation_true_value_days",
    }
    if set(paired.columns) != expected_columns or paired.isna().any().any():
        raise ValueError(
            "Normalized allocation value requires one complete numerator and "
            "denominator per confirmation seed"
        )
    denominator = paired["oracle_allocation_true_value_days"].to_numpy(float)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise ValueError("Oracle allocation values must be finite and positive")
    values = (
        paired["allocated_true_value_days"].to_numpy(float) / denominator
    )
    if not np.isfinite(values).all():
        raise ValueError("Normalized allocation values must be finite")

    token = f"allocation|{selected_variant}|normalized_allocation_value|True"
    seed = int.from_bytes(
        hashlib.sha256(f"{bootstrap_seed}|{token}".encode()).digest()[:4],
        "big",
    )
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(
            0,
            len(values),
            size=(int(bootstrap_samples), len(values)),
        )
    ].mean(axis=1)
    row = {
        "record_type": "uncertainty_summary",
        "phase": "confirmation_summary",
        "layer": "allocation",
        "variant": selected_variant,
        "metric": "normalized_allocation_value",
        "value": float(values.mean()),
        "uses_oracle": True,
        "oracle_unblinded_after_freeze": True,
        "n_runs": int(len(values)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "notes": (
            "derived from frozen run-level allocation values with the "
            "registered Phase-9 reporting seed"
        ),
    }
    return pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)


def _estimate(row: pd.Series) -> dict[str, float]:
    return {
        "estimate": _number(row["value"]),
        "ci_lower": _number(row["ci_lower"]),
        "ci_upper": _number(row["ci_upper"]),
        "n_runs": int(_number(row["n_runs"])),
    }


def _format_number(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_interval(row: pd.Series) -> str:
    def field(name: str) -> float:
        if hasattr(row, name):
            return float(getattr(row, name))
        return float(row[name])

    return (
        f"{_format_number(field('estimate'))} "
        f"[{_format_number(field('ci_lower'))}--{_format_number(field('ci_upper'))}]"
    )


def _latex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _write_main_table(frame: pd.DataFrame, path: Path) -> None:
    rows = "\n".join(
        f"{row.layer} & {row.display_metric} & {_format_interval(row)} & "
        f"{row.evidence} \\\\"
        for row in frame.itertuples(index=False)
    )
    path.write_text(
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Prespecified Phase~9 synthetic confirmation results. "
        "Intervals are 95\\% run-level bootstrap intervals over ten untouched "
        "confirmation seeds. Oracle quantities were opened only after the "
        "candidate and decision records were frozen.}\n"
        "\\label{tab:prometheus-main-results}\n"
        "\\begin{tabular}{llll}\n"
        "\\toprule\n"
        "Layer & Metric & Estimate [95\\% CI] & Evidence \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n",
        encoding="utf-8",
    )


def _contrastive_table(
    discovery: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    rank_discovery = _one(
        discovery,
        record_type="discovery_summary",
        candidate_variant="global_rank_only",
    )
    con_discovery = _one(
        discovery,
        record_type="discovery_summary",
        candidate_variant="global_rank_plus_contrastive",
    )
    records = [{
        "metric": "Discovery validation loss",
        "rank_only": _number(rank_discovery["selection_value"]),
        "contrastive": _number(con_discovery["selection_value"]),
        "difference": (
            _number(rank_discovery["selection_value"])
            - _number(con_discovery["selection_value"])
        ),
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "rank_only_sd": _number(rank_discovery["selection_std"]),
        "contrastive_sd": _number(con_discovery["selection_std"]),
        "evidence": "Non-oracle discovery; lower is better",
    }]
    for oracle, display in (
        (False, "Held-out DR global concordance"),
        (True, "Oracle global concordance"),
    ):
        rank = _summary_row(
            metrics,
            layer="causal_ranking",
            variant="global_rank_only",
            metric="global_pairwise_concordance",
            uses_oracle=oracle,
        )
        contrastive = _summary_row(
            metrics,
            layer="causal_ranking",
            variant="global_rank_plus_contrastive",
            metric="global_pairwise_concordance",
            uses_oracle=oracle,
        )
        paired = _one(
            metrics,
            record_type="paired_uncertainty_summary",
            layer="causal_ranking",
            variant="global_rank_only",
            metric="selected_minus_comparator:global_pairwise_concordance",
            paired_comparator="global_rank_plus_contrastive",
            uses_oracle=oracle,
        )
        records.append({
            "metric": display,
            "rank_only": _number(rank["value"]),
            "contrastive": _number(contrastive["value"]),
            "difference": _number(paired["value"]),
            "ci_lower": _number(paired["ci_lower"]),
            "ci_upper": _number(paired["ci_upper"]),
            "rank_only_sd": np.nan,
            "contrastive_sd": np.nan,
            "evidence": (
                "Synthetic oracle after freeze" if oracle else "Non-oracle confirmation"
            ),
        })
    return pd.DataFrame(records)


def _write_contrastive_table(frame: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in frame.itertuples(index=False):
        if np.isfinite(row.rank_only_sd):
            rank = f"{row.rank_only:.6f} $\\pm$ {row.rank_only_sd:.6f}"
            contrastive = (
                f"{row.contrastive:.6f} $\\pm$ {row.contrastive_sd:.6f}"
            )
            difference = f"{row.difference:.6f}"
        else:
            rank = f"{row.rank_only:.6f}"
            contrastive = f"{row.contrastive:.6f}"
            difference = (
                f"{row.difference:.6f} "
                f"[{row.ci_lower:.6f}, {row.ci_upper:.6f}]"
            )
        lines.append(
            f"{row.metric} & {rank} & {contrastive} & {difference} \\\\"
        )
    path.write_text(
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Prespecified contrastive ablation. Differences are rank-only "
        "minus contrastive; confirmation intervals are paired 95\\% bootstrap "
        "intervals over the same ten seeds.}\n"
        "\\label{tab:prometheus-contrastive-ablation}\n"
        "\\begin{tabular}{llll}\n"
        "\\toprule\n"
        "Metric & Rank-only & Rank+contrastive & Difference [95\\% CI] \\\\\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n",
        encoding="utf-8",
    )


def _stability_table(metrics: pd.DataFrame, selected: str) -> pd.DataFrame:
    selected_rows = metrics.loc[
        metrics["record_type"].astype(str).eq("stability_summary")
        & metrics["variant"].astype(str).eq(selected)
    ].copy()
    expected = set(STABILITY_LABELS)
    observed = set(selected_rows["metric"].astype(str))
    if observed != expected:
        raise ValueError(
            f"Unexpected stability metrics: missing={expected-observed}, extra={observed-expected}"
        )
    selected_rows["display_metric"] = selected_rows["metric"].map(STABILITY_LABELS)
    selected_rows["estimate"] = selected_rows["value"].map(_number)
    selected_rows["ci_lower"] = selected_rows["ci_lower"].map(_number)
    selected_rows["ci_upper"] = selected_rows["ci_upper"].map(_number)
    selected_rows["n_pair_comparisons"] = (
        selected_rows["n_runs"].map(_number).astype(int)
    )
    return selected_rows[[
        "display_metric", "estimate", "ci_lower", "ci_upper",
        "n_pair_comparisons",
    ]].reset_index(drop=True)


def _write_stability_table(frame: pd.DataFrame, path: Path) -> None:
    rows = "\n".join(
        f"{row.display_metric} & {_format_interval(row)} & "
        f"{row.n_pair_comparisons} \\\\"
        for row in frame.itertuples(index=False)
    )
    path.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Fixed-dataset learner-seed stability of the locked model.}\n"
        "\\label{tab:prometheus-stability}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Metric & Mean [95\\% CI] & Pair comparisons \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def _diagnostic_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    gates = diagnostics.loc[diagnostics["row_type"].astype(str).eq("gate")].copy()
    observed = set(gates["diagnostic"].astype(str))
    expected = set(DIAGNOSTIC_LABELS)
    if observed != expected:
        raise ValueError(
            f"Unexpected diagnostic gates: missing={expected-observed}, extra={observed-expected}"
        )
    if not _bool_mask(gates["passed"], True).all():
        failed = gates.loc[~_bool_mask(gates["passed"], True), "diagnostic"].tolist()
        raise ValueError(f"Cannot publish a passing-gate table; failed gates: {failed}")
    gates["display_diagnostic"] = gates["diagnostic"].map(DIAGNOSTIC_LABELS)
    gates["estimate"] = gates["value"].map(_number)
    gates["threshold_value"] = gates["threshold"].map(_number)
    gates["passed"] = True
    return gates[[
        "display_diagnostic", "estimate", "threshold_value", "operator", "passed"
    ]].reset_index(drop=True)


def _operator_label(value: str, threshold: float) -> str:
    if value == "componentwise_less_than_or_equal":
        return f"$\\leq {_format_number(threshold)}$"
    if value == "equals":
        return f"$= {_format_number(threshold)}$"
    if value == "greater_than_or_equal_and_permuted_below_maximum":
        return "Prespecified compound gate"
    return _latex_escape(value)


def _write_diagnostic_table(frame: pd.DataFrame, path: Path) -> None:
    rows = "\n".join(
        f"{row.display_diagnostic} & {_format_number(row.estimate)} & "
        f"{_operator_label(row.operator, row.threshold_value)} & Pass \\\\"
        for row in frame.itertuples(index=False)
    )
    path.write_text(
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Prespecified non-oracle diagnostic gates. These diagnostics "
        "were ineligible for final model selection.}\n"
        "\\label{tab:prometheus-negative-controls}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Diagnostic & Observed & Criterion & Status \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n",
        encoding="utf-8",
    )


def _write_protocol_table(manifest: dict[str, Any], path: Path) -> None:
    rows = (
        ("Structural/DGP scenarios", manifest["scenario_count"]),
        ("Patients per scenario", manifest["population_size"]),
        ("Scenario-level patient records",
         manifest["scenario_count"] * manifest["population_size"]),
        ("Non-oracle discovery seeds", len(manifest["phase9b_discovery_run_seeds"])),
        ("Untouched confirmation seeds", len(manifest["phase9b_confirmation_run_seeds"])),
        ("Fixed-dataset learner seeds", len(manifest["phase9b_fixed_dataset_run_seeds"])),
        ("Phase 8 gates passed", f"{manifest['phase8_required_gates_passed']}/{manifest['phase8_required_gates']}"),
        ("Phase 9 gates passed", f"{manifest['phase9b_required_gates_passed']}/{manifest['phase9b_required_gates']}"),
        ("Locked candidate", _latex_escape(manifest["phase9b_selected_candidate"])),
    )
    body = "\n".join(f"{label} & {value} \\\\" for label, value in rows)
    path.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Frozen synthetic evaluation protocol.}\n"
        "\\label{tab:prometheus-protocol}\n"
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Design element & Value \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def _write_results_text(
    main: pd.DataFrame,
    contrastive: pd.DataFrame,
    stability: pd.DataFrame,
    manifest: dict[str, Any],
    path: Path,
) -> None:
    def main_row(metric: str) -> pd.Series:
        return _one(main, metric=metric)

    def contrastive_row(metric: str) -> pd.Series:
        return _one(contrastive, metric=metric)

    heldout = main.loc[
        main.metric.eq("global_pairwise_concordance") & ~main.uses_oracle
    ].iloc[0]
    oracle = main.loc[
        main.metric.eq("global_pairwise_concordance") & main.uses_oracle
    ].iloc[0]
    oracle_cross = main.loc[
        main.metric.eq("cross_profile_pairwise_concordance") & main.uses_oracle
    ].iloc[0]
    recommendation = main_row("recommendation_rate")
    profile_agreement = main_row("oracle_profile_exact_agreement")
    recommendation_regret = main_row("mean_recommendation_regret_days")
    deferral = main_row("conditional_deferral_rate_among_recommended")
    normalized_allocation = main_row("normalized_allocation_value")
    score_stability = _one(stability, display_metric="Rank-score Spearman")
    discovery = contrastive_row("Discovery validation loss")
    paired_oracle = contrastive_row("Oracle global concordance")
    path.write_text(
        "\\paragraph{Synthetic Evaluation Results.}\n"
        f"The pipeline executed structural and diagnostic checks in "
        f"{manifest['scenario_count']} fully synthetic scenarios with "
        f"{manifest['population_size']} patients per scenario. The frozen "
        "Phase~9 campaign used the baseline-identifiable scenario. Five non-oracle "
        "discovery seeds selected the rank-only "
        f"configuration (mean validation loss {discovery['rank_only']:.6f} versus "
        f"{discovery['contrastive']:.6f} for rank+contrastive). After the candidate "
        "and decision stack were frozen, ten untouched confirmation seeds yielded "
        f"a held-out DR global concordance of {_format_interval(heldout)} and a "
        f"synthetic-oracle global concordance of {_format_interval(oracle)}. "
        f"Synthetic-oracle cross-profile concordance was "
        f"{_format_interval(oracle_cross)}. "
        "The paired rank-only-minus-contrastive difference in oracle global "
        f"concordance was {paired_oracle['difference']:.6f} "
        f"[{paired_oracle['ci_lower']:.6f}, {paired_oracle['ci_upper']:.6f}], "
        "so the interval included zero. Fixed-dataset learner-seed analysis gave "
        f"a rank-score Spearman correlation of {_format_interval(score_stability)}. "
        f"The recommendation rate was {_format_interval(recommendation)}, with "
        f"synthetic oracle-profile exact agreement {_format_interval(profile_agreement)}; "
        f"mean recommendation regret was {_format_interval(recommendation_regret)}. "
        f"The conditional allocation deferral rate was {_format_interval(deferral)}, "
        f"and normalized allocation value was "
        f"{_format_interval(normalized_allocation)}. "
        f"All {manifest['phase8_required_gates']} Phase~8 diagnostic gates and all "
        f"{manifest['phase9b_required_gates']} Phase~9 protocol-integrity gates "
        "passed. These findings are methodological evidence on synthetic data only "
        "and do not establish clinical effectiveness, Italian-population validity, "
        "ACG equivalence, or deployment readiness.\n",
        encoding="utf-8",
    )


def _decision_flow_table(metrics: pd.DataFrame, selected: str) -> pd.DataFrame:
    labels = (
        ("patients", "Eligible cohort"),
        ("actionable_recommendations", "Actionable recommendations"),
        ("allocated_recommendations", "Allocated recommendations"),
        ("deferred_recommendations", "Deferred recommendations"),
    )
    records = []
    for metric, display in labels:
        row = _summary_row(
            metrics,
            layer="allocation",
            variant=selected,
            metric=metric,
            uses_oracle=False,
        )
        records.append({
            "metric": metric,
            "display_metric": display,
            **_estimate(row),
            "uses_oracle": False,
        })
    return pd.DataFrame(records)


def _budget_value_table(metrics: pd.DataFrame, selected: str) -> pd.DataFrame:
    multipliers = (
        (0.25, "0p25"),
        (0.50, "0p5"),
        (0.75, "0p75"),
        (1.00, "1p0"),
        (1.25, "1p25"),
    )
    records = []
    for multiplier, suffix in multipliers:
        metric = f"budget_curve_learned_true_value:{suffix}"
        row = _summary_row(
            metrics,
            layer="allocation",
            variant=selected,
            metric=metric,
            uses_oracle=True,
        )
        records.append({
            "budget_multiplier": multiplier,
            "metric": metric,
            **_estimate(row),
            "uses_oracle": True,
        })
    return pd.DataFrame(records)


def _paper_plot_style(plt: Any) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(
        output / f"figure_{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "PROMETHEUS paper report",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        output / f"figure_{stem}.png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "PROMETHEUS paper report"},
    )


def _write_figure_wrapper(output: Path, stem: str, *, wide: bool) -> None:
    environment = "figure*" if wide else "figure"
    width = r"\textwidth" if wide else r"\columnwidth"
    (output / f"figure_{stem}.tex").write_text(
        f"\\begin{{{environment}}}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width={width}]{{figure_{stem}.pdf}}\n"
        f"\\caption{{{FIGURE_STEM_TO_CAPTION[stem]}}}\n"
        f"\\label{{fig:prometheus-{stem.replace('_', '-')}}}\n"
        f"\\Description{{{FIGURE_DESCRIPTIONS[stem]}}}\n"
        f"\\end{{{environment}}}\n",
        encoding="utf-8",
    )


def _plot_contrastive(plt: Any, frame: pd.DataFrame, output: Path) -> None:
    navy = "#1F4E79"
    orange = "#E69F00"
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))

    discovery = _one(frame, metric="Discovery validation loss")
    x = np.arange(2)
    axes[0].errorbar(
        x,
        [discovery["rank_only"], discovery["contrastive"]],
        yerr=[discovery["rank_only_sd"], discovery["contrastive_sd"]],
        fmt="o",
        markersize=6,
        capsize=4,
        color=navy,
        ecolor="#5B6B7A",
        linewidth=1.4,
    )
    axes[0].set_xticks(x, ["Rank-only", "Rank + contrastive"])
    axes[0].set_ylabel("Validation pairwise loss")
    axes[0].set_title("A. Non-oracle discovery (lower is better)", loc="left")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)

    comparison = frame.loc[
        frame.metric.isin((
            "Held-out DR global concordance", "Oracle global concordance"
        ))
    ].copy()
    comparison["label"] = comparison.metric.map({
        "Held-out DR global concordance": "Held-out DR",
        "Oracle global concordance": "Synthetic oracle",
    })
    comparison = comparison.set_index("label").loc[
        ["Held-out DR", "Synthetic oracle"]
    ].reset_index()
    y = np.arange(len(comparison))
    difference = comparison.difference.to_numpy(float)
    lower = difference - comparison.ci_lower.to_numpy(float)
    upper = comparison.ci_upper.to_numpy(float) - difference
    axes[1].errorbar(
        difference,
        y,
        xerr=np.vstack((lower, upper)),
        fmt="o",
        markersize=6,
        capsize=4,
        color=orange,
        ecolor="#8A6D1F",
        linewidth=1.4,
    )
    axes[1].axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y, comparison.label)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Concordance difference\n(rank-only $-$ contrastive)")
    axes[1].set_title("B. Paired confirmation difference", loc="left")
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    fig.suptitle("Contrastive ablation — fully synthetic evaluation", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output, "contrastive_ablation")
    plt.close(fig)


def _plot_stability(plt: Any, frame: pd.DataFrame, output: Path) -> None:
    order = (
        "Rank-score Spearman",
        "Top-10\\% Jaccard",
        "Top-20\\% Jaccard",
        "Recommended-patient Jaccard",
        "Exact recommended-profile agreement",
        "Allocated-patient Jaccard",
    )
    plotted = frame.set_index("display_metric").loc[list(order)].reset_index()
    labels = plotted.display_metric.map({
        "Rank-score Spearman": "Rank-score Spearman",
        "Top-10\\% Jaccard": "Top-10% Jaccard",
        "Top-20\\% Jaccard": "Top-20% Jaccard",
        "Recommended-patient Jaccard": "Recommended patient\nJaccard",
        "Exact recommended-profile agreement": "Exact profile\nagreement",
        "Allocated-patient Jaccard": "Allocated patient\nJaccard",
    })
    y = np.arange(len(plotted))
    estimate = plotted.estimate.to_numpy(float)
    lower = estimate - plotted.ci_lower.to_numpy(float)
    upper = plotted.ci_upper.to_numpy(float) - estimate
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    ax.errorbar(
        estimate,
        y,
        xerr=np.vstack((lower, upper)),
        fmt="o",
        markersize=5.5,
        capsize=3.5,
        color="#1F4E79",
        ecolor="#5B6B7A",
        linewidth=1.3,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Agreement / correlation")
    ax.set_title("Fixed-dataset stability\n(fully synthetic evaluation)", loc="left")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    _save_figure(fig, output, "stability")
    plt.close(fig)


def _plot_decision_flow(plt: Any, frame: pd.DataFrame, output: Path) -> None:
    plotted = frame.iloc[::-1].reset_index(drop=True)
    cohort = float(frame.loc[frame.metric.eq("patients"), "estimate"].iloc[0])
    y = np.arange(len(plotted))
    estimate = plotted.estimate.to_numpy(float)
    colors = ["#999999", "#D55E00", "#009E73", "#1F4E79"]
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    ax.barh(y, estimate, color=colors, alpha=0.92)
    ax.set_yticks(y, plotted.display_metric)
    ax.set_xlim(0.0, cohort * 1.16)
    ax.set_xlabel("Mean patients per confirmation run")
    ax.set_title("Recommendation-to-allocation flow\n(fully synthetic)", loc="left")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    for index, value in enumerate(estimate):
        ax.text(
            value + cohort * 0.03,
            index,
            f"{value:,.0f} ({100.0 * value / cohort:.1f}%)",
            va="center",
            fontsize=7.5,
        )
    fig.tight_layout()
    _save_figure(fig, output, "decision_flow")
    plt.close(fig)


def _plot_budget_value(plt: Any, frame: pd.DataFrame, output: Path) -> None:
    x = frame.budget_multiplier.to_numpy(float)
    mean = frame.estimate.to_numpy(float)
    lower = frame.ci_lower.to_numpy(float)
    upper = frame.ci_upper.to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    ax.fill_between(x, lower, upper, color="#56B4E9", alpha=0.25, linewidth=0)
    ax.plot(x, mean, color="#1F4E79", marker="o", linewidth=1.8, markersize=4.5)
    ax.axvline(1.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xlabel("Shared-budget multiplier")
    ax.set_ylabel("Aggregate synthetic true value (days)")
    ax.set_ylim(0.0, float(upper.max()) * 1.08)
    ax.set_title("Allocation value–budget curve\n(post-freeze synthetic oracle)", loc="left")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    _save_figure(fig, output, "budget_value")
    plt.close(fig)


def _generate_figures(
    output: Path,
    contrastive: pd.DataFrame,
    stability: pd.DataFrame,
    decision_flow: pd.DataFrame,
    budget_value: pd.DataFrame,
) -> None:
    previous_config = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="prometheus-mpl-") as cache_dir:
        os.environ["MPLCONFIGDIR"] = cache_dir
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            _paper_plot_style(plt)
            _plot_contrastive(plt, contrastive, output)
            _plot_stability(plt, stability, output)
            _plot_decision_flow(plt, decision_flow, output)
            _plot_budget_value(plt, budget_value, output)
        except ImportError as error:
            raise RuntimeError(
                "Paper figures require the reporting dependencies; install with "
                "`python -m pip install -e .[report]`."
            ) from error
        finally:
            if previous_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_config
    for stem in FIGURE_STEM_TO_CAPTION:
        _write_figure_wrapper(
            output,
            stem,
            wide=stem == "contrastive_ablation",
        )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    required_truths = {
        "status": "complete",
        "source_population_fully_synthetic": True,
        "phase8_diagnostics_oracle_used": False,
        "phase8_diagnostics_eligible_for_model_selection": False,
        "phase9b_oracle_used_for_candidate_selection": False,
        "phase9b_confirmation_opened_after_candidate_lock": True,
    }
    for key, expected in required_truths.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Artifact is not eligible for paper reporting: {key}="
                f"{manifest.get(key)!r}, expected {expected!r}"
            )
    if manifest.get("phase8_failed_gates") or manifest.get("phase9b_failed_gates"):
        raise ValueError("Artifact has failed protocol gates")
    if manifest.get("phase8_required_gates_passed") != manifest.get(
        "phase8_required_gates"
    ):
        raise ValueError("Not all Phase-8 gates passed")
    if manifest.get("phase9b_required_gates_passed") != manifest.get(
        "phase9b_required_gates"
    ):
        raise ValueError("Not all Phase-9 gates passed")


def generate_paper_report(
    artifact_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Write traceable CSV and LaTeX tables from one completed artifact."""

    source = Path(artifact_dir).resolve()
    output = Path(output_dir).resolve()
    missing = [name for name in REQUIRED_INPUTS if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Paper report inputs are missing: {missing}")

    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        (source / "profile_candidate_declaration.json").read_text(encoding="utf-8")
    )
    _validate_manifest(manifest)
    if candidate.get("status") != "locked_before_confirmation":
        raise ValueError("Candidate declaration was not locked before confirmation")
    if candidate.get("oracle_used_for_selection") is not False:
        raise ValueError("Candidate declaration used oracle information for selection")
    if candidate.get("selected_primary_variant") != manifest.get(
        "phase9b_selected_candidate"
    ):
        raise ValueError("Candidate declaration and run manifest disagree")
    discovery = pd.read_csv(source / "profile_experiment_discovery.csv")
    metrics = pd.read_csv(source / "profile_experiment_metrics.csv")
    diagnostics = pd.read_csv(source / "profile_diagnostic_campaign.csv")
    metrics = _with_normalized_allocation_summary(
        metrics,
        selected_variant=manifest["phase9b_selected_candidate"],
        bootstrap_samples=int(manifest.get("phase9b_bootstrap_samples", 2_000)),
    )

    main_records = []
    for layer, variant, metric, oracle, display, evidence in MAIN_METRICS:
        row = _summary_row(
            metrics,
            layer=layer.lower().replace(" ", "_"),
            variant=variant,
            metric=metric,
            uses_oracle=oracle,
        )
        main_records.append({
            "layer": layer,
            "variant": variant,
            "metric": metric,
            "display_metric": display,
            **_estimate(row),
            "uses_oracle": oracle,
            "evidence": evidence,
        })
    main = pd.DataFrame(main_records)
    contrastive = _contrastive_table(discovery, metrics)
    stability = _stability_table(metrics, manifest["phase9b_selected_candidate"])
    diagnostics_table = _diagnostic_table(diagnostics)
    decision_flow = _decision_flow_table(
        metrics, manifest["phase9b_selected_candidate"]
    )
    budget_value = _budget_value_table(
        metrics, manifest["phase9b_selected_candidate"]
    )

    output.mkdir(parents=True, exist_ok=True)
    main.to_csv(output / "paper_main_results.csv", index=False)
    contrastive.to_csv(output / "paper_contrastive_ablation.csv", index=False)
    stability.to_csv(output / "paper_stability_results.csv", index=False)
    diagnostics_table.to_csv(output / "paper_negative_controls.csv", index=False)
    decision_flow.to_csv(output / "paper_decision_flow.csv", index=False)
    budget_value.to_csv(output / "paper_budget_value.csv", index=False)
    _write_protocol_table(manifest, output / "table_protocol.tex")
    _write_main_table(main, output / "table_main_results.tex")
    _write_contrastive_table(contrastive, output / "table_contrastive_ablation.tex")
    _write_stability_table(stability, output / "table_stability.tex")
    _write_diagnostic_table(diagnostics_table, output / "table_negative_controls.tex")
    _write_results_text(
        main,
        contrastive,
        stability,
        manifest,
        output / "paper_results_text.tex",
    )
    _generate_figures(
        output,
        contrastive,
        stability,
        decision_flow,
        budget_value,
    )

    output_names = sorted(
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name != "paper_results_manifest.json"
    )
    report_manifest = {
        "report_version": "prometheus_paper_report_v2",
        "status": "synthetic_methodological_evidence_only",
        "source_run_id": manifest["run_id"],
        "source_protocol_version": candidate["protocol_version"],
        "candidate_configuration_sha256": candidate[
            "candidate_configuration_sha256"
        ],
        "source_artifact": str(source),
        "source_population_fully_synthetic": True,
        "selected_candidate": manifest["phase9b_selected_candidate"],
        "oracle_used_for_selection": False,
        "clinical_claims_allowed": False,
        "italian_population_claims_allowed": False,
        "acg_comparison_claims_allowed": False,
        "deployment_claims_allowed": False,
        "input_sha256": {
            name: _sha256(source / name) for name in REQUIRED_INPUTS
        },
        "output_sha256": {
            name: _sha256(output / name) for name in output_names
        },
    }
    (output / "paper_results_manifest.json").write_text(
        json.dumps(report_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output
