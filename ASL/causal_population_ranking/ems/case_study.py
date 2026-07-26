"""End-to-end real-data-informed semi-synthetic EMS application case."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .allocation import evaluate_frozen_ems_case, freeze_ems_allocation
from .calibration import load_ems_calibration
from .constraint_sensitivity import (
    evaluate_ems_constraint_sensitivity,
    freeze_ems_constraint_sensitivity,
    plot_ems_constraint_sensitivity,
    write_ems_constraint_sensitivity_report,
)
from .policy_benchmark import (
    build_ems_policy_inputs,
    evaluate_ems_policy_benchmark,
    evaluate_ems_policy_score_diagnostics,
    freeze_ems_policy_benchmark,
)
from .ranking import fit_ems_direct_ranker
from .supervision import (
    build_ems_causal_supervision,
    sample_direct_ranking_pairs,
)
from .synthetic import generate_ems_semisynthetic_cohort


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_seeds(value: Any, prefix: str = "") -> dict[str, int]:
    seeds: dict[str, int] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("_seed"):
                seeds[path] = int(child)
            else:
                seeds.update(_collect_seeds(child, path))
    return seeds


def _validate_contract(config: Mapping[str, Any]) -> dict[str, int]:
    expected = {
        "source_data_level": "aggregate_only",
        "causal_validation_on_observed_ems_data": False,
        "priority_score_is_ordinal": True,
        "allocation_utility_is_separate_from_raw_priority_score": True,
        "policy_benchmark_uses_common_constraints": True,
        "resource_cost_units_are_simulated": True,
        "ranking_and_calibration_frozen_across_constraint_regimes": True,
        "individual_cate_estimated_then_sorted": False,
        "potential_outcomes_are_simulated": True,
        "oracle_inputs_allowed_for_training": False,
        "clinical_effectiveness_claims_allowed": False,
        "italian_population_claims_allowed": False,
        "deployment_readiness_claims_allowed": False,
    }
    contract = config.get("scientific_contract", {})
    disagreements = {
        key: (contract.get(key), required)
        for key, required in expected.items()
        if contract.get(key) != required
    }
    if disagreements:
        raise ValueError(f"Invalid EMS scientific contract: {disagreements}")
    seeds = _collect_seeds(config)
    if not seeds or len(seeds) != len(set(seeds.values())):
        raise ValueError("Every stochastic EMS component requires a unique seed")
    return seeds


def _smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["simulation"]["missions"] = 800
    result["causal_supervision"].update({
        "outcome_model_max_iter": 8,
        "propensity_model_max_iter": 8,
    })
    result["ranking"].update({
        "maximum_train_pairs": 600,
        "maximum_validation_pairs": 250,
        "maximum_evaluation_pairs": 1500,
        "epochs": 5,
        "patience": 2,
        "batch_size": 256,
    })
    result["allocation"]["oracle_solver_time_limit_seconds"] = 10
    result["policy_benchmark"]["model_max_iter"] = 8
    # A smoke run validates the complete code path with one nominal resource
    # regime.  The full grid belongs to the non-smoke sensitivity analysis and
    # otherwise dominates the runtime of multi-seed smoke campaigns.
    result["constraint_sensitivity"]["budget_multipliers"] = [1.00]
    result["constraint_sensitivity"]["capacity_multipliers"] = [1.00]
    return result


def _write_report(
    path: Path,
    calibration_summary: Mapping[str, Any],
    nonoracle: pd.DataFrame,
    evaluation: pd.DataFrame,
    methodology: pd.DataFrame,
    policy_benchmark: pd.DataFrame,
    sensitivity_summary: Mapping[str, Any],
    *,
    smoke: bool,
) -> None:
    def metric(method: str, name: str) -> float:
        match = evaluation.loc[
            evaluation.method.eq(method) & evaluation.metric.eq(name),
            "value",
        ]
        return float(match.iloc[0])

    interventions = f"{int(calibration_summary['interventions']):,}".replace(",", ".")
    events = f"{int(calibration_summary['events']):,}".replace(",", ".")
    prometheus_milp = policy_benchmark.loc[
        policy_benchmark.Policy.eq("PROMETHEUS")
        & policy_benchmark.Optimizer.eq("MILP")
    ].iloc[0]
    noncausal_milp = policy_benchmark.loc[
        policy_benchmark.Policy.isin([
            "Risk-first", "Need-first", "Outcome-first",
        ])
        & policy_benchmark.Optimizer.eq("MILP")
    ].sort_values("Normalized value", ascending=False)
    best_noncausal = noncausal_milp.iloc[0]
    normalized_gap = (
        float(prometheus_milp["Normalized value"])
        - float(best_noncausal["Normalized value"])
    )
    true_value_gain = (
        float(prometheus_milp["True value"])
        - float(best_noncausal["True value"])
    )
    if prometheus_milp["Normalized value"] > best_noncausal["Normalized value"]:
        central_result = (
            "A parità di budget, capacità e MILP, PROMETHEUS conserva una "
            "quota maggiore del welfare oracle rispetto alle policy Risk-first, "
            "Need-first e Outcome-first in questo run semi-sintetico. Il margine "
            f"sul miglior comparatore è {normalized_gap:.3f} in valore "
            f"normalizzato e {true_value_gain:.3f} unità di true value simulato."
        )
    else:
        central_result = (
            "In questo run semi-sintetico PROMETHEUS non supera la migliore "
            "policy non causale a parità di budget, capacità e MILP."
        )
    lines = [
        "# Caso applicativo PROMETHEUS EMS/118",
        "",
        "## Inquadramento",
        "",
        (
            "Studio semi-sintetico informato da dati aggregati del sistema EMS/118 "
            "di Benevento. Non costituisce una validazione causale sui dati reali, "
            "una stima di efficacia clinica, né evidenza di prontezza al deployment."
        ),
        "",
        "## Base osservata",
        "",
        (
            f"I workbook riconciliati contengono {interventions} interventi, "
            f"{events} eventi, "
            f"{int(calibration_summary['municipalities'])} comuni e "
            f"{int(calibration_summary['vehicles'])} identificativi di mezzo."
        ),
        "",
        "## Esperimento riproducibile",
        "",
        (
            "Le missioni individuali, l'assegnazione ai tre livelli di risposta, "
            "gli esiti e i potenziali outcome sono simulati. Il modello apprende "
            "direttamente un ordine di priorità da coppie causali doubly robust; "
            "il punteggio è ordinale e non è un effetto di trattamento calibrato."
        ),
        "",
        (
            f"Modalità eseguita: {'smoke test' if smoke else 'protocollo completo'}. "
            f"Accordo pairwise non-oracle sul test: "
            f"{float(nonoracle.value.iloc[0]):.3f}."
        ),
        "",
        "## Diagnostica del ranking ordinale",
        "",
        (
            "Le metriche seguenti usano esclusivamente la verità simulata, aperta "
            "dopo il congelamento di punteggi e allocazione. Descrivono "
            "l'allocatore gerarchico originale e sono secondarie rispetto al "
            "benchmark comune riportato sotto."
        ),
        "",
        (
            f"Concordanza pairwise delle opportunità: "
            f"{metric('direct_causal_ranking', 'opportunity_pairwise_concordance'):.3f}."
        ),
        (
            f"Valore allocativo normalizzato: "
            f"{metric('direct_causal_ranking', 'normalized_allocation_value'):.3f}; "
            f"baseline per severità: "
            f"{metric('severity_only', 'normalized_allocation_value'):.3f}; "
            f"baseline casuale seeded: "
            f"{metric('seeded_random', 'normalized_allocation_value'):.3f}."
        ),
        "",
        "## Confronto metodologico centrale",
        "",
        "| Metodo | Valore usato dall'allocatore | Causale | Personalizzato | Profile-specific |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in methodology.itertuples(index=False):
        lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |"
        )
    lines.extend([
        "",
        (
            "Per PROMETHEUS il valore MILP è una utilità allocativa ottenuta "
            "calibrando il punteggio ordinale su uno split dedicato. Non è il "
            "punteggio grezzo e non è dichiarato come CATE individuale."
        ),
        (
            "Tutte le righe MILP usano lo stesso numero di missioni servite e "
            "medicalizzate e quindi lo stesso costo effettivo; budget e capacità "
            "sono comuni anche alle righe greedy."
        ),
        "",
        "| Policy | Optimizer | True value | Normalized value | Regret | Benefit/cost | Served | Medicalized | Cost used |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in policy_benchmark.itertuples(index=False):
        lines.append(
            f"| {row[0]} | {row[1]} | {float(row[2]):.3f} | "
            f"{float(row[3]):.3f} | {float(row[4]):.3f} | "
            f"{float(row[5]):.3f} | {int(row[6])} | {int(row[7])} | "
            f"{float(row[8]):.1f} |"
        )
    lines.extend([
        "",
        central_result,
        (
            f"Il miglior comparatore non causale MILP è "
            f"{best_noncausal['Policy']} con valore normalizzato "
            f"{float(best_noncausal['Normalized value']):.3f}; "
            f"PROMETHEUS raggiunge "
            f"{float(prometheus_milp['Normalized value']):.3f}."
        ),
        (
            "La differenza è descrittiva per un singolo run con seed fissati: "
            "senza una campagna multi-seed e intervalli di incertezza non "
            "costituisce una dimostrazione stabile di superiorità."
        ),
        "",
        "## Sensibilità ai vincoli operativi",
        "",
        (
            f"Ranking, calibrazione e valori delle policy sono stati mantenuti "
            f"fissi in {int(sensitivity_summary['scenario_count'])} regimi. "
            f"PROMETHEUS-MILP supera Risk-first in "
            f"{int(sensitivity_summary['prometheus_wins_vs_risk_first'])} "
            f"regimi, Need-first in "
            f"{int(sensitivity_summary['prometheus_wins_vs_need_first'])} e "
            f"Outcome-first in "
            f"{int(sensitivity_summary['prometheus_wins_vs_outcome_first'])}."
        ),
        (
            f"Il welfare normalizzato mediano di PROMETHEUS è "
            f"{float(sensitivity_summary['prometheus_median_normalized_value']):.3f}; "
            f"il minimo osservato è "
            f"{float(sensitivity_summary['prometheus_minimum_normalized_value']):.3f}."
        ),
        (
            "Queste quantità sono evaluation-only su verità simulata; mostrano "
            "riuso operativo del ranking, non robustezza clinica esterna."
        ),
        "",
        (
            "Questi risultati descrivono soltanto il benchmark semi-sintetico "
            "generato da questo run e non sono generalizzabili alla popolazione "
            "italiana o all'efficacia operativa del servizio."
        ),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ems_case_study(
    config_path: str | Path = "configs/applications/ems_case_study.yaml",
    output_dir: str | Path | None = None,
    *,
    smoke: bool = False,
    include_constraint_sensitivity: bool = True,
) -> Path:
    """Run the isolated EMS case and emit frozen, auditable artifacts."""

    config_file = Path(config_path).resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if smoke:
        config = _smoke_config(config)
    seeds = _validate_contract(config)
    project_root = next(
        (
            parent for parent in config_file.parents
            if (parent / "pyproject.toml").is_file()
        ),
        config_file.parent,
    )
    data_dir = Path(config["data"]["directory"])
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    calibration = load_ems_calibration(data_dir)
    cohort = generate_ems_semisynthetic_cohort(calibration, config)
    supervision = build_ems_causal_supervision(cohort.learner_data, config)
    ranking = config["ranking"]
    pair_common = {
        "pair_type_fractions": ranking.get("pair_type_fractions", {
            "within_unit": 0.30,
            "within_treatment": 0.35,
            "global_cross_treatment": 0.35,
        }),
        "top_region_fraction": float(ranking.get("top_region_fraction", 0.25)),
        "top_pair_multiplier": float(ranking.get("top_pair_multiplier", 2.0)),
        "gap_clip_quantile": float(ranking.get("gap_clip_quantile", 0.95)),
    }
    train_pairs = sample_direct_ranking_pairs(
        supervision.opportunities,
        supervision.repeat_signal_columns,
        split="rank_train",
        maximum_pairs=int(ranking["maximum_train_pairs"]),
        minimum_signal_difference=float(ranking["minimum_signal_difference"]),
        minimum_repeat_agreement=float(ranking["minimum_repeat_agreement"]),
        seed=int(ranking["pair_seed"]),
        **pair_common,
    )
    validation_pairs = sample_direct_ranking_pairs(
        supervision.opportunities,
        supervision.repeat_signal_columns,
        split="validation",
        maximum_pairs=int(ranking["maximum_validation_pairs"]),
        minimum_signal_difference=float(ranking["minimum_signal_difference"]),
        minimum_repeat_agreement=float(ranking["minimum_repeat_agreement"]),
        seed=int(ranking["validation_pair_seed"]),
        **pair_common,
    )
    ranker = fit_ems_direct_ranker(
        supervision.opportunities,
        train_pairs,
        validation_pairs,
        config,
    )
    test = supervision.opportunities.loc[
        supervision.opportunities.split.eq("test")
    ].reset_index(drop=True)
    priority_scores = ranker.score(test)
    frozen_scores = test[
        ["mission_id", "opportunity_tier", "opportunity_type"]
    ].copy()
    frozen_scores["raw_priority_score"] = priority_scores
    frozen = freeze_ems_allocation(test, priority_scores, config)
    policy_inputs = build_ems_policy_inputs(
        cohort.learner_data,
        supervision,
        ranker,
        config,
    )
    frozen_policy_benchmark = freeze_ems_policy_benchmark(
        policy_inputs,
        config,
    )
    frozen_constraint_sensitivity = (
        freeze_ems_constraint_sensitivity(policy_inputs, config)
        if include_constraint_sensitivity else None
    )

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = project_root / config["reporting"]["output_root"] / stamp
    else:
        output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evaluation_dir = output / "evaluation_only"
    evaluation_dir.mkdir()

    calibration_summary = calibration.summary()
    _json(output / "calibration_summary.json", calibration_summary)
    calibration.severity_counts.rename_axis("severity_code").reset_index().to_csv(
        output / "source_severity_counts.csv", index=False
    )
    calibration.event_counts.rename_axis("event_type").reset_index().to_csv(
        output / "source_event_counts.csv", index=False
    )
    calibration.municipality_pathology.to_csv(
        output / "source_municipality_pathology.csv", index=False
    )
    calibration.vehicle_severity.to_csv(
        output / "source_vehicle_severity.csv", index=False
    )
    calibration.vehicle_assessment.to_csv(
        output / "source_vehicle_assessment.csv", index=False
    )
    _json(output / "dgp_audit.json", cohort.audit)
    _json(output / "supervision_audit.json", supervision.audit)
    _json(output / "ranking_audit.json", ranker.audit)
    policy_inputs.methodology.to_csv(
        output / "policy_methodology.csv",
        index=False,
    )
    policy_inputs.allocator_values.to_csv(
        output / "frozen_policy_allocator_values.csv",
        index=False,
        lineterminator="\n",
    )
    frozen_policy_benchmark.decisions.to_csv(
        output / "frozen_policy_benchmark_decisions.csv",
        index=False,
        lineterminator="\n",
    )
    _json(output / "policy_value_audit.json", policy_inputs.audit)
    policy_values_path = output / "frozen_policy_allocator_values.csv"
    policy_decisions_path = output / "frozen_policy_benchmark_decisions.csv"
    policy_benchmark_freeze = {
        **frozen_policy_benchmark.audit,
        "allocator_values_sha256": _sha256(policy_values_path),
        "decisions_sha256": _sha256(policy_decisions_path),
    }
    _json(output / "policy_benchmark_freeze.json", policy_benchmark_freeze)
    constraint_freeze = None
    if frozen_constraint_sensitivity is not None:
        constraint_scenarios_path = output / "constraint_scenarios.csv"
        constraint_decisions_path = output / "frozen_constraint_decisions.csv"
        constraint_summary_path = output / "frozen_constraint_decision_summary.csv"
        frozen_constraint_sensitivity.scenarios.to_csv(
            constraint_scenarios_path,
            index=False,
        )
        frozen_constraint_sensitivity.selected_decisions.to_csv(
            constraint_decisions_path,
            index=False,
            lineterminator="\n",
        )
        frozen_constraint_sensitivity.decision_summary.to_csv(
            constraint_summary_path,
            index=False,
            lineterminator="\n",
        )
        constraint_freeze = {
            **frozen_constraint_sensitivity.audit,
            "scenario_catalog_sha256": _sha256(constraint_scenarios_path),
            "selected_decisions_sha256": _sha256(constraint_decisions_path),
            "decision_summary_sha256": _sha256(constraint_summary_path),
        }
        _json(output / "constraint_sensitivity_freeze.json", constraint_freeze)
    score_path = output / "frozen_test_priority_scores.csv"
    allocation_path = output / "frozen_test_allocation.csv"
    frozen_scores.to_csv(score_path, index=False, lineterminator="\n")
    frozen.decisions.to_csv(allocation_path, index=False, lineterminator="\n")
    score_freeze = {
        "status": "frozen_before_oracle_evaluation",
        "sha256": _sha256(score_path),
        "rows": int(len(frozen_scores)),
        "score_semantics": "ordinal_not_calibrated_treatment_effect",
        "individual_cate_estimated_then_sorted": False,
    }
    allocation_freeze = {
        **frozen.audit,
        "sha256": _sha256(allocation_path),
    }
    _json(output / "score_freeze.json", score_freeze)
    _json(output / "allocation_freeze.json", allocation_freeze)

    test_pairs = sample_direct_ranking_pairs(
        supervision.opportunities,
        supervision.repeat_signal_columns,
        split="test",
        maximum_pairs=int(ranking["maximum_evaluation_pairs"]),
        minimum_signal_difference=float(ranking["minimum_signal_difference"]),
        minimum_repeat_agreement=float(ranking["minimum_repeat_agreement"]),
        seed=int(ranking["evaluation_pair_seed"]),
        **pair_common,
    )
    heldout_gap = (
        priority_scores[test_pairs.high_index.to_numpy(int)]
        - priority_scores[test_pairs.low_index.to_numpy(int)]
    )
    correctness = (heldout_gap > 0.0).astype(float) + 0.5 * (heldout_gap == 0.0)
    rows = []
    groups = [("all", np.ones(len(test_pairs), dtype=bool))]
    if "pair_type" in test_pairs:
        groups.extend((name, test_pairs.pair_type.eq(name).to_numpy()) for name in sorted(test_pairs.pair_type.unique()))
    for name, mask in groups:
        if not mask.any():
            continue
        weights = test_pairs.loc[mask, "weight"].to_numpy(float)
        rows.append({
            "method": "direct_causal_ranking",
            "metric": "heldout_dr_pairwise_concordance",
            "pair_type": name,
            "value": float(np.average(correctness[mask], weights=weights)),
            "pairs": int(mask.sum()),
            "split": "test",
            "oracle_access_for_policy_construction": False,
            "oracle_access_for_evaluation": False,
        })
    nonoracle = pd.DataFrame(rows)
    nonoracle.to_csv(output / "nonoracle_metrics.csv", index=False)

    evaluation, policy_decisions, oracle_audit = evaluate_frozen_ems_case(
        frozen,
        test,
        priority_scores,
        cohort.evaluation_only,
        config,
    )
    evaluation.to_csv(evaluation_dir / "metrics.csv", index=False)
    policy_decisions.to_csv(evaluation_dir / "policy_allocations.csv", index=False)
    _json(evaluation_dir / "oracle_solver_audit.json", oracle_audit)
    central_benchmark, benchmark_decisions, benchmark_oracle_audit = (
        evaluate_ems_policy_benchmark(
            frozen_policy_benchmark,
            cohort.evaluation_only,
            config,
        )
    )
    central_benchmark.to_csv(
        evaluation_dir / "policy_benchmark.csv",
        index=False,
    )
    benchmark_decisions.to_csv(
        evaluation_dir / "policy_benchmark_decisions.csv",
        index=False,
    )
    _json(
        evaluation_dir / "policy_benchmark_oracle_audit.json",
        benchmark_oracle_audit,
    )
    score_diagnostics, score_detail, score_diagnostics_audit = (
        evaluate_ems_policy_score_diagnostics(
            policy_inputs,
            cohort.evaluation_only,
        )
    )
    score_diagnostics.to_csv(
        evaluation_dir / "policy_score_diagnostics.csv",
        index=False,
    )
    score_detail.to_csv(
        evaluation_dir / "policy_score_evaluation.csv.gz",
        index=False,
        compression="gzip",
    )
    _json(
        evaluation_dir / "policy_score_diagnostics_audit.json",
        score_diagnostics_audit,
    )
    sensitivity_summary = None
    if frozen_constraint_sensitivity is not None:
        (
            constraint_metrics,
            constraint_stability,
            constraint_marginal,
            constraint_all_decisions,
            constraint_audit,
        ) = evaluate_ems_constraint_sensitivity(
            frozen_constraint_sensitivity,
            policy_inputs,
            cohort.evaluation_only,
            config,
        )
        constraint_dir = evaluation_dir / "constraint_sensitivity"
        constraint_dir.mkdir()
        constraint_metrics.to_csv(
            constraint_dir / "metrics.csv",
            index=False,
        )
        constraint_stability.to_csv(
            constraint_dir / "selection_stability.csv",
            index=False,
        )
        constraint_marginal.to_csv(
            constraint_dir / "budget_marginal_value.csv",
            index=False,
        )
        constraint_all_decisions.to_csv(
            constraint_dir / "selected_decisions.csv",
            index=False,
        )
        _json(constraint_dir / "evaluation_audit.json", constraint_audit)
        sensitivity_summary = write_ems_constraint_sensitivity_report(
            constraint_metrics,
            frozen_constraint_sensitivity.scenarios,
            constraint_dir / "constraint_sensitivity_report.md",
        )
        _json(constraint_dir / "summary.json", sensitivity_summary)
        plot_ems_constraint_sensitivity(
            constraint_metrics,
            frozen_constraint_sensitivity.scenarios,
            constraint_dir / "figures",
        )
        _write_report(
            output / "case_study_report.md",
            calibration_summary,
            nonoracle,
            evaluation,
            policy_inputs.methodology,
            central_benchmark,
            sensitivity_summary,
            smoke=smoke,
        )

    written = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "status": "completed",
        "config_version": config["config_version"],
        "mode": "smoke" if smoke else "full",
        "scientific_contract": config["scientific_contract"],
        "all_explicit_seeds": seeds,
        "source_sha256": calibration.source_sha256,
        "freeze_before_oracle": {
            "score_sha256": score_freeze["sha256"],
            "allocation_sha256": allocation_freeze["sha256"],
            "policy_allocator_values_sha256": policy_benchmark_freeze[
                "allocator_values_sha256"
            ],
            "policy_decisions_sha256": policy_benchmark_freeze[
                "decisions_sha256"
            ],
            **({
                "constraint_scenario_catalog_sha256": constraint_freeze[
                    "scenario_catalog_sha256"
                ],
                "constraint_decisions_sha256": constraint_freeze[
                    "selected_decisions_sha256"
                ],
            } if constraint_freeze is not None else {}),
        },
        "constraint_sensitivity_included": bool(include_constraint_sensitivity),
        "mission_level_observed_ems_data_used": False,
        "output_sha256": {
            path.relative_to(output).as_posix(): _sha256(path)
            for path in written
        },
    }
    _json(output / "manifest.json", manifest)
    return output
