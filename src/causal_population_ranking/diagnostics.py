"""Non-oracle negative controls and ablations for the unified pipeline."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ranking import sample_profile_pairs, train_profile_rankers


ORACLE_PREFIXES = ("true_", "oracle_", "potential_outcome", "latent_")
DIAGNOSTIC_CONDITIONS = (
    "standard",
    "within_profile_training_only",
    "permuted_training_pair_labels",
)
SEED_COMPONENTS = (
    "model_seed",
    "pair_seed",
    "validation_pair_seed",
    "contrastive_seed",
    "gbdt_seed",
    "random_seed",
    "label_permutation_seed",
    "test_pair_seed",
    "need_population_seed",
    "need_reference_seed",
    "need_model_seed",
    "need_split_seed",
    "pipeline_split_seed",
    "nuisance_seed",
)
ROW_COLUMNS = (
    "row_type",
    "family",
    "diagnostic",
    "scenario",
    "base_seed",
    "training_condition",
    "variant",
    "pair_scope",
    "partition",
    "metric",
    "value",
    "comparator_value",
    "threshold",
    "operator",
    "passed",
    "pair_count",
    "training_label_changes",
    "oracle_used",
    "eligible_for_model_selection",
    "notes",
)


@dataclass(frozen=True)
class DiagnosticCampaignResult:
    rows: pd.DataFrame
    audit: dict[str, Any]


def _row(**values: Any) -> dict[str, Any]:
    row = {column: np.nan for column in ROW_COLUMNS}
    row.update(
        {
            "oracle_used": False,
            "eligible_for_model_selection": False,
            "notes": "",
        }
    )
    row.update(values)
    return row


def _oracle_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in map(str, frame.columns)
        if column != "oracle_used" and column.startswith(ORACLE_PREFIXES)
    )


def derive_diagnostic_seeds(base_seed: int) -> dict[str, int]:
    """Derive explicit component seeds from one frozen campaign seed."""

    state = np.random.SeedSequence(int(base_seed)).generate_state(
        len(SEED_COMPONENTS), dtype=np.uint32
    )
    values = [int(value) for value in state]
    if len(set(values)) != len(values):
        raise RuntimeError("Diagnostic component-seed derivation produced a collision")
    return dict(zip(SEED_COMPONENTS, values))


def _canonical_score_hash(scores: pd.DataFrame) -> str:
    columns = ["patient_id", "care_profile_id", "split", "method", "method_score"]
    stable = scores.loc[:, columns].sort_values(columns[:-1], kind="stable")
    return hashlib.sha256(stable.to_csv(index=False).encode("utf-8")).hexdigest()


def _diagnostic_settings(
    ranker_settings: Mapping[str, Any],
    condition: str,
    variants: Sequence[str],
) -> dict[str, Any]:
    if condition not in DIAGNOSTIC_CONDITIONS:
        raise ValueError(f"Unknown diagnostic training condition: {condition}")
    settings = copy.deepcopy(dict(ranker_settings))
    configured = tuple(map(str, variants))
    if not configured or not set(configured).issubset(
        {"global_rank_only", "global_rank_plus_contrastive"}
    ):
        raise ValueError("Diagnostic variants must be global direct rankers")
    settings["variants"] = list(configured)
    settings["primary_variant"] = (
        "global_rank_plus_contrastive"
        if "global_rank_plus_contrastive" in configured
        else "global_rank_only"
    )
    settings["allow_cross_profile_training_pairs"] = (
        condition != "within_profile_training_only"
    )
    return settings


def _heldout_test_frame(supervision: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "care_profile_id",
        "split",
        "causal_supervision_status",
        "dr_pseudo_outcome",
        "dr_reliability_weight",
    }
    missing = sorted(required.difference(supervision.columns))
    if missing:
        raise ValueError(f"Diagnostic supervision is missing: {missing}")
    frame = supervision.loc[
        supervision.split.astype(str).eq("test")
        & supervision.causal_supervision_status.astype(str).eq("supported")
    ].copy()
    return frame.reset_index(drop=True)


def _pair_concordance_rows(
    *,
    scores: pd.DataFrame,
    test: pd.DataFrame,
    pairs,
    scenario: str,
    base_seed: int,
    condition: str,
    training_label_changes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not len(pairs):
        return [
            _row(
                row_type="replicate_metric",
                family="negative_control_or_ablation",
                diagnostic="heldout_pair_concordance",
                scenario=scenario,
                base_seed=int(base_seed),
                training_condition=condition,
                variant="none",
                pair_scope="all",
                partition="test",
                metric="heldout_pair_concordance",
                value=np.nan,
                pair_count=0,
                training_label_changes=int(training_label_changes),
                notes="No stable repeated-DR test pairs; no effect was fabricated.",
            )
        ]
    left_keys = pd.MultiIndex.from_arrays(
        [
            test.iloc[pairs.left].patient_id.astype(str),
            test.iloc[pairs.left].care_profile_id.astype(str),
        ]
    )
    right_keys = pd.MultiIndex.from_arrays(
        [
            test.iloc[pairs.right].patient_id.astype(str),
            test.iloc[pairs.right].care_profile_id.astype(str),
        ]
    )
    for variant in sorted(scores.method.astype(str).unique()):
        method = scores.loc[scores.method.astype(str).eq(variant)].copy()
        method["patient_id"] = method.patient_id.astype(str)
        method["care_profile_id"] = method.care_profile_id.astype(str)
        indexed = method.set_index(["patient_id", "care_profile_id"])["method_score"]
        if indexed.index.duplicated().any():
            raise ValueError("Diagnostic scores must be unique by patient and profile")
        left_score = indexed.reindex(left_keys).to_numpy(float)
        right_score = indexed.reindex(right_keys).to_numpy(float)
        finite = np.isfinite(left_score) & np.isfinite(right_score)
        for scope, scope_mask in (
            ("all", np.ones(len(pairs), dtype=bool)),
            ("within_profile", ~pairs.is_cross_profile),
            ("cross_profile", pairs.is_cross_profile),
        ):
            mask = finite & scope_mask
            if not mask.any():
                continue
            signed = pairs.direction[mask] * (left_score[mask] - right_score[mask])
            concordant = (signed > 0.0).astype(float)
            concordant[signed == 0.0] = 0.5
            weights = pairs.weight[mask]
            value = float(np.average(concordant, weights=weights))
            rows.append(
                _row(
                    row_type="replicate_metric",
                    family="negative_control_or_ablation",
                    diagnostic="heldout_pair_concordance",
                    scenario=scenario,
                    base_seed=int(base_seed),
                    training_condition=condition,
                    variant=variant,
                    pair_scope=scope,
                    partition="test",
                    metric="heldout_pair_concordance",
                    value=value,
                    pair_count=int(mask.sum()),
                    training_label_changes=int(training_label_changes),
                    notes=(
                        "Test DR directions were not used for fitting, early stopping, "
                        "calibration or model selection."
                    ),
                )
            )
    return rows


def run_nonoracle_ranker_diagnostics(
    learner: pd.DataFrame,
    supervision: pd.DataFrame,
    opportunities: pd.DataFrame,
    patient_splits: pd.DataFrame,
    ranker_settings: Mapping[str, Any],
    *,
    scenario: str,
    conditions: Sequence[str],
    base_seeds: Sequence[int],
    campaign_settings: Mapping[str, Any],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    profile_ids: Sequence[str],
) -> DiagnosticCampaignResult:
    """Fit seeded diagnostic rankers and score only untouched non-oracle test pairs."""

    frames = {
        "learner": learner,
        "supervision": supervision,
        "opportunities": opportunities,
        "patient_splits": patient_splits,
    }
    leaked = {name: _oracle_columns(frame) for name, frame in frames.items()}
    if any(leaked.values()):
        raise ValueError(f"Oracle columns cannot enter diagnostic fitting: {leaked}")
    conditions = tuple(map(str, conditions))
    if not conditions or not set(conditions).issubset(DIAGNOSTIC_CONDITIONS):
        raise ValueError("Diagnostic conditions are empty or invalid")
    base_seeds = tuple(map(int, base_seeds))
    if not base_seeds or len(set(base_seeds)) != len(base_seeds):
        raise ValueError("Diagnostic campaign requires unique explicit base seeds")
    variants = tuple(map(str, campaign_settings["ranker_variants"]))
    test = _heldout_test_frame(supervision)
    rows: list[dict[str, Any]] = []
    derived_registry: dict[str, dict[str, int]] = {}
    audits: list[dict[str, Any]] = []
    reproducibility_reference: tuple[
        dict[str, Any], dict[str, Any], str
    ] | None = None

    for base_seed in base_seeds:
        derived = derive_diagnostic_seeds(base_seed)
        derived_registry[str(base_seed)] = derived
        pairs = sample_profile_pairs(
            test,
            pairs=int(campaign_settings["heldout_test_pairs"]),
            seed=int(derived["test_pair_seed"]),
            minimum_signal_gap=float(ranker_settings["minimum_signal_gap"]),
            minimum_direction_agreement=float(
                ranker_settings["minimum_direction_agreement"]
            ),
            within_profile_fraction=float(ranker_settings["within_profile_fraction"]),
            allow_cross_profile=True,
            allow_same_patient_cross_profile=bool(
                ranker_settings["allow_same_patient_cross_profile_pairs"]
            ),
            maximum_pair_weight=float(ranker_settings["maximum_pair_weight"]),
        )
        for condition in conditions:
            condition_variants = (
                ("global_rank_only",)
                if condition != "standard"
                else variants
            )
            settings = _diagnostic_settings(
                ranker_settings, condition, condition_variants
            )
            permutation_seed = (
                int(derived["label_permutation_seed"])
                if condition == "permuted_training_pair_labels"
                else None
            )
            result = train_profile_rankers(
                learner,
                supervision,
                opportunities,
                patient_splits,
                settings,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                profile_ids=profile_ids,
                model_seed=int(derived["model_seed"]),
                pair_seed=int(derived["pair_seed"]),
                validation_pair_seed=int(derived["validation_pair_seed"]),
                contrastive_seed=int(derived["contrastive_seed"]),
                gbdt_seed=int(derived["gbdt_seed"]),
                random_seed=int(derived["random_seed"]),
                training_pair_label_permutation_seed=permutation_seed,
            )
            if result.audit["status"] != "trained":
                raise RuntimeError(
                    f"Diagnostic ranker did not train for {scenario}/{condition}"
                )
            audits.append(result.audit)
            changes = int(result.audit["training_pair_label_changes"])
            rows.extend(
                _pair_concordance_rows(
                    scores=result.scores,
                    test=test,
                    pairs=pairs,
                    scenario=scenario,
                    base_seed=base_seed,
                    condition=condition,
                    training_label_changes=changes,
                )
            )
            if (
                reproducibility_reference is None
                and base_seed == base_seeds[0]
                and condition == "standard"
            ):
                reproducibility_reference = (
                    derived,
                    settings,
                    _canonical_score_hash(result.scores),
                )

    if reproducibility_reference is not None:
        derived, settings, original_hash = reproducibility_reference
        first_rows = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["base_seed"] == base_seeds[0]
            and row["training_condition"] == "standard"
        ]
        repeated = train_profile_rankers(
            learner,
            supervision,
            opportunities,
            patient_splits,
            settings,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            profile_ids=profile_ids,
            model_seed=int(derived["model_seed"]),
            pair_seed=int(derived["pair_seed"]),
            validation_pair_seed=int(derived["validation_pair_seed"]),
            contrastive_seed=int(derived["contrastive_seed"]),
            gbdt_seed=int(derived["gbdt_seed"]),
            random_seed=int(derived["random_seed"]),
        )
        exact = original_hash == _canonical_score_hash(repeated.scores)
        rows.append(
            _row(
                row_type="replicate_check",
                family="reproducibility",
                diagnostic="seeded_exact_reproduction",
                scenario=scenario,
                base_seed=int(base_seeds[0]),
                training_condition="standard",
                variant="all_configured_diagnostic_rankers",
                pair_scope="all",
                partition="all_scored_opportunities",
                metric="exact_score_hash_match",
                value=float(exact),
                operator="equals",
                threshold=1.0,
                passed=bool(exact),
                pair_count=int(len(first_rows)),
                notes="Repeated fit used identical data, settings and derived seeds.",
            )
        )

    oracle_used = any(
        bool(audit.get("oracle_used_for_training", audit.get("oracle_used", False)))
        or bool(audit.get("oracle_used_for_pair_construction", False))
        or bool(audit.get("oracle_used_for_early_stopping", False))
        for audit in audits
    )
    return DiagnosticCampaignResult(
        rows=pd.DataFrame(rows, columns=ROW_COLUMNS),
        audit={
            "scenario": str(scenario),
            "conditions": list(conditions),
            "base_seeds": list(base_seeds),
            "derived_component_seeds": derived_registry,
            "evaluation_partition": "test",
            "test_partition_used_for_fitting_or_early_stopping": False,
            "oracle_columns_detected": leaked,
            "oracle_used": bool(oracle_used),
            "eligible_for_model_selection": False,
            "ranker_fit_count": int(len(audits) + (1 if reproducibility_reference else 0)),
        },
    )


def recommendation_ablation_rows(
    *,
    scenario: str,
    calibration_diagnostics: pd.DataFrame,
    threshold_results: Mapping[float, Any],
) -> pd.DataFrame:
    """Put calibrator candidates and frozen threshold sensitivities in one table."""

    if _oracle_columns(calibration_diagnostics):
        raise ValueError("Oracle columns cannot enter calibration ablations")
    rows: list[dict[str, Any]] = []
    for _, candidate in calibration_diagnostics.iterrows():
        method = str(candidate["method"])
        for metric in (
            "validation_dr_mae",
            "validation_reliability_weighted_dr_mae",
            "validation_rank_correlation",
        ):
            rows.append(
                _row(
                    row_type="ablation_metric",
                    family="calibration_ablation",
                    diagnostic="validation_calibrator_candidates",
                    scenario=scenario,
                    training_condition="frozen_primary_ranker",
                    variant=method,
                    partition="validation",
                    metric=metric,
                    value=float(candidate[metric]),
                    notes=(
                        "Selected by the frozen validation-only rule."
                        if bool(candidate["selected"])
                        else "Prespecified non-selected calibrator candidate."
                    ),
                )
            )
    for threshold, result in sorted(threshold_results.items()):
        audit = result.audit
        rows.append(
            _row(
                row_type="ablation_metric",
                family="threshold_ablation",
                diagnostic="recommendation_threshold_sensitivity",
                scenario=scenario,
                training_condition="frozen_ranker_and_calibrator",
                variant=f"strictly_greater_than_{float(threshold):g}_days",
                partition="all_patients_nonoracle",
                metric="recommendation_rate",
                value=float(audit["patients_recommended"] / max(audit["patients"], 1)),
                comparator_value=float(audit["patients_recommended"]),
                threshold=float(threshold),
                operator="strictly_greater_than",
                notes="Sensitivity only; it cannot select the primary threshold.",
            )
        )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def hidden_confounding_sensitivity_row(
    *, scenario: str, conditional_exchangeability: bool
) -> pd.DataFrame:
    declared_failure = not bool(conditional_exchangeability)
    return pd.DataFrame(
        [
            _row(
                row_type="sensitivity_declaration",
                family="identification_sensitivity",
                diagnostic="hidden_confounding",
                scenario=scenario,
                training_condition="standard",
                variant="not_eligible_for_selection",
                partition="nonoracle_metadata",
                metric="identification_failure_declared",
                value=float(declared_failure),
                passed=declared_failure,
                notes=(
                    "Conditional exchangeability is deliberately violated; ranking "
                    "performance cannot validate identification."
                ),
            )
        ],
        columns=ROW_COLUMNS,
    )


def need_stratification_ablation_rows(
    *,
    rules_assessments: pd.DataFrame,
    supervised_assessments: pd.DataFrame,
    reference_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Compare need modes against the allowed noisy panel reference, never truth."""

    frames = {
        "rules": rules_assessments,
        "supervised": supervised_assessments,
        "reference": reference_labels,
    }
    leaked = {name: _oracle_columns(frame) for name, frame in frames.items()}
    if any(leaked.values()):
        raise ValueError(f"Oracle columns cannot enter need ablation: {leaked}")
    required_assessment = {"patient_id", "baseline_need_level"}
    for name in ("rules", "supervised"):
        missing = sorted(required_assessment.difference(frames[name].columns))
        if missing:
            raise ValueError(f"{name} need assessments are missing: {missing}")
    required_reference = {
        "patient_id",
        "clinician_assigned_need_level",
        "need_reference_source",
    }
    missing = sorted(required_reference.difference(reference_labels.columns))
    if missing:
        raise ValueError(f"Need reference labels are missing: {missing}")
    reference = reference_labels.loc[:, sorted(required_reference)].copy()
    reference["patient_id"] = reference.patient_id.astype(str)
    if reference.need_reference_source.astype(str).str.strip().eq("").any():
        raise ValueError("Need ablation requires complete panel-reference provenance")
    rows: list[dict[str, Any]] = []
    for mode, assessments in (
        ("rules", rules_assessments),
        ("supervised", supervised_assessments),
    ):
        predicted = assessments[["patient_id", "baseline_need_level"]].copy()
        predicted["patient_id"] = predicted.patient_id.astype(str)
        joined = reference.merge(
            predicted, on="patient_id", how="inner", validate="one_to_one"
        )
        if len(joined) != len(reference):
            raise ValueError("Every diagnostic patient requires a need assessment")
        observed = pd.to_numeric(
            joined.clinician_assigned_need_level, errors="raise"
        ).to_numpy(float)
        estimate = pd.to_numeric(
            joined.baseline_need_level, errors="raise"
        ).to_numpy(float)
        metrics = {
            "panel_reference_ordinal_mae": float(np.mean(np.abs(estimate - observed))),
            "panel_reference_exact_accuracy": float(np.mean(estimate == observed)),
            "panel_reference_within_one_level_accuracy": float(
                np.mean(np.abs(estimate - observed) <= 1.0)
            ),
        }
        for metric, value in metrics.items():
            rows.append(
                _row(
                    row_type="ablation_metric",
                    family="need_ablation",
                    diagnostic="rules_vs_supervised_need",
                    scenario="separate_need_diagnostic_cohort",
                    training_condition="out_of_fold_or_deterministic_rules",
                    variant=mode,
                    partition="nonoracle_noisy_panel_reference",
                    metric=metric,
                    value=value,
                    pair_count=int(len(joined)),
                    notes=(
                        "The panel proxy is a learner reference, not synthetic truth or "
                        "clinical validation."
                    ),
                )
            )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def _median_concordance(
    rows: pd.DataFrame,
    *,
    scenario: str,
    condition: str,
    variant: str,
    scope: str = "all",
) -> float:
    selected = rows.loc[
        rows.scenario.astype(str).eq(scenario)
        & rows.training_condition.astype(str).eq(condition)
        & rows.variant.astype(str).eq(variant)
        & rows.pair_scope.astype(str).eq(scope)
        & rows.metric.astype(str).eq("heldout_pair_concordance"),
        "value",
    ]
    values = pd.to_numeric(selected, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def finalize_diagnostic_campaign(
    rows: pd.DataFrame,
    *,
    settings: Mapping[str, Any],
    recommendation_summary: pd.DataFrame,
    allocation_summary: pd.DataFrame,
) -> DiagnosticCampaignResult:
    """Add descriptive contrasts and evaluate the prespecified Phase-8 gates."""

    frame = rows.reindex(columns=ROW_COLUMNS).copy()
    appended: list[dict[str, Any]] = []

    for scenario in ("baseline_identifiable", "no_shared_response"):
        for scope in ("all", "within_profile", "cross_profile"):
            rank_only = _median_concordance(
                frame,
                scenario=scenario,
                condition="standard",
                variant="global_rank_only",
                scope=scope,
            )
            contrastive = _median_concordance(
                frame,
                scenario=scenario,
                condition="standard",
                variant="global_rank_plus_contrastive",
                scope=scope,
            )
            within_only = _median_concordance(
                frame,
                scenario=scenario,
                condition="within_profile_training_only",
                variant="global_rank_only",
                scope=scope,
            )
            if np.isfinite(rank_only) and np.isfinite(contrastive):
                appended.append(
                    _row(
                        row_type="ablation_contrast",
                        family="ranker_ablation",
                        diagnostic="rank_only_vs_contrastive_v2",
                        scenario=scenario,
                        training_condition="standard",
                        variant="contrastive_minus_rank_only",
                        pair_scope=scope,
                        partition="test",
                        metric="median_concordance_delta",
                        value=float(contrastive - rank_only),
                        comparator_value=rank_only,
                        notes="Descriptive ablation; not a final-oracle selection rule.",
                    )
                )
            if np.isfinite(rank_only) and np.isfinite(within_only):
                appended.append(
                    _row(
                        row_type="ablation_contrast",
                        family="ranker_ablation",
                        diagnostic="within_only_vs_cross_profile_training",
                        scenario=scenario,
                        training_condition="standard_minus_within_only",
                        variant="global_rank_only",
                        pair_scope=scope,
                        partition="test",
                        metric="median_concordance_delta",
                        value=float(rank_only - within_only),
                        comparator_value=within_only,
                        notes="Descriptive ablation on identical held-out test pairs.",
                    )
                )

    need_mae = frame.loc[
        frame.family.astype(str).eq("need_ablation")
        & frame.metric.astype(str).eq("panel_reference_ordinal_mae")
    ]
    rules_need = pd.to_numeric(
        need_mae.loc[need_mae.variant.astype(str).eq("rules"), "value"],
        errors="coerce",
    ).dropna()
    supervised_need = pd.to_numeric(
        need_mae.loc[need_mae.variant.astype(str).eq("supervised"), "value"],
        errors="coerce",
    ).dropna()
    if len(rules_need) == 1 and len(supervised_need) == 1:
        appended.append(
            _row(
                row_type="ablation_contrast",
                family="need_ablation",
                diagnostic="rules_vs_supervised_need",
                scenario="separate_need_diagnostic_cohort",
                training_condition="supervised_minus_rules",
                variant="ordinal_need_model",
                partition="nonoracle_noisy_panel_reference",
                metric="panel_reference_ordinal_mae_delta",
                value=float(supervised_need.iloc[0] - rules_need.iloc[0]),
                comparator_value=float(rules_need.iloc[0]),
                notes="Single reserved-seed ablation; not clinical validation.",
            )
        )

    frame = pd.concat(
        [frame, pd.DataFrame(appended, columns=ROW_COLUMNS)], ignore_index=True
    )
    controls = settings["controls"]
    gate_rows: list[dict[str, Any]] = []

    def add_gate(name: str, value: float, threshold: float, operator: str, passed: bool,
                 notes: str = "") -> None:
        gate_rows.append(
            _row(
                row_type="gate",
                family="phase8_exit_gate",
                diagnostic=name,
                scenario="campaign",
                training_condition="prespecified",
                variant="aggregate",
                pair_scope="all",
                partition="test_nonoracle_or_structural",
                metric=name,
                value=float(value),
                threshold=float(threshold),
                operator=operator,
                passed=bool(passed),
                notes=notes,
            )
        )

    for scenario, name in (
        ("sharp_null", "sharp_null_behavior"),
        ("placebo_outcome", "placebo_outcome_behavior"),
    ):
        medians = [
            _median_concordance(
                frame,
                scenario=scenario,
                condition="standard",
                variant=variant,
            )
            for variant in ("global_rank_only", "global_rank_plus_contrastive")
        ]
        finite = [value for value in medians if np.isfinite(value)]
        no_stable_pairs = bool((
            frame.scenario.astype(str).eq(scenario)
            & frame.training_condition.astype(str).eq("standard")
            & frame.variant.astype(str).eq("none")
            & pd.to_numeric(frame.pair_count, errors="coerce").fillna(0).eq(0)
        ).any())
        maximum_distance = (
            max(abs(value - 0.5) for value in finite) if finite else 0.0
        )
        summary = recommendation_summary.loc[
            recommendation_summary.scenario.astype(str).eq(scenario)
        ]
        recommendation_rate = (
            float(summary.iloc[0].patients_recommended / max(summary.iloc[0].patients, 1))
            if len(summary)
            else np.inf
        )
        max_distance = float(controls["null_max_median_concordance_distance_from_chance"])
        max_rate = float(controls["null_max_recommendation_rate"])
        passed = (
            (bool(finite) or no_stable_pairs)
            and maximum_distance <= max_distance
            and recommendation_rate <= max_rate
        )
        add_gate(
            name,
            max(maximum_distance, recommendation_rate),
            max(max_distance, max_rate),
            "componentwise_less_than_or_equal",
            passed,
            notes=(
                f"max_concordance_distance={maximum_distance:.6g}; "
                f"recommendation_rate={recommendation_rate:.6g}; "
                f"limits=({max_distance:.6g},{max_rate:.6g})"
            ),
        )

    unpermuted = _median_concordance(
        frame,
        scenario="baseline_identifiable",
        condition="standard",
        variant="global_rank_only",
    )
    permuted = _median_concordance(
        frame,
        scenario="baseline_identifiable",
        condition="permuted_training_pair_labels",
        variant="global_rank_only",
    )
    minimum_delta = float(controls["minimum_unpermuted_minus_permuted_concordance"])
    maximum_permuted = float(controls["permuted_max_median_concordance"])
    permutation_changes = pd.to_numeric(
        frame.loc[
            frame.training_condition.astype(str).eq(
                "permuted_training_pair_labels"
            ),
            "training_label_changes",
        ],
        errors="coerce",
    ).fillna(0)
    delta = unpermuted - permuted
    permutation_passed = (
        np.isfinite(delta)
        and delta >= minimum_delta
        and permuted <= maximum_permuted
        and bool((permutation_changes > 0).all())
    )
    add_gate(
        "training_pair_label_permutation",
        delta if np.isfinite(delta) else -np.inf,
        minimum_delta,
        "greater_than_or_equal_and_permuted_below_maximum",
        bool(permutation_passed),
        notes=(
            f"unpermuted={unpermuted:.6g}; permuted={permuted:.6g}; "
            f"permuted_max={maximum_permuted:.6g}"
        ),
    )

    reproducible = frame.loc[
        frame.metric.astype(str).eq("exact_score_hash_match"), "passed"
    ]
    reproduction_passed = len(reproducible) > 0 and reproducible.fillna(False).astype(bool).all()
    add_gate(
        "seeded_reproducibility",
        float(reproduction_passed),
        1.0,
        "equals",
        bool(reproduction_passed),
    )

    oracle_passed = not frame.oracle_used.fillna(False).astype(bool).any()
    add_gate(
        "oracle_isolation",
        float(oracle_passed),
        1.0,
        "equals",
        bool(oracle_passed),
    )

    zero_violations = float(
        pd.to_numeric(allocation_summary.constraint_violations, errors="coerce").sum()
    )
    allocation_oracle = allocation_summary.oracle_used.fillna(False).astype(bool).any()
    add_gate(
        "zero_allocation_violations",
        zero_violations,
        0.0,
        "equals",
        zero_violations == 0.0 and not allocation_oracle,
    )

    hidden = frame.loc[
        frame.metric.astype(str).eq("identification_failure_declared")
    ]
    hidden_passed = (
        len(hidden) == 1
        and float(hidden.iloc[0].value) == 1.0
        and not bool(hidden.iloc[0].eligible_for_model_selection)
    )
    add_gate(
        "hidden_confounding_declared_failure",
        float(hidden_passed),
        1.0,
        "equals",
        bool(hidden_passed),
    )

    threshold_rows = frame.loc[
        frame.metric.astype(str).eq("recommendation_rate")
        & frame.family.astype(str).eq("threshold_ablation")
    ].sort_values("threshold", kind="stable")
    rates = pd.to_numeric(threshold_rows.value, errors="coerce").to_numpy(float)
    threshold_passed = (
        len(rates) == len(settings["threshold_sensitivity_days"])
        and np.isfinite(rates).all()
        and bool(np.all(np.diff(rates) <= 1e-12))
    )
    add_gate(
        "threshold_sensitivity_monotone",
        float(threshold_passed),
        1.0,
        "equals",
        bool(threshold_passed),
        notes="Recommendation rate cannot increase as the strict threshold increases.",
    )

    gate_frame = pd.DataFrame(gate_rows, columns=ROW_COLUMNS)
    frame = pd.concat([frame, gate_frame], ignore_index=True)
    all_passed = bool(gate_frame.passed.astype(bool).all())
    return DiagnosticCampaignResult(
        rows=frame,
        audit={
            "status": "passed" if all_passed else "failed",
            "required_gate_count": int(len(gate_frame)),
            "required_gates_passed": int(gate_frame.passed.astype(bool).sum()),
            "failed_gates": gate_frame.loc[
                ~gate_frame.passed.astype(bool), "diagnostic"
            ].astype(str).tolist(),
            "evaluation_partition": "test_nonoracle",
            "oracle_used": False,
            "eligible_for_model_selection": False,
            "failure_policy": "stop_before_large_experiments_and_version_diagnostic_protocol",
        },
    )


__all__ = [
    "DIAGNOSTIC_CONDITIONS",
    "DiagnosticCampaignResult",
    "derive_diagnostic_seeds",
    "finalize_diagnostic_campaign",
    "hidden_confounding_sensitivity_row",
    "need_stratification_ablation_rows",
    "recommendation_ablation_rows",
    "run_nonoracle_ranker_diagnostics",
]
