# PROMETHEUS

PROMETHEUS is a direct global causal-ranking method over patient-action
opportunities. The integrated default path operationalizes the DM 77 research rule
engine without treating a DM 77 level as a treatment or as a model output.

```text
pre-index patient data
  -> DM77 multidimensional need assessment
  -> pre-index current care state
  -> authoritative catalog eligibility
  -> action-specific nuisance models and cross-fitted DR signals
  -> split-local robust DR aggregation
  -> direct global patient-action ranking
  -> fixed-set validation checkpoint and validation-only calibration
  -> shared-budget and capacity-constrained allocation
```

The three central concepts remain separate:

| Concept | Field | Role |
| --- | --- | --- |
| Multidimensional need | `dm77_need_level` | Rules-based context and action eligibility |
| Services already active | `current_care_state` | Pre-index network input and transition consistency |
| Causal treatment | `care_action_id` / `action_id` | Concrete care modification ranked by PROMETHEUS |

The network learns `s_theta(X_i, a)` with a shared clinical encoder and a stable
numeric embedding for each `action_id`. Its raw score is an ordinal global priority,
not a calibrated individual treatment effect. It directly compares both within-action
and cross-action pairs when all actions share the same outcome semantics. The derived
DM 77 level is not a ranker feature, treatment, pair target, or learned output.

The optional causal-contrastive v2 regularizer has a projection head separate from
the ranking score, requests balanced within-/cross-action contrastive pairs, and
weights them using agreement across repeated cross-fitted DR signals. It never uses
synthetic ground truth for training or selection.

## Quick start

```powershell
py -m pip install -e .
py -m pytest
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_smoke.yaml
```

For the causal-contrastive v2 end-to-end smoke:

```powershell
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_contrastive_v2_smoke.yaml
```

The package CLI uses the integrated full configuration by default:

```powershell
causal-ranking run
causal-ranking run --config configs\prometheus\dm77_integrated_smoke.yaml
```

Multi-seed comparisons, fixed-dataset stability, and future adjudicated-panel DM77
evaluation use the scripts under `src/scripts/`; see
`docs/prometheus_validation_roadmap.md` for the exact evidence and claim boundaries.

Without an editable installation, set the source path first:

```powershell
$env:PYTHONPATH="src"
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_smoke.yaml
```

The smoke configuration runs entirely on CPU with a seeded, fully synthetic
longitudinal population. The larger methodological configuration is
`configs/prometheus/dm77_integrated_default.yaml`. The computable care catalog is
`configs/dm77/care_action_catalog.yaml`.

## Integrated outputs

Every integrated run writes at least:

- `dm77_need_assessment.csv`;
- `patient_current_care_state.csv`;
- `dm77_action_eligibility.csv`;
- `patient_action_opportunities.csv`;
- `action_specific_dr_signals.csv`;
- `action_specific_dr_diagnostics.csv`;
- `global_ranking.csv`;
- `calibration_diagnostics.csv`;
- `calibration_score_group_diagnostics.csv`;
- `baseline_comparison.csv`;
- `baseline_policy_assignments.csv`;
- `observed_test_rate_metrics.csv` and `actionwise_ope_metrics.csv`;
- `allocator_ablation.csv` and `allocator_ablation_diagnostics.json`;
- `budget_value_curve.csv` and `budget_curve_summary.csv`;
- `subgroup_policy_metrics.csv` and `worst_group_summary.csv`;
- `risk_causal_policy_discordance.csv`;
- `allocation_decisions.csv`;
- `protected_and_manual_review_cases.csv`;
- `run_manifest.json` and the versioned model checkpoint.

The additional non-oracle comparators include a direct pairwise GBDT ranker,
pooled and independent DR-GBDT models, an action-conditioned DR ExtraTrees model,
and a shallow DR policy tree. They are comparators only: the primary PROMETHEUS
pipeline remains direct causal ranking and never estimates an individual CATE and
sorts it. Method ablations and falsification controls are declared in
`configs/prometheus/dm77_method_ablation_suites.yaml`.

All prespecified method suites can be launched with one resumable command:

```powershell
py src\scripts\run_all_dm77_experiments.py --profile full
```

The `quick` profile runs every condition with seed 17; `full` runs the seven
method/negative-control suites over all configured seeds; `publication` additionally
runs the much larger rank-only and contrastive-v2 primary/stress/stability suites.
Completed plan entries are skipped by default, and the final summary is generated
automatically. Progress is recorded in
`artifacts/dm77_method_ablations/master_status.json`.

Level VI/protected pathways bypass discretionary ranking. Manual-review cases are not
automatically allocated. Mandatory actions bypass causal ranking. The discretionary
allocator applies an explicit selected action to `current_care_state`; it does not
construct a final package as `1 + sum(selected increments)`.

## Preserved numeric-transition experiments

The previous `1_to_2`, ..., `5_to_6` fully synthetic path remains available only when
its configuration explicitly declares `opportunity_source: synthetic_legacy`:

```powershell
py src\scripts\run_prometheus.py --config configs\prometheus\global_smoke.yaml
```

The older local-transition baseline is also preserved:

```powershell
py src\scripts\run_prometheus.py --legacy-local --config configs\prometheus\smoke.yaml
causal-ranking run-legacy-local --config configs\prometheus\smoke.yaml
```

Legacy experiment suites and summaries use the scripts under `src/scripts/` and keep
their original method version and numeric-transition semantics.

## Scope

All integrated results are methodological results from actual seeded runs. Synthetic
potential outcomes, `true_cate`, latent effects, and individual effects are physically
separated and joined only for post-allocation evaluation. They never enter nuisance fitting, pair construction,
ranker training, calibration, or observational allocation.

The DM 77 thresholds and care-action rules in this repository are transparent research
operationalizations, not official executable national rules. The project does not
claim clinical effectiveness, validity for the Italian population, superiority over
ACG, or deployment readiness. See
`docs/dm77_action_integration.md`, `docs/dm77_implementation.md`, and
`docs/synthetic_population_protocol.md` for the contracts and limitations. The
evidence hierarchy, target-trial requirements, external-validation modes, and
prospective path are specified in `docs/prometheus_validation_roadmap.md`.
