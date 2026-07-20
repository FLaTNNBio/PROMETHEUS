# PROMETHEUS global multi-transition implementation handoff

## Outcome

The primary pipeline now implements a single multivalued observational treatment,
transition-specific repeated nuisance cross-fitting, pooled patient-transition
opportunities, balanced within/cross direct causal ranking, pooled monotone
calibration, and exact shared-budget allocation with precedence.

A separate DM 77 research operationalization now produces six need levels or an
abstention, reason codes, multidimensional domain scores, a manual-review queue, and a
protected level-VI palliative pathway. It does not change the causal ranker's target or
use the derived need level as a model feature. See `docs/dm77_implementation.md`.

The default source is now a seeded fully synthetic 24-month longitudinal population;
it consumes no real-person records and does not require the prepared Synthea cache.
Structural validation is a hard gate, while broad plausibility and population-shift
checks remain explicitly distinct from external validity. See
`docs/synthetic_population_protocol.md`.

The preserved old formulation is explicitly a legacy local baseline. The main CLI and
`scripts/run_prometheus.py` invoke the global runner by default.

## File inventory

### Created

- `docs/global_multitransition_audit.md`
- `docs/global_multitransition_handoff.md`
- `docs/dm77_implementation.md`
- `docs/synthetic_population_protocol.md`
- `src/causal_population_ranking/dm77/`
- `src/causal_population_ranking/synthetic/`
- `configs/dm77/default.yaml`
- `configs/dm77/intervention_catalog.yaml`
- `configs/dm77/regional_overrides.example.yaml`
- `src/causal_population_ranking/causal_utilization/global_simulation.py`
- `src/causal_population_ranking/causal_utilization/global_support.py`
- `src/causal_population_ranking/causal_utilization/global_prometheus_runner.py`
- `src/causal_population_ranking/ranking/opportunities.py`
- `src/causal_population_ranking/ranking/global_pairs.py`
- `src/causal_population_ranking/ranking/global_ranker.py`
- `src/causal_population_ranking/ranking/calibration.py`
- `src/causal_population_ranking/allocation/__init__.py`
- `src/causal_population_ranking/allocation/global_allocator.py`
- `src/causal_population_ranking/evaluation/global_metrics.py`
- `configs/prometheus/global_default.yaml`
- `configs/prometheus/global_smoke.yaml`
- `configs/prometheus/global_experiment_suites.yaml`
- `scripts/plot_prometheus_global_experiments.py`
- `tests/test_global_multitransition.py`
- `tests/test_fully_synthetic_population.py`
- `tests/test_dm77_stratification.py`

### Modified

- `pyproject.toml` and `src/causal_population_ranking/__init__.py`: global method
  description and version `0.2.0`.
- `README.md`, `docs/prometheus_protocol.md`, `reports/model_card.md`,
  `reports/README.md`: global method documentation.
- `reports/prometheus_protocol_run.md`: clearly labelled as an archived actual legacy
  run rather than current behavior.
- `src/causal_population_ranking/cli.py` and `scripts/run_prometheus.py`: global runner
  by default and explicit `run-legacy-local`/`--legacy-local` access.
- `scripts/run_prometheus_suites.py`, `scripts/summarize_prometheus_suites.py`: global
  ablation execution and global metric summaries.
- `src/causal_population_ranking/ranking/prometheus_ranker.py`: the shared network
  forward pass accepts a scalar or one mixed transition index per row.
- `src/causal_population_ranking/ranking/__init__.py` and
  `src/causal_population_ranking/evaluation/__init__.py`: exports for global APIs.
- `src/causal_population_ranking/causal_utilization/__init__.py`: exposes the global
  simulation and runner while retaining legacy exports.
- `src/causal_population_ranking/nuisance/cross_fitting.py`: deterministic convergent
  linear nuisance models; partitioning and repeated grouped cross-fitting are unchanged.
- `src/causal_population_ranking/causal_utilization/simulation.py`: private legacy
  helper renamed from `_orthogonalized` to `_remove_linear_component`.

### Deprecated from the main path, retained and usable as legacy baselines

- `causal_utilization/simulation.py`
- `causal_utilization/initial_state.py`
- `causal_utilization/eligibility.py`
- `causal_utilization/stratifier.py`
- `causal_utilization/capacity_thresholds.py`
- `causal_utilization/policy_benchmarks.py`
- `causal_utilization/prometheus_runner.py`
- the original local mode YAML files under `configs/prometheus/`

### Intentionally unchanged

- `ranking/losses.py`, `ranking/causal_signals.py`, and `ranking/pair_sampler.py` remain
  the tested legacy/local primitives; pooled logic lives in `global_pairs.py`.
- `evaluation/metrics.py` remains the legacy/local metric implementation; global
  metrics live in `evaluation/global_metrics.py`.
- `data/*`, `reproducibility.py`, `causal_utilization/helpers.py`, and
  `causal_utilization/support.py` retain their prior responsibilities.
- All pre-existing test files remain unchanged and pass as legacy regressions.
- `scripts/build_causal_dataset.py`, `scripts/generate_synthea.py`, and the legacy
  plotting script remain unchanged.
- The external `synthea/` checkout was not modified.

## Commands

From an editable installation:

```powershell
py -m pytest
py scripts\run_prometheus.py --config configs\prometheus\global_smoke.yaml
py scripts\run_prometheus_suites.py --suite global_pair_ablation --suite-config configs\prometheus\global_experiment_suites.yaml
```

From a plain checkout without installation:

```powershell
$env:PYTHONPATH="src"
py -m pytest
py scripts\run_prometheus.py --config configs\prometheus\global_smoke.yaml
py scripts\run_prometheus_suites.py --suite global_pair_ablation --suite-config configs\prometheus\global_experiment_suites.yaml
```

Suite post-processing:

```powershell
py scripts\summarize_prometheus_suites.py --runs-dir artifacts\prometheus_global_suites --output-dir artifacts\prometheus_global_analysis
py scripts\plot_prometheus_global_experiments.py --summary artifacts\prometheus_global_analysis\global_mean_se.csv
```

## Verification performed

- Full test suite: **74 passed**.
- Final end-to-end smoke run:
  `artifacts/prometheus_global_smoke/20260718_115130_prometheus_global_prometheus_global_causal_contrastive_baseline_population_baseline_s7`.
- Combined population-shift smoke run:
  `artifacts/prometheus_global_smoke/20260718_115144_prometheus_global_prometheus_global_causal_contrastive_baseline_population_combined_shift_s7`.
- The smoke sampler used 150 valid within-transition and 150 valid cross-transition
  ranking pairs. Every cross block had `k != l`; block counts were balanced.
- The exact HiGHS MILP terminated optimally with zero budget, eligibility, support,
  precedence, and transition-capacity violations.
- Two fixed-seed smoke runs produced identical learner, training-opportunity,
  validation, operational opportunity, allocation, calibration, metric, and policy
  files in the reproducibility check.
- No tests failed. No required acceptance component is knowingly left unimplemented.

The smoke metrics are real diagnostic results from a tiny one-epoch run, not claimed
performance: cross-transition concordance `0.4843`, global allocation value `352.14`
days, and global regret `264.76` days. The source contains 800 fully synthetic profiles
and 19,200 patient-month rows. All 33 structural/plausibility checks passed with zero
warnings. The DM 77 research rules represented all six levels (60/258/380/39/61/2),
and both level-VI profiles had every standard-eligibility flag disabled. There were
zero budget, eligibility, support, and precedence violations, and no oracle header was
present in source or operational outputs.

The combined population shift also passed all 33 source checks without warnings,
represented all six DM 77 levels (24/184/374/107/106/5), and completed with zero
budget, eligibility, support, and precedence violations. Its metrics remain diagnostic,
not evidence of external validity or performance.

## Concrete examples

Synthetic stable ranking pair examples:

```text
within: u=(p001, 1_to_2), Gamma_u=8.5 days
        v=(p017, 1_to_2), Gamma_v=3.0 days -> q_uv=+1

cross:  u=(p001, 2_to_3), Gamma_u=6.0 days
        v=(p044, 4_to_5), Gamma_v=1.0 day  -> q_uv=+1
```

Actual pooled smoke calibrator example:

```text
raw score -0.27847 -> calibrated benefit 1.89638 days
out_of_bounds = clip; monotonicity_check = true
```

Actual exact-allocation example:

```text
patient 0d9d9f81-b2ab-7b08-6614-300d15c6f992
selected 1_to_2: calibrated 21.0230, cost 1.0
selected 2_to_3: calibrated 59.0459, cost 1.5
final package = 3
```

The second increment is selected only with its prerequisite, illustrating the
precedence constraint.

## Oracle isolation audit

The operational learner, opportunity, calibration, support, and allocation APIs have
no oracle argument. The runner writes `learner_multivalued_dataset.csv` and
`test_global_opportunities.csv` before opening the separate ground-truth frame for
evaluation. The operational table header contains no `true_*`, `oracle_*`,
`potential_*`, or latent-rank column. The manifest records:

```json
{
  "oracle_used_for_training": false,
  "oracle_used_for_pair_construction": false,
  "oracle_used_for_calibration": false,
  "oracle_used_for_allocation": false,
  "oracle_table_join_stage": "evaluation_after_learned_allocation_fixed"
}
```

Oracle effects are used only to compute global concordance, learned allocation value,
the same-constraint oracle allocation, regret, and fraction of oracle benefit.
