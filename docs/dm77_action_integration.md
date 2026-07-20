# DM 77 patient-action integration

## Operational contract

The integrated PROMETHEUS path keeps need, care state, and treatment as different
objects:

1. `DM77NeedAssessment` describes multidimensional need and routing status.
2. `CurrentCareState` describes services active strictly before the index date.
3. `CareAction` is a versioned, concrete modification of care.
4. `PatientActionOpportunity` is a catalog-admissible `(patient, action)` candidate.
5. `AllocationDecision` separates the causally recommended action from the action
   selected under budget and capacity.

The stable `action_id -> transition_index` mapping is the order of actions in
`configs/dm77/care_action_catalog.yaml`. Python APIs and artifacts expose action IDs;
the numeric index is used only for the transition-conditioned embedding.

## Authoritative flow

`assess_dm77_population` evaluates the transparent research DM 77 rules.
`derive_current_care_states` loads/simulates pre-index active services without reading
the derived need level. `derive_dm77_action_eligibility` then evaluates every catalog
action against both objects, clinical rules, contraindications, prerequisites,
protected routing, and manual review. `build_patient_action_opportunities` consumes
only this authoritative table.

The old `eligible_1_to_2`, ..., `eligible_5_to_6` fields are not consulted in the
integrated path. They remain confined to `opportunity_source: synthetic_legacy`.

## Link to the causal network

For each ranked action, the fully synthetic observational DGP creates an
action-specific binary comparison between that action and no new action, with one
observed common 365-day outcome. In the identifiable baseline the evaluation CATE is
the deterministic function `true_cate = f_a(X)` of observed pre-index covariates.
`latent_individual_effect` is exactly zero. Treatment assignment uses observed
pre-index covariates and current care state, never `true_cate`, potential outcomes,
or another oracle. The `targeted_selection_observed` scenario adds an observed
targeting function that can be correlated with benefit without opening the oracle.
The `strong_observed_confounding` scenario strengthens this observed targeting while
retaining conditional exchangeability; `poor_overlap` is a separate positivity stress.
Only `hidden_confounding` and `combined_stress` introduce a latent `U` affecting
assignment, outcome, and response.

The ground-truth artifact keeps `true_cate`, `latent_individual_effect`, and their sum
`individual_effect` as separate action-specific fields. Primary ranking and policy
metrics use `true_cate`; the latent sum is a secondary sensitivity target only in the
hidden-confounding scenarios. Nuisance models, propensity estimates, outcome models,
overlap diagnostics, and repeated cross-fitted doubly robust (DR) signals are all
action-specific.

The global network receives:

- numeric pre-index patient features;
- one-hot `current_care_state` features;
- a numeric `transition_index` mapped from `action_id`.

It does not receive `dm77_need_level`, latent `U`, or any oracle column. The shared encoder and
action embedding learn a direct ordinal score `s_theta(X_i, a)`. Pairwise supervision
contains both same-action and different-action pairs. Catalog loading blocks global
cross-action training unless all discretionary actions share outcome, horizon,
direction, and unit.

Repeated raw DR signals are preserved for audit. Robust supervision is formed with
split-local winsorization, median aggregation by default, direction-agreement
filtering, and optional inverse-dispersion reliability weights. No test quantity sets
training or validation winsorization bounds. Cross-action pairs exclude opportunities
from the same patient by default, and labels are signs of robust DR differences.

One fixed validation pair set is constructed before training and reused at every
epoch. Checkpoint selection uses either fixed-pair global concordance or held-out DR
policy value, as selected in configuration; training pairs can still be resampled.
The checkpoint diagnostics record the initial metric, selected epoch, within-,
cross-, and global concordance, and held-out DR policy value.

The baseline has contrastive regularization disabled. If enabled, integrated configs
must select the maintained causal-contrastive v2 mode, require stable relationships
across nuisance repetitions, and filter training rows by overlap support. The v2 mode
keeps a separate auxiliary contrastive projection, requests a balanced 50/50 budget
of within- and cross-action pairs, and reliability-weights those pairs from repeated
DR agreement. Both heads share the pre-index clinical encoder and action embedding,
but the raw priority is produced only by the ranking projection and scoring head.

The maintained contrastive profiles are:

- `configs/prometheus/dm77_integrated_contrastive_v2_smoke.yaml` for the end-to-end
  check;
- `configs/prometheus/dm77_integrated_contrastive_v2_default.yaml` for the frozen
  full profile;
- `configs/prometheus/dm77_integrated_contrastive_v2_experiment_suites.yaml` for
  evaluation on new, previously unused final seeds.

The contrastive primary method is recorded as
`prometheus_global_causal_contrastive_v2`. It uses `pair_scope: balanced_mixed`, a
separate contrastive head, and reliability weights derived only from repeated
training DR signals. Its frozen profile uses `lambda_con: 0.01`. This value and the
final seeds are declared in configuration before final evaluation; oracle quantities
remain evaluation-only. The same run also fits and reports the
`unified_global_ranker` rank-only baseline using the same training and validation
opportunity sets.

## Allocation

Four mappings are fitted only on validation scores and validation DR targets: pooled
isotonic, reliability-weighted pooled isotonic, monotonic binned calibration, and
action-mean shrinkage. Their validation DR errors and rank correlations are compared
without oracle access. Synthetic-oracle error, bias, and selected-benefit bias are
computed only after every observational allocation is fixed. Ranking and concordance
always use the raw ordinal score.

The default cost-aware MILP uses the selected calibrated field. The alternative
`equal_cost_rank_only` mode gives every candidate unit cost and creates the allocation
objective from the raw global rank, so cardinal calibration cannot affect selection.

The allocator enforces a shared budget, per-pool capacities, catalog/state
compatibility, prerequisites, empirical support, mutual exclusions, and the configured
per-patient action limit (one by default). It emits both `recommended_action` and
`allocated_action`. The resulting state is obtained by applying the selected action's
`to_state` to the pre-index state.

Prerequisites are normally required to be active before the index date. When a future
catalog adapter explicitly supplies `joint_prerequisite_actions` and raises the
per-patient limit, the MILP adds implication constraints so the dependent action can
be selected only together with every named complementary action.

Level VI and other protected pathways bypass discretionary ranking. Manual-review
cases receive `allocation_status=manual_review`. Mandatory actions are excluded from
causal ranking and handled before the discretionary decision.

## Reproducible smoke

```powershell
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_smoke.yaml
```

The causal-contrastive v2 smoke is:

```powershell
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_contrastive_v2_smoke.yaml
```

The frozen v2 full single-seed profile is:

```powershell
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_contrastive_v2_default.yaml
```

The run manifest records action counts, true cross-action pair counts, allocations,
protected/manual-review routing, constraint violations, method version, DGP scenario,
checkpoint contract, calibration choice, and explicit oracle non-use declarations.
`baseline_comparison.csv` compares random, risk-only, action-mean, independent-action,
DM77 fixed-rule, historical-propensity, cost-normalized-risk, unified-local,
unified-global, direct pairwise GBDT, pooled/independent DR-GBDT, DR ExtraTrees,
shallow DR policy tree, and evaluation-only oracle priorities. It reports
within-action, cross-action, global, and hard cross-action concordance plus allocation
value and regret. Synthetic evaluation also reports Benefit, Value, Regret, and
OracleTop recovery and NDCG at 5%, 10%, and 20%. `observed_test_rate_metrics.csv`
contains held-out observational AUTOC/QINI RATE estimates with patient bootstrap;
`actionwise_ope_metrics.csv` contains OR/IPW/DR action-wise policy estimates.
Budget curves, MILP-vs-greedy allocator comparisons, subgroup/worst-group diagnostics,
and risk-versus-causal selection discordance are separate artifacts. All reported
numeric results must be read from an actual run.

The primary and stress suites are reproducible with:

```powershell
py src\scripts\run_prometheus_suites.py --suite dm77_primary_multi_seed --suite-config configs\prometheus\dm77_integrated_experiment_suites.yaml --output-root artifacts\dm77_action_suites
py src\scripts\run_prometheus_suites.py --suite dm77_initial_stress_ablation --suite-config configs\prometheus\dm77_integrated_experiment_suites.yaml --output-root artifacts\dm77_action_suites
py src\scripts\run_prometheus_suites.py --suite dm77_seed_stability --suite-config configs\prometheus\dm77_integrated_experiment_suites.yaml --output-root artifacts\dm77_action_suites
py src\scripts\summarize_dm77_action_suites.py --runs-dir artifacts\dm77_action_suites --output-dir artifacts\dm77_action_suite_summary
```

Prespecified signal, pair, current-care, permuted-label, and negative-control suites
are in `configs/prometheus/dm77_method_ablation_suites.yaml`. For example, from
Windows `cmd.exe` use a single line:

```bat
py src\scripts\run_prometheus_suites.py --suite causal_signal_estimator_ablation --suite-config configs\prometheus\dm77_method_ablation_suites.yaml --output-root artifacts\dm77_method_ablations
py src\scripts\run_prometheus_suites.py --suite dgp_negative_controls --suite-config configs\prometheus\dm77_method_ablation_suites.yaml --output-root artifacts\dm77_method_ablations
```

To run every method suite sequentially, resume completed runs, execute tests first,
and build the final summary automatically:

```bat
py src\scripts\run_all_dm77_experiments.py --profile full
```

Use `--profile quick` for all conditions with seed 17. The `publication` profile also
includes the large 20,000-patient primary and contrastive-v2 suites and is intentionally
not the default.

The primary suite uses ten seeds. The stress suite uses five seeds over identifiable
baseline, strong observed confounding, poor overlap, risk-benefit misalignment,
observed targeted selection, hidden confounding, and combined stress. The stability
suite fixes `dataset_seed` and changes ten nuisance/model seeds. Summaries contain
mean, standard deviation, median, 95% normal-approximation intervals, paired
differences, win rates, and seed-pair stability diagnostics per scenario.

## Scope and open limitations

The current action DGP and current-care state are fully synthetic research mechanisms.
The DM 77 thresholds and care rules are configurable operationalizations, not official
executable national thresholds. Actions share a single methodological hospital-free-
days estimand; the protected palliative action deliberately has a different outcome
and never enters cross-action ranking. Real-data use would require versioned action
definitions, index dates, measurement validation, overlap/positivity checks, local
governance, professional review, and external validation.

No clinical-effectiveness, Italian-population, ACG-superiority, or deployment-readiness
claim is made.
