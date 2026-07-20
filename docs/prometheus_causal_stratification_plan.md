# PROMETHEUS causal population-stratification plan

## Status and authority

This is the active implementation and evidence plan for the revised PROMETHEUS
scientific objective. The care-profile implementation is the only public pipeline;
superseded phase runners and user-facing action-level experiment commands are removed
as their required foundations are absorbed into the current path.

## Scientific objective

PROMETHEUS will become an end-to-end causal population-stratification framework that
produces a final DM77-aligned care-profile recommendation while keeping need,
actionability and capacity separate.

It must not classify patients directly into a final care level. The final pathway is:

```text
pre-index patient data
  -> baseline need stratification
  -> clinically admissible care profiles
  -> profile-specific causal supervision
  -> direct global patient-profile causal ranking
  -> supported actionable-profile recommendation
  -> capacity-constrained profile allocation
```

The central invariant is:

```text
baseline need != causal recommendation != operational allocation
```

The required patient-level output contract is:

```text
baseline_need_level
recommended_actionable_level
allocated_care_level
```

Capacity may change `allocated_care_level`; it must never overwrite
`recommended_actionable_level`.

## Frozen methodological principles

- The primary method remains direct pairwise causal ranking. The main pipeline must
  never estimate an individual CATE with a regression model and then sort it.
- The ranking unit is one admissible `(patient_id, care_profile_id)` opportunity.
- The primary treatment is assignment to a versioned target care profile relative to
  maintenance of the pre-index pathway (`no_new_profile`).
- `care_action_id` remains a concrete component of a profile bundle; it is not the
  primary treatment and cannot be treated as a profile alias.
- The raw model output `raw_priority_score` is ordinal. It is not baseline risk, an
  observed outcome, an ITE, a CATE, a calibrated benefit or a final care level.
- Validation-only calibration may map the direct-ranking score to an approximate
  incremental-benefit scale for recommendation and cost-aware allocation. This does
  not convert the primary model into a CATE regression pipeline.
- All learner inputs are measured strictly before the index date.
- Synthetic `true_*`, `oracle_*`, potential outcomes, latent response and oracle rank
  remain physically separate and evaluation-only.
- Eligibility is externally governed and cannot depend on the learned score, future
  outcomes or allocation.
- Risk, baseline need, causal recommendation and allocation remain distinct outputs.
- Every stochastic component receives an explicit seed.
- Metrics and reports are generated only from actual executions.
- Synthetic evidence cannot establish clinical effectiveness, Italian-population
  validity, ACG equivalence/superiority or deployment readiness.

## Target schemas

### Care-profile catalogue

The catalogue must contain, at minimum:

```text
care_profile_id
care_profile_level
care_profile_name
included_care_actions
outcome_name
outcome_horizon
outcome_unit
benefit_direction
resource_cost
capacity_pool
clinical_prerequisites
```

It must be versioned and validate every referenced `care_action_id`. Bundle-level
cost, capacity and prerequisite semantics must be explicit rather than inferred by
summing fields silently. Protected profiles with incompatible outcome semantics must
belong to a separate ranking domain.

### Baseline need

```text
patient_id
baseline_need_level
baseline_need_score
baseline_need_probabilities
baseline_need_explanation
baseline_need_mode
baseline_need_status
baseline_need_model_version
```

Rules mode uses transparent DM77-inspired research thresholds and emits rule
provenance. Supervised mode uses an explicitly named ordinal reference such as
`clinician_assigned_need_level`. In rules mode, any probability vector must be
labelled as deterministic/degenerate rather than learned uncertainty.

### Care history and observed treatment

```text
patient_id
current_care_profile
observed_treatment_profile
index_date
observed_outcome
```

`current_care_profile`, `baseline_need_level` and `observed_treatment_profile` are
different variables. The observational comparison for profile `l` includes eligible
patients assigned to `l` or to `no_new_profile`; it must not use a mixture of other
profiles as the comparator.

### Patient-profile opportunities

```text
patient_id
care_profile_id
care_profile_level
baseline_need_level
current_care_profile
eligibility
eligibility_reasons
protected_pathway
manual_review
urgent_case
mandatory_care
common_outcome_contract
empirical_support
```

Mandatory, protected, urgent, manual-review, unsupported and prerequisite-violating
rows do not enter discretionary automated ranking.

### Recommendation

```text
patient_id
recommended_actionable_level
recommended_profile_id
recommendation_priority_score
calibrated_incremental_benefit
recommendation_support
recommendation_abstained
recommendation_reason
recommendation_policy_version
```

For each patient, the recommendation stage compares admissible profiles using the
validation-calibrated benefit associated with the direct-ranking score. A profile is
recommended only when benefit exceeds the frozen minimum threshold and empirical
support is adequate. Otherwise the versioned policy either retains baseline care or
abstains. A raw ordinal score must never be compared directly with zero.

### Allocation

```text
patient_id
recommended_profile_id
recommended_actionable_level
allocated_profile_id
allocated_care_level
allocation_status
allocation_reason
deferred_recommendation
```

The allocator receives frozen recommendation records. It applies eligibility, shared
budget, profile capacities, capacity pools, costs, prerequisites, protected routing,
mutual exclusions and at most one primary profile per patient. Non-allocation leaves
the recommendation unchanged.

## Outcome and comparability contract

The primary synthetic outcome remains days alive outside acute hospital care over
365 days, with larger values preferred. Cross-profile pairs are permitted only inside
one declared ranking domain whose profiles share:

```text
outcome_name
outcome_horizon
outcome_unit
benefit_direction
```

If any field differs, the pair generator must block the pair and the runner must
report separate ranking domains. The protected palliative pathway cannot be forced
onto the primary hospital-free-days scale.

## Evidence partitions

The revised contract requires a new versioned seed registry before any profile-model
selection. It must separate:

- consumed implementation/debug seeds;
- negative-control diagnostic seeds;
- non-oracle discovery seeds;
- sealed confirmation seeds;
- fixed-dataset stability seeds.

The existing action-level seed registry stays historical. Its still-sealed seeds are
not automatically reassigned to the profile protocol. Final synthetic oracle tables
are opened only after model, calibration, recommendation policy and allocator are all
frozen.

## Implementation phases

### Execution and file-organization rule

The numbered phases below are planning checkpoints, not separate programs. There is
one pipeline command:

```text
causal-ranking run --config configs/pipeline.yaml
```

Every completed component is integrated into that runner. New phases must extend the
same configuration schema and pipeline; they must not create `run_phaseN.py`,
`phaseN_smoke.yaml`, or a separate phase report. Experiment variants are expressed as
configurations of the whole pipeline. Code may be split into package modules only by
stable responsibility (data, supervision, ranking, recommendation, allocation),
never merely by development phase.

### Phase 0 — Freeze the revised scientific contract

Deliverables:

- versioned baseline-need definition and allowed feature schema;
- versioned care-profile catalogue and action-bundle mapping;
- profile-versus-maintenance target-trial template;
- common-outcome and ranking-domain contract;
- recommendation threshold, support, fallback and abstention semantics;
- allocation and substitution/deferment semantics;
- oracle boundary, artifact versions and claim contract;
- new seed registry and decision log;
- a single explicit runtime configuration with `decision_unit: care_profile`.

Decisions that must be explicit include whether de-intensification is supported,
whether multiple profiles may map to one DM77 level, how bundle costs aggregate,
whether allocation may substitute an alternative profile, and how unmatched current
care is represented.

Exit gate: all schemas, treatment versions, estimands, ranking domains, thresholds,
fallbacks, seeds and forbidden claims are reviewable without consulting source code.

### Phase 1 — Implement and validate baseline need stratification

Implement one interface with:

```text
need_stratifier.mode = rules
need_stratifier.mode = supervised
```

Rules mode wraps versioned transparent thresholds and reason codes. Supervised mode
uses a seeded ordinal model trained only on allowed pre-index predictors and an
explicit reference label. Observational training must use out-of-fold/cross-fitted
need outputs where in-sample predictions would leak label information.

For synthetic experiments, generate a physically separate true need variable plus a
non-trivial noisy reference-label process. Oracle need scores and latent label rules
must not enter fitting or threshold selection.

Required metrics:

- quadratic weighted Cohen kappa;
- macro F1 and balanced accuracy;
- ordinal MAE and within-one-level accuracy;
- Spearman correlation and confusion matrix;
- prespecified subgroup diagnostics.

Exit gate: rules and supervised outputs satisfy the same schema; leakage tests pass;
need outputs are invariant to capacity, causal scores, recommendation and allocation.

### Phase 2 — Migrate action structures to care profiles

Add distinct profile models, catalogue loader, bundle validation, profile eligibility,
comparability domains and patient-profile opportunities. Keep concrete actions only
as versioned components of profile bundles.

Required compatibility behavior:

- no implicit conversion from `care_action_id` to `care_profile_id`;
- one-action profiles still require explicit profile records;
- superseded configurations and runners are absent from the executable repository;
- new manifests and artifacts declare `decision_unit: care_profile`;
- historical artifact schemas do not change.

Exit gate: missing actions, invalid bundles, incompatible outcomes, eligibility
leakage and duplicate patient-profile keys fail deterministically.

### Phase 3 — Extend the synthetic DGP

Generate, in causal order:

```text
baseline covariates
true baseline need and reference need label
current care profile
eligible profiles
historically assigned profile or no_new_profile
one common observed outcome
profile-specific heterogeneous effects
resource costs and capacities
```

Required scenarios:

- need/current-care alignment;
- unmet need;
- over-intensive historical care;
- risk-benefit alignment and misalignment;
- shared and profile-specific causal response;
- strong observed confounding;
- poor overlap;
- sharp null and placebo outcome;
- hidden confounding as a declared identification failure;
- capacity scarcity and heterogeneous profile costs.

The DGP must not be tailored to make the neural model outperform its comparators.
Truth remains in a separate table until ranking, recommendation and allocation freeze.

Exit gate: structural validation, scenario-direction tests, reproducibility, common
outcome, non-trivial assignment and oracle-isolation tests pass.

### Phase 4 — Rebuild profile-specific causal supervision

For each profile, construct exactly:

```text
profile assignment versus maintenance of pre-index current care
```

Fit propensity and arm-specific outcome models with repeated patient-grouped
cross-fitting. Produce repeated DR pseudo-outcomes on the common outcome scale,
split-local robust aggregation, reliability and overlap diagnostics.

The pseudo-outcome is noisy causal supervision, not an observed ITE or oracle effect.
No `true_*`, potential outcome, latent response or oracle rank can enter nuisance
fitting, robustification, pair construction or data-adequacy thresholds.

Exit gate: every profile has adequate split-specific arm counts and prespecified ESS,
support filters are fixed without oracle data, and held-out outcome mutations cannot
alter upstream nuisance predictions.

### Phase 5 — Adapt the global patient-profile ranker

The ranker input is:

```text
pre-index patient covariates
baseline_need_level
current_care_profile
candidate care_profile_id
```

Adapt the shared encoder and treatment embedding to care profiles. Preserve direct
pairwise ranking as the primary objective and construct within-profile plus valid
cross-profile pairs from repeated DR directions. Keep a fixed validation-pair set.

Required variants:

- `independent_profile_rankers`;
- `global_rank_only`;
- `global_rank_plus_contrastive`;
- `direct_pairwise_gbdt`;
- `risk_based_ranking`;
- `baseline_need_based_ranking`;
- `random`;
- `oracle_evaluation_only`.

Contrastive v2 remains auxiliary, shares the clinical encoder and profile embedding,
and uses a separate projection head unless an explicit ablation states otherwise.
The rank-only model remains a required baseline.

Exit gate: profile-ID, gradient-isolation, within/cross comparability, pair-label,
seed, split and oracle-exclusion tests pass.

### Phase 6 — Implement actionable-profile recommendation

Create a recommendation module independent of the allocator. For each patient it:

1. receives all admissible, supported profile opportunities;
2. preserves their `raw_priority_score` ordering;
3. applies the selected validation-only calibrator;
4. selects the greatest calibrated incremental benefit;
5. checks the frozen benefit/support threshold;
6. emits a profile, baseline fallback or explicit abstention.

Calibration selection may use only validation DR diagnostics. The recommendation
record must be written before test oracle data are opened and before the allocator is
called.

Required metrics include recommended-level distribution, escalation/de-intensification
rate where applicable, abstention rate, support diagnostics, evaluation-only
oracle-profile within-one-level agreement, expected benefit and recommendation regret.

Exit gate: the raw ordinal score is never interpreted as a positive effect; insufficient
support triggers the declared fallback; capacity changes cannot change a frozen
recommendation.

### Phase 7 — Separate recommendation from allocation

Implement a profile allocator over frozen recommendations. Report both cardinal
cost-aware allocation and fixed-capacity ordinal diagnostics without conflating them.
Profile bundles must consume their declared resources once, with no action-component
double counting.

Required outputs include budget-value curves, allocation regret, selected-profile
overlap, capacity utilization, recommendation-to-allocation gap and deferred
recommendations. Every constraint and solver setting must be recorded.

Exit gate: zero eligibility, budget, capacity, prerequisite, protected-pathway,
mutual-exclusion and per-patient violations; reducing capacity may defer allocation
but leaves `recommended_actionable_level` unchanged.

### Phase 8 — Run negative controls and ablations

Run prespecified:

- sharp-null effects;
- placebo outcome;
- seeded permutation of training pair labels;
- no shared causal-response structure;
- within-profile-only versus cross-profile ranking;
- rank-only versus contrastive v2;
- rules versus supervised need stratification;
- calibration and threshold ablations;
- hidden confounding as a declared failure/sensitivity analysis.

Negative controls must be judged using non-oracle held-out diagnostics across multiple
reserved seeds. Failure stops optimization and starts a diagnostic protocol version.

Exit gate: null/placebo behavior, oracle isolation, label permutation, reproducibility
and zero-allocation-violation rules pass before large experiments.

Implementation result: the unified run `20260720T181643Z` executed the five frozen
diagnostic seeds on untouched non-oracle test DR pairs and passed all 8/8 exit gates.
The same campaign reports the within/cross-profile, rank-only/contrastive, need-mode,
calibrator and threshold ablations without using them for model selection. Hidden
confounding is recorded as an identification failure and is ineligible for
optimization. These are bounded synthetic diagnostics, not confirmation evidence.

### Phase 9 — Prespecified synthetic evaluation

Execute in order:

1. bounded CPU smoke runs for software correctness;
2. discovery runs selected only by non-oracle validation records;
3. a locked candidate/configuration declaration;
4. untouched confirmation and fixed-dataset stability runs;
5. oracle unblinding after need model, ranker, calibration, recommendation and
   allocator are frozen.

Final reporting covers four distinct layers:

| Layer | Required metrics |
| --- | --- |
| Baseline need | weighted kappa, macro F1, balanced accuracy, ordinal MAE, within-one-level accuracy, Spearman, confusion matrix, subgroup performance |
| Causal ranking | within/cross/global concordance, AUTOC, QINI, RATE, Benefit@Capacity, policy value, regret, boundary stability, ranking overlap |
| Recommendation | level distribution, escalation/de-intensification, abstention, support, oracle-optimal-profile agreement, expected benefit, recommendation regret |
| Allocation | budget-value curves, allocation regret, selected-profile overlap, capacity utilization, constraint violations, recommendation-allocation gap and deferrals |

Oracle agreement is synthetic evaluation, not real-world clinical accuracy.

Exit gate: every result is regenerated from actual runs; paired uncertainty and seed
provenance are present; the candidate was not selected with confirmation oracle data.

Implementation result: run `20260720T181643Z` executed the frozen 5/10/5
discovery/confirmation/stability design. Discovery selected `global_rank_only` by
mean validation pairwise loss (`0.682957` versus `0.683029`) and wrote the hashed
candidate declaration before any confirmation run. All 15 confirmation/stability
runs wrote a complete decision-stack freeze record before oracle evaluation, and all
9/9 protocol-integrity gates passed. The fixed-dataset analysis nevertheless found
weak ranking stability (mean score Spearman `-0.0650`, top-10% Jaccard `0.0448`).
This is a substantive synthetic limitation: the candidate remains locked, and any
remediation requires a new protocol/version and new seeds rather than post-hoc
reselection.

### Phase 10 — Future clinical and ACG validation

Validate baseline need against an independently adjudicated clinical reference before
making a DM77 validity claim. An authorized ACG comparison, if later available on the
same governed cohort and index date, primarily concerns `baseline_need_level` through
ordinal/risk-group agreement, utilization gradients and subgroup consistency.

`recommended_actionable_level` is a causal policy output and is not equivalent to an
ACG risk category. Real-world causal claims require profile-specific target-trial
protocols, retrospective sensitivity analyses, temporal/geographic validation and
prospective evaluation.

Exit gate: none at repository-only stage; this remains future work requiring external
data, governance and clinical adjudication.

## Cross-cutting acceptance tests

The new profile path must fail if:

- baseline need uses post-index data, outcome, DR signals, oracle effects,
  recommendation or allocation;
- a concrete `care_action_id` is treated as the primary causal treatment;
- a profile references an unknown action or violates composition prerequisites;
- eligibility depends on a learned score, future outcome or allocation;
- oracle data enter training, validation selection, calibration, recommendation,
  allocation or thresholds;
- cross-profile pairs have incompatible outcome semantics;
- a raw ordinal score is interpreted as benefit above zero;
- capacity overwrites `recommended_actionable_level`;
- allocation violates any declared operational constraint;
- any stochastic component lacks an explicit seed.

Each important implementation phase requires focused unit tests, the full regression
suite and a bounded end-to-end smoke before it can be marked complete.

## Claim contract

### Wording allowed after software implementation and synthetic checks

> PROMETHEUS combines a DM77-aligned baseline need stratification with a direct causal
> actionability layer that ranks admissible care profiles and separates the supported
> recommendation from capacity-constrained allocation.

This wording is allowed only after the corresponding modules and tests exist. Until
then, documentation must say that it is the target design.

### Forbidden without external evidence

- PROMETHEUS is clinically validated for Italy;
- PROMETHEUS is equivalent or superior to ACG;
- PROMETHEUS reproduces ACG stratification;
- PROMETHEUS improves outcomes in practice;
- PROMETHEUS is deployment-ready.

## Historical and active status

| Workstream | Status | Interpretation |
| --- | --- | --- |
| New Phase 0 | Complete | Scientific contract, catalogues, target-trial template, decisions, oracle boundary and seed registry are frozen |
| New Phase 1 | Complete | Rules/supervised need interface, noisy reference process, cross-fitting and ordinal evaluation are integrated foundations |
| New Phase 2 | Complete | Profile schemas, catalogue validation, current-profile resolution, eligibility and opportunities are integrated foundations |
| New Phase 3 | Complete | The single pipeline runs exact treatment assignment, common outcomes, frozen scenarios and physical oracle separation; evidence is summarized in `prometheus_implementation_status.md` |
| New Phase 4 | Complete | The single pipeline builds exact profile-versus-maintenance tasks, seeded patient splits, repeated nuisance cross-fitting, robust DR signals and non-oracle arm-count/overlap/ESS support gates |
| New Phase 5 | Complete | The single pipeline trains the direct global patient-profile ranker from frozen supported DR supervision, fixes within/cross validation pairs and writes ordinal scores before oracle materialization |
| New Phase 6 | Complete | The single pipeline selects a finite monotone calibrator from validation DR diagnostics, preserves raw-score ordering, applies the frozen support/benefit policy and writes one recommendation or explicit baseline fallback per patient before oracle evaluation |
| New Phase 7 | Complete | The single pipeline solves a cardinal MILP over frozen recommendations, accounts once for missing profile components, enforces budget/pool/profile/patient constraints and reports separate raw-score fixed-count diagnostics before oracle evaluation |
| New Phase 8 | Complete | The single pipeline runs five consumed diagnostic seeds on untouched non-oracle test pairs, enforces 8 null/placebo/permutation/reproducibility/oracle/allocation/identification/threshold gates and records all prespecified ablations without model selection |
| New Phase 9 | Complete | The single pipeline ran 5 non-oracle discovery, 10 untouched confirmation and 5 fixed-dataset stability runs; 15/15 decisions were frozen before oracle evaluation and 9/9 integrity gates passed, while weak ranker stability remains an explicit limitation |
| New Phase 10 | Not started | No external clinical or authorized ACG performance evidence is claimed |

## Required execution order

```text
freeze revised contract
-> implement baseline need module
-> add profile schemas/catalogue/adapters
-> update synthetic DGP
-> migrate causal supervision
-> adapt global ranker
-> implement recommendation
-> adapt allocation
-> add negative controls and ablations
-> pass all contract tests
-> run bounded smokes
-> run discovery
-> freeze candidate
-> open confirmation
-> regenerate reports from actual runs
```

No large experimental run begins before the relevant contract tests and bounded
end-to-end smoke pass.
