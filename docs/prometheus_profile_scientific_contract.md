# PROMETHEUS patient-profile scientific contract v1

## Status

This document freezes the scientific contract. The synthetic implementation now
executes through validation-only calibration, recommendation, capacity-constrained
allocation, prespecified non-oracle negative controls and the locked synthetic
discovery/confirmation protocol. External clinical validation remains unimplemented.

The machine-readable authority is the `scientific_contract` section of
`configs/pipeline.yaml`. The baseline-need contract is embedded in the same file;
care profiles and component actions share `configs/care_catalog.yaml`.

## Central separation

PROMETHEUS must expose three independent patient-level concepts:

| Output | Question | Capacity may change it? |
| --- | --- | ---: |
| `baseline_need_level` | What is the patient's DM77-aligned descriptive level of need? | No |
| `recommended_actionable_level` | Which supported admissible target profile has the greatest validation-calibrated incremental benefit? | No |
| `allocated_care_level` | Which recommended profile is actually activated under operational constraints? | Yes |

Example:

```text
baseline_need_level          = III
recommended_actionable_level = IV
allocated_care_level         = III
```

This means that a level-IV profile remains recommended but was deferred. It must not
be rewritten as a level-III recommendation.

## Decision unit and treatment

The primary decision unit is:

```text
(patient_id, care_profile_id)
```

The treatment is assignment to one versioned target care profile. The comparator is
maintenance of the pre-index care pathway, encoded as `no_new_profile`. A comparison
against a pooled mixture of other profiles is prohibited.

`care_action_id` remains a lower-level component. A profile can contain one or more
ordered component actions. A one-action profile still requires its own
`care_profile_id`; action and profile identifiers are never interchangeable.

## Care-profile catalogue

`configs/care_catalog.yaml` is a synthetic research operationalization,
not an official executable DM77 catalogue. It defines:

- one maintenance reference profile at level I;
- target profiles for structured follow-up and chronic management;
- three distinct level-IV alternatives, because profile and level are not one-to-one;
- a sequential high-intensity home-support bundle at level V;
- a protected palliative profile in a separate outcome/ranking domain.

`resource_cost` is the sum of the declared bundle-component action costs. Patient-
profile opportunities must compute the actual incremental cost and capacity
requirements from only those ordered bundle components not already active in
`current_care_profile`. Existing services cannot be charged twice.

## Baseline need stratification

The need module uses pre-index variables only and supports two versioned modes.

### Rules mode

Rules mode wraps the existing transparent DM77-inspired research thresholds. It emits
the assigned level, dimension/rule explanation and manual-review status. Its
probability vector is deterministic one-hot output and must not be described as model
uncertainty.

### Supervised mode

Supervised mode uses the explicit reference label
`clinician_assigned_need_level` and a cumulative binary ordinal model. Five threshold
models estimate `P(level > k)` for `k=1,...,5`; monotone cumulative probabilities are
converted into six level probabilities. Training and split seeds are explicit.

In profile-DGP training, predicted need features must be out-of-fold when a supervised
model is fitted on the same observational population. Future outcomes, post-index
utilization, treatment, pseudo-outcomes, ranking, recommendation, allocation and
oracle fields are forbidden.

Synthetic `true_baseline_need_level` remains evaluation-only. The learner receives a
seeded noisy panel-proxy label whose construction is not a trivial inversion of the
oracle rule.

## Current care, need and observed treatment

The data model contains separately:

```text
baseline_need_level
current_care_profile
observed_treatment_profile
```

No equality is assumed. Under-served and over-intensive current pathways are explicit
DGP scenarios. The new primary synthetic DGP generates `current_care_profile`
directly; it must not infer an ambiguous profile solely from the old
`remote_multidisciplinary_management` state.

Version 1 creates automatic causal candidates only for same-level alternatives or
higher target-profile levels relative to current care. De-intensification is not an
automated v1 treatment. The over-intensive scenario therefore tests separation and
appropriate abstention, not learned down-titration.

## Eligibility

Eligibility is a governed deterministic function of:

```text
pre-index covariates
baseline_need_level
current_care_profile
care-profile catalogue
external clinical rules
```

It cannot read outcome, future utilization, priority score, calibrated benefit,
recommendation, allocation or oracle data. The ranker cannot alter eligibility.

Mandatory, protected, urgent and manual-review cases bypass discretionary automated
ranking. A profile is also excluded when its missing bundle components cannot be
applied in order, action-level prerequisites fail or empirical support is inadequate.

## Profile-specific target-trial template

Every automatically ranked `care_profile_id` must instantiate the following protocol
before a real-data analysis or synthetic analogue is accepted.

| Element | Frozen requirement |
| --- | --- |
| Eligibility | Profile-specific baseline criteria evaluated at time zero using pre-index data only |
| Treatment strategy | Assignment to one exact, versioned target profile and its ordered missing components |
| Comparator strategy | Maintain the versioned pre-index `current_care_profile` (`no_new_profile`) |
| Time zero | Date on which eligibility and profile assignment are both defined |
| Follow-up | 365 days for the primary standard ranking domain |
| Outcome | Days alive outside acute hospital care; higher is better |
| Causal contrast | Assignment to target profile versus maintenance among patients eligible for that profile |
| Primary conditional estimand | Conditional expected incremental outcome benefit on the common day scale |
| Ranking estimand | Relative ordering of admissible patient-profile opportunities induced by the conditional benefit |
| Nuisance models | Profile-specific propensity and arm-specific outcomes with repeated patient-grouped cross-fitting |
| Support | Declared propensity interval and profile-level ESS threshold |
| Treatment versions | Exact profile/catalogue version and component-action ordering |
| Censoring/switching/adherence | Must be declared in a real protocol; the first synthetic DGP has complete follow-up and assigned-strategy consistency by construction |
| Interference | No cross-patient interference in the first synthetic DGP; this remains an explicit limitation |

Identification requires consistency, positivity, conditional exchangeability and no
interference under the declared treatment versions. `hidden_confounding` deliberately
breaks conditional exchangeability and is a failure/sensitivity scenario.

## Causal supervision and ranking

Each empirically supported profile-specific binary learner produces repeated
cross-fitted DR pseudo-outcomes. The current implementation fixes four
patient-disjoint splits, fits nuisance models only from `nuisance_train`, predicts
downstream splits by fold ensemble, and robustifies signals independently within
each split. Exact other-profile assignments never enter the `no_new_profile`
comparator. Arm counts, the `[0.05, 0.95]` overlap interval, split ESS and the frozen
profile ESS threshold of 30 are evaluated without oracle data; unsupported profiles
receive no fabricated supervision.

These pseudo-outcomes are noisy individual supervision signals, not observed ITEs,
oracle effects or calibrated patient-level CATE estimates. All causal-supervision
artifacts are frozen and checksummed before evaluation-only truth is materialized.

The global ranker consumes pre-index covariates, leakage-safe baseline need, current
care profile and target profile ID. Its primary loss remains direct pairwise causal
ranking. It creates within-profile pairs and cross-profile pairs only inside a common
ranking domain.

The raw output remains:

```text
raw_priority_score
```

It is ordinal and has no causal zero. The rank-only model is mandatory. Siamese
contrastive v2 remains an auxiliary loss with a separate projection head unless a
declared ablation tests sharing.

## Outcome comparability

Cross-profile comparison requires equality of outcome name, horizon, unit, direction
and ranking domain. Standard target profiles use days alive outside acute hospital
care over 365 days. The palliative profile uses a different 90-day goal-concordant-care
outcome, bypasses discretionary ranking and cannot receive a score on the standard
global scale.

## Validation-only calibration

Calibration maps the direct-ranking score to
`calibrated_incremental_benefit` using only validation DR targets. Candidate methods
are pooled isotonic, reliability-weighted pooled isotonic, monotonic binned and
profile-mean shrinkage. The selected method minimizes validation DR MAE among finite
monotone candidates, then uses validation rank correlation and method name as frozen
tie-breakers.

Calibration never replaces or retroactively changes `raw_priority_score`. For patient
recommendation, a score outside the validation calibration range causes abstention;
clipped extrapolation is not treated as evidence of benefit.

The implementation reports the validation 90th percentile of absolute calibration
residuals as a population-level dispersion diagnostic. It is not an individual-effect
interval, has no coverage interpretation and does not alter the frozen recommendation
threshold.

## Actionable-profile recommendation

Recommendation is a separate pre-allocation stage. For each patient it:

1. retains only eligible, supported, same-or-higher-level target profiles;
2. computes validation-calibrated incremental benefit;
3. selects the maximum calibrated benefit;
4. requires benefit strictly greater than 2.0 days;
5. requires patient support, score-range support and profile ESS of at least 30;
6. resolves exact ties by raw ordinal priority and then profile ID;
7. emits the frozen recommendation record before allocation and oracle access.

The 2-day threshold is a synthetic methodological threshold aligned with the initial
pair-resolution scale. It is not a clinically validated minimal important difference.
Thresholds 0, 2 and 5 days may be reported as prespecified sensitivity analyses but
cannot choose a model using final oracle outcomes.

If no candidate passes, the system explicitly abstains:

```text
recommended_actionable_level = baseline_need_level
recommended_profile_id       = null
recommendation_abstained      = true
```

The abstention flag is essential: fallback to baseline need is not evidence that a
profile has positive causal benefit.

## Allocation

The allocator receives frozen recommendation records and may consider only the one
recommended profile for each patient. It cannot silently substitute a cheaper or
lower-ranked profile. When the recommendation cannot be activated, it records a
deferral and leaves the actual pathway at `current_care_profile`:

```text
recommended_profile_id = profile_iv_remote_monitoring
allocated_profile_id   = null
allocated_care_level   = current_care_profile_level
deferred_recommendation = true
```

The MILP uses calibrated benefit as its cardinal objective and enforces shared budget,
profile and component-pool capacity, incremental missing-component cost,
prerequisites, protected routing, mutual exclusions and at most one primary profile
per patient. Tie-breaking uses a recorded explicit seed plus stable patient/profile
keys. Fixed-capacity ordinal analysis remains a separate diagnostic that ignores
calibration and monetary cost.

The bounded synthetic implementation fixes a shared budget of 0.75 resource units per
population member. Profile capacity inherits the available units of the profile's
declared primary component pool while every component pool remains separately
binding. Budget multipliers 0.25/0.50/0.75/1.00/1.25 form the cardinal value curve;
fixed population fractions 0.05/0.10/0.20 form the separate raw-score ordinal
diagnostic. These are methodological defaults, not Italian operational estimates.

## Negative controls and ablations

Phase 8 uses the five consumed diagnostic base seeds `1201`, `1223`, `1249`, `1277`
and `1301`. Every model, pair sampler, label permutation and need-model component gets
an explicit seed deterministically derived from its registered base seed. Ranker
diagnostics are evaluated on 256 stable repeated-DR pairs from the untouched `test`
partition; those rows do not enter fitting, early stopping, calibration or model
selection.

The sharp-null and placebo controls require the median held-out concordance of each
global direct ranker to remain within 0.15 of chance and the patient recommendation
rate to remain at or below 0.10. The permuted-label control requires its median
concordance to remain at or below 0.60 and not exceed the corresponding unpermuted
median. Exact seeded reproduction, oracle isolation, zero allocation violations,
hidden-confounding declaration and monotone threshold sensitivity are separate exit
gates. A failed gate stops the pipeline before large experiments and requires a new
diagnostic-protocol version; thresholds are not relaxed automatically.

Prespecified descriptive ablations compare within-profile-only with cross-profile
training, rank-only with contrastive v2, all four validation calibrators and strict
recommendation thresholds of 0, 2 and 5 days. Rules and supervised need modes are
compared on a separate 5,000-record synthetic diagnostic cohort against only the
allowed noisy panel-reference label; synthetic true need is not opened. Ablation
results cannot select the primary model. Hidden confounding remains a declared
conditional-exchangeability failure and is ineligible for optimization.

## Oracle and information boundary

The following cannot enter need fitting, eligibility, nuisance fitting, pair creation,
training, checkpointing, tuning, calibration, recommendation, allocation or thresholds:

```text
true_*
oracle_*
potential_outcome*
latent_*
```

Final synthetic truth is opened only after the need model, ranker, calibrator,
recommendation policy and allocation policy are frozen and their recommendation and
allocation files have been written. Separate evaluation joins compute need, ranking,
recommendation and allocation metrics.

## Reproducibility and evidence

The `seed_registry` section of `configs/pipeline.yaml` is the authority. Discovery was
non-oracle. The confirmation and fixed-dataset seeds were opened only after the
candidate declaration was written and hashed; every run then froze the complete
decision stack before its oracle evaluation join. Legacy seeds are not silently
recycled.

Every result table and report must come from actual executions. Contract documents
contain no performance claims. Any change to treatment, comparator, outcome,
recommendation threshold, support rule or allocation substitution policy requires a
new contract version and a documented seed-policy decision.

## Runtime and claims

`decision_unit: care_profile` is the only supported runtime. Concrete care actions
remain versioned components of profile bundles but cannot be selected as an
alternative primary treatment contract.

Allowed target-design wording is:

> PROMETHEUS combines a DM77-aligned baseline need stratification with a direct causal
> actionability layer that ranks admissible care profiles and separates recommendation
> from allocation.

This wording describes the implemented synthetic design. Clinical validity for
Italy, ACG equivalence/superiority, improved real-world outcomes, ranker stability
and deployment readiness remain forbidden claims. The Phase-9 fixed-dataset result
shows weak ranking stability and cannot be converted into an effectiveness claim.
