# PROMETHEUS decision log

## Status

Decision-log version: `prometheus_profile_decisions_v1`  
Frozen: 2026-07-20  
Scope: scientific and compatibility decisions for the current pipeline, exposed
through one execution path.

## Frozen decisions

| ID | Decision | Reason and consequence |
| --- | --- | --- |
| P0-01 | The primary decision unit is `(patient_id, care_profile_id)`. | The project targets final actionable profile stratification rather than isolated action prioritization. |
| P0-02 | `care_action_id` remains a profile component and is never a profile alias. | Concrete intervention provenance is preserved without retaining the old treatment semantics. |
| P0-03 | Multiple care profiles may share one DM77 level. | A level is a broad care stratum; distinct treatment versions remain separate profiles. |
| P0-04 | The comparator is `no_new_profile`, meaning maintenance of the exact pre-index pathway. | Other-profile mixtures would make the causal contrast ambiguous. |
| P0-05 | Rules mode is the initial synthetic default; supervised ordinal need stratification remains supported. | Rules are auditable, while supervised fitting requires its explicit reference-label and cross-fitting contract. |
| P0-06 | Leakage-safe baseline need is an allowed ranker input. | Need and causal actionability remain distinct outputs. |
| P0-07 | Version 1 permits escalation and same-level alternatives, not automated de-intensification. | Down-titration requires separate outcomes, safety rules and treatment versions. |
| P0-08 | Cross-profile ranking requires an identical outcome contract and ranking domain. | Global ordinal scores are meaningful only on a common benefit scale. |
| P0-09 | Recommendation uses maximum validation-calibrated benefit, not the raw score alone. | An ordinal score has no meaningful causal zero. |
| P0-10 | Recommendation requires calibrated benefit strictly greater than 2.0 days plus empirical support. | This is a synthetic methodological threshold, not a clinical MCID. |
| P0-11 | Scores outside the validation calibration range cause abstention. | Clipped extrapolation is insufficient recommendation evidence. |
| P0-12 | Failed benefit or support gates yield explicit abstention, baseline-need fallback and no profile ID. | The flag prevents fallback from being misread as causal evidence. |
| P0-13 | Recommendations are written and frozen before allocation. | Scarcity cannot alter the actionability recommendation. |
| P0-14 | The allocator may allocate only the frozen recommended profile. | A cheaper substitute would be a new causal decision that was not recommended. |
| P0-15 | A deferred patient remains recommended; allocated care stays at the current-care level. | This preserves the recommendation-allocation gap. |
| P0-16 | Bundle resources are charged only for missing components relative to current care. | Existing services cannot consume activation budget or capacity twice. |
| P0-17 | The profile path has dedicated artifacts, manifests and seeds. | Action-level and profile-level evidence cannot be pooled silently. |
| P0-18 | `care_profile` is the only executable decision unit. | Superseded action-level configurations cannot be selected accidentally. |
| P0-19 | Development checkpoints are integrated into one pipeline. | Experiments always execute the complete currently implemented path. |
| P0-20 | Clinical, Italian-population, ACG and deployment claims remain forbidden. | The repository contains a synthetic research design, not external evidence. |
| P0-21 | Phase-8 gates use untouched non-oracle test DR diagnostics and cannot select the primary model. | Passing a diagnostic guardrail is not confirmation evidence or hyperparameter selection. |
| P0-22 | Hidden confounding is a declared identification failure and is never an optimization scenario. | Predictive ranking diagnostics cannot repair violated conditional exchangeability. |
| P0-23 | Phase-9 discovery selects only by mean validation pairwise loss over five frozen seeds. | `global_rank_only` was locked before confirmation; neither test nor oracle results may change it. |
| P0-24 | Weak fixed-dataset ranking stability is reported as a limitation and does not reopen the locked candidate. | Any stabilization work requires a new protocol version and fresh seeds, preventing post-confirmation optimization. |

## Resolved implementation choices

| Choice | Resolution |
| --- | --- |
| Ordinal need estimator | Cumulative binary ordinal logistic model with monotone probability projection. |
| Ambiguous current-care state | Explicit manual review; the system never guesses a profile. |
| Nuisance supervision | Repeated partitioned cross-fitting with robust DR aggregation and non-oracle support gates. |
| Direct ranker | Shared patient encoder and profile embedding with direct pairwise ranking; contrastive learning is auxiliary. |
| Recommendation uncertainty diagnostic | Validation 90th percentile of absolute calibration residuals, descriptive only and without individual coverage claims. |
| Cardinal allocator | Binary MILP over the single frozen recommended profile per patient, maximizing total validation-calibrated benefit. |
| Shared budget | `0.75` resource units per population member in the bounded synthetic implementation. |
| Profile capacity | Derived from the target profile's declared primary component-capacity pool; component pools remain separately binding. |
| Resource accounting | Scenario-adjusted cost and capacity of missing bundle components only. |
| Allocation tie breaking | Seeded stable epsilon followed by patient/profile identifiers; seed is recorded in the run manifest. |
| Budget diagnostics | Prespecified multipliers `0.25`, `0.50`, `0.75`, `1.00`, `1.25` of the primary shared budget. |
| Ordinal diagnostics | Fixed population fractions `0.05`, `0.10`, `0.20`, using raw priority only and ignoring calibration and monetary cost. |
| Negative-control seeds | Five consumed base seeds: `1201`, `1223`, `1249`, `1277`, `1301`; component seeds use deterministic `SeedSequence` derivation and are recorded. |
| Held-out diagnostic pairs | Up to 256 stable repeated-DR pairs from `test`, which is excluded from fitting, early stopping, calibration and selection. |
| Null/placebo gates | Median concordance distance from chance at most `0.15` for each global direct ranker and recommendation rate at most `0.10`. |
| Permuted-label gate | Median concordance at most `0.60` and no higher than the paired unpermuted median across the frozen seeds. |
| Need-mode ablation | Rules and out-of-fold supervised need are compared on a separate 5,000-record cohort using only the noisy non-oracle panel reference. |
| Diagnostic failure policy | Stop before large experiments and version the protocol; do not loosen thresholds automatically. |
| Phase-9 discovery | Five non-oracle validation runs selected `global_rank_only` (`0.682957` mean loss versus `0.683029`). |
| Phase-9 confirmation | Ten independent confirmation seeds plus five learner seeds on fixed dataset seed `5003`; all decision stacks are hashed before oracle evaluation. |
| Phase-9 uncertainty | Run-level and paired percentile bootstrap with 1,000 replicates and seed `2039`. |

Operational quantities above are synthetic methodological defaults. They are not
estimates of Italian healthcare budgets or capacity.

## Deferred implementation choices

| Item | Resolution phase |
| --- | --- |
| External clinical and ACG validation protocol | Phase 10 with authorized governed data |

## Known limitations frozen with v1

- The care-profile catalogue is a synthetic research operationalization and has not
  been clinically adjudicated.
- The 2-day recommendation threshold and allocation quantities are methodological,
  not clinically or operationally validated.
- Version 1 cannot recommend de-intensification.
- Bundle adherence, treatment switching, censoring and interference are simplified.
- A supervised need model has no real clinician-labelled dataset in this repository.
- No authorized ACG outputs are available.

## Change control

Changing the decision unit, treatment, comparator, outcome, ranking domain,
de-intensification policy, recommendation threshold, support gate, budget policy,
resource accounting or allocation substitution rule requires:

1. a new contract and decision-log version;
2. updated structural tests;
3. a seed-registry disposition decision;
4. bounded smoke validation before experiments;
5. reports regenerated only from new actual runs.
