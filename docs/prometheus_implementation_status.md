# PROMETHEUS implementation status

This is the single status record for the current care-profile implementation. The
development phases are checkpoints in the active plan, not independently executable
programs.

## Implemented

- The scientific contract, care-profile and component-action catalogues, target-trial
  assumptions, recommendation/allocation semantics, oracle boundary and seed registry
  are frozen.
- Configuration is reduced to two current files: `configs/pipeline.yaml` contains
  execution parameters, the baseline-need and scientific contracts, and the seed
  registry; `configs/care_catalog.yaml` contains states, actions and care profiles.
- Baseline need has one interface with transparent rules and supervised ordinal
  modes. The supervised path is patient-disjoint and out-of-fold.
- Current care, eligibility and opportunities use explicit versioned care profiles;
  concrete actions remain profile components.
- The synthetic DGP produces exact profile-versus-`no_new_profile` assignments, one
  common outcome, heterogeneous effects, resource data and fourteen frozen scenarios.
- Every supported profile comparison uses seeded patient-disjoint splits, repeated
  grouped cross-fitting, propensity and arm-specific outcome models, repeated DR
  pseudo-outcomes and split-local robust aggregation.
- Fixed non-oracle arm-count, overlap and ESS gates mark unsupported comparisons
  without fitting or inventing supervision. Learner and causal-supervision artifacts
  are checksummed before evaluation-only truth is written.
- Evaluation has one public API backed by two responsibility-based modules:
  `baseline.py` for ordinal need and `ranking.py` for held-out DR and explicitly
  oracle-only causal-ranking diagnostics. Historical action/transition metric files
  and the unimplemented clinical-panel validator are not retained.
- The DM 77-aligned layer has four responsibility-based modules: `need.py` for the
  baseline-need models, `need_rules.py` for transparent rules, `care_profiles.py`
  for versioned profile/component contracts, and `opportunities.py` for current
  care, eligibility and opportunity construction. Rule settings live directly in
  the baseline-need contract; no generic DM77 `default.yaml` remains.
- One direct global patient-profile ranker now consumes only supported rank-train DR
  supervision and pre-index features. It uses a shared clinical encoder, learned
  profile embedding, direct pairwise objective and optional auxiliary contrastive
  projection head. Validation pairs are fixed and patient-disjoint; preprocessing,
  pair construction, fitting and checkpoint selection are non-oracle.
- The integrated comparison set contains independent profile rankers, global
  rank-only, global rank-plus-contrastive, direct pairwise GBDT, risk, baseline need
  and seeded random. Only the configured primary global method writes
  `raw_priority_score`; risk and need remain explicitly non-causal comparators. The
  oracle comparator stays deferred to final evaluation.
- Validation-only calibration compares the four frozen finite monotone candidates
  using DR MAE, rank correlation and deterministic method-name tie breaking. The
  selected mapping never overwrites the ordinal score and abstains rather than
  extrapolating beyond the validation score range.
- A separate pre-allocation stage writes one actionable-profile recommendation or
  explicit baseline-need fallback per patient. It applies the strict 2-day synthetic
  threshold, profile overlap/ESS support, same-or-higher-level policy and deterministic
  ties without reading capacity or oracle data. Validation residual Q90 is recorded
  only as a non-individual dispersion diagnostic.
- A deterministic binary MILP allocates only each patient's frozen recommended
  profile. It maximizes total validation-calibrated benefit under shared budget,
  component-pool, derived profile-capacity and one-profile-per-patient constraints;
  missing bundle components are charged exactly once. Unselected recommendations are
  explicitly deferred and leave allocated care at the current-care level.
- Cardinal budget-value curves remain distinct from fixed-count ordinal diagnostics.
  The latter use only the raw score and a seeded stable tie break, never calibration
  or monetary cost. Allocation truth metrics are computed only after the allocation
  contract and decisions are frozen.
- One compact diagnostic module runs the five consumed Phase-8 base seeds. It fits
  standard, within-profile-only and training-label-permuted global rankers as
  prespecified, evaluates only untouched non-oracle test DR pairs, records exact
  reproduction and keeps every diagnostic ineligible for model selection. The same
  root artifact contains need-mode, calibrator and threshold ablations; no additional
  runner or phase configuration exists.

The current unified runner was executed at
`artifacts/prometheus_pipeline/20260720T181643Z`: all 14 scenarios had zero critical
structural failures and all 12 prespecified direction checks passed. The primary
identifiable scenario supported all 6 profile comparisons. Across stress scenarios,
77/84 comparisons were fit and 67/84 passed every support gate; `poor_overlap`
supported 1/6 and `over_intensive_care` 0/6, as explicitly recorded rather than
silently pooled. The ranker trained in 13/14 scenarios and scored 20,686 supported
opportunities; `over_intensive_care` correctly did not train because no profile was
supported. Validation calibration was available in 12/14 scenarios; the unsupported
`over_intensive_care` scenario and the 17-row `poor_overlap` validation case abstained
instead of weakening the minimum 30-row rule. Across all scenarios 6,352 of 16,800
patient records received a profile recommendation and 10,448 explicitly abstained,
with zero raw-to-calibrated ordering violations, de-intensifications, capacity inputs
or oracle inputs. In the primary scenario the frozen validation set contains 192
within-profile and 192 cross-profile pairs, and all seven non-oracle variants scored
1,608 supported opportunities; 634/1,200 patients received a recommendation and 566
abstained.

The allocator considered all 6,352 frozen recommendations, activated 4,252 and
deferred 2,100. It recorded zero eligibility, budget, pool-capacity,
profile-capacity, prerequisite, protected-pathway, mutual-exclusion, per-patient,
double-charge, recommendation-overwrite and profile-substitution violations. In the
primary scenario it activated 485/634 recommendations and deferred 149; shared-budget
utilization was 100%. The capacity-scarcity scenario increased conditional deferral
from 23.5% to 75.2% without changing the meaning of the frozen recommendation. The
complete current suite passes 77 tests. These are actual fully synthetic engineering
results, not operational or clinical evidence.

The Phase-8 campaign passed all 8/8 prespecified exit gates: sharp-null behavior,
placebo behavior, training-pair-label permutation, exact seeded reproducibility,
oracle isolation, zero allocation violations, hidden-confounding failure declaration
and monotone threshold sensitivity. Sharp-null recommendations were 34/1,200 (2.83%)
and placebo recommendations were 0/1,200. Across five seeds the baseline rank-only
median held-out concordance was 0.5319 versus 0.5289 after label permutation, a small
0.0030 gap that passes the frozen diagnostic gate but is not effectiveness evidence.

The descriptive ablations are not model-selection records. In the primary scenario,
cross-profile training improved the rank-only median concordance by 0.0230 over
within-only training, while contrastive v2 was 0.0172 below rank-only. Under no shared
response the corresponding differences were +0.0493 and -0.0467. On the separate
5,000-record need diagnostic cohort, supervised out-of-fold ordinal MAE against the
noisy panel proxy was 0.5668 versus 0.8034 for rules; this is not comparison against
synthetic truth or clinical labels. Recommendation rates at strict thresholds 0, 2
and 5 days were 52.83%, 52.83% and 52.67%, respectively.

## Phase-9 prespecified synthetic evaluation

The same runner executed five non-oracle discovery runs, wrote and hashed the
candidate declaration, and only then opened ten untouched confirmation seeds and
five learner seeds on fixed dataset seed `5003`. Discovery selected
`global_rank_only` by a small mean validation-loss difference (`0.682957` versus
`0.683029` for the contrastive variant). The final artifact contains 15 freeze
records, one for every confirmation/stability run, and passed all 9/9 protocol
integrity gates. It reports 3,642 run-level metrics plus run-level, paired and
stability uncertainty summaries across baseline need, causal ranking,
recommendation and allocation.

Across the ten confirmation runs, the rules baseline had weighted kappa `0.7083`
(95% run-level bootstrap interval `0.6948` to `0.7193`) and ordinal MAE `0.6901`
(`0.6767` to `0.7057`) against synthetic need truth. The selected ranker had oracle
global concordance `0.6665` (`0.6336` to `0.6960`), while concordance against held-out
DR supervision was `0.5086` (`0.4965` to `0.5215`). Its paired oracle global-
concordance difference from the contrastive comparator was `0.0014` with an interval
crossing zero (`-0.0008` to `0.0034`). These are synthetic evaluation summaries,
not calibrated individual effects or evidence of clinical effectiveness.

The fixed-dataset result exposes a material limitation: mean rank-score Spearman
across learner-seed pairs was `-0.0650` (`-0.2651` to `0.1637`), with top-10% and
top-20% Jaccard overlap `0.0448` and `0.0811`. Recommended-patient Jaccard was
`0.8966` and allocated-patient Jaccard `0.6749`, but downstream thresholding does not
repair unstable raw ranking. The candidate is not changed after unblinding; any
stability remediation must use a new protocol version and fresh seeds.

## Single execution path

The current command is:

```powershell
py src\scripts\run_pipeline.py --config configs\pipeline.yaml
```

The pipeline now runs capacity-constrained allocation, Phase-8 non-oracle diagnostics
and the full Phase-9 discovery/lock/confirmation/stability campaign through this one
command and configuration.

## Evidence limits

The current synthetic evaluation, allocation, negative-control and confirmation
campaigns do not establish clinical or operational performance. They also expose
weak raw-ranker stability. The repository makes no clinical-effectiveness,
Italian-population, ACG-comparison, fairness, stability or deployment-readiness claim.
