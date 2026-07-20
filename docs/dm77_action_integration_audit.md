# DM 77 action-integration audit

## Previous flow

The global runner generated numeric nested eligibility columns
`eligible_1_to_2` through `eligible_5_to_6` in the causal DGP. Those columns directly
defined transition learners and test opportunities. DM 77 assessment ran afterwards:
it produced level, reasons, protected routing, and a descriptive intervention table,
but only level-VI/manual-review exclusion affected the causal path.

The neural ranker already operated on generic numeric transition indices and supported
within/cross comparisons. Nuisance fitting consumed a generic binary
`transition_treatment` column. These components can therefore be retained behind an
action-ID adapter.

The old global allocator cannot be reused for actions because it hard-codes five nested
increments, precedence, and `final_package = 1 + sum(selected increments)`.

## Files and dependencies

- DM 77 assessment/catalogue: `src/causal_population_ranking/dm77/`.
- Synthetic numeric eligibility and observed DGP:
  `causal_utilization/global_simulation.py`.
- Opportunity construction/orchestration:
  `causal_utilization/global_prometheus_runner.py`.
- Nuisance and support: `nuisance/cross_fitting.py`,
  `causal_utilization/global_support.py`.
- Pairing/ranker/calibration: `ranking/global_pairs.py`, `ranking/global_ranker.py`,
  `ranking/calibration.py`.
- Nested allocator: `allocation/global_allocator.py`.
- Configuration/CLI/suites: `src/causal_population_ranking/config.py`,
  `src/scripts/run_prometheus.py`, `src/scripts/run_prometheus_suites.py`.
- Regression coverage: `tests/test_global_multitransition.py` and related legacy tests.

## Required change

Add a versioned care-action catalogue, patient current-care state, authoritative
DM77-plus-state action eligibility, action-index mapping, action-specific observational
DGP/nuisance signals, action opportunities, and an action allocator that applies a
selected action to the current state. Protected, manual-review, and mandatory pathways
must bypass discretionary ranking.

## Causal DGP defect found and correction

The first integrated synthetic DGP mixed an unobserved row-level random response term
into the quantity exported as `true_benefit`. More seriously, the observational
assignment logit directly read that effect matrix. Conditional exchangeability given
the learner covariates therefore did not hold even though the run was treated as the
identifiable baseline, and the primary target mixed an observed-covariate CATE with
latent individual variation.

The corrected baseline defines each action CATE as `f_a(X)` using observed pre-index
features only, sets `latent_individual_effect` to zero, and constructs assignment from
observed covariates/current care state without reading any truth array. A separate
observed-targeting scenario permits realistic selection correlated with benefit.
Latent `U` exists only in explicitly named hidden-confounding stress scenarios and is
saved exclusively in the evaluation table. The runner fixes decisions before joining
ground truth, and all primary metrics use `true_cate`.

Regression tests verify deterministic baseline CATE across assignment seeds, zero
baseline latent effect, nonzero hidden-scenario latent effect, learner/oracle physical
separation, no same-patient cross-action pairs, robust-DR audit fields, the four
validation-only calibrators, equal-cost rank-only allocation, and mandatory-action
precedence.

## Compatibility risks and controls

- Preserve the numeric nested-increment implementation under
  `opportunity_source: synthetic_legacy`.
- Keep action IDs external and numeric indices internal to the neural network.
- Do not change legacy artifact schemas; the integrated runner writes new artifact
  names.
- Validate common outcome semantics before enabling cross-action comparisons.
- Centralize method-version constants and keep legacy suite discovery on the legacy
  version.
