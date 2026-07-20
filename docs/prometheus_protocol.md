# PROMETHEUS global multi-transition protocol

## Observational contract

By default, `X_i` is generated without real-person records by the versioned structured
longitudinal generator. A 24-month monthly panel is generated first; 12-month counts and
utilization trends are then reconstructed from that panel and validated as exact
identities. The source generator has its own seed and produces no treatment, outcome,
or oracle columns.

There is one row per patient: pre-treatment covariates `X_i`, one historical package
`T_i in {1,...,6}`, and `Y_i`, days alive and outside acute-care hospitalization over
365 days. `T_i` is not an initial severity or care class. The global path contains no
`baseline_care_level` or `current_care_level`.

Potential outcomes are generated recursively on the common day scale:

```text
mu_i,1 = f0(X_i)
mu_i,t = mu_i,1 + sum_{r < t} tau_i,r
Y_i = mu_i,T_i + noise_i
```

The learner and oracle tables are physically separate. The learner has a single
multivalued treatment and nested, treatment-independent, pre-treatment eligibility.
The oracle table is opened only after the learned operational allocation is fixed.

## Splits and nuisance signals

One seeded patient split is assigned before constructing transition populations:
`nuisance_train`, `rank_train`, `validation`, and `test`. For transition `k`, nuisance
estimation sees eligible patients whose historical treatment is `k` or `k+1`, with
`D_i^(k) = 1[T_i=k+1]`. Repeated patient-grouped cross-fitting estimates propensity
and arm-specific outcomes. DR signals remain in outcome days and are never centered,
standardized, or quantiled separately by transition.

## Global ranking

Rows of the ranking table are `(patient_id, transition_index)` opportunities. The
network combines a clinical encoder, transition embedding, interaction projection,
and scoring head. Every batch may contain mixed transition indices.

Stable ranking pairs are sampled with explicit budgets for each `(k,l)` block. Within
pairs have `k=l`; cross pairs have `k!=l`. A pair is retained only when its aggregated
DR gap exceeds the configured day threshold and the direction agrees across enough
nuisance repetitions. The objective is

```text
(1-beta_cross) * L_within + beta_cross * L_cross
```

The causal contrastive variant uses pooled day-gap thresholds for positive and
negative pairs, including cross-transition pairs. Early stopping includes held-out
cross-transition DR concordance. The raw score is a globally comparable ordinal
priority, not a calibrated treatment-effect estimate.

## Calibration and allocation

Global models fit one isotonic calibrator on validation scores and held-out validation
DR signals. Local baselines use separate transition calibrators because their raw
scores have no common scale. No oracle target enters calibration.

The allocator uses binary increments `Z_i,k` and exact MILP optimization. It enforces
clinical eligibility, optional empirical support, precedence `Z_i,k <= Z_i,k-1`, one
shared budget, and optional transition capacities. Non-positive calibrated increments
are excluded when configured. The final package is `1 + sum_k Z_i,k`; it is an
allocation result, not a severity class.

## Evaluation

Primary synthetic metrics are cross-transition concordance, global allocation
value, benefit per capacity unit, global regret against an oracle solution with the
same constraints, and fraction of oracle benefit. Within-transition AUTOC and
Benefit@capacity remain secondary diagnostics. Oracle effects are evaluation-only.

Population-shift scenarios vary age/need, social fragility, and utilization. Broad
plausibility checks and scenario robustness are internal methodological diagnostics;
they do not establish representativeness of the Italian population.
