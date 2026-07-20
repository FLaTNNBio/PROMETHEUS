# PROMETHEUS

PROMETHEUS is a research pipeline for direct causal ranking of admissible
patient-care-profile opportunities. It keeps three outputs separate:

```text
baseline need != causal recommendation != operational allocation
```

The primary decision unit is `(patient_id, care_profile_id)`. The raw priority score
is ordinal: it is not a calibrated individual treatment effect and is never compared
directly with zero.

## Current pipeline

The repository currently implements one integrated path through the synthetic
care-profile DGP:

```text
seeded pre-index synthetic population
  -> baseline-need assessment
  -> current care profile
  -> governed profile eligibility
  -> exact observed profile or no_new_profile
  -> common observed outcome
  -> patient-disjoint protocol splits
  -> repeated cross-fitted profile-specific nuisance models
  -> robust doubly-robust ranking supervision
  -> stable within-profile and cross-profile DR pairs
  -> direct global patient-profile ranker over two prespecified variants
  -> ordinal priority scores for supported opportunities
  -> validation-only monotone calibration selection
  -> actionable profile recommendation or explicit baseline fallback
  -> cardinal MILP allocation over frozen recommendations
  -> separate fixed-count ordinal diagnostics
  -> five-seed held-out negative controls and prespecified ablations
  -> frozen diagnostic gates before large experiments
  -> five-run non-oracle discovery and hashed candidate declaration
  -> ten untouched confirmations plus five fixed-dataset stability runs
  -> paired uncertainty across four reporting layers
  -> frozen allocation/truth physical boundary
```

Phases 8 and 9 run through this same pipeline, without separate phase runners. The
prespecified discovery selected `global_rank_only`; the candidate was written and
hashed before confirmation seeds were opened. The completed synthetic run is
`artifacts/prometheus_pipeline/20260720T181643Z` and passed all 9 protocol-integrity
gates. Fixed-dataset ranking stability was weak, so the result does not support a
stability, clinical or deployment claim.

## Run

```powershell
py -m pip install -e .
causal-ranking run --config configs\pipeline.yaml
```

The equivalent source command is:

```powershell
py src\scripts\run_pipeline.py --config configs\pipeline.yaml
```

An experiment is defined by a configuration using the same pipeline schema. The
canonical configuration runs all frozen synthetic scenarios; a smaller experiment
may use an ordered subset. There is no command-line switch for development phases.
`configs/` contains only `pipeline.yaml` (experiment, contracts and seeds) and
`care_catalog.yaml` (care states, component actions and target profiles).

## Verify

```powershell
py -m pytest
```

The current suite contains 77 passing tests.

All stochastic components require explicit registered seeds. Synthetic truth,
potential outcomes and latent response are written only under `evaluation_only/`
after learner-safe data, causal supervision, validation pairs, the fitted ranker,
priority scores, selected validation calibrator and actionable recommendations have
been frozen and checksummed. Synthetic truth is opened only after the allocation
policy and allocation decisions have also been frozen.

## Scope

The current evidence is fully synthetic and methodological. It does not establish
clinical effectiveness, validity for the Italian population, equivalence or
superiority to ACG, fairness, or deployment readiness. The active scientific plan is
in `docs/prometheus_causal_stratification_plan.md`.
