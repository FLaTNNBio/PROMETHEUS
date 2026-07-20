# PROMETHEUS validation roadmap

## Evidence levels and permitted claims

PROMETHEUS separates software correctness, causal identification, operational utility,
and clinical impact. Evidence from one level must not be promoted to a claim belonging
to a later level.

| Phase | Primary evidence | Permitted claim |
| --- | --- | --- |
| Software verification | tests, schemas, constraint checks | the pipeline executes reproducibly and respects its contract |
| Synthetic validation | oracle policy value, regret, controlled failures | the method can recover known priorities under stated synthetic assumptions |
| DM77 clinical validation | adjudicated panel agreement | the computable DM77 operationalization agrees with the chosen clinical reference |
| Retrospective target trials | held-out DR policy value and sensitivity analyses | recommendations have estimated value in the specified observational data under explicit assumptions |
| Temporal/geographic validation | out-of-time/out-of-site replication | conclusions are robust or transportable under the tested condition |
| Silent deployment | completeness, latency, support, routing and disparities | the system can run safely in the observed workflow without changing care |
| Human-in-the-loop evaluation | acceptance, override, workflow and early safety | recommendations are interpretable and operationally manageable |
| Controlled impact study | prespecified clinical endpoints | use of the system changes outcomes under the trial design |

The present repository supports the first two phases with fully synthetic data. It
provides a metric implementation for a future adjudicated DM77 panel, but it contains
no panel ratings, retrospective clinical cohort, silent deployment, or controlled
trial data. Therefore it establishes none of the later claims.

## Current synthetic protocol

The primary evaluation is policy value under shared constraints, accompanied by
`Regret@Budget`. Ranking diagnostics include within-, cross-, global-, and hard
cross-action concordance. For top fractions 5%, 10%, and 20%, the runner reports:

- `Benefit@q`: mean true CATE among the top-q opportunities;
- `OracleTop-q Recovery`: fraction of the oracle top-q recovered, synthetic only;
- `Value@q`: sum of true CATE in the selected top-q;
- `Regret@q`: oracle top-q value minus learned top-q value.

Exact oracle recovery is secondary because multiple near-tied opportunities can lead
to operationally equivalent value. The raw score remains ordinal. Only the distinct
`calibrated_incremental_benefit` is assessed for bias, slope, intercept, and score-group
calibration.

### Stability

The stability suite fixes `dataset_seed` while changing the nuisance/model seed. This
ensures that runs contain the same patient-action keys and true CATEs. It reports:

- Spearman and Kendall correlation over the complete ranking;
- Jaccard of the top 5%, 10%, and 20%;
- identical-allocation fraction and allocation Jaccard;
- absolute policy-value difference;
- allocation instability near the selection boundary.

Runs with different synthetic populations are not treated as ranking-stability pairs.

### Confounding and failure scenarios

`strong_observed_confounding` uses strong observed targeting while retaining
conditional exchangeability. `poor_overlap` stresses positivity separately.
`hidden_confounding` and `combined_stress` are failure/sensitivity scenarios: latent
`U` affects assignment, outcome, and response but is absent from learner features. No
oracle recovery is expected under hidden confounding.

Different outcome names, follow-up horizons, units, or benefit directions are
structural incompatibilities. They must prevent cross-action pair construction rather
than become ordinary performance benchmarks.

## Interval coverage protocol

Coverage is defined only for aggregate estimands:

- held-out policy value;
- paired value difference against a baseline;
- action- or score-group average effect;
- `Benefit@Budget`.

For each independently generated synthetic replicate, the policy must be fixed before
interval estimation. Evaluation then uses a patient-clustered bootstrap or another
prespecified valid estimator on an independent evaluation partition. Coverage is the
fraction of intervals containing that replicate's synthetic oracle estimand. The
10-seed comparison suite estimates variability; it must not be labelled a coverage
study without a prespecified Monte Carlo reference and sufficient independent
replicates. PROMETHEUS does not claim coverage of individual treatment-effect
intervals because it does not produce such intervals.

## DM77 clinical validation

DM77 need assessment is validated separately from causal ranking. The reference input
must be a panel-adjudicated assessment rather than a single reviewer's label. The
implemented evaluator reports:

- raw level agreement and quadratic weighted Cohen kappa;
- optional quadratic weighted multi-rater Fleiss kappa;
- ordinal mean absolute error and errors greater than one level;
- sensitivity, specificity, and false-negative rates for critical levels;
- false negatives for protected pathways and manual review;
- exact and Jaccard agreement for eligible action sets.

The evaluator explicitly records that neither the causal ranker nor clinical
effectiveness is being assessed. Adjudication procedures, reviewer independence, case
sampling, missingness, and confidence intervals must be prespecified in a real study.

The future panel analysis can be run with:

```powershell
py src\scripts\evaluate_dm77_panel.py --rule-assessments dm77_need_assessment.csv --adjudicated-reference panel_reference.csv --output-dir artifacts\dm77_panel_validation --panel-rater-columns reviewer_1 reviewer_2 reviewer_3
```

The reference table must contain `patient_id`, `reference_dm77_need_level`,
`reference_manual_review`, and `reference_protected_pathway`. Optional
`reference_eligible_action_ids` is compared with the pipe-delimited
`eligible_action_ids` supplied in the rule table. The repository provides the metric
calculation only; it does not provide or simulate a clinical reference panel.

## Retrospective target-trial layer

There is no single generic PROMETHEUS target trial. Each `care_action_id` requires a
separate protocol containing:

- eligibility criteria and treatment versions;
- the action and `no_new_action` comparator;
- index/time-zero definition aligned with eligibility and assignment;
- follow-up, outcome, causal contrast, and estimand;
- baseline confounders measured before time zero;
- censoring, treatment switching, missing-data, and positivity strategies;
- mapping from every protocol element to source data;
- negative controls and sensitivity analyses for unmeasured confounding.

The TARGET Statement is used as a reporting checklist for observational target-trial
emulations, not as proof that identification holds. Its checklist requires the target
trial protocol, causal estimand, identifying assumptions, data mapping, estimates,
precision, and sensitivity analyses:
<https://www.bmj.com/content/390/bmj-2025-087179>.

## External validation experiments

External validation must identify which of three experiments was performed:

1. **Frozen transport:** feature transformation, ranker, calibration, thresholds, and
   catalog remain fixed.
2. **Local recalibration:** the ranker remains fixed while calibration, baseline rates,
   capacities, or costs are updated using local data.
3. **Local re-estimation:** nuisance models or the ranker are refitted.

Treatment definition, outcome, time zero, follow-up, unit, direction, and provenance
must be semantically equivalent; matching column names is insufficient. Results from
these three experiments must not be pooled under one portability claim.

## Prospective evaluation and reporting

Silent deployment can study data completeness, latency, support, protected/manual
review routing, organizational failure, stability, and subgroup disparities. It
cannot establish improved outcomes because recommendations do not alter care.

DECIDE-AI is applicable to reporting early clinical evaluation of AI decision-support
systems and emphasizes small-scale clinical performance, safety, and human factors; it
does not replace a controlled impact study:
<https://www.nature.com/articles/s41591-022-01772-9>.

TRIPOD+AI supersedes TRIPOD 2015 for transparent reporting of prediction-model studies,
and PROBAST+AI addresses quality, risk of bias, and applicability of prediction models.
They are relevant to predictive components but do not cover the full causal-policy
claim:
<https://www.bmj.com/content/385/bmj-2023-078378> and
<https://www.bmj.com/content/388/bmj-2024-082505>.

A later randomized evaluation should use an appropriate AI trial reporting extension.
Cluster randomization or a stepped-wedge design may be considered when team-level
behavior would contaminate patient-level randomization; the design choice requires a
separate protocol, power calculation, governance approval, and clinical oversight.
