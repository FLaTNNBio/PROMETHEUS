# PROMETHEUS global model card

PROMETHEUS directly ranks patient-transition opportunities across five adjacent care
package additions. It uses repeated grouped nuisance cross-fitting, fixed doubly robust
signals in a common day scale, balanced within/cross pair sampling, and optional pooled
causal contrastive regularization. Its output is an ordinal global priority, not a
calibrated individual treatment-effect estimate.

Validation-only isotonic regression provides an approximate cardinal benefit for
joint allocation. The exact allocator respects shared budget, eligibility, empirical
support, precedence, and optional transition capacities. Risk and causal actionability
remain separate outputs.

## Intended use

Methodological research with fully synthetic longitudinal data and controlled
comparisons. The default source generator consumes no real-person records. It is not a
clinical decision tool.

## Assumptions and limitations

- Identification relies on consistency, positivity, and no unmeasured confounding.
- `hidden_confounding` is an explicit stress scenario where the last assumption fails.
- Potential outcomes, true effects, assignment probabilities, and oracle allocations
  are evaluation-only.
- Every feature must be measured before the index date.
- Isotonic calibration is an approximate validation-signal mapping, not proof of
  individual-effect calibration.
- Results depend on the DGP, overlap, costs, budget, and sampled population.
- Structural identities and broad plausibility checks do not prove clinical realism.
- The generator is not calibrated to the Italian population and does not provide a
  formal differential-privacy guarantee; no such guarantee is needed to protect source
  records because the generator consumes none, but the distinction remains explicit.

The prototype does not establish clinical effectiveness, validity for the Italian
population, superiority over ACG or other systems, or deployment readiness.
