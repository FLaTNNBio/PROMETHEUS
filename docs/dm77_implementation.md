# DM 77 evolution of PROMETHEUS

## Implemented outcome

The project now exposes two different and explicitly separated questions:

| Output | Question | Semantics |
| --- | --- | --- |
| `dm77_need_level` | What is the person's multidimensional level of need? | Need stratum I-VI or abstention |
| prognostic inputs/baselines | Who is at higher future risk? | Risk, separate from actionability |
| PROMETHEUS score | Which eligible patient-action opportunity has higher causal priority? | Ordinal priority, not an individual treatment effect |
| allocation | Which supported opportunities can be selected under constraints? | Operational decision under budget and capacity |

The derived DM 77 level is never used as the historical treatment, ranker feature,
pair-construction target, causal calibration target, or oracle. PROMETHEUS remains a
direct causal-ranking method and does not estimate an individual CATE and sort it.

## Regulatory basis and scope

The level labels follow the six-level population-stratification model in Annex 1 of
Italian DM 77/2022: health, minimal/time-limited complexity, medium complexity,
medium-high complexity with possible social fragility, high complexity with possible
social fragility/non-self-sufficiency, and palliative care.

Official source: [Gazzetta Ufficiale, DM 77/2022 Annex 1](https://www.gazzettaufficiale.it/atto/serie_generale/caricaArticolo?art.codiceRedazionale=22G00085&art.dataPubblicazioneGazzetta=2022-06-22&art.flagTipoArticolo=1&art.idArticolo=1&art.idGruppo=0&art.idSottoArticolo=1&art.idSottoArticolo1=10&art.progressivo=0&art.versione=1).

DM 77 does not supply the executable numeric cut-offs implemented here. The thresholds
under `configs/dm77/default.yaml` are therefore labelled
`research_default_not_official`. They are transparent and configurable, but not a
validated national algorithm. Local changes require documented clinical, social-care,
legal, data-governance, and fairness review. An example override file is provided only
as a template and is not validated.

## Strict pre-index data contract

The assessment reads only:

- age and recorded clinical burden;
- prior inpatient and emergency use;
- prior medication burden;
- frailty and functional limitation;
- cognitive impairment and non-self-sufficiency;
- social fragility, caregiver availability, and housing instability;
- a pre-index recorded palliative-care need.

Treatment, observed outcome, potential outcomes, true effects, oracle probabilities,
causal scores, and latent ranks are ignored even if present. A per-run audit lists such
columns as present-but-ignored.

The default fully synthetic source generator creates all functional and social domains
from a versioned dependency structure and an explicit seed before treatment assignment.
They are not measurements from, or a calibration to, the Italian population. In the
optional cohort-cache mode, missing domains can be augmented only for methodological
simulation; on a future real-data path, missing required inputs must cause abstention
rather than automatic DM 77 imputation.

## Rules and protected routing

Rules are evaluated in this order:

1. a recorded palliative need routes to level VI and the protected palliative pathway;
2. non-self-sufficiency or high functional/multidimensional complexity routes to V;
3. medium-high clinical complexity combined with functional or social fragility routes to IV;
4. chronicity, early frailty, cognitive impairment, or moderate functional limitation routes to III;
5. minimal or episodic recorded need routes to II;
6. absence of recorded complexity indicators routes to I.

Each assessment contains the three dimension scores, bands, data completeness, status,
and reason codes. Rules cannot replace a professional multidimensional assessment.
Level VI is not interpreted as the highest ordinary service intensity. It is excluded
from the standard causal allocation and linked only to the protected palliative entry
in the intervention catalogue. Cases with missing required data are also excluded from
automatic standard allocation and enter a manual-review queue.

## Computable care-action catalogue

`configs/dm77/care_action_catalog.yaml` is the authoritative integrated catalogue. It
contains versioned care states, action IDs, eligible need levels, state transitions,
clinical rules, contraindications, prerequisites, mutual exclusions, costs, capacity
pools, protected/mandatory flags, and outcome semantics. Its rules determine which
actions are admissible; they do not determine the causal priority score.

`configs/dm77/intervention_catalog.yaml` is retained only for the preserved numeric
`synthetic_legacy` path. The integrated application path never converts a DM 77 level
into a `k_to_k+1` treatment.

The protected palliative action has a dedicated outcome and horizon and is excluded
from cross-action causal ranking. The current 365-day hospital-free-days outcome is not
sufficient to evaluate palliative care quality.

## Run artifacts

Every integrated global run writes:

- `dm77_need_assessment.csv`;
- `patient_current_care_state.csv`;
- `dm77_action_eligibility.csv`;
- `patient_action_opportunities.csv`;
- `action_specific_dr_signals.csv`;
- `global_ranking.csv`;
- `calibration_diagnostics.csv`;
- `allocation_decisions.csv`;
- `protected_and_manual_review_cases.csv`;
- `run_manifest.json` and explicit oracle non-use declarations.

All metrics and counts in run reports come from the run itself. No clinical
effectiveness, Italian-population validity, ACG superiority, or deployment-readiness
claim is made.

## Verification

Run:

```powershell
py -m pytest
py src\scripts\run_prometheus.py --config configs\prometheus\dm77_integrated_smoke.yaml
```

Tests cover levels I-VI, missing-data abstention, treatment/outcome/oracle invariance,
catalogue routing, need-versus-current-state eligibility, deterministic fully synthetic
covariates, absence of the derived level from ranker features, protected/manual-review
routing, true cross-action comparisons, allocation feasibility, and the end-to-end
output contract. See `docs/dm77_action_integration.md` for the network and allocator
details.
