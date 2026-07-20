# Fully synthetic population protocol

## Purpose and claim boundary

The default PROMETHEUS run does not require Synthea, a prepared cohort cache, or real
patient records. `structured_longitudinal_population_v1` generates a complete adult
population from an explicit seed. Its purpose is reproducible methodological testing of
the DM 77 and causal-ranking architecture.

The generator is internally coherent and auditable; it is not “unassailable,” a digital
twin of a real person, or evidence of Italian-population representativeness. Realism is
separated into three claims:

1. structural validity: exact ranges, identities, temporal aggregation, and leakage rules;
2. internal plausibility: broad distributions and expected dependency directions;
3. external validity: deliberately not established without independent aggregate targets
   and external validation.

Only the first claim is enforced as a hard run gate. The second is reported as broad,
advisory validation. The third remains false in every manifest.

## Generative sequence

The generator follows this dependency order:

```text
seed
-> age, sex, deprivation, rurality, health literacy, smoking, BMI
-> named chronic conditions and additional morbidity
-> baseline frailty and social fragility
-> 24 monthly encounter, emergency, inpatient, procedure, care-plan and medication histories
-> exact 12-month aggregates and utilization trend
-> cognitive, functional, autonomy, caregiver, housing and palliative indicators
-> DM 77 assessment
-> multivalued treatment/outcome DGP
-> direct global causal ranking and constrained allocation
```

The source-population generator stops before the DM 77 assessment and before causal
simulation. It never creates treatment, outcomes, potential outcomes, true effects,
propensities, scores, or latent ranks. Oracle data are created later by the causal DGP
and kept in a physically separate evaluation table.

## Longitudinal contract

Every synthetic person has one row for each configured month. Acute encounters are a
subset of total encounters by construction. The following pre-index snapshot fields are
recomputed from the last 12 monthly rows and checked exactly:

- encounter count;
- inpatient count;
- emergency count;
- procedure count;
- care-plan count;
- difference between mean encounters in the latest and previous six months.

Active medication burden uses the maximum active count in the lookback. Days since the
last encounter are derived from the most recent non-zero month. All downstream features
are measured before the index date.

## Multidimensional domains

The snapshot includes clinical burden, specific chronic-condition indicators,
utilization, medication burden, frailty, functional limitation, cognition,
non-self-sufficiency, deprivation, social fragility, caregiver availability, housing
instability, rurality, health literacy, and recorded palliative need. Dependency
directions are deliberate: for example, age influences chronicity; chronicity and
frailty influence utilization; frailty influences functional limitation; and social
fragility reduces caregiver availability in expectation.

These are synthetic structural assumptions, not learned facts about Italy.

## Validation gates

`synthetic_validation_checks.csv` records each observed value, expectation, severity,
and result. A run stops for any critical failure involving:

- duplicate or non-synthetic identifiers;
- direct identifier or oracle fields;
- missing/non-finite source features;
- invalid ranges or indicator values;
- incomplete or duplicated patient-month panels;
- failure of exact 12-month aggregation or trend reconstruction.

Broad distribution ranges and Spearman dependency directions are advisory. They can
warn but cannot be relabelled as population validity. The validation thresholds never
read treatment, outcome, oracle effects, or downstream performance.

## Population-shift stress scenarios

The configurable scenarios are:

- `baseline`;
- `older_high_need`;
- `social_fragility_shift`;
- `utilization_surge`;
- `combined_shift`.

The suite `synthetic_population_stress` runs the primary ranker across these scenarios
and multiple seeds. These shifts test sensitivity to the synthetic population design;
they do not substitute for external data.

## Privacy and lineage

The generator consumes zero real-person rows and emits only IDs in a deterministic
`syn_<seed>_<row>` namespace. It contains no names, addresses, phone numbers, e-mail,
tax/health identifiers, or birth dates. Therefore it cannot memorize input patient
records because no such inputs exist.

The run still does not claim differential privacy or zero re-identification risk. The
privacy artifact reports these statements explicitly to avoid confusing synthetic
lineage with a formal privacy theorem.

## Reproducibility and artifacts

The global seed and configured offset determine the source-population seed. Every run
writes:

- `synthetic_patient_features.csv`;
- `synthetic_monthly_history.csv`;
- `synthetic_generation_metadata.json`;
- `synthetic_validation_checks.csv`;
- `synthetic_validation_report.json`;
- `synthetic_privacy_audit.json`.

To reproduce the bounded run:

```powershell
py scripts\run_prometheus.py --config configs\prometheus\global_smoke.yaml
```

One bounded shifted-population run can be launched directly:

```powershell
py scripts\run_prometheus.py --config configs\prometheus\global_smoke.yaml --synthetic-scenario combined_shift
```

To exercise population shifts:

```powershell
py scripts\run_prometheus_suites.py --suite synthetic_population_stress --suite-config configs\prometheus\global_experiment_suites.yaml
```
