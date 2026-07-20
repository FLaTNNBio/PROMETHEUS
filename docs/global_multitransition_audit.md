# PROMETHEUS global multi-transition migration audit

| file | current responsibility | obsolete assumption | planned modification |
|---|---|---|---|
| `causal_utilization/simulation.py` | Five separate binary transition DGPs | Separate populations, baseline care gate, transition-specific outcome scale | Keep as an explicitly legacy baseline; add `global_simulation.py` with one multivalued treatment and physically separate oracle data |
| `causal_utilization/initial_state.py` | Generates a current/baseline care stratum | A preliminary stratum determines the only applicable escalation | Keep for legacy experiments only; never import it from the global pipeline |
| `causal_utilization/eligibility.py` | Gates eligibility with `baseline_care_level == lower` | Eligibility depends on a synthetic current-care class | Keep legacy behavior; implement treatment-independent nested eligibility in `global_simulation.py` |
| `nuisance/cross_fitting.py` | Patient-grouped repeated nuisance cross-fitting | Expects a transition-specific binary learner table | Reuse after deriving adjacent binary populations from the single multivalued dataset |
| `ranking/losses.py` | Within-transition ranking and local-bin contrastive pairs | No cross-transition comparisons; transition-wise quantile bins | Preserve legacy functions; add pooled balanced samplers in `ranking/global_pairs.py` |
| `ranking/prometheus_ranker.py` | Shared encoder with transition embeddings | Scalar-only transition input, local losses, local AUTOC checkpointing | Make the network accept mixed transition tensors and add a global ranker trained on pooled opportunities |
| `causal_utilization/prometheus_runner.py` | Legacy local training and per-transition top-B selection | Raw scores are local; baseline plus one-step escalation | Preserve as the legacy runner; make `global_prometheus_runner.py` the CLI/script default |
| `causal_utilization/stratifier.py` | Applies one-step escalation thresholds | Final decision is separate per transition and tied to baseline level | Legacy only; replace with a joint shared-budget allocator |
| `causal_utilization/policy_benchmarks.py` | Equal learned transition-count baselines | Separate resource counts, no joint precedence optimization | Legacy only; evaluate global baselines with the same allocator and constraints |
| `evaluation/metrics.py` | Transition metrics and macro averages | Global cross-transition ordering is absent | Preserve local diagnostics; add `evaluation/global_metrics.py` |
| `config.py` and `configs/prometheus/*.yaml` | Loads local PROMETHEUS settings | No shared budget, pooled pair, calibration, or eligibility configuration | Add validated global configuration files while retaining legacy examples |
| `scripts/run_prometheus.py` and CLI | Invoke legacy runner | Legacy pipeline is the main behavior | Point the default command to the global runner and expose an explicit legacy switch |
| suite/summary/plot scripts | Aggregate local AUTOC and macro metrics | Local macro performance is primary | Add global suite settings and summarize cross-transition concordance, value, and regret |
| tests | Protect the five local transition formulation | Several tests intentionally assert no cross-transition pairs | Retain them as legacy regression tests and add global contract, sampler, ranker, allocator, and smoke tests |

The migration keeps synthetic oracle quantities in an evaluation-only frame. Global
training, pair construction, validation, calibration, support prediction, and
allocation accept learner or opportunity structures that contain no oracle fields.
