# Archived legacy PROMETHEUS local-transition run

This file records an actual run of the preserved legacy formulation. It is not the
current global multi-transition PROMETHEUS behavior. Current run reports are written
inside each `artifacts/prometheus_global*` run directory.

## Legacy objective

Training mode: `prometheus_causal_contrastive`

The implementation uses repeated patient-grouped nuisance cross-fitting, median-aggregated
transition-specific doubly robust signals, and ranking pairs whose direction is stable
across repetitions. A unified transition-conditioned ranker combines standard pairwise
causal ranking loss with optional causal contrastive regularization on the latent
representation. Scores are ordinal and are comparable only within a transition; they are
not calibrated treatment-effect estimates.

No oracle columns, potential outcomes, or latent ranks enter fitting, pair construction,
response-bin fitting, early stopping, or capacity allocation. This is a semi-synthetic
methodological experiment, not evidence of clinical effectiveness or deployment readiness.

## Transition performance

| transition | capacity | selected | AUTOC | heldout DR AUTOC | concordance | overlap coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 1_to_2 | 0.4 | 165 | 4.621327257006832e-05 | 0.13210051393520525 | 0.46943215025036716 | 1.0 |
| 2_to_3 | 0.3 | 90 | 0.05116627509864499 | 0.07364022776006796 | 0.6619273856527214 | 1.0 |
| 3_to_4 | 0.2 | 45 | -0.01634361411416722 | 0.041008245733112306 | 0.45829022310172673 | 1.0 |
| 4_to_5 | 0.1 | 17 | 0.02854033106102729 | 0.03485618477558449 | 0.6016173659321863 | 1.0 |
| 5_to_6 | 0.05 | 5 | 0.04143111999602313 | 0.10792198402689496 | 0.6586451908966672 | 1.0 |

## Macro average

| active_contrastive_margin_fraction | autoc | average_contrastive_negative_distance | average_contrastive_positive_distance | benefit_at_10pct | benefit_at_20pct | benefit_at_5pct | benefit_at_operational_capacity | boundary_error_at_10pct | boundary_error_at_20pct | boundary_error_at_5pct | contrastive_negative_pair_count | contrastive_positive_pair_count | dr_signal_repetitions | enrichment_at_10pct | enrichment_at_20pct | enrichment_at_5pct | heldout_dr_benefit_at_10pct | heldout_dr_benefit_at_20pct | heldout_dr_benefit_at_5pct | heldout_dr_pairwise_concordance | heldout_dr_policy_value_at_10pct | heldout_dr_policy_value_at_20pct | heldout_dr_policy_value_at_5pct | kendall | mean_dr_signal_between_repeat_sd | ndcg | observed_autoc | operational_capacity | operational_selected_n | overlap_coverage | pairwise_concordance | policy_regret_at_10pct | policy_regret_at_20pct | policy_regret_at_5pct | policy_regret_at_operational_capacity | policy_value_at_operational_capacity | spearman | tail_regret_at_10pct | tail_regret_at_20pct | tail_regret_at_5pct | top_capacity_overlap_with_oracle | top_overlap_at_10pct | top_overlap_at_20pct | top_overlap_at_5pct | top_weighted_concordance_at_10pct | top_weighted_concordance_at_20pct | top_weighted_concordance_at_5pct | total_benefit_at_10pct | total_benefit_at_20pct | total_benefit_at_5pct | valid_pair_fraction | valid_training_pair_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9970993876457215 | 0.02096806506281965 | 0.508422976732254 | 0.503354299068451 | -0.005988693255577576 | -0.005184371148335215 | 0.0167913575736133 | -0.012591661922195912 | 0.4207087427563618 | 0.5832552280171328 | 0.6250375409423029 | 245.8 | 245.8 | 5.0 | 1.9220970295551603 | 1.5207232224356737 | 3.5476190476190474 | 0.17765031275272833 | -0.0005530378915073131 | 0.04050940965142567 | 0.5235753210181722 | 0.017182284002380123 | -6.35838806110789e-05 | 0.0013495608332620228 | 0.1399649263334675 | 0.10873466205504952 | 0.9067044269770671 | 0.077905431246173 | 0.20999999999999996 | 64.4 | 1.0 | 0.5699824631667337 | 2.755834239127651 | 4.199070906064579 | 1.4717291173037341 | 0.015953147089458164 | 1.7091672400630564 | 0.1993916059885285 | 2.755834239127651 | 4.199070906064579 | 1.4717291173037341 | 0.2778728461081402 | 0.192209702955516 | 0.30414464448713474 | 0.17738095238095236 | 0.599957 | 0.586279 | 0.592939 | -0.806992873840839 | -1.9294403194341552 | -0.0987978658412493 | 0.995097670609481 | 0.9707083333333333 |

## Assigned levels

| assigned_causal_level | recommended_package | n | mean_risk |
| --- | --- | --- | --- |
| 1 | ordinary prevention | 248 | -0.5597454241624689 |
| 2 | light monitoring | 480 | -0.5328953450569868 |
| 3 | structured disease management | 449 | -0.473500471879845 |
| 4 | multidisciplinary integrated care | 434 | -0.19813807219109114 |
| 5 | intensive case management / home care | 396 | 0.12630990586587312 |
| 6 | level-six pathway | 393 | 1.636594257765251 |

## Evaluation-only policy comparison

| actionable_test_n | current_care_policy_value | learned_policy_value | risk_only_policy_value | burden_only_policy_value | cate_sort_policy_value | random_capacity_policy_value | oracle_capacity_policy_value | learned_increment_vs_current_care | learned_minus_risk_only | learned_minus_burden_only | learned_minus_cate_sort | learned_minus_random | fraction_of_oracle_increment | benchmark_definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2012 | 1.7126186883302397 | 1.6998379819453808 | 1.701010681534708 | 1.7014702487796947 | 1.7017522228584756 | 1.6975879392968414 | 1.7102548686977497 | -0.012780706384858842 | -0.0011726995893270864 | -0.0016322668343138336 | -0.001914240913094778 | 0.002250042648539452 | 5.406802708040265 | current care is the no-escalation baseline; risk, burden, and random use exact learned resources; oracle is evaluation-only |
