import numpy as np
import pandas as pd

from causal_population_ranking.evaluation.metrics import evaluate_ranker, pairwise_concordance, tail_metrics


def test_manual_twenty_unit_orientation_and_top_k():
    truth = np.arange(20, dtype=float)
    good = evaluate_ranker(truth, truth, truth)
    bad = evaluate_ranker(-truth, truth, truth)
    assert good["pairwise_concordance"] == 1.0
    assert bad["pairwise_concordance"] == 0.0
    assert good["top_overlap_at_10pct"] == 1.0


def test_pairwise_concordance_does_not_penalize_oracle_ties():
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    score = np.array([0.0, 0.1, 0.9, 1.0])
    assert pairwise_concordance(score, truth, max_pairs=100000, seed=3) == 1.0


def test_explicit_patient_alignment_pattern():
    prediction_ids = pd.Series(["b", "a"])
    truth = pd.DataFrame({"patient_id": ["a", "b"], "value": [1, 2]})
    aligned = truth.set_index("patient_id").loc[prediction_ids].reset_index()
    assert aligned.patient_id.tolist() == prediction_ids.tolist()
    assert aligned.value.tolist() == [2, 1]


def test_tail_metrics_perfect_and_inverse_on_twenty_units():
    benefit = np.arange(20, dtype=float)
    perfect = tail_metrics(benefit, benefit, capacities=(0.10,), max_pairs=20000)
    inverse = tail_metrics(-benefit, benefit, capacities=(0.10,), max_pairs=20000)
    assert perfect["top_weighted_concordance_at_10pct"] == 1.0
    assert perfect["tail_regret_at_10pct"] == 0.0
    assert inverse["top_weighted_concordance_at_10pct"] == 0.0
