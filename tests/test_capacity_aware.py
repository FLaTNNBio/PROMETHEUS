import numpy as np

from causal_population_ranking.causal_utilization.capacity_thresholds import select_at_capacity


def test_all_capacity_allocations_respect_their_budget_even_with_score_ties():
    score = np.ones(20)
    support = np.array([True] * 17 + [False] * 3)
    for capacity in (0.05, 0.10, 0.30, 7):
        selected = select_at_capacity(score, capacity, support)
        budget = capacity if isinstance(capacity, int) else int(np.floor(capacity * len(score)))
        assert selected.sum() <= budget
        assert not selected[~support].any()
