from causal_population_ranking.ems.ranking import EMSDirectRanker, fit_ems_direct_ranker


def test_ems_ranker_import_is_not_circular():
    assert EMSDirectRanker is not None
    assert callable(fit_ems_direct_ranker)
