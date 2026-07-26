import numpy as np
import pandas as pd

from causal_population_ranking.applications.contrastive_ranker import (
    sample_stable_ranking_pairs,
)


def test_pair_sampler_covers_within_unit_and_global_comparisons():
    rng = np.random.default_rng(17)
    rows = []
    for unit in range(40):
        unit_effect = rng.normal(0, 0.25)
        for treatment in range(3):
            signal = unit_effect + 0.7 * treatment + 0.08 * unit * (treatment == 2)
            rows.append({
                "unit_id": f"u{unit:03d}",
                "treatment": f"t{treatment}",
                "split": "rank_train",
                "_dr_repeat_0": signal - 0.02,
                "_dr_repeat_1": signal,
                "_dr_repeat_2": signal + 0.02,
            })
    opportunities = pd.DataFrame(rows)
    pairs = sample_stable_ranking_pairs(
        opportunities,
        ["_dr_repeat_0", "_dr_repeat_1", "_dr_repeat_2"],
        split="rank_train",
        unit_id_column="unit_id",
        treatment_column="treatment",
        maximum_pairs=360,
        minimum_signal_difference=0.05,
        minimum_repeat_agreement=0.80,
        pair_type_fractions={
            "within_unit": 0.30,
            "within_treatment": 0.35,
            "global_cross_treatment": 0.35,
        },
        top_region_fraction=0.25,
        top_pair_multiplier=2.0,
        gap_clip_quantile=0.95,
        seed=22,
    )
    assert set(pairs.pair_type) == {
        "within_unit", "within_treatment", "global_cross_treatment"
    }
    assert (pairs.repeat_agreement >= 0.80).all()
    assert np.isclose(pairs.weight.mean(), 1.0)

    within = pairs.loc[pairs.pair_type.eq("within_unit")]
    high = opportunities.iloc[within.high_index.to_numpy(int)]
    low = opportunities.iloc[within.low_index.to_numpy(int)]
    assert np.array_equal(high.unit_id.to_numpy(), low.unit_id.to_numpy())
    assert np.all(high.treatment.to_numpy() != low.treatment.to_numpy())
