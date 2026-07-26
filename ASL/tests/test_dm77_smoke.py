from pathlib import Path

import yaml

from causal_population_ranking.applications.dm77_case import run_dm77_case


def test_dm77_smoke_pipeline(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/applications/dual_application.yaml").read_text()
    )
    config["dm77"]["patients_smoke"] = 700
    result = run_dm77_case(config, tmp_path / "dm77", smoke=True)
    table = result["results"]
    assert "PROMETHEUS-Contrastive" in set(table.method)
    assert "Open utilization bands" in set(table.method)
    assert "X-learner GBDT" in set(table.method)
    assert table.normalized_value.notna().all()
    prometheus = table.loc[table.method.eq("PROMETHEUS-Contrastive")].iloc[0]
    assert not bool(prometheus.oracle_access_for_policy_construction)
    audit = result["ranking_audit"]
    assert audit["architecture_family"] == "film_treatment_conditioned_siamese_contrastive_ranker"
    assert audit["checkpoint_received_contrastive_updates"]
    assert set(audit["train_pair_types"]) == {
        "within_unit", "within_treatment", "global_cross_treatment"
    }
    assert audit["best_validation_selection_metric"] > 0.0

