from __future__ import annotations

from scripts.run_all_dm77_experiments import experiment_groups
from scripts.run_prometheus_suites import _apply, build_suite_plan, run_suite


METHOD_CONFIG = "configs/prometheus/dm77_method_ablation_suites.yaml"


def test_signal_estimator_suite_plan_is_deterministic_and_complete():
    _, plan = build_suite_plan(
        "causal_signal_estimator_ablation", METHOD_CONFIG
    )
    assert len(plan) == 12
    assert [condition["causal_signal_estimator"] for _, condition in plan[:4]] == [
        "dr", "outcome_regression", "ipw", "naive_outcome",
    ]
    assert [seed for seed, _ in plan] == [17] * 4 + [37] * 4 + [71] * 4


def test_suite_resume_skips_completed_plan_entry(tmp_path, monkeypatch):
    completed = (
        tmp_path / "causal_signal_estimator_ablation" / "s17_r000" / "finished"
    )
    completed.mkdir(parents=True)
    (completed / "run_manifest.json").write_text("{}", encoding="utf-8")

    def fail_if_called(_):
        raise AssertionError("A completed plan entry must not be executed")

    monkeypatch.setattr(
        "scripts.run_prometheus_suites.run_global_prometheus", fail_if_called
    )
    result = run_suite(
        "causal_signal_estimator_ablation", METHOD_CONFIG, tmp_path,
        seeds=[17], max_runs=1, resume=True,
    )
    assert result["planned_runs"] == 1
    assert result["executed_runs"] == 0
    assert result["skipped_completed_runs"] == 1


def test_master_profiles_cover_every_method_suite():
    quick = experiment_groups("quick")
    full = experiment_groups("full")
    publication = experiment_groups("publication")
    assert len(quick) == len(full) == 7
    assert all(seeds == [17] for _, _, seeds in quick)
    assert all(seeds is None for _, _, seeds in full)
    assert len(publication) == 13


def test_checkpoint_metric_axis_updates_only_experiment_copy(tmp_path):
    base = {
        "run": {"seed": 42},
        "checkpoint": {"metric": "global_concordance"},
    }

    configured = _apply(
        base,
        {"checkpoint_metric": "dr_policy_value"},
        seed=17,
        output_root=tmp_path,
        run_index=3,
    )

    assert configured["checkpoint"]["metric"] == "dr_policy_value"
    assert configured["run"]["seed"] == 17
    assert configured["run"]["output_root"].endswith("s17_r003")
    assert base["checkpoint"]["metric"] == "global_concordance"
