import json

import numpy as np
import pytest

from solace.evaluation import run_evaluation, write_report
from solace.evaluation.runner import paired_comparison


def test_evaluation_report(smoke_cfg, scen, tmp_path):
    results = run_evaluation(smoke_cfg, scen, ["uncontrolled", "tou", "mpc"], seeds=[0, 1], max_days=1)
    for method, res in results.items():
        agg = res["aggregate"]
        # Unit of analysis is the test day: deterministic methods are run once, not once per seed.
        assert agg["unmet_energy_pct"]["n"] == 1.0, method
        assert len(res["episodes"]) == 1
        assert 0.0 <= agg["jain"]["mean"] <= 1.0
        assert agg["latency_ms"]["mean"] > 0.0
    path = write_report(results, tmp_path, "test")
    assert path.read_text(encoding="utf-8").count("|") > 10
    data = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(data) == {"uncontrolled", "tou", "mpc"}


def test_seed_dependent_methods_are_averaged_per_day(smoke_cfg, scen):
    res = run_evaluation(smoke_cfg, scen, ["bandits"], seeds=[0, 1, 2], max_days=1)["bandits"]
    assert len(res["episodes"]) == 3  # one run per seed
    assert res["aggregate"]["unmet_energy_pct"]["n"] == 1.0  # averaged into one row per day
    expected = np.mean([runs[0]["unmet_energy_pct"] for runs in res["episodes"]])
    assert res["days"][0]["unmet_energy_pct"] == pytest.approx(expected)


def test_paired_comparison_on_identical_methods():
    days = [{"unmet_energy_pct": v} for v in (1.0, 2.0, 3.0)]
    results = {"a": {"days": days}, "b": {"days": [{"unmet_energy_pct": v + 1.0} for v in (1.0, 2.0, 3.0)]}}
    cmp = paired_comparison(results, "a", "unmet_energy_pct")
    assert cmp["b"]["mean_diff"] == pytest.approx(1.0)
    assert 0.0 <= cmp["b"]["wilcoxon_p"] <= 1.0


def test_unknown_method_and_missing_checkpoint(smoke_cfg, scen):
    with pytest.raises(KeyError):
        run_evaluation(smoke_cfg, scen, ["nope"], seeds=[0], max_days=1)
    with pytest.raises(ValueError):
        run_evaluation(smoke_cfg, scen, ["solace"], seeds=[0], max_days=1)
