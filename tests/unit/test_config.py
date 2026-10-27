from pathlib import Path

import pytest
from pydantic import ValidationError

from solace.config import ExperimentConfig, dump_config, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_default_config_matches_paper_values():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.feeder.n_transformers == 4 and cfg.feeder.transformer_kva == 630.0
    assert cfg.data.split == (300, 35, 30)
    assert [cfg.fleet[c].count for c in ("2W", "3W", "4W")] == [620, 220, 160]
    t = cfg.training
    assert (t.lr_actor, t.lr_critic, t.gamma, t.gae_lambda, t.clip_eps) == (3e-4, 5e-4, 0.99, 0.95, 0.2)
    assert (t.rounds, t.local_iters, t.d_fairness, t.d_degradation_rel) == (300, 5, 0.06, 1.1)
    assert cfg.n_aggregators == 8
    assert cfg.env.beta_projection == 0.05


def test_smoke_extends_default_and_merges_nested_fleet():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.fleet["2W"].count == 6
    assert cfg.fleet["2W"].battery_kwh == (2.0, 4.0)  # inherited from default.yaml
    assert cfg.training.lr_actor == 3e-4


def test_overrides_and_validation(tmp_path):
    with pytest.raises(ValidationError):
        load_config(ROOT / "configs" / "smoke.yaml", {"data": {"split": [10, 10, 10]}})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_dump_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    dump_config(cfg, tmp_path / "c.yaml")
    again = load_config(tmp_path / "c.yaml")
    assert again == cfg
    assert isinstance(again, ExperimentConfig)
    assert cfg.scenario_dir.name == "smoke"
