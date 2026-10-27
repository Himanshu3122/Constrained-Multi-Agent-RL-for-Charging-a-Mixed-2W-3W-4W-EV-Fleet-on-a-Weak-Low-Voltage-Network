"""Shared fixtures: a tiny configuration and scenario set (synthetic, CPU only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solace.config import ExperimentConfig, load_config
from solace.data import ScenarioSet, build_scenarios

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_cfg() -> ExperimentConfig:
    return load_config(ROOT / "configs" / "smoke.yaml",
                       {"training": {"rounds": 1, "ppo_epochs": 1, "local_iters": 1}})


@pytest.fixture(scope="session")
def scen(smoke_cfg: ExperimentConfig) -> ScenarioSet:
    return build_scenarios(smoke_cfg, np.random.default_rng(0))
