"""Feeder topology.

The paper's testbed is an 11 kV radial feeder with four 630 kVA, 11/0.415 kV
transformers, each feeding an LV network (248 LV buses and 1,340 households in
total). The topology and impedances are not published, so ``build_feeder``
creates a synthetic radial network with configurable laterals and segment
impedances (assumption A1). A JSON loader supports user-provided feeders, for
example a conversion of the IEEE 34-bus test feeder.

Each LV network is modeled as a single-phase equivalent (balanced) radial
network; the transformer secondary is the slack bus at 1.0 p.u.
(assumption A2). Per-unit base: S_base = transformer rating, V_base = 415 V.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from solace.config import FeederConfig
from solace.grid.lindistflow import path_impedance_matrices


@dataclass
class LVNetwork:
    """Radial LV network below one distribution transformer.

    Bus 0 is the transformer secondary (slack). For bus ``k >= 1``, ``parent[k]``
    is its upstream bus and ``r_pu[k]``, ``x_pu[k]`` the impedance of the line
    ``parent[k] -> k``.
    """

    transformer_id: int
    parent: np.ndarray
    r_pu: np.ndarray
    x_pu: np.ndarray
    households: np.ndarray
    s_rated_pu: float = 1.0
    R_path: np.ndarray = field(init=False, repr=False)
    X_path: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.parent)
        if not (len(self.r_pu) == len(self.x_pu) == len(self.households) == n):
            raise ValueError("parent, r_pu, x_pu and households must have equal length")
        if self.parent[0] != -1:
            raise ValueError("bus 0 must be the slack bus with parent -1")
        for k in range(1, n):
            if not 0 <= self.parent[k] < k:
                raise ValueError("buses must be ordered so that parent[k] < k (radial, topologically sorted)")
        self.R_path, self.X_path = path_impedance_matrices(self.parent, self.r_pu, self.x_pu)

    @property
    def n_buses(self) -> int:
        return len(self.parent)


@dataclass
class Feeder:
    networks: list[LVNetwork]
    s_base_kva: float
    v_base_kv: float
    v_min: float
    v_max: float
    polygon_sides: int

    @property
    def n_transformers(self) -> int:
        return len(self.networks)

    @property
    def total_buses(self) -> int:
        return sum(net.n_buses - 1 for net in self.networks)


def _synthetic_network(t: int, cfg: FeederConfig, households: int, rng: np.random.Generator) -> LVNetwork:
    n_load_buses = cfg.lv_buses_per_transformer
    z_base = (cfg.v_base_kv * 1e3) ** 2 / (cfg.transformer_kva * 1e3)
    r_seg = cfg.segment_r_ohm / z_base
    x_seg = cfg.segment_x_ohm / z_base
    parent = np.full(n_load_buses + 1, -1, dtype=int)
    tips = [0] * cfg.laterals_per_transformer
    for k in range(1, n_load_buses + 1):
        lateral = (k - 1) % cfg.laterals_per_transformer
        parent[k] = tips[lateral]
        tips[lateral] = k
    r = np.full(n_load_buses + 1, r_seg)
    x = np.full(n_load_buses + 1, x_seg)
    r[0] = x[0] = 0.0
    hh = np.zeros(n_load_buses + 1, dtype=int)
    hh[1:] = households // n_load_buses
    extra = rng.choice(np.arange(1, n_load_buses + 1), size=households - hh.sum(), replace=False)
    hh[extra] += 1
    return LVNetwork(transformer_id=t, parent=parent, r_pu=r, x_pu=x, households=hh)


def build_feeder(cfg: FeederConfig, rng: np.random.Generator) -> Feeder:
    """Build the feeder described by ``cfg`` (synthetic or JSON)."""
    if cfg.source == "json":
        if not cfg.json_path:
            raise ValueError("feeder.source='json' requires feeder.json_path")
        return load_feeder_json(cfg.json_path, cfg)
    per_tx = np.full(cfg.n_transformers, cfg.households // cfg.n_transformers)
    per_tx[: cfg.households % cfg.n_transformers] += 1
    networks = [_synthetic_network(t, cfg, int(per_tx[t]), rng) for t in range(cfg.n_transformers)]
    return Feeder(networks, cfg.transformer_kva, cfg.v_base_kv, cfg.v_min, cfg.v_max, cfg.polygon_sides)


def load_feeder_json(path: str | Path, cfg: FeederConfig) -> Feeder:
    """Load a feeder from JSON.

    Schema::

        {"s_base_kva": 630, "v_base_kv": 0.415,
         "networks": [{"parent": [-1, 0, 1, ...], "r_pu": [...], "x_pu": [...],
                       "households": [...], "s_rated_pu": 1.0}, ...]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    networks = [
        LVNetwork(
            transformer_id=i,
            parent=np.asarray(n["parent"], dtype=int),
            r_pu=np.asarray(n["r_pu"], dtype=float),
            x_pu=np.asarray(n["x_pu"], dtype=float),
            households=np.asarray(n["households"], dtype=int),
            s_rated_pu=float(n.get("s_rated_pu", 1.0)),
        )
        for i, n in enumerate(data["networks"])
    ]
    return Feeder(
        networks,
        float(data.get("s_base_kva", cfg.transformer_kva)),
        float(data.get("v_base_kv", cfg.v_base_kv)),
        cfg.v_min,
        cfg.v_max,
        cfg.polygon_sides,
    )
