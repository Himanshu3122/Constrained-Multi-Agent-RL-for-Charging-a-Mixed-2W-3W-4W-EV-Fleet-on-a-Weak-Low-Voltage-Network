"""The EV fleet (paper Table II): one record per vehicle with a fixed home bus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solace.config import VEHICLE_CLASSES, ExperimentConfig
from solace.grid.feeder import Feeder


@dataclass
class Fleet:
    vehicle_class: np.ndarray  # str per vehicle
    chemistry: np.ndarray  # str per vehicle
    battery_kwh: np.ndarray
    charger_kw: np.ndarray
    v2g_capable: np.ndarray  # bool
    transformer: np.ndarray  # int
    bus: np.ndarray  # int, local bus index within the transformer's LV network (>= 1)
    aggregator: np.ndarray  # int, assigned per transformer (paper: 2 per transformer)

    def __len__(self) -> int:
        return len(self.battery_kwh)

    def class_index(self) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(VEHICLE_CLASSES)}
        return np.array([lookup[c] for c in self.vehicle_class])


def build_fleet(cfg: ExperimentConfig, feeder: Feeder, rng: np.random.Generator) -> Fleet:
    """Sample vehicles and place them on buses in proportion to household counts.

    Battery size is drawn uniformly within the Table II range and the charger
    rating uniformly from the listed ratings (assumption A3).
    """
    classes, chem, batt, charger, v2g = [], [], [], [], []
    for cls in VEHICLE_CLASSES:
        spec = cfg.fleet[cls]
        for _ in range(spec.count):
            classes.append(cls)
            chem.append(spec.chemistry)
            batt.append(rng.uniform(*spec.battery_kwh))
            charger.append(float(rng.choice(spec.charger_kw)))
            v2g.append(spec.v2g_capable)
    n = len(classes)
    hh_tx = np.array([net.households.sum() for net in feeder.networks], dtype=float)
    transformer = rng.choice(feeder.n_transformers, size=n, p=hh_tx / hh_tx.sum())
    bus = np.empty(n, dtype=int)
    for i, t in enumerate(transformer):
        weights = feeder.networks[t].households[1:].astype(float)
        bus[i] = 1 + rng.choice(len(weights), p=weights / weights.sum())
    per_tx = cfg.training.aggregators_per_transformer
    aggregator = np.empty(n, dtype=int)
    for t in range(feeder.n_transformers):
        idx = np.flatnonzero(transformer == t)
        aggregator[idx] = t * per_tx + (np.arange(len(idx)) % per_tx)
    order = rng.permutation(n)  # avoid class-sorted agent order
    return Fleet(
        vehicle_class=np.array(classes)[order],
        chemistry=np.array(chem)[order],
        battery_kwh=np.array(batt)[order],
        charger_kw=np.array(charger)[order],
        v2g_capable=np.array(v2g)[order],
        transformer=transformer,
        bus=bus,
        aggregator=aggregator,
    )
