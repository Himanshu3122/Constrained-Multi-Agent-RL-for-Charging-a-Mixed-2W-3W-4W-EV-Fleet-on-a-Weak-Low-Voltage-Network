"""Day scenarios and the train/validation/test split (paper: 300/35/30 days)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from solace.config import ExperimentConfig
from solace.data.fleet import Fleet, build_fleet
from solace.data.profiles import (
    household_shape,
    load_irradiance_csv,
    pv_from_irradiance,
    pv_shape,
    step_hours,
    tariff_profile,
)
from solace.data.sessions import SESSION_FIELDS, AcnSampler, generate_day_sessions
from solace.grid.feeder import Feeder, build_feeder

SPLITS = ("train", "val", "test")


@dataclass
class DayScenario:
    """Inputs for one 96-step episode."""

    day: int
    base_p_kw: list[np.ndarray]  # per transformer: (steps, n_buses) active load, bus 0 = 0
    base_q_kvar: list[np.ndarray]
    tariff: np.ndarray  # (steps,) INR/kWh
    pv_norm: np.ndarray  # (steps,) normalized PV output (already netted out of base_p_kw)
    sessions: list[dict] = field(default_factory=list)


@dataclass
class ScenarioSet:
    feeder: Feeder
    fleet: Fleet
    days: dict[str, list[DayScenario]]

    def all_days(self) -> list[DayScenario]:
        return [d for s in SPLITS for d in self.days[s]]


def _day_profiles(
    cfg: ExperimentConfig, feeder: Feeder, rng: np.random.Generator, pv_norm: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    hours = step_hours(cfg.data)
    shape = household_shape(hours)
    daily = max(0.5, rng.normal(1.0, cfg.data.base_load_noise))
    tan_phi = np.tan(np.arccos(cfg.feeder.household_pf))
    p_list, q_list = [], []
    for net in feeder.networks:
        hh = net.households.astype(float)
        noise = rng.normal(1.0, cfg.data.base_load_noise, size=(len(hours), net.n_buses))
        p = daily * cfg.data.household_peak_kw * shape[:, None] * hh[None, :] * np.clip(noise, 0.5, 1.5)
        pv_bus = cfg.data.pv_kw_per_transformer * pv_norm[:, None] * hh[None, :] / max(hh.sum(), 1.0)
        q = p * tan_phi
        p_net = p - pv_bus
        p_net[:, 0] = 0.0
        q[:, 0] = 0.0
        p_list.append(p_net)
        q_list.append(q)
    return p_list, q_list


def build_scenarios(cfg: ExperimentConfig, rng: np.random.Generator) -> ScenarioSet:
    """Build feeder, fleet and all day scenarios split into train/val/test."""
    feeder = build_feeder(cfg.feeder, rng)
    fleet = build_fleet(cfg, feeder, rng)
    acn = AcnSampler.from_csv(cfg.data.acn_path) if cfg.data.acn_path else None
    ghi = load_irradiance_csv(cfg.data.irradiance_path) if cfg.data.irradiance_path else None
    hours = step_hours(cfg.data)
    tariff = tariff_profile(hours, cfg.data.tariff)
    days: list[DayScenario] = []
    for d in range(sum(cfg.data.split)):
        if ghi is not None:
            pv = pv_from_irradiance(ghi[d % len(ghi)], hours)
        else:
            pv = pv_shape(hours) * rng.uniform(0.4, 1.0)
        p, q = _day_profiles(cfg, feeder, rng, pv)
        sessions = generate_day_sessions(cfg, fleet, rng, acn)
        days.append(DayScenario(day=d, base_p_kw=p, base_q_kvar=q, tariff=tariff.copy(), pv_norm=pv,
                                sessions=sessions))
    n_tr, n_va, _ = cfg.data.split
    # Split by day before any fitting so that test days never leak into training.
    split = {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}
    return ScenarioSet(feeder=feeder, fleet=fleet, days=split)


def save_scenarios(scen: ScenarioSet, out_dir: str | Path) -> Path:
    """Save scenarios as one ``.npz`` (arrays) plus ``sessions.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for i, net in enumerate(scen.feeder.networks):
        arrays[f"net{i}_parent"] = net.parent
        arrays[f"net{i}_r"] = net.r_pu
        arrays[f"net{i}_x"] = net.x_pu
        arrays[f"net{i}_hh"] = net.households
    for name in ("battery_kwh", "charger_kw", "v2g_capable", "transformer", "bus", "aggregator"):
        arrays[f"fleet_{name}"] = getattr(scen.fleet, name)
    arrays["fleet_vehicle_class"] = scen.fleet.vehicle_class.astype("U2")
    arrays["fleet_chemistry"] = scen.fleet.chemistry.astype("U3")
    meta: dict = {"feeder": {"s_base_kva": scen.feeder.s_base_kva, "v_base_kv": scen.feeder.v_base_kv,
                             "v_min": scen.feeder.v_min, "v_max": scen.feeder.v_max,
                             "polygon_sides": scen.feeder.polygon_sides,
                             "n_networks": scen.feeder.n_transformers},
                  "splits": {}}
    for split in SPLITS:
        meta["splits"][split] = []
        for day in scen.days[split]:
            key = f"d{day.day}"
            arrays[f"{key}_tariff"] = day.tariff
            arrays[f"{key}_pv"] = day.pv_norm
            for i in range(len(day.base_p_kw)):
                arrays[f"{key}_p{i}"] = day.base_p_kw[i]
                arrays[f"{key}_q{i}"] = day.base_q_kvar[i]
            meta["splits"][split].append({"day": day.day, "sessions": [
                {k: s[k] for k in SESSION_FIELDS} for s in day.sessions]})
    np.savez_compressed(out / "scenarios.npz", **arrays)
    (out / "sessions.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


def load_scenarios(in_dir: str | Path) -> ScenarioSet:
    from solace.grid.feeder import LVNetwork

    src = Path(in_dir)
    if not (src / "scenarios.npz").is_file():
        raise FileNotFoundError(f"{src}/scenarios.npz not found; run scripts/preprocess.py first")
    arrays = np.load(src / "scenarios.npz")
    meta = json.loads((src / "sessions.json").read_text(encoding="utf-8"))
    fm = meta["feeder"]
    networks = [
        LVNetwork(i, arrays[f"net{i}_parent"], arrays[f"net{i}_r"], arrays[f"net{i}_x"], arrays[f"net{i}_hh"])
        for i in range(fm["n_networks"])
    ]
    feeder = Feeder(networks, fm["s_base_kva"], fm["v_base_kv"], fm["v_min"], fm["v_max"], fm["polygon_sides"])
    fleet = Fleet(**{name: arrays[f"fleet_{name}"] for name in (
        "vehicle_class", "chemistry", "battery_kwh", "charger_kw", "v2g_capable", "transformer", "bus",
        "aggregator")})
    days: dict[str, list[DayScenario]] = {}
    for split in SPLITS:
        days[split] = []
        for entry in meta["splits"][split]:
            key = f"d{entry['day']}"
            days[split].append(DayScenario(
                day=entry["day"],
                base_p_kw=[arrays[f"{key}_p{i}"] for i in range(fm["n_networks"])],
                base_q_kvar=[arrays[f"{key}_q{i}"] for i in range(fm["n_networks"])],
                tariff=arrays[f"{key}_tariff"], pv_norm=arrays[f"{key}_pv"], sessions=entry["sessions"]))
    return ScenarioSet(feeder=feeder, fleet=fleet, days=days)
