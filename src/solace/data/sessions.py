"""Charging-session generation and per-session constraints.

* 4W sessions: the paper fits arrival, dwell and energy distributions to
  ACN-Data. ``AcnSampler`` resamples a user-supplied ACN-Data export; when it
  is absent, the parametric generator below is used (labelled synthetic).
* 2W/3W sessions: the paper calibrates a generator to aggregate Indian
  statistics (78% 2W session share, 18:00 session peak, 21:00 energy peak).
  The functional form is not published; this module uses Gaussian-mixture
  arrival times and log-normal dwell/energy (assumption A4).

Constraints C_i (deadline, min_soc, v2g_allowed, soc_floor, max_discharge_kwh,
priority) are sampled here for RL training and evaluation; in the full system
they come from the intent layer (``solace.intent``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from solace.config import ExperimentConfig, IntentSamplingConfig
from solace.data.fleet import Fleet

PRIORITIES = ("normal", "cost", "urgent")

SESSION_FIELDS = (
    "vehicle", "arrival", "departure", "soc_arrival", "energy_kwh",
    "deadline", "min_soc", "v2g_allowed", "soc_floor", "max_discharge_kwh", "priority",
)


@dataclass
class AcnSampler:
    """Empirical (arrival hour, dwell hours, energy kWh) triples from ACN-Data."""

    arrival_hour: np.ndarray
    dwell_h: np.ndarray
    energy_kwh: np.ndarray

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        arrival_col: str = "connectionTime",
        departure_col: str = "disconnectTime",
        energy_col: str = "kWhDelivered",
    ) -> AcnSampler:
        """Read an ACN-Data CSV export. Timestamps must be ISO-8601 or RFC-2822 parseable."""
        from email.utils import parsedate_to_datetime

        def parse(ts: str) -> datetime:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return parsedate_to_datetime(ts)

        arr, dwell, energy = [], [], []
        with Path(path).open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    a, d, e = parse(row[arrival_col]), parse(row[departure_col]), float(row[energy_col])
                except (KeyError, ValueError, TypeError):
                    continue
                hours = (d - a).total_seconds() / 3600.0
                if hours <= 0.25 or e <= 0.1:
                    continue
                arr.append(a.hour + a.minute / 60.0)
                dwell.append(hours)
                energy.append(e)
        if not arr:
            raise ValueError(f"{path}: no usable sessions found")
        return cls(np.array(arr), np.array(dwell), np.array(energy))

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        i = rng.integers(len(self.arrival_hour))
        return float(self.arrival_hour[i]), float(self.dwell_h[i]), float(self.energy_kwh[i])


def _lognormal_with_mean(rng: np.random.Generator, mean: float, sigma: float) -> float:
    return float(rng.lognormal(np.log(mean) - 0.5 * sigma**2, sigma))


def _sample_arrival_hour(rng: np.random.Generator, modes: list[tuple[float, float, float]]) -> float:
    weights = np.array([m[0] for m in modes], dtype=float)
    k = rng.choice(len(modes), p=weights / weights.sum())
    return float(rng.normal(modes[k][1], modes[k][2]) % 24.0)


def sample_constraint(
    rng: np.random.Generator,
    cfg: IntentSamplingConfig,
    arrival: int,
    departure: int,
    soc_arrival: float,
    target_soc: float,
    capacity_kwh: float,
    v2g_capable: bool,
) -> dict[str, float | int | bool | str]:
    deadline = departure
    if rng.random() < cfg.p_early_deadline and departure - arrival > 4:
        deadline = arrival + int(0.7 * (departure - arrival))
    v2g = bool(v2g_capable and rng.random() < cfg.p_v2g_opt_in)
    floor = float(rng.uniform(*cfg.soc_floor_range))
    floor = min(floor, max(0.0, target_soc - 0.05))
    max_dis = float(rng.uniform(*cfg.max_discharge_frac_range) * capacity_kwh) if v2g else 0.0
    u = rng.random()
    if u < cfg.p_priority_urgent:
        priority = "urgent"
    elif u < cfg.p_priority_urgent + cfg.p_priority_cost:
        priority = "cost"
    else:
        priority = "normal"
    return {
        "deadline": int(deadline), "min_soc": float(target_soc), "v2g_allowed": v2g,
        "soc_floor": floor, "max_discharge_kwh": max_dis, "priority": priority,
    }


def generate_day_sessions(
    cfg: ExperimentConfig, fleet: Fleet, rng: np.random.Generator, acn: AcnSampler | None = None
) -> list[dict[str, float | int | bool | str]]:
    """Sample all sessions of one episode (non-overlapping per vehicle)."""
    steps = cfg.data.steps_per_day
    dt_h = cfg.dt_hours
    sessions: list[dict[str, float | int | bool | str]] = []
    for v in range(len(fleet)):
        cls = str(fleet.vehicle_class[v])
        spec = cfg.fleet[cls]
        cap = float(fleet.battery_kwh[v])
        n_sessions = int(np.floor(spec.sessions_per_day)) + int(rng.random() < spec.sessions_per_day % 1.0)
        occupied = np.zeros(steps, dtype=bool)
        for _ in range(n_sessions):
            if cls == "4W" and acn is not None:
                hour, dwell_h, energy = acn.sample(rng)
            else:
                hour = _sample_arrival_hour(rng, spec.arrival_modes)
                dwell_h = _lognormal_with_mean(rng, spec.mean_dwell_h, 0.4)
                energy = _lognormal_with_mean(rng, spec.mean_energy_kwh, 0.3)
            arrival = int(((hour - cfg.data.episode_start_hour) % 24.0) / dt_h)
            departure = min(steps, arrival + max(1, int(round(dwell_h / dt_h))))
            if arrival >= steps - 1 or occupied[arrival:departure].any():
                continue
            occupied[arrival:departure] = True
            soc_arr = float(rng.uniform(0.1, 0.6))
            energy = float(np.clip(energy, 0.05 * cap, (0.98 - soc_arr) * cap))
            target = soc_arr + energy / cap
            con = sample_constraint(
                rng, cfg.intent.sampling, arrival, departure, soc_arr, target, cap, bool(fleet.v2g_capable[v])
            )
            sessions.append({"vehicle": v, "arrival": arrival, "departure": departure,
                             "soc_arrival": soc_arr, "energy_kwh": energy, **con})
    return sessions
