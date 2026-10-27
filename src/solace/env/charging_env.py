"""Constrained Dec-POMDP environment (paper eqs. (1)-(8)).

One agent per vehicle; a vehicle is active while one of its sessions is
plugged in. Each 15-min step:

1. raw actions a_tilde in [-1, 1] arrive (p = a * P_rated, consumption positive);
2. Layer 3 projects them per transformer onto LinDistFlow, kVA-polygon and
   battery bounds (eq. (11)) unless ``use_projection`` is False;
3. batteries evolve by eq. (2); degradation follows eq. (7);
4. the network is solved with full AC power flow, and violations are counted;
5. the per-step cost eq. (6) plus the soft deadline term gives the reward;
   constraint costs c_D (degradation) and c_F = 1 - Jain (at episode end)
   feed the Lagrangian (eqs. (9)-(10)).

The API mirrors the PettingZoo parallel API (reset/step over all agents) but
uses arrays instead of per-agent dicts for speed. Inactive agents receive a
zero observation (except time features) and their actions are ignored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from solace.config import ExperimentConfig
from solace.data.scenarios import DayScenario, ScenarioSet
from solace.degradation import DegradationModel
from solace.grid.powerflow import ac_power_flow
from solace.safety.projection import ProjectionGroup, build_transformer_problem, project_groups

OBS_DIM = 17
PRIORITY_CODE = {"normal": 0, "cost": 1, "urgent": 2}


@dataclass
class StepResult:
    obs: np.ndarray  # (N, OBS_DIM)
    reward: float  # shared team reward
    cost_degradation: np.ndarray  # (N,) normalized c_D per agent
    cost_fairness: float  # c_F = 1 - Jain, non-zero only at the last step
    done: bool
    a_hat: np.ndarray
    info: dict


@dataclass
class EpisodeMetrics:
    peak_loading: list[float] = field(default_factory=list)  # per transformer, fraction of rating
    voltage_violations: int = 0  # bus-steps outside [v_min, v_max] under AC power flow
    max_voltage_deviation: float = 0.0  # p.u. beyond the limit
    energy_cost_inr: float = 0.0
    degradation_cost_inr: float = 0.0
    degradation_loss: float = 0.0  # summed fractional capacity loss
    requested_kwh: float = 0.0
    unmet_kwh: float = 0.0
    delivered_ratio: list[float] = field(default_factory=list)
    delivered_vehicle: list[int] = field(default_factory=list)
    intent_satisfied: list[bool] = field(default_factory=list)
    projection_distance_kw: list[float] = field(default_factory=list)
    relaxed_steps: int = 0
    projection_latency_s: list[float] = field(default_factory=list)
    loading_trace: list[list[float]] = field(default_factory=list)  # per step: loading of each transformer
    class_requested_kwh: dict[str, float] = field(default_factory=lambda: {"2W": 0.0, "3W": 0.0, "4W": 0.0})
    class_unmet_kwh: dict[str, float] = field(default_factory=lambda: {"2W": 0.0, "3W": 0.0, "4W": 0.0})

    def jain(self) -> float:
        x = np.asarray(self.delivered_ratio, dtype=float)
        if len(x) == 0 or np.sum(x**2) == 0:
            return 1.0 if len(x) == 0 else 0.0
        return float(x.sum() ** 2 / (len(x) * np.sum(x**2)))

    def summary(self) -> dict[str, float]:
        def mean(values: list) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "peak_loading_pct": 100.0 * max(self.peak_loading) if self.peak_loading else 0.0,
            "voltage_violations": float(self.voltage_violations),
            "max_voltage_deviation_pu": self.max_voltage_deviation,
            "unmet_energy_pct": float(100.0 * self.unmet_kwh / self.requested_kwh) if self.requested_kwh else 0.0,
            "jain": self.jain(),
            "cost_kinr": self.energy_cost_inr / 1e3,
            "degradation_kinr": self.degradation_cost_inr / 1e3,
            "intent_satisfaction_pct": 100.0 * mean(self.intent_satisfied),
            "projection_distance_kw": mean(self.projection_distance_kw),
            "relaxed_steps": float(self.relaxed_steps),
            "projection_latency_ms": 1e3 * mean(self.projection_latency_s),
            "sessions": float(len(self.delivered_ratio)),
            **{f"unmet_energy_pct_{c}": (100.0 * self.class_unmet_kwh[c] / self.class_requested_kwh[c]
                                         if self.class_requested_kwh[c] > 0 else 0.0)
               for c in ("2W", "3W", "4W")},
        }


def jain_index(x: np.ndarray) -> float:
    """Jain's fairness index (paper eq. (8))."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 1.0
    denom = len(x) * np.sum(x**2)
    return float(x.sum() ** 2 / denom) if denom > 0 else 0.0


class ChargingEnv:
    def __init__(self, cfg: ExperimentConfig, scen: ScenarioSet, use_projection: bool | None = None,
                 degradation_reference: float | None = None):
        self.cfg = cfg
        self.feeder = scen.feeder
        self.fleet = scen.fleet
        self.n = len(self.fleet)
        self.steps = cfg.data.steps_per_day
        self.dt = cfg.dt_hours
        self.use_projection = cfg.env.use_projection if use_projection is None else use_projection
        self.s_base = self.feeder.s_base_kva
        self.p_rated = self.fleet.charger_kw.astype(float)
        self.cap = self.fleet.battery_kwh.astype(float)
        self.v2g_capable = self.fleet.v2g_capable.astype(bool)
        self.degradation = DegradationModel(cfg.degradation, self.fleet.chemistry, self.cap)
        # c_D is normalized by a per-agent reference episode loss (set from a no-V2G run).
        self.degradation_reference = degradation_reference or 1e-6
        self._tx_members = [np.flatnonzero(self.fleet.transformer == t) for t in range(self.feeder.n_transformers)]
        self.day: DayScenario | None = None
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def _hour(self, t: int) -> float:
        return (self.cfg.data.episode_start_hour + t * self.dt) % 24.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds on the normalized action for every agent (zero when inactive)."""
        lo = np.zeros(self.n)
        hi = np.zeros(self.n)
        act = self.active
        e_max = self.cap
        hi[act] = np.clip((e_max[act] - self.energy[act]) / (self.cfg.env.eta_charge * self.dt * self.p_rated[act]),
                          0.0, 1.0)
        dis = act & self.v2g_allowed & self.v2g_capable
        room = np.minimum(self.energy - self.soc_floor * self.cap, self.max_discharge - self.discharged)
        room = np.clip(room, 0.0, None)
        lo[dis] = -np.clip(room[dis] * self.cfg.env.eta_discharge / (self.dt * self.p_rated[dis]), 0.0, 1.0)
        return lo, hi

    def projection_groups(self, lo: np.ndarray, hi: np.ndarray) -> list[ProjectionGroup]:
        groups = []
        t = self.t
        for tx, members in enumerate(self._tx_members):
            idx = members[self.active[members]]
            if len(idx) == 0:
                continue
            net = self.feeder.networks[tx]
            prob = build_transformer_problem(
                R_path=net.R_path, X_path=net.X_path, bus=self.fleet.bus[idx],
                p_rated_pu=self.p_rated[idx] / self.s_base, lo=lo[idx], hi=hi[idx],
                base_p_pu=self.day.base_p_kw[tx][t] / self.s_base, base_q_pu=self.day.base_q_kvar[tx][t] / self.s_base,
                s_rated_pu=net.s_rated_pu, v_min=self.feeder.v_min, v_max=self.feeder.v_max,
                polygon_sides=self.feeder.polygon_sides)
            groups.append(ProjectionGroup(agent_idx=idx, problem=prob))
        return groups

    def observe(self) -> np.ndarray:
        t = min(self.t, self.steps - 1)
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        hour = self._hour(t)
        normal = self.cfg.data.tariff.normal_inr_per_kwh
        act = self.active
        soc = self.energy / self.cap
        obs[:, 0] = act
        obs[:, 1] = np.where(act, soc, 0.0)
        obs[:, 2] = np.where(act, self.min_soc, 0.0)
        obs[:, 3] = np.where(act, np.clip(self.min_soc - soc, 0.0, 1.0), 0.0)
        obs[:, 4] = np.where(act, np.clip(self.deadline - t, 0, self.steps) / self.steps, 0.0)
        obs[:, 5] = np.where(act, np.clip(self.departure - t, 0, self.steps) / self.steps, 0.0)
        obs[:, 6] = np.where(act, self.v2g_allowed & self.v2g_capable, 0.0)
        obs[:, 7] = np.where(act, self.soc_floor, 0.0)
        room = np.clip(self.max_discharge - self.discharged, 0.0, None) / self.cap
        obs[:, 8] = np.where(act, room, 0.0)
        obs[:, 9] = np.where(act, self.priority == 1, 0.0)
        obs[:, 10] = np.where(act, self.priority == 2, 0.0)
        obs[:, 11] = self.day.tariff[t] / normal
        obs[:, 12] = self.day.pv_norm[t]
        obs[:, 13] = self.tx_loading[self.fleet.transformer]
        obs[:, 14] = np.sin(2 * np.pi * hour / 24.0)
        obs[:, 15] = np.cos(2 * np.pi * hour / 24.0)
        obs[:, 16] = self.p_rated / 30.0
        return obs

    # ------------------------------------------------------------------ API
    def reset(self, day: DayScenario) -> np.ndarray:
        self.day = day
        self.t = 0
        n = self.n
        self.energy = np.zeros(n)
        self.active = np.zeros(n, dtype=bool)
        self.session_id = np.full(n, -1)
        self.min_soc = np.zeros(n)
        self.soc_arrival = np.zeros(n)
        self.deadline = np.zeros(n, dtype=int)
        self.departure = np.zeros(n, dtype=int)
        self.v2g_allowed = np.zeros(n, dtype=bool)
        self.soc_floor = np.zeros(n)
        self.max_discharge = np.zeros(n)
        self.discharged = np.zeros(n)
        self.priority = np.zeros(n, dtype=int)
        self.evaluated = np.ones(n, dtype=bool)
        self.tx_loading = np.zeros(self.feeder.n_transformers)
        self.last_a_hat = np.zeros(n)
        self.degradation.reset()
        self.metrics = EpisodeMetrics(peak_loading=[0.0] * self.feeder.n_transformers)
        self._by_arrival: dict[int, list[int]] = {}
        for k, s in enumerate(day.sessions):
            self._by_arrival.setdefault(int(s["arrival"]), []).append(k)
        self._activate_arrivals()
        return self.observe()

    def _activate_arrivals(self) -> None:
        for k in self._by_arrival.get(self.t, []):
            s = self.day.sessions[k]
            v = int(s["vehicle"])
            self.active[v] = True
            self.session_id[v] = k
            self.energy[v] = float(s["soc_arrival"]) * self.cap[v]
            self.soc_arrival[v] = float(s["soc_arrival"])
            self.min_soc[v] = min(1.0, float(s["min_soc"]))
            self.deadline[v] = int(s["deadline"])
            self.departure[v] = int(s["departure"])
            self.v2g_allowed[v] = bool(s["v2g_allowed"])
            self.soc_floor[v] = float(s["soc_floor"])
            self.max_discharge[v] = float(s["max_discharge_kwh"])
            self.discharged[v] = 0.0
            self.priority[v] = PRIORITY_CODE.get(str(s["priority"]), 0)
            self.evaluated[v] = False

    def step(self, a_raw: np.ndarray) -> StepResult:
        if self.day is None:
            raise RuntimeError("call reset() before step()")
        cfg = self.cfg
        t = self.t
        a_raw = np.clip(np.asarray(a_raw, dtype=float), -1.0, 1.0) * self.active
        lo, hi = self.bounds()
        tic = time.perf_counter()
        groups: list[ProjectionGroup] = []
        if self.use_projection:
            groups = self.projection_groups(lo, hi)
            a_hat = project_groups(a_raw, groups)
            self.metrics.relaxed_steps += sum(g.problem.relaxed for g in groups)
        else:
            # Without Layer 3 only physical battery limits apply (no grid constraints).
            a_hat = np.clip(a_raw, lo, hi)
        self.metrics.projection_latency_s.append(time.perf_counter() - tic)
        diff = a_hat - a_raw
        proj_sq = float(np.sum(diff[self.active] ** 2))
        self.metrics.projection_distance_kw.append(float(np.linalg.norm(diff * self.p_rated)))

        # Battery dynamics, eq. (2).
        p_kw = a_hat * self.p_rated
        p_ch = np.clip(p_kw, 0.0, None)
        p_dis = np.clip(-p_kw, 0.0, None)
        self.energy = np.clip(self.energy + self.dt * (cfg.env.eta_charge * p_ch - p_dis / cfg.env.eta_discharge),
                              0.0, self.cap)
        self.discharged += p_dis * self.dt / cfg.env.eta_discharge
        soc = self.energy / self.cap
        loss = self.degradation.step(p_kw, soc, self.dt, self.active)
        deg_cost = self.degradation.cost_inr(loss)

        # AC power flow per transformer.
        violations = 0
        for tx, net in enumerate(self.feeder.networks):
            members = self._tx_members[tx]
            ev_bus = np.zeros(net.n_buses)
            np.add.at(ev_bus, self.fleet.bus[members], p_kw[members])
            p_load = (self.day.base_p_kw[tx][t] + ev_bus) / self.s_base
            q_load = self.day.base_q_kvar[tx][t] / self.s_base
            pf = ac_power_flow(net.parent, net.r_pu, net.x_pu, p_load, q_load)
            loading = pf.transformer_loading_pu / net.s_rated_pu
            self.tx_loading[tx] = loading
            self.metrics.peak_loading[tx] = max(self.metrics.peak_loading[tx], loading)
            v = pf.v_mag[1:]
            under = self.feeder.v_min - v
            over = v - self.feeder.v_max
            bad = (under > cfg.env.ac_violation_tol) | (over > cfg.env.ac_violation_tol)
            violations += int(bad.sum())
            if bad.any():
                self.metrics.max_voltage_deviation = max(self.metrics.max_voltage_deviation,
                                                         float(np.max(np.maximum(under, over))))
        self.metrics.voltage_violations += violations
        self.metrics.loading_trace.append(self.tx_loading.tolist())

        tariff = self.day.tariff[t]
        export = cfg.data.tariff.v2g_export_multiplier
        energy_cost = float(tariff * self.dt * (p_ch.sum() - export * p_dis.sum()))
        self.metrics.energy_cost_inr += energy_cost
        self.metrics.degradation_cost_inr += float(deg_cost[self.active].sum())
        self.metrics.degradation_loss += float(loss[self.active].sum())

        # Sessions reaching their deadline are scored (soft deadline term).
        unmet_now = 0.0
        due = self.active & ~self.evaluated & (self.deadline <= t + 1)
        for v in np.flatnonzero(due):
            requested = max(0.0, (self.min_soc[v] - self.soc_arrival[v]) * self.cap[v])
            delivered = (soc[v] - self.soc_arrival[v]) * self.cap[v]
            unmet = max(0.0, requested - delivered)
            unmet_now += unmet
            self.metrics.requested_kwh += requested
            self.metrics.unmet_kwh += unmet
            cls = str(self.fleet.vehicle_class[v])
            self.metrics.class_requested_kwh[cls] += requested
            self.metrics.class_unmet_kwh[cls] += unmet
            ratio = 1.0 if requested <= 1e-9 else float(np.clip(delivered / requested, 0.0, 1.0))
            self.metrics.delivered_ratio.append(ratio)
            self.metrics.delivered_vehicle.append(int(v))
            ok = soc[v] >= self.min_soc[v] - 0.01 and self.discharged[v] <= self.max_discharge[v] + 1e-6
            self.metrics.intent_satisfied.append(bool(ok))
            self.evaluated[v] = True

        cost_inr = energy_cost + float(deg_cost[self.active].sum()) + cfg.env.unmet_penalty_inr_per_kwh * unmet_now
        cost_inr += cfg.env.violation_penalty_inr * violations
        reward = -cfg.env.reward_scale * cost_inr - cfg.env.beta_projection * proj_sq
        c_d = np.where(self.active, loss, 0.0) / self.degradation_reference

        # Departures and new arrivals.
        departing = self.active & (self.departure <= t + 1)
        self.active[departing] = False
        self.last_a_hat = a_hat
        self.t += 1
        done = self.t >= self.steps
        c_f = 0.0
        if done:
            c_f = 1.0 - self.metrics.jain()
        else:
            self._activate_arrivals()
        info = {"violations": violations, "energy_cost_inr": energy_cost, "projection_sq": proj_sq,
                "a_raw": a_raw, "groups": groups}
        return StepResult(self.observe(), float(reward), c_d, float(c_f), done, a_hat, info)
