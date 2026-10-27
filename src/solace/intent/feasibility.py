"""Per-EV feasibility LP and counter-offers (paper Section III-B, Fig. 1).

For one EV and a candidate constraint, the LP maximizes the energy reachable
by the deadline subject to charger power, battery capacity, the plug-in
window and a per-step headroom forecast:

    max  e0 + sum_t dt * eta * c_t
    s.t. 0 <= c_t <= min(P_charger, h_t),   e0 + dt * eta * sum_{u<=t} c_u <= E_cap

The constraint is feasible if the maximum reachable SoC >= min_soc. Otherwise
a counter-offer is computed: the earliest feasible deadline within the plug-in
window, or, if none exists, the maximum reachable SoC by the requested
deadline. The headroom forecast model is not specified in the paper; callers
pass per-step headroom in kW (assumption A14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from solace.intent.schema import ChargingConstraint, SessionContext, parse_hhmm


@dataclass
class FeasibilityResult:
    feasible: bool
    max_reachable_soc: float
    counter_offer: ChargingConstraint | None = None
    reason: str = ""


def _steps_between(start: str, end: str, dt_minutes: int) -> int:
    s, e = parse_hhmm(start), parse_hhmm(end)
    if e <= s:
        e += 24 * 60
    return (e - s) // dt_minutes


def _minutes_to_hhmm(minutes: int) -> str:
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class FeasibilityChecker:
    def __init__(self, dt_minutes: int = 15, eta_charge: float = 0.95):
        self.dt_minutes = dt_minutes
        self.dt_h = dt_minutes / 60.0
        self.eta = eta_charge

    def max_reachable_soc(self, ctx: SessionContext, steps: int, headroom_kw: np.ndarray | None = None) -> float:
        """Solve the per-EV LP for ``steps`` steps from plug-in."""
        if steps <= 0:
            return ctx.soc
        cap_kw = np.full(steps, ctx.charger_kw)
        if headroom_kw is not None:
            h = np.asarray(headroom_kw, dtype=float)[:steps]
            cap_kw[: len(h)] = np.minimum(cap_kw[: len(h)], np.clip(h, 0.0, None))
        e0 = ctx.soc * ctx.battery_kwh
        gain = self.dt_h * self.eta
        # Cumulative capacity rows: gain * sum_{u<=t} c_u <= E_cap - e0
        A_ub = np.tril(np.ones((steps, steps))) * gain
        b_ub = np.full(steps, ctx.battery_kwh - e0)
        res = linprog(c=-np.full(steps, gain), A_ub=A_ub, b_ub=b_ub,
                      bounds=list(zip(np.zeros(steps), cap_kw, strict=True)), method="highs")
        if not res.success:
            raise RuntimeError(f"feasibility LP failed: {res.message}")
        return float((e0 + gain * res.x.sum()) / ctx.battery_kwh)

    def check(self, con: ChargingConstraint, ctx: SessionContext,
              headroom_kw: np.ndarray | None = None) -> FeasibilityResult:
        steps = _steps_between(ctx.plug_in_time, con.deadline, self.dt_minutes)
        reach = self.max_reachable_soc(ctx, steps, headroom_kw)
        if reach >= con.min_soc - 1e-6:
            return FeasibilityResult(True, reach)
        window = steps
        if ctx.departure_time is not None:
            window = max(steps, _steps_between(ctx.plug_in_time, ctx.departure_time, self.dt_minutes))
        for k in range(steps + 1, window + 1):
            if self.max_reachable_soc(ctx, k, headroom_kw) >= con.min_soc - 1e-6:
                later = _minutes_to_hhmm(parse_hhmm(ctx.plug_in_time) + k * self.dt_minutes)
                offer = con.model_copy(update={"deadline": later})
                return FeasibilityResult(False, reach, offer, f"min_soc reachable by {later}")
        offered_soc = float(np.floor(reach * 100) / 100)
        offer = con.model_copy(update={"min_soc": offered_soc, "soc_floor": min(con.soc_floor, offered_soc)})
        return FeasibilityResult(False, reach, offer, f"at most {offered_soc:.2f} SoC by {con.deadline}")


def per_ev_headroom(s_rated_kva: float, base_forecast_kw: np.ndarray, expected_evs: np.ndarray,
                    margin: float = 0.05, polygon_sides: int = 16) -> np.ndarray:
    """Simple headroom-share forecast: transformer headroom split over expected concurrent EVs."""
    usable = s_rated_kva * np.cos(np.pi / polygon_sides) * (1.0 - margin)
    return np.clip(usable - np.asarray(base_forecast_kw), 0.0, None) / np.maximum(np.asarray(expected_evs), 1.0)
