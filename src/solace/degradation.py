"""Semi-empirical battery degradation model (paper eq. (7)).

    dQ = B * exp(-(Ea + alpha * C) / (R T)) * [A_{t+1}^z - A_t^z] + k_cal(sigma) * dt

The paper cites Wang et al. (2011) for LFP and Xu et al. (2018) but does not
report parameter values. We follow Wang et al.'s convention that A is the
Ah throughput of a 2.2 Ah reference cell and the loss is in percent, i.e.
A = 2.2 * (equivalent full cycles) and dQ_fraction = dQ_percent / 100.
The default coefficients in ``solace.config.DegradationConfig`` must be
checked against the original publications before any quantitative use
(assumption A8). NMC uses the same functional form with its own config entry.
"""

from __future__ import annotations

import numpy as np

from solace.config import DegradationConfig

GAS_CONSTANT = 8.314  # J / (mol K)
REFERENCE_CELL_AH = 2.2


class DegradationModel:
    def __init__(self, cfg: DegradationConfig, chemistry: np.ndarray, capacity_kwh: np.ndarray):
        self.cfg = cfg
        self.capacity_kwh = np.asarray(capacity_kwh, dtype=float)
        n = len(self.capacity_kwh)
        params = [cfg.chemistries[str(c)] for c in chemistry]
        self.B = np.array([p.B for p in params])
        self.Ea = np.array([p.Ea_j_per_mol for p in params])
        self.alpha = np.array([p.alpha_j_per_mol_per_c for p in params])
        self.z = np.array([p.z for p in params])
        self.cal = np.array([p.calendar_per_day for p in params])
        self.cal_slope = np.array([p.calendar_soc_slope for p in params])
        self.temperature_k = cfg.temperature_c + 273.15
        # kappa_i: replacement cost per unit (fractional) capacity loss.
        self.kappa = cfg.battery_cost_inr_per_kwh * self.capacity_kwh / cfg.end_of_life_loss
        self.equivalent_cycles = np.full(n, cfg.initial_equivalent_cycles)

    def reset(self) -> None:
        """Start an episode from the configured battery age (A^z is concave, so age matters)."""
        self.equivalent_cycles = np.full(len(self.capacity_kwh), self.cfg.initial_equivalent_cycles)

    def step(self, power_kw: np.ndarray, soc: np.ndarray, dt_h: float, plugged: np.ndarray) -> np.ndarray:
        """Return fractional capacity loss per vehicle for one step and update throughput."""
        throughput_kwh = np.abs(power_kw) * dt_h
        cycles_new = self.equivalent_cycles + throughput_kwh / self.capacity_kwh
        c_rate = np.abs(power_kw) / self.capacity_kwh
        arrhenius = np.exp(-(self.Ea + self.alpha * c_rate) / (GAS_CONSTANT * self.temperature_k))
        a_old = REFERENCE_CELL_AH * self.equivalent_cycles
        a_new = REFERENCE_CELL_AH * cycles_new
        cycle_loss = self.B * arrhenius * (a_new**self.z - a_old**self.z) / 100.0
        calendar = self.cal * (dt_h / 24.0) * (1.0 + self.cal_slope * np.clip(soc - 0.5, 0.0, None))
        self.equivalent_cycles = cycles_new
        return np.where(plugged, cycle_loss + calendar, calendar)

    def cost_inr(self, loss: np.ndarray) -> np.ndarray:
        return self.kappa * loss
