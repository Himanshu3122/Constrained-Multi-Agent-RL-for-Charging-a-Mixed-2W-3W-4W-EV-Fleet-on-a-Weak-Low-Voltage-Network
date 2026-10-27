"""Rule-based baselines (paper baselines (i) and (ii))."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from solace.env.charging_env import ChargingEnv


class Policy(Protocol):
    """Maps the environment state to raw normalized actions a_tilde in [-1, 1]."""

    name: str

    def reset(self, env: ChargingEnv) -> None: ...

    def act(self, env: ChargingEnv, obs: np.ndarray) -> np.ndarray: ...


def _steps_needed(env: ChargingEnv) -> np.ndarray:
    """Steps at rated power needed to reach min_soc from the current SoC."""
    need_kwh = np.clip(env.min_soc * env.cap - env.energy, 0.0, None)
    per_step = env.cfg.env.eta_charge * env.dt * env.p_rated
    return np.ceil(need_kwh / per_step)


class UncontrolledPolicy:
    """(i) Charge at rated power from arrival until the requested energy is delivered."""

    name = "uncontrolled"

    def reset(self, env: ChargingEnv) -> None:
        return None

    def act(self, env: ChargingEnv, obs: np.ndarray) -> np.ndarray:
        return (env.active & (env.energy < env.min_soc * env.cap - 1e-9)).astype(float)


class ToUPolicy:
    """(ii) Delay charging to off-peak hours when the deadline allows."""

    name = "tou"

    def reset(self, env: ChargingEnv) -> None:
        tariff = env.day.tariff
        self.offpeak = tariff <= tariff.min() + 1e-9
        # offpeak_after[t] = number of off-peak steps in [t, T)
        self.offpeak_after = np.cumsum(self.offpeak[::-1])[::-1]

    def act(self, env: ChargingEnv, obs: np.ndarray) -> np.ndarray:
        t = env.t
        steps = len(self.offpeak)
        need = _steps_needed(env)
        a = np.zeros(env.n)
        for v in np.flatnonzero(env.active & (need > 0)):
            deadline = min(int(env.deadline[v]), steps)
            remaining_offpeak = self.offpeak_after[t] - (self.offpeak_after[deadline] if deadline < steps else 0)
            if self.offpeak[t] or remaining_offpeak <= need[v] or deadline - t <= need[v]:
                a[v] = 1.0
        return a
