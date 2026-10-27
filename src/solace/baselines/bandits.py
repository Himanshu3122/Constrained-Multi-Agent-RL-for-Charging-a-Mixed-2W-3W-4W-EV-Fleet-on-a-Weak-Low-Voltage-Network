"""(iii) Decentralized Thompson-sampling bandits (approximation of Zafar et al.).

The paper describes the baseline as a fully decentralized, model-free
two-armed (charge / idle) Thompson-sampling bandit per EV. The reward
definition of the original work is not given in the manuscript, so this
implementation uses a local reward (assumption A12): charging is rewarded when
the EV's transformer was below ``loading_threshold`` and the tariff was not at
peak; idling is rewarded otherwise. An urgency override charges when the
remaining time to the deadline only just suffices. Posteriors persist across
episodes, so the bandits keep learning during evaluation, as online bandits do.
"""

from __future__ import annotations

import numpy as np

from solace.baselines.rule_based import _steps_needed
from solace.env.charging_env import ChargingEnv


class ThompsonBanditPolicy:
    name = "bandits"

    def __init__(self, n_agents: int, seed: int = 0, loading_threshold: float = 0.9):
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones((n_agents, 2))  # arm 0 = idle, arm 1 = charge
        self.beta = np.ones((n_agents, 2))
        self.loading_threshold = loading_threshold
        self._last_arm: np.ndarray | None = None

    def reset(self, env: ChargingEnv) -> None:
        self._last_arm = None

    def _update(self, env: ChargingEnv) -> None:
        if self._last_arm is None:
            return
        t_prev = max(env.t - 1, 0)
        good_time = (env.tx_loading[env.fleet.transformer] < self.loading_threshold) & (
            env.day.tariff[t_prev] <= env.cfg.data.tariff.normal_inr_per_kwh + 1e-9)
        played = self._last_arm >= 0
        arm = np.clip(self._last_arm, 0, 1)
        success = np.where(arm == 1, good_time, ~good_time) & played
        failure = ~success & played
        rows = np.flatnonzero(played)
        self.alpha[rows, arm[rows]] += success[rows]
        self.beta[rows, arm[rows]] += failure[rows]

    def act(self, env: ChargingEnv, obs: np.ndarray) -> np.ndarray:
        self._update(env)
        need = _steps_needed(env)
        sample = self.rng.beta(self.alpha, self.beta)
        charge = sample[:, 1] > sample[:, 0]
        urgent = (env.deadline - env.t) <= need
        a = np.where(env.active & (need > 0) & (charge | urgent), 1.0, 0.0)
        self._last_arm = np.where(env.active & (need > 0), charge.astype(int), -1)
        return a
