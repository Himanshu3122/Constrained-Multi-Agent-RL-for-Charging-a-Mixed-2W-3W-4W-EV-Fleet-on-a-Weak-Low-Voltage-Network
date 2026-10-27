"""Episode rollouts and generalized advantage estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from solace.data.scenarios import DayScenario
from solace.env.charging_env import ChargingEnv, EpisodeMetrics
from solace.safety.projection import ProjectionGroup


@dataclass
class Rollout:
    obs: np.ndarray  # (T, N, D)
    actions: np.ndarray  # (T, N) sampled (pre-clip) raw actions
    logp: np.ndarray  # (T, N)
    a_hat: np.ndarray  # (T, N) projected actions
    active: np.ndarray  # (T, N) bool
    reward: np.ndarray  # (T,)
    cost_d: np.ndarray  # (T, N)
    cost_f: np.ndarray  # (T,)
    groups: list[list[ProjectionGroup]] = field(default_factory=list)
    metrics: EpisodeMetrics | None = None

    @property
    def steps(self) -> int:
        return self.obs.shape[0]


ActFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def collect_rollout(env: ChargingEnv, day: DayScenario, act_fn: ActFn) -> Rollout:
    """Run one episode. ``act_fn(obs) -> (sampled_actions, log_probs)``."""
    obs = env.reset(day)
    O, A, LP, AH, ACT, R, CD, CF, G = [], [], [], [], [], [], [], [], []
    done = False
    while not done:
        active = env.active.copy()
        a, logp = act_fn(obs)
        res = env.step(a)
        O.append(obs)
        A.append(a)
        LP.append(logp)
        AH.append(res.a_hat)
        ACT.append(active)
        R.append(res.reward)
        CD.append(res.cost_degradation)
        CF.append(res.cost_fairness)
        G.append(res.info["groups"])
        obs, done = res.obs, res.done
    return Rollout(np.stack(O), np.stack(A), np.stack(LP), np.stack(AH), np.stack(ACT), np.array(R),
                   np.stack(CD), np.array(CF), G, env.metrics)


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE over a finite episode. rewards, values: (T, N). Returns (advantages, returns)."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1:])
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values
