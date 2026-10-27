"""Federated safe MARL training (paper Algorithm 1).

Engineering realization (assumption A11):

* aggregators are simulated sequentially in one process (the paper uses
  separate processes);
* each local iteration runs ``n_envs`` joint episodes in which every EV acts
  with the actor of its own aggregator; each aggregator then updates only on
  its own agents' transitions;
* after ``local_iters`` iterations the server averages actor and critic
  parameters weighted by transition counts (eq. (15)) and updates the Lagrange
  multipliers by projected dual ascent on the weighted constraint estimates;
* with ``training.federated: false`` a single learner owns all agents
  ("centralized training" ablation).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from solace.baselines.rule_based import UncontrolledPolicy
from solace.config import ExperimentConfig
from solace.data.scenarios import ScenarioSet
from solace.env.charging_env import ChargingEnv
from solace.training.checkpoint import save_checkpoint
from solace.training.mappo import AggregatorLearner
from solace.training.rollout import Rollout, collect_rollout
from solace.utils.logging import get_logger, kv

log = get_logger("training")


def degradation_reference(cfg: ExperimentConfig, scen: ScenarioSet, max_days: int = 3) -> float:
    """Per-agent episode capacity loss of uncontrolled (no-V2G) charging, the d_D reference."""
    env = ChargingEnv(cfg, scen, use_projection=False)
    policy = UncontrolledPolicy()
    losses = []
    for day in scen.days["train"][:max_days]:
        obs = env.reset(day)
        policy.reset(env)
        done = False
        while not done:
            res = env.step(policy.act(env, obs))
            obs, done = res.obs, res.done
        losses.append(env.metrics.degradation_loss / env.n)
    ref = float(np.mean(losses)) if losses else 0.0
    return ref if ref > 0 else 1e-6


def fed_average(models: list[torch.nn.Module], weights: list[float]) -> dict[str, torch.Tensor]:
    total = sum(weights)
    w = [x / total for x in weights] if total > 0 else [1.0 / len(models)] * len(models)
    states = [m.state_dict() for m in models]
    return {k: sum(wi * s[k].float() for wi, s in zip(w, states, strict=True)).to(states[0][k].dtype)
            for k in states[0]}


class FederatedTrainer:
    def __init__(self, cfg: ExperimentConfig, scen: ScenarioSet, out_dir: str | Path | None = None,
                 seed: int | None = None):
        self.cfg = cfg
        self.scen = scen
        self.seed = cfg.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        self.out_dir = Path(out_dir) if out_dir else None
        self.ref = degradation_reference(cfg, scen)
        self.envs = [ChargingEnv(cfg, scen, degradation_reference=self.ref) for _ in range(cfg.training.n_envs)]
        fleet = scen.fleet
        class_idx = fleet.class_index()
        if cfg.training.federated:
            owners = [fleet.aggregator == m for m in range(cfg.n_aggregators)]
            owners = [o for o in owners if o.any()]
        else:
            owners = [np.ones(len(fleet), dtype=bool)]
        self.owned = owners
        self.learners = [AggregatorLearner(cfg, fleet.transformer, class_idx, o) for o in owners]
        self.global_actor = copy.deepcopy(self.learners[0].actor)
        self.global_critic = copy.deepcopy(self.learners[0].critic)
        self.agent_owner = np.zeros(len(fleet), dtype=int)
        for m, o in enumerate(owners):
            self.agent_owner[o] = m
        self.lambdas = {"F": 0.0, "D": 0.0}
        self.history: list[dict[str, Any]] = []
        log.info(kv(event="init", aggregators=len(self.learners), agents=len(fleet), degradation_ref=self.ref))

    def _act(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(obs.shape[0])
        std = np.ones(obs.shape[0])
        for m, learner in enumerate(self.learners):
            mu, sd = learner.policy(obs)
            own = self.agent_owner == m
            mean[own] = mu[own]
            std[own] = sd[own]
        a = self.rng.normal(mean, std)
        logp = -0.5 * ((a - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
        return a, logp

    def _sample_days(self) -> list:
        train = self.scen.days["train"]
        if not train:
            raise ValueError("training split is empty")
        return [train[i] for i in self.rng.integers(len(train), size=len(self.envs))]

    def train(self) -> list[dict[str, Any]]:
        tcfg = self.cfg.training
        proj_grad = tcfg.projection_gradient and self.cfg.env.use_projection
        t_start = time.perf_counter()
        for rnd in range(1, tcfg.rounds + 1):
            for learner in self.learners:
                learner.actor.load_state_dict(self.global_actor.state_dict())
                learner.critic.load_state_dict(self.global_critic.state_dict())
            last_stats: list[dict[str, float]] = []
            episode_metrics = []
            for _ in range(tcfg.local_iters):
                rollouts: list[Rollout] = [collect_rollout(env, day, self._act)
                                           for env, day in zip(self.envs, self._sample_days(), strict=True)]
                episode_metrics.extend(r.metrics.summary() for r in rollouts)
                last_stats = [lr.update(rollouts, self.lambdas, proj_grad) for lr in self.learners]
            weights = [s["n_transitions"] for s in last_stats]
            self.global_actor.load_state_dict(fed_average([lr.actor for lr in self.learners], weights))
            self.global_critic.load_state_dict(fed_average([lr.critic for lr in self.learners], weights))
            total = sum(weights) or 1.0
            j_f = sum(w * s["J_F"] for w, s in zip(weights, last_stats, strict=True)) / total
            j_d = sum(w * s["J_D"] for w, s in zip(weights, last_stats, strict=True)) / total
            if tcfg.use_fairness_constraint:
                self.lambdas["F"] = self._dual_step(self.lambdas["F"], j_f, tcfg.d_fairness)
            if tcfg.use_degradation_constraint:
                self.lambdas["D"] = self._dual_step(self.lambdas["D"], j_d, tcfg.d_degradation_rel)
            summary = {k: float(np.mean([m[k] for m in episode_metrics])) for k in episode_metrics[0]}
            record = {"round": rnd, "lambda_F": self.lambdas["F"], "lambda_D": self.lambdas["D"],
                      "J_F": j_f, "J_D": j_d,
                      "policy_loss": float(np.mean([s["policy_loss"] for s in last_stats])),
                      "value_loss": float(np.mean([s["value_loss"] for s in last_stats])),
                      "projection_loss": float(np.mean([s["projection_loss"] for s in last_stats])),
                      "elapsed_s": time.perf_counter() - t_start,
                      "env_steps": rnd * tcfg.local_iters * len(self.envs) * self.cfg.data.steps_per_day,
                      **{f"train_{k}": v for k, v in summary.items()}}
            self.history.append(record)
            log.info(kv(event="round", **{k: record[k] for k in (
                "round", "lambda_F", "lambda_D", "J_F", "J_D", "policy_loss", "value_loss",
                "train_unmet_energy_pct", "train_peak_loading_pct", "train_jain")}))
            if self.out_dir:
                self.save(self.out_dir / "latest.pt", rnd)
        if self.out_dir:
            self.save(self.out_dir / "final.pt", tcfg.rounds)
        return self.history

    def _dual_step(self, lam: float, j_hat: float, budget: float) -> float:
        """Projected dual ascent, eq. (10). With ``dual_normalize`` the violation is divided by the
        budget so that the step size is comparable across constraints (revision; audit C3)."""
        tcfg = self.cfg.training
        viol = j_hat - budget
        if tcfg.dual_normalize:
            viol /= max(abs(budget), 1e-9)
        return float(min(tcfg.lambda_max, max(0.0, lam + tcfg.dual_lr * viol)))

    def save(self, path: Path, rnd: int) -> Path:
        return save_checkpoint(path, self.global_actor, self.global_critic, self.cfg, self.lambdas, rnd,
                               self.ref, self.history)
