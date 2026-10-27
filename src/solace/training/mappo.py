"""Constrained MAPPO update for one aggregator (paper eqs. (9)-(14)).

Loss terms per minibatch:

* clipped PPO surrogate on the Lagrangian advantage
  A^L = A^R - lambda_F A^F - lambda_D A^D  (eq. (14)), each term by GAE;
* entropy bonus;
* projection-correction term beta * ||Proj(mu) - mu||^2 on the policy mean,
  differentiated through the safety QP via its KKT conditions. This is how
  this implementation realizes "backpropagating through (11)" in Algorithm 1
  (assumption A11: the paper does not state where the gradient enters);
* value regression on three heads (reward, fairness cost, degradation cost).

Only transitions of agents owned by this aggregator (and active) enter the
loss, so raw trajectories of other aggregators are not used for its update.
"""

from __future__ import annotations

import numpy as np
import torch

from solace.agents.networks import ClassActors, build_critic
from solace.config import ExperimentConfig
from solace.env.charging_env import OBS_DIM, jain_index
from solace.safety.projection import differentiable_project
from solace.training.rollout import Rollout, gae


class AggregatorLearner:
    def __init__(self, cfg: ExperimentConfig, transformer: np.ndarray, class_idx: np.ndarray, owned: np.ndarray):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.n_agents = len(transformer)
        self.class_idx = torch.as_tensor(class_idx, dtype=torch.long, device=self.device)
        self.owned = owned.astype(bool)
        self.actor = ClassActors(OBS_DIM, cfg.agent).to(self.device)
        self.critic = build_critic(OBS_DIM, self.n_agents, cfg.agent,
                                   torch.as_tensor(transformer, device=self.device)).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.training.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.training.lr_critic)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def policy(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mean, std = self.actor(o, self.class_idx)
        return mean.cpu().numpy(), std.cpu().numpy()

    # ------------------------------------------------------------------ update
    def _targets(self, rollouts: list[Rollout], lambdas: dict[str, float]):
        tcfg = self.cfg.training
        adv_l, rets = [], []
        for ro in rollouts:
            with torch.no_grad():
                v = self.critic(torch.as_tensor(ro.obs, dtype=torch.float32, device=self.device),
                                torch.as_tensor(ro.a_hat, dtype=torch.float32, device=self.device)).cpu().numpy()
            T, N = ro.active.shape
            r = np.repeat(ro.reward[:, None], N, axis=1)
            f = np.repeat(ro.cost_f[:, None], N, axis=1)
            a_r, ret_r = gae(r, v[..., 0], tcfg.gamma, tcfg.gae_lambda)
            a_f, ret_f = gae(f, v[..., 1], tcfg.gamma, tcfg.gae_lambda)
            a_d, ret_d = gae(ro.cost_d, v[..., 2], tcfg.gamma, tcfg.gae_lambda)
            adv_l.append(a_r - lambdas["F"] * a_f - lambdas["D"] * a_d)
            rets.append(np.stack([ret_r, ret_f, ret_d], -1))
        return np.concatenate(adv_l), np.concatenate(rets)

    def update(self, rollouts: list[Rollout], lambdas: dict[str, float], projection_gradient: bool) -> dict[str, float]:
        tcfg = self.cfg.training
        dev = self.device
        obs = torch.as_tensor(np.concatenate([r.obs for r in rollouts]), dtype=torch.float32, device=dev)
        act = torch.as_tensor(np.concatenate([r.actions for r in rollouts]), dtype=torch.float32, device=dev)
        logp_old = torch.as_tensor(np.concatenate([r.logp for r in rollouts]), dtype=torch.float32, device=dev)
        a_hat = torch.as_tensor(np.concatenate([r.a_hat for r in rollouts]), dtype=torch.float32, device=dev)
        active = np.concatenate([r.active for r in rollouts])
        mask_np = active & self.owned[None, :]
        mask = torch.as_tensor(mask_np, dtype=torch.float32, device=dev)
        groups = [g for r in rollouts for g in r.groups]
        adv_np, ret_np = self._targets(rollouts, lambdas)
        if mask_np.any():
            mu, sd = adv_np[mask_np].mean(), adv_np[mask_np].std() + 1e-8
            adv_np = (adv_np - mu) / sd
        adv = torch.as_tensor(adv_np, dtype=torch.float32, device=dev)
        ret = torch.as_tensor(ret_np, dtype=torch.float32, device=dev)
        K = obs.shape[0]
        n_mb = max(1, min(tcfg.minibatches, K))
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "projection_loss": 0.0, "updates": 0}
        use_proj = projection_gradient and any(len(g) for g in groups)
        for _ in range(tcfg.ppo_epochs):
            for mb in np.array_split(np.random.permutation(K), n_mb):
                m = mask[mb]
                denom = m.sum().clamp(min=1.0)
                dist = self.actor.distribution(obs[mb], self.class_idx)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - tcfg.clip_eps, 1 + tcfg.clip_eps) * adv[mb]
                policy_loss = -(torch.min(s1, s2) * m).sum() / denom
                entropy = (dist.entropy() * m).sum() / denom
                loss_a = policy_loss - tcfg.entropy_coef * entropy
                proj_loss = torch.zeros((), device=dev)
                if use_proj:
                    sub = np.arange(len(mb))
                    k = tcfg.projection_grad_batch
                    if k is not None and k < len(mb):
                        sub = np.sort(np.random.choice(len(mb), size=k, replace=False))  # unbiased subsample
                    mean = dist.mean[sub].clamp(-1.0, 1.0)
                    projected = differentiable_project(mean, [groups[mb[i]] for i in sub])
                    m_sub = m[sub]
                    proj_loss = self.cfg.env.beta_projection * (((projected - mean) ** 2) * m_sub).sum() \
                        / m_sub.sum().clamp(min=1.0)
                    loss_a = loss_a + proj_loss
                self.opt_actor.zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), tcfg.max_grad_norm)
                self.opt_actor.step()

                v = self.critic(obs[mb], a_hat[mb])
                value_loss = (((v - ret[mb]) ** 2) * m.unsqueeze(-1)).sum() / (3 * denom)
                self.opt_critic.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), tcfg.max_grad_norm)
                self.opt_critic.step()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["projection_loss"] += proj_loss.item()
                stats["updates"] += 1
        for k in ("policy_loss", "value_loss", "entropy", "projection_loss"):
            stats[k] /= max(stats["updates"], 1)
        stats["n_transitions"] = float(mask_np.sum())
        stats.update(self.constraint_estimates(rollouts))
        return stats

    def constraint_estimates(self, rollouts: list[Rollout]) -> dict[str, float]:
        """J_hat for the Lagrangian: fairness (1 - Jain over own sessions) and per-agent degradation."""
        owned_ids = np.flatnonzero(self.owned)
        f_vals, d_vals = [], []
        for ro in rollouts:
            met = ro.metrics
            ratios = [x for x, v in zip(met.delivered_ratio, met.delivered_vehicle, strict=True) if self.owned[v]]
            f_vals.append(1.0 - jain_index(np.array(ratios)))
            d_vals.append(float(ro.cost_d[:, owned_ids].sum(0).mean()) if len(owned_ids) else 0.0)
        return {"J_F": float(np.mean(f_vals)), "J_D": float(np.mean(d_vals))}
