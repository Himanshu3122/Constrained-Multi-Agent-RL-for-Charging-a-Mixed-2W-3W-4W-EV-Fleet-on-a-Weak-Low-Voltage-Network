"""Actor and critic networks (paper Sections III-C and V-F).

* ``ClassActors``: one Gaussian actor per vehicle class (2W, 3W, 4W) with
  parameters shared across all vehicles of the class (paper: three actor
  networks regardless of fleet size). MLP 2x128, tanh (paper Table III).
* ``GATMeanFieldCritic``: eq. (12)-(13). Graph attention over EVs under the same
  transformer, a mean-field summary (mean projected action) of all other EVs,
  and global features g_t (transformer loading, tariff, time). It outputs
  three values per agent: reward, fairness cost and degradation cost, used for
  the Lagrangian advantage (eq. (14)).
* ``ConcatCritic``: MAPPO baseline critic on the concatenated state of all
  agents; its input grows with the fleet size.
* ``LocalCritic``: IPPO baseline (own observation only).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from solace.config import VEHICLE_CLASSES, AgentConfig

N_VALUE_HEADS = 3  # reward, fairness cost, degradation cost
GLOBAL_FEATURES = [11, 12, 13, 14, 15]  # tariff, PV, transformer loading, time (sin, cos)


def mlp(sizes: list[int], activation: type[nn.Module] = nn.Tanh,
        out_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    if out_activation is not None:
        layers.append(out_activation)
    return nn.Sequential(*layers)


class ClassActors(nn.Module):
    def __init__(self, obs_dim: int, cfg: AgentConfig):
        super().__init__()
        self.actors = nn.ModuleDict({c: mlp([obs_dim, *cfg.actor_hidden, 1], out_activation=nn.Tanh())
                                     for c in VEHICLE_CLASSES})
        self.log_std = nn.ParameterDict({c: nn.Parameter(torch.tensor(cfg.init_log_std)) for c in VEHICLE_CLASSES})

    def forward(self, obs: torch.Tensor, class_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs (..., N, D), class_idx (N,) -> mean, std of shape (..., N)."""
        means = torch.stack([self.actors[c](obs).squeeze(-1) for c in VEHICLE_CLASSES], dim=-1)
        stds = torch.stack([self.log_std[c].exp() for c in VEHICLE_CLASSES])
        idx = class_idx.expand(means.shape[:-1]).unsqueeze(-1)
        mean = torch.gather(means, -1, idx).squeeze(-1)
        std = stds[class_idx].expand_as(mean)
        return mean, std

    def distribution(self, obs: torch.Tensor, class_idx: torch.Tensor) -> Normal:
        mean, std = self(obs, class_idx)
        return Normal(mean, std)


class GATMeanFieldCritic(nn.Module):
    def __init__(self, obs_dim: int, cfg: AgentConfig, transformer: torch.Tensor):
        super().__init__()
        h = cfg.gat_hidden
        if h % cfg.gat_heads:
            raise ValueError("gat_hidden must be divisible by gat_heads")
        self.heads = cfg.gat_heads
        self.head_dim = h // cfg.gat_heads
        self.use_mean_field = cfg.use_mean_field
        self.embed = mlp([obs_dim, h, h])
        self.W = nn.Linear(h, h, bias=False)
        self.att_src = nn.Parameter(torch.randn(self.heads, self.head_dim) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(self.heads, self.head_dim) * 0.1)
        self.mf = nn.Linear(2, cfg.mean_field_dim)
        in_dim = h + h + (cfg.mean_field_dim if cfg.use_mean_field else 0) + len(GLOBAL_FEATURES)
        self.head = mlp([in_dim, h, N_VALUE_HEADS])
        self.register_buffer("transformer", transformer.long())

    def forward(self, obs: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        """obs (B, N, D), a_hat (B, N) -> values (B, N, 3)."""
        B, N, _ = obs.shape
        active = obs[..., 0] > 0.5
        h = self.embed(obs)
        wh = self.W(h).view(B, N, self.heads, self.head_dim)
        agg = torch.zeros_like(h)
        mf_in = torch.zeros(B, N, 2, device=obs.device)
        for tx in torch.unique(self.transformer):
            idx = torch.nonzero(self.transformer == tx).squeeze(-1)
            w_t = wh[:, idx]  # (B, n, H, d)
            src = (w_t * self.att_src).sum(-1)  # (B, n, H)
            dst = (w_t * self.att_dst).sum(-1)
            e = nn.functional.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), 0.2)  # (B, n_i, n_j, H)
            mask = active[:, idx]
            pair = mask.unsqueeze(2) & mask.unsqueeze(1)
            eye = torch.eye(len(idx), dtype=torch.bool, device=obs.device).unsqueeze(0)
            pair = pair | eye  # every agent attends at least to itself
            e = e.masked_fill(~pair.unsqueeze(-1), float("-inf"))
            alpha = torch.softmax(e, dim=2)
            out = torch.einsum("bijh,bjhd->bihd", alpha, w_t).reshape(B, len(idx), -1)
            agg[:, idx] = out
            if self.use_mean_field:
                others = (self.transformer != tx)
                act_o = (active[:, others]).float()
                count = act_o.sum(-1, keepdim=True)
                mean_a = (a_hat[:, others] * act_o).sum(-1, keepdim=True) / count.clamp(min=1.0)
                frac = count / max(int(others.sum()), 1)
                mf_in[:, idx] = torch.cat([mean_a, frac], -1).unsqueeze(1).expand(B, len(idx), 2)
        parts = [h, agg]
        if self.use_mean_field:
            parts.append(torch.tanh(self.mf(mf_in)))
        parts.append(obs[..., GLOBAL_FEATURES])
        return self.head(torch.cat(parts, -1))


class ConcatCritic(nn.Module):
    """Centralized critic on the concatenation of all agents' observations (MAPPO baseline)."""

    def __init__(self, obs_dim: int, n_agents: int, cfg: AgentConfig):
        super().__init__()
        self.n = n_agents
        self.net = mlp([obs_dim * n_agents, cfg.gat_hidden, cfg.gat_hidden, n_agents * N_VALUE_HEADS])

    def forward(self, obs: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        return self.net(obs.reshape(B, -1)).view(B, self.n, N_VALUE_HEADS)


class LocalCritic(nn.Module):
    """Decentralized critic on the agent's own observation (IPPO baseline)."""

    def __init__(self, obs_dim: int, cfg: AgentConfig):
        super().__init__()
        self.net = mlp([obs_dim, cfg.gat_hidden, cfg.gat_hidden, N_VALUE_HEADS])

    def forward(self, obs: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def build_critic(obs_dim: int, n_agents: int, cfg: AgentConfig, transformer: torch.Tensor) -> nn.Module:
    if cfg.critic_type == "gat_mf":
        return GATMeanFieldCritic(obs_dim, cfg, transformer)
    if cfg.critic_type == "concat":
        return ConcatCritic(obs_dim, n_agents, cfg)
    return LocalCritic(obs_dim, cfg)
