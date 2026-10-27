import numpy as np
import torch

from solace.agents import ClassActors, ConcatCritic, GATMeanFieldCritic, LocalCritic
from solace.config import AgentConfig
from solace.training.rollout import gae

D = 17


def obs_batch(B=2, N=6, active=None):
    torch.manual_seed(0)
    o = torch.randn(B, N, D)
    o[..., 0] = 1.0 if active is None else torch.as_tensor(active, dtype=torch.float32)
    return o


def test_class_actors_select_per_class_network():
    cfg = AgentConfig(actor_hidden=[16, 16])
    actors = ClassActors(D, cfg)
    cls = torch.tensor([0, 1, 2, 0, 1, 2])
    mean, std = actors(obs_batch(), cls)
    assert mean.shape == (2, 6) and std.shape == (2, 6)
    assert torch.all(mean.abs() <= 1.0)
    direct = actors.actors["3W"](obs_batch()[:, 1]).squeeze(-1)
    assert torch.allclose(mean[:, 1], direct)


def test_gat_critic_permutation_equivariant_within_transformer():
    cfg = AgentConfig(gat_hidden=16, gat_heads=4, mean_field_dim=8)
    tx = torch.tensor([0, 0, 0, 1, 1, 1])
    critic = GATMeanFieldCritic(D, cfg, tx)
    o = obs_batch()
    a = torch.rand(2, 6)
    v = critic(o, a)
    assert v.shape == (2, 6, 3)
    perm = torch.tensor([2, 0, 1, 3, 4, 5])
    v_perm = critic(o[:, perm], a[:, perm])
    assert torch.allclose(v_perm, v[:, perm], atol=1e-5)


def test_gat_critic_ignores_inactive_neighbours():
    cfg = AgentConfig(gat_hidden=16, gat_heads=4, mean_field_dim=8, use_mean_field=False)
    tx = torch.zeros(4, dtype=torch.long)
    critic = GATMeanFieldCritic(D, cfg, tx)
    o = obs_batch(B=1, N=4, active=[1, 1, 0, 0])
    a = torch.zeros(1, 4)
    v1 = critic(o, a)
    o2 = o.clone()
    o2[0, 3, 1:] += 5.0  # change an inactive agent's features
    v2 = critic(o2, a)
    assert torch.allclose(v1[0, :2], v2[0, :2], atol=1e-6)


def test_baseline_critics_shapes():
    cfg = AgentConfig(gat_hidden=16)
    assert ConcatCritic(D, 6, cfg)(obs_batch(), torch.zeros(2, 6)).shape == (2, 6, 3)
    assert LocalCritic(D, cfg)(obs_batch(), torch.zeros(2, 6)).shape == (2, 6, 3)


def test_gae_reduces_to_reward_to_go():
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.zeros((3, 1))
    adv, ret = gae(r, v, gamma=1.0, lam=1.0)
    assert np.allclose(adv[:, 0], [6.0, 5.0, 3.0])
    assert np.allclose(ret, adv)
