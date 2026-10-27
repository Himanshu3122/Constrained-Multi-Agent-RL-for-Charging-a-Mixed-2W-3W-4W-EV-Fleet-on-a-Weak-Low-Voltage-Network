import numpy as np
import torch

from solace.data import load_scenarios, save_scenarios
from solace.env import ChargingEnv
from solace.training import FederatedTrainer, LearnedPolicy, load_checkpoint


def test_federated_training_checkpoint_roundtrip(smoke_cfg, scen, tmp_path):
    save_scenarios(scen, tmp_path / "scen")
    loaded = load_scenarios(tmp_path / "scen")
    assert len(loaded.fleet) == len(scen.fleet)
    assert len(loaded.days["test"][0].sessions) == len(scen.days["test"][0].sessions)

    torch.manual_seed(0)
    trainer = FederatedTrainer(smoke_cfg, loaded, out_dir=tmp_path / "ckpt")
    assert len(trainer.learners) == smoke_cfg.n_aggregators  # 2 aggregators on 1 transformer
    before = {k: v.clone() for k, v in trainer.global_actor.state_dict().items()}
    history = trainer.train()
    assert len(history) == smoke_cfg.training.rounds
    rec = history[-1]
    assert rec["lambda_F"] >= 0 and rec["lambda_D"] >= 0
    assert "projection_loss" in rec and rec["projection_loss"] >= 0
    after = trainer.global_actor.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)  # FedAvg produced new parameters

    ckpt = load_checkpoint(tmp_path / "ckpt" / "final.pt")
    assert ckpt["round"] == smoke_cfg.training.rounds and ckpt["degradation_reference"] > 0
    policy = LearnedPolicy(tmp_path / "ckpt" / "final.pt")
    env = ChargingEnv(smoke_cfg, loaded)
    obs = env.reset(loaded.days["test"][0])
    policy.reset(env)
    a = policy.act(env, obs)
    assert a.shape == (env.n,) and np.all(np.abs(a) <= 1.0)
    assert np.all(a[~env.active] == 0.0)


def test_fed_average_is_weighted_mean():
    from solace.training.federated import fed_average

    a, b = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    with torch.no_grad():
        for p in a.parameters():
            p.fill_(1.0)
        for p in b.parameters():
            p.fill_(5.0)
    avg = fed_average([a, b], [1.0, 3.0])
    assert all(torch.allclose(v, torch.full_like(v, 4.0)) for v in avg.values())  # 0.25*1 + 0.75*5


def test_dual_step_normalized_and_capped(smoke_cfg, scen):
    tcfg = smoke_cfg.training.model_copy(update={"dual_lr": 0.5, "dual_normalize": True, "lambda_max": 1.0})
    trainer = FederatedTrainer(smoke_cfg.model_copy(update={"training": tcfg}), scen)
    assert trainer._dual_step(0.0, 0.12, 0.06) == 0.5  # (0.12 - 0.06) / 0.06 = 1.0 -> 0.5
    assert trainer._dual_step(0.9, 0.12, 0.06) == 1.0  # capped
    assert trainer._dual_step(0.2, 0.0, 0.06) == 0.0  # projected onto lambda >= 0


def test_centralized_ablation_single_learner(smoke_cfg, scen):
    cfg = smoke_cfg.model_copy(update={"training": smoke_cfg.training.model_copy(update={"federated": False})})
    trainer = FederatedTrainer(cfg, scen)
    assert len(trainer.learners) == 1
    trainer.train()
