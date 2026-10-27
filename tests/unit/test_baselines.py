import numpy as np

from solace.baselines import (
    CentralizedMPCPolicy,
    FairMPCPolicy,
    SolverGroundedPolicy,
    ThompsonBanditPolicy,
    ToUPolicy,
)
from solace.baselines.horizon_qp import linear_wear_cost
from solace.env import ChargingEnv


def play(env, day, policy):
    obs = env.reset(day)
    policy.reset(env)
    actions = []
    while True:
        a = policy.act(env, obs)
        actions.append(a.copy())
        res = env.step(a)
        obs = res.obs
        if res.done:
            return np.array(actions)


def test_tou_does_not_charge_at_peak_when_slack_allows(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen, use_projection=False)
    pol = ToUPolicy()
    acts = play(env, scen.days["train"][0], pol)
    tariff = scen.days["train"][0].tariff
    assert np.all(np.isin(acts, [0.0, 1.0]))
    assert acts[tariff == tariff.min()].sum() >= acts[tariff == tariff.max()].sum()


def test_bandits_learn_state_persists(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen, use_projection=False)
    pol = ThompsonBanditPolicy(env.n, seed=0)
    play(env, scen.days["train"][0], pol)
    assert (pol.alpha + pol.beta).sum() > 4 * env.n  # posteriors were updated


def test_horizon_qp_actions_feasible(smoke_cfg, scen):
    for policy in (CentralizedMPCPolicy(smoke_cfg), FairMPCPolicy(smoke_cfg), SolverGroundedPolicy(smoke_cfg, scen)):
        env = ChargingEnv(smoke_cfg, scen, use_projection=True)
        acts = play(env, scen.days["test"][0], policy)
        assert np.all(np.abs(acts) <= 1.0 + 1e-9)
        assert env.metrics.summary()["unmet_energy_pct"] < 50.0


def test_solver_grounded_first_day_forecast_has_no_test_day_information(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen)
    pol = SolverGroundedPolicy(smoke_cfg, scen)
    day = scen.days["test"][0]
    env.reset(day)
    pol.reset(env)
    env.t = 40
    p, _ = pol._base_forecast(env, 0)
    train_mean = np.mean([d.base_p_kw[0] for d in scen.days["train"]], axis=0)
    assert np.allclose(p[41:], train_mean[41:])  # forecast = training-split mean, not the true future
    assert not np.allclose(p[41:], day.base_p_kw[0][41:])
    assert np.allclose(p[40], day.base_p_kw[0][40])  # current step is measured


def test_fair_mpc_objective_includes_fairness_weight(smoke_cfg):
    assert FairMPCPolicy(smoke_cfg).fairness_weight > 0
    assert CentralizedMPCPolicy(smoke_cfg).fairness_weight == 0


def test_linear_wear_cost_positive_and_age_dependent(smoke_cfg):
    young = linear_wear_cost(smoke_cfg.model_copy(update={"degradation": smoke_cfg.degradation.model_copy(
        update={"initial_equivalent_cycles": 10.0})}), "NMC")
    old = linear_wear_cost(smoke_cfg, "NMC")
    assert young > old > 0.0
