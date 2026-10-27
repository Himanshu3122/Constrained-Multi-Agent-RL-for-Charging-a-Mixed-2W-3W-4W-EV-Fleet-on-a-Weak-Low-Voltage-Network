import numpy as np
import pytest

from solace.baselines import UncontrolledPolicy
from solace.env import OBS_DIM, ChargingEnv
from solace.env.charging_env import jain_index


def run(env, day, policy_fn):
    obs = env.reset(day)
    steps = 0
    while True:
        res = env.step(policy_fn(env, obs))
        obs = res.obs
        steps += 1
        if res.done:
            return res, steps


def test_reset_shapes_and_episode_length(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen)
    obs = env.reset(scen.days["train"][0])
    assert obs.shape == (len(scen.fleet), OBS_DIM)
    res, steps = run(env, scen.days["train"][0], lambda e, o: np.zeros(e.n))
    assert steps == smoke_cfg.data.steps_per_day and res.done


def test_battery_energy_update_follows_eq2(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen, use_projection=False)
    day = scen.days["train"][0]
    env.reset(day)
    while not env.active.any():
        env.step(np.zeros(env.n))
    v = int(np.flatnonzero(env.active)[0])
    e0 = env.energy[v]
    lo, hi = env.bounds()
    a = np.zeros(env.n)
    a[v] = hi[v]
    env.step(a)
    expected = e0 + env.dt * smoke_cfg.env.eta_charge * hi[v] * env.p_rated[v]
    assert env.energy[v] == pytest.approx(min(expected, env.cap[v]))


def test_bounds_respect_v2g_permission_and_floor(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen)
    env.reset(scen.days["train"][0])
    for _ in range(env.steps):
        lo, hi = env.bounds()
        assert np.all(lo[~(env.v2g_allowed & env.v2g_capable)] == 0.0)
        assert np.all(lo <= 0.0) and np.all(hi >= 0.0) and np.all(hi <= 1.0) and np.all(lo >= -1.0)
        assert np.all(lo[~env.active] == 0.0) and np.all(hi[~env.active] == 0.0)
        if env.step(lo).done:  # discharge as far as allowed
            break
    assert np.all(env.energy >= env.soc_floor * env.cap * (env.discharged > 0) - 1e-6)


def test_uncontrolled_meets_feasible_requests(smoke_cfg, scen):
    env = ChargingEnv(smoke_cfg, scen, use_projection=False)
    pol = UncontrolledPolicy()
    run(env, scen.days["test"][0], pol.act)
    s = env.metrics.summary()
    assert s["sessions"] == len(scen.days["test"][0].sessions)
    assert 0.0 <= s["unmet_energy_pct"] <= 100.0
    assert 0.0 <= s["jain"] <= 1.0


def test_projection_reduces_grid_stress_on_full_feeder():
    from solace.config import load_config
    from solace.data import build_scenarios

    cfg = load_config("configs/default.yaml", {"data": {"n_days": 1, "split": [0, 0, 1]},
                                               "feeder": {"n_transformers": 1, "households": 335},
                                               "fleet": {"2W": {"count": 155}, "3W": {"count": 55},
                                                         "4W": {"count": 40}}})
    scen = build_scenarios(cfg, np.random.default_rng(0))
    day = scen.days["test"][0]
    free = ChargingEnv(cfg, scen, use_projection=False)
    safe = ChargingEnv(cfg, scen, use_projection=True)
    run(free, day, lambda e, o: e.active.astype(float))
    run(safe, day, lambda e, o: e.active.astype(float))
    f, s = free.metrics.summary(), safe.metrics.summary()
    assert s["peak_loading_pct"] < f["peak_loading_pct"]
    assert s["voltage_violations"] <= f["voltage_violations"]
    assert s["max_voltage_deviation_pu"] <= 0.005  # residual violations come from linearization only


def test_jain_index():
    assert jain_index(np.ones(5)) == pytest.approx(1.0)
    assert jain_index(np.array([1.0, 0, 0, 0])) == pytest.approx(0.25)
    assert jain_index(np.array([])) == 1.0
