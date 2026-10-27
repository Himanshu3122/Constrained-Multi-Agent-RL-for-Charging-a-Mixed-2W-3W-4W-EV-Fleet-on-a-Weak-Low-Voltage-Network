"""LinDistFlow vs AC power flow error on the study feeder (revision; reviewer comment R2-16e).

For every step of the test days, uncontrolled charging is projected by Layer 3
and executed. The voltages predicted by LinDistFlow (eq. (4)) for the executed
loads are compared with the AC backward/forward sweep, and the transformer
power is compared with its lossless linear estimate. Reports error statistics
binned by transformer loading.

    python scripts/linearization_error.py --config configs/study.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from solace.baselines import UncontrolledPolicy
from solace.config import load_config
from solace.data import load_scenarios
from solace.env import ChargingEnv
from solace.grid import ac_power_flow, linear_voltage_sq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--out", default="results/linearization_error.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scen = load_scenarios(cfg.scenario_dir)
    env = ChargingEnv(cfg, scen, use_projection=True)
    pol = UncontrolledPolicy()
    loading, v_err, v_under, p_err = [], [], [], []
    for day in scen.days["test"]:
        obs = env.reset(day)
        pol.reset(env)
        done = False
        while not done:
            t = env.t
            res = env.step(pol.act(env, obs))
            p_ev = res.a_hat * env.p_rated
            for tx, net in enumerate(scen.feeder.networks):
                members = np.flatnonzero(scen.fleet.transformer == tx)
                ev_bus = np.zeros(net.n_buses)
                np.add.at(ev_bus, scen.fleet.bus[members], p_ev[members])
                p = (day.base_p_kw[tx][t] + ev_bus) / scen.feeder.s_base_kva
                q = day.base_q_kvar[tx][t] / scen.feeder.s_base_kva
                ac = ac_power_flow(net.parent, net.r_pu, net.x_pu, p, q)
                lin = np.sqrt(np.clip(linear_voltage_sq(net.R_path, net.X_path, p, q), 0.0, None))
                loading.append(ac.transformer_loading_pu / net.s_rated_pu)
                v_err.append(float(np.max(np.abs(lin[1:] - ac.v_mag[1:]))))
                v_under.append(float(np.max(lin[1:] - ac.v_mag[1:])))  # >0: linear model is optimistic
                s_lin = np.hypot(p.sum(), q.sum())
                p_err.append(float(ac.transformer_loading_pu - s_lin))
            obs, done = res.obs, res.done
    loading = np.array(loading)
    v_err = np.array(v_err)
    p_err = np.array(p_err)
    bins = [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.2)]
    table = []
    for lo, hi in bins:
        m = (loading >= lo) & (loading < hi)
        if m.any():
            table.append({"loading_range": [lo, hi], "steps": int(m.sum()),
                          "max_voltage_error_pu": float(v_err[m].max()),
                          "mean_voltage_error_pu": float(v_err[m].mean()),
                          "mean_power_gap_pu": float(p_err[m].mean()), "max_power_gap_pu": float(p_err[m].max())})
    out = {"steps": len(loading), "max_voltage_error_pu": float(v_err.max()),
           "linear_optimistic_fraction": float(np.mean(np.array(v_under) > 0)), "by_loading": table}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
