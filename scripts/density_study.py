"""Fixed-feeder EV-density study of Layer 3 (revision; reviewer comment R1-06).

The feeder (topology, impedances, households, transformer rating) is held fixed
while the number of EVs is scaled. For each density, uncontrolled charging
requests are executed (i) without and (ii) with the safety projection, so the
study isolates what Layer 3 does as grid stress grows. Reports peak loading,
AC voltage violations, unmet energy, Jain index, relaxed steps and projection
latency per 15-min step.

    python scripts/density_study.py --config configs/study.yaml --factors 0.5 1 1.5 2 3 --days 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from solace.baselines import UncontrolledPolicy
from solace.config import load_config
from solace.data import build_scenarios
from solace.env import ChargingEnv


def run(cfg, scen, use_projection: bool) -> dict[str, float]:
    env = ChargingEnv(cfg, scen, use_projection=use_projection)
    pol = UncontrolledPolicy()
    rows = []
    for day in scen.days["test"]:
        obs = env.reset(day)
        pol.reset(env)
        done = False
        while not done:
            res = env.step(pol.act(env, obs))
            obs, done = res.obs, res.done
        rows.append(env.metrics.summary())
    keys = ("peak_loading_pct", "voltage_violations", "max_voltage_deviation_pu", "unmet_energy_pct", "jain",
            "relaxed_steps", "projection_latency_ms")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {
        "peak_loading_pct_max": float(np.max([r["peak_loading_pct"] for r in rows]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--factors", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/density_study.json")
    args = ap.parse_args()
    base = load_config(args.config)
    out = []
    for f in args.factors:
        fleet = {c: {"count": max(1, round(base.fleet[c].count * f))} for c in base.fleet}
        cfg = load_config(args.config, {"fleet": fleet, "data": {"n_days": args.days, "split": [0, 0, args.days]}})
        scen = build_scenarios(cfg, np.random.default_rng(args.seed))
        n_ev = len(scen.fleet)
        row = {"factor": f, "n_ev": n_ev, "without_projection": run(cfg, scen, False),
               "with_projection": run(cfg, scen, True)}
        out.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
