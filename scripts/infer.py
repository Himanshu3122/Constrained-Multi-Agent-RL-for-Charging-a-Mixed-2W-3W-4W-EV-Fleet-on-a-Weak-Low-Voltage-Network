"""Inference examples: Layer 1 on a request, and one dispatch step with Layer 2 + Layer 3.

    python scripts/infer.py intent --text "Need 80% by 7 am, discharge up to 3 kWh but not below 40%" \
        --context '{"vehicle_class": "4W", "battery_kwh": 40, "soc": 0.3, "charger_kw": 7.4,
                    "plug_in_time": "20:00", "departure_time": "09:00", "v2g_capable": true}'
    python scripts/infer.py dispatch --config configs/smoke.yaml \
        --checkpoint checkpoints/smoke/seed0/final.pt --step 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from solace.config import load_config
from solace.data import load_scenarios
from solace.env import ChargingEnv
from solace.intent import FeasibilityChecker, IntentPipeline, RuleBasedParser, SessionContext
from solace.training import LearnedPolicy


def run_intent(args: argparse.Namespace) -> None:
    ctx = SessionContext.model_validate(json.loads(args.context))
    if args.parser == "hf":
        from solace.intent import HFParser

        parser = HFParser(args.hf_model, args.hf_adapter)
    else:
        parser = RuleBasedParser()
    res = IntentPipeline(parser, FeasibilityChecker()).process(args.text, ctx)
    print(json.dumps({"status": res.status, "constraint": res.constraint.model_dump(mode="json"),
                      "requested": res.requested.model_dump(mode="json"),
                      "max_reachable_soc": round(res.max_reachable_soc, 4), "attempts": res.attempts,
                      "flagged_for_review": res.flagged_for_review, "reason": res.reason}, indent=2,
                     ensure_ascii=False))


def run_dispatch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    scen = load_scenarios(Path(args.scenarios) if args.scenarios else cfg.scenario_dir)
    policy = LearnedPolicy(args.checkpoint)
    env = ChargingEnv(cfg, scen, use_projection=True)
    day = scen.days["test"][0]
    obs = env.reset(day)
    policy.reset(env)
    for _ in range(min(args.step, env.steps - 1)):
        obs = env.step(policy.act(env, obs)).obs
    a_raw = policy.act(env, obs)
    active = np.flatnonzero(env.active)
    res = env.step(a_raw)
    rows = [{"vehicle": int(v), "class": str(scen.fleet.vehicle_class[v]),
             "raw_action": round(float(res.info["a_raw"][v]), 4), "projected_action": round(float(res.a_hat[v]), 4),
             "setpoint_kw": round(float(res.a_hat[v] * scen.fleet.charger_kw[v]), 3)} for v in active]
    print(json.dumps({"step": int(env.t - 1), "transformer_loading": env.tx_loading.round(4).tolist(),
                      "voltage_violations": res.info["violations"], "setpoints": rows}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("intent")
    pi.add_argument("--text", required=True)
    pi.add_argument("--context", required=True, help="SessionContext as JSON")
    pi.add_argument("--parser", choices=["rule", "hf"], default="rule")
    pi.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B-Instruct")
    pi.add_argument("--hf-adapter", default=None)
    pd = sub.add_parser("dispatch")
    pd.add_argument("--config", required=True)
    pd.add_argument("--checkpoint", required=True)
    pd.add_argument("--scenarios")
    pd.add_argument("--step", type=int, default=48)
    args = ap.parse_args()
    run_intent(args) if args.cmd == "intent" else run_dispatch(args)


if __name__ == "__main__":
    main()
