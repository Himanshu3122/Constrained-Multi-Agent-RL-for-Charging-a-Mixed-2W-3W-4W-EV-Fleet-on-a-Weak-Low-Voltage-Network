"""End-to-end smoke test on a tiny CPU workload (configs/smoke.yaml).

Runs: config load -> scenario build/save/load -> federated training (2 rounds)
-> checkpoint reload -> evaluation of every implemented method on one test day
-> intent pipeline on the demo corpus -> API requests via FastAPI's TestClient.
Exits non-zero on the first failure. Validates the software pipeline only.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from solace.config import load_config
from solace.data import build_scenarios, load_scenarios, save_scenarios
from solace.evaluation import run_evaluation, write_report
from solace.intent import ChargingConstraint, FeasibilityChecker, IntentPipeline, RuleBasedParser, SessionContext
from solace.intent.corpus import generate_demo_corpus
from solace.intent.metrics import intent_metrics
from solace.training import FederatedTrainer, LearnedPolicy
from solace.utils import seed_everything


def step(name: str, fn):  # noqa: ANN001, ANN201
    t0 = time.perf_counter()
    try:
        out = fn()
    except Exception as exc:  # report and stop
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
    print(f"[PASS] {name} ({time.perf_counter() - t0:.1f}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--workdir", default="results/smoke_run", help="all smoke artifacts go here")
    args = ap.parse_args()
    work = Path(args.workdir)
    cfg = step("load config", lambda: load_config(args.config))
    rng = seed_everything(cfg.seed)
    scen = step("build scenarios", lambda: build_scenarios(cfg, rng))
    step("save scenarios", lambda: save_scenarios(scen, work / "scenarios"))
    scen = step("load scenarios", lambda: load_scenarios(work / "scenarios"))
    trainer = step("init trainer", lambda: FederatedTrainer(cfg, scen, out_dir=work / "checkpoints"))
    history = step("train (federated, smoke rounds)", trainer.train)
    ckpt = work / "checkpoints" / "final.pt"
    step("reload checkpoint", lambda: LearnedPolicy(ckpt))
    methods = ["uncontrolled", "tou", "bandits", "solver_grounded", "mpc", "solace"]
    results = step("evaluate all methods", lambda: run_evaluation(cfg, scen, methods, [cfg.seed],
                                                                   {"solace": str(ckpt)}, "test", 1))
    step("write report", lambda: write_report(results, work / "evaluation", "Smoke evaluation (tiny synthetic data)"))

    def intent_check() -> dict:
        pipe = IntentPipeline(RuleBasedParser(), FeasibilityChecker())
        recs = generate_demo_corpus(2, seed=cfg.seed)
        preds, golds = [], []
        for r in recs:
            res = pipe.process(r["text"], SessionContext.model_validate(r["context"]))
            preds.append(res.requested)
            golds.append(ChargingConstraint.model_validate(r["constraint"]))
        return intent_metrics(preds, golds)

    intent = step("intent pipeline on demo corpus", intent_check)

    def api_check() -> None:
        from fastapi.testclient import TestClient

        from solace.api import create_app

        client = TestClient(create_app(RuleBasedParser()))
        assert client.get("/health").json()["status"] == "ok"
        ctx = {"vehicle_class": "4W", "battery_kwh": 40, "soc": 0.3, "charger_kw": 7.4, "plug_in_time": "20:00",
               "departure_time": "09:00", "v2g_capable": True}
        r = client.post("/v1/intent", json={"text": "Need 80% by 7 am", "context": ctx})
        assert r.status_code == 200, r.text
        net = scen.feeder.networks[0]
        body = {"network": {"parent": net.parent.tolist(), "r_pu": net.r_pu.tolist(), "x_pu": net.x_pu.tolist(),
                            "s_base_kva": scen.feeder.s_base_kva},
                "base_p_kw": [0.0] + [5.0] * (net.n_buses - 1), "base_q_kvar": [0.0] + [1.5] * (net.n_buses - 1),
                "evs": [{"bus": 1, "rated_kw": 7.4, "lo": -1.0, "hi": 1.0, "raw_action": 1.0}]}
        r = client.post("/v1/dispatch", json=body)
        assert r.status_code == 200, r.text

    step("API (TestClient)", api_check)
    summary = {
        "rounds": len(history),
        "final_round": history[-1],
        "evaluation": {m: {k: v["mean"] for k, v in res["aggregate"].items()} for m, res in results.items()},
        "intent_demo_metrics": {k: intent[k] for k in ("field_f1", "exact_match", "intent_type_accuracy", "n")},
        "note": "Tiny synthetic smoke workload; numbers validate the pipeline, not the paper.",
    }
    (work / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"smoke test finished; summary in {work / 'smoke_summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
