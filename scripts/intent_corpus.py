"""Generate the DEMO trilingual corpus and evaluate a parser on a corpus file.

    python scripts/intent_corpus.py generate --n-per-cell 5 --out data/corpus/demo.jsonl
    python scripts/intent_corpus.py evaluate --corpus data/corpus/demo.jsonl --parser rule

The demo corpus is not the paper's corpus (which is not released).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from solace.intent import ChargingConstraint, FeasibilityChecker, IntentPipeline, RuleBasedParser, SessionContext
from solace.intent.corpus import generate_demo_corpus, load_jsonl, save_jsonl
from solace.intent.metrics import intent_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n-per-cell", type=int, default=5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/corpus/demo.jsonl")
    e = sub.add_parser("evaluate")
    e.add_argument("--corpus", required=True)
    e.add_argument("--parser", choices=["rule", "hf"], default="rule")
    e.add_argument("--hf-model", default="meta-llama/Llama-3.1-8B-Instruct")
    e.add_argument("--hf-adapter", default=None)
    e.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()
    if args.cmd == "generate":
        path = save_jsonl(generate_demo_corpus(args.n_per_cell, args.seed), args.out)
        print(f"wrote {path} (demo data)")
        return
    if args.parser == "hf":
        from solace.intent import HFParser

        parser = HFParser(args.hf_model, args.hf_adapter)
    else:
        parser = RuleBasedParser()
    pipe = IntentPipeline(parser, FeasibilityChecker())
    preds, golds, counter = [], [], 0
    for rec in load_jsonl(args.corpus):
        res = pipe.process(rec["text"], SessionContext.model_validate(rec["context"]))
        preds.append(res.requested)
        golds.append(ChargingConstraint.model_validate(rec["constraint"]))
        counter += res.status == "counter_offer"
    metrics = intent_metrics(preds, golds)
    metrics["counter_offer_rate"] = counter / len(preds)
    metrics["parser"] = args.parser
    text = json.dumps(metrics, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
