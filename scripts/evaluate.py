"""Evaluate methods on the held-out test days (paper Section VII metrics).

    python scripts/evaluate.py --config configs/smoke.yaml \
        --checkpoint "checkpoints/smoke/seed{seed}/final.pt"

``--checkpoint`` is used by learned methods (``solace``); ``{seed}`` is replaced
by each evaluation seed. Extra learned methods can be added with
``--learned NAME=PATH`` (e.g. ``mappo=checkpoints/mappo/seed{seed}/final.pt``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from solace.config import load_config
from solace.data import load_scenarios
from solace.evaluation import METHODS, run_evaluation, write_report
from solace.evaluation.runner import register_learned
from solace.utils import RunDirectory, get_logger
from solace.utils.logging import kv

log = get_logger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="*", help="default: evaluation.methods from the config")
    ap.add_argument("--checkpoint", help="SOLACE checkpoint path (may contain {seed})")
    ap.add_argument("--learned", nargs="*", default=[], help="extra learned methods NAME=PATH")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--scenarios", help="scenario directory (default: <data_dir>/<name>)")
    ap.add_argument("--out", help="results directory (default: <results_dir>/<name>)")
    ap.add_argument("--keep-traces", action="store_true", help="store per-step transformer loading per episode")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scen = load_scenarios(Path(args.scenarios) if args.scenarios else cfg.scenario_dir)
    methods = list(args.methods or cfg.evaluation.methods)
    checkpoints: dict[str, str] = {}
    if args.checkpoint:
        checkpoints["solace"] = args.checkpoint
    for item in args.learned:
        name, _, path = item.partition("=")
        if not path:
            ap.error(f"--learned expects NAME=PATH, got '{item}'")
        register_learned(name)
        checkpoints[name] = path
        methods.append(name)
    missing = [m for m in methods if METHODS.get(m) and METHODS[m].needs_checkpoint and m not in checkpoints]
    if missing:
        log.warning(kv(event="skipping_methods_without_checkpoint", methods=",".join(missing)))
        methods = [m for m in methods if m not in missing]
    results = run_evaluation(cfg, scen, methods, cfg.evaluation.seeds, checkpoints, args.split,
                             args.max_days or cfg.evaluation.max_test_days, keep_traces=args.keep_traces)
    run = RunDirectory.create(Path(args.out or Path(cfg.results_dir)), cfg.name, cfg)
    report = write_report(results, run.path, f"Evaluation: {cfg.name} ({args.split} split)")
    log.info(kv(event="done", report=str(report)))


if __name__ == "__main__":
    main()
