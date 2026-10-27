"""Train SOLACE (federated constrained MAPPO, Algorithm 1) or a learned baseline.

    python scripts/train.py --config configs/smoke.yaml --seed 0
    python scripts/train.py --config configs/baselines/mappo.yaml --seed 0

Outputs <checkpoint_dir>/<name>/seed<seed>/{latest.pt, final.pt, history.json, config.yaml, metadata.json}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from solace.config import load_config
from solace.data import load_scenarios
from solace.training import FederatedTrainer
from solace.utils import RunDirectory, get_logger, seed_everything
from solace.utils.logging import kv

log = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="overrides config.seed")
    ap.add_argument("--scenarios", help="scenario directory (default: <data_dir>/<scenario_set or name>)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    cfg = cfg.model_copy(update={"seed": seed})
    seed_everything(seed)
    scen_dir = Path(args.scenarios) if args.scenarios else cfg.scenario_dir
    scen = load_scenarios(scen_dir)
    run = RunDirectory.create(Path(cfg.checkpoint_dir) / cfg.name, f"seed{seed}", cfg)
    trainer = FederatedTrainer(cfg, scen, out_dir=run.path, seed=seed)
    history = trainer.train()
    run.write_json("history.json", history)
    log.info(kv(event="done", checkpoint=str(run.path / "final.pt"), rounds=len(history)))


if __name__ == "__main__":
    main()
