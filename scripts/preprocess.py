"""Build the feeder, fleet and day scenarios (train/val/test) and save them.

    python scripts/preprocess.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from solace.config import load_config
from solace.data import build_scenarios, save_scenarios
from solace.utils import get_logger, seed_everything
from solace.utils.logging import kv

log = get_logger("preprocess")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="output directory (default: <data_dir>/<name>)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = seed_everything(cfg.seed)
    scen = build_scenarios(cfg, rng)
    out = Path(args.out) if args.out else cfg.scenario_dir
    save_scenarios(scen, out)
    log.info(kv(event="saved", out=str(out), vehicles=len(scen.fleet), buses=scen.feeder.total_buses,
                train=len(scen.days["train"]), val=len(scen.days["val"]), test=len(scen.days["test"])))


if __name__ == "__main__":
    main()
