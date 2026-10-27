"""Train every learned method of the reduced-scale study (configs/study*.yaml) for several seeds.

    python scripts/preprocess.py --config configs/study.yaml
    python scripts/run_study.py --seeds 0 1 2 --parallel 4

Each run is ``scripts/train.py --config <cfg> --seed <s>``; logs go to results/study_logs/.
Runs are launched in parallel with a bounded number of worker processes, each limited to
``--threads`` torch/BLAS threads.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["configs/study.yaml", "configs/study_nofair.yaml", "configs/study_mappo.yaml",
           "configs/study_ppo_proj.yaml"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    logs = ROOT / "results" / "study_logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    queue = [(c, s) for s in args.seeds for c in args.configs]
    running: list[tuple[subprocess.Popen, str, float]] = []
    failures = 0
    while queue or running:
        while queue and len(running) < args.parallel:
            cfg, seed = queue.pop(0)
            tag = f"{Path(cfg).stem}_seed{seed}"
            fh = (logs / f"{tag}.log").open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "scripts/train.py", "--config", cfg, "--seed", str(seed)],
                                    cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((proc, tag, time.time()))
            print(f"started {tag}", flush=True)
        time.sleep(5)
        for item in list(running):
            proc, tag, t0 = item
            if proc.poll() is not None:
                running.remove(item)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                failures += proc.returncode != 0
                print(f"finished {tag}: {status} after {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
