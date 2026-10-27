"""Run directories with config snapshots and environment metadata."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solace.config import ExperimentConfig, dump_config


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for mod in ("numpy", "scipy", "torch", "osqp"):
        try:
            meta[mod] = __import__(mod).__version__
        except ImportError:
            meta[mod] = None
    return meta


@dataclass
class RunDirectory:
    """``<root>/<name>/`` holding ``config.yaml``, ``metadata.json`` and outputs."""

    path: Path

    @classmethod
    def create(cls, root: str | Path, name: str, cfg: ExperimentConfig) -> RunDirectory:
        path = Path(root) / name
        path.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, path / "config.yaml")
        meta = environment_metadata()
        meta["git_commit"] = _git_commit(Path.cwd())
        meta["seed"] = cfg.seed
        (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return cls(path)

    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path / name
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
