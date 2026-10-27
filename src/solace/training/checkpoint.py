"""Checkpoint save/load and the deployable learned policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from solace.agents.networks import ClassActors
from solace.config import ExperimentConfig
from solace.env.charging_env import OBS_DIM, ChargingEnv

CHECKPOINT_VERSION = 1


def save_checkpoint(path: str | Path, actor: torch.nn.Module, critic: torch.nn.Module, cfg: ExperimentConfig,
                    lambdas: dict[str, float], round_idx: int, degradation_reference: float,
                    history: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": CHECKPOINT_VERSION,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "config": cfg.model_dump(mode="json"),
        "lambdas": dict(lambdas),
        "round": round_idx,
        "degradation_reference": degradation_reference,
        "history": history,
    }, path)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {ckpt.get('version')} in {path}")
    return ckpt


class LearnedPolicy:
    """Deterministic execution of the per-class actors (decentralized execution).

    The actor uses only each agent's local observation; Layer 3 (inside the
    environment or the dispatch API) projects the joint action.
    """

    def __init__(self, checkpoint: str | Path, name: str = "solace"):
        ckpt = load_checkpoint(checkpoint)
        self.cfg = ExperimentConfig.model_validate(ckpt["config"])
        self.actor = ClassActors(OBS_DIM, self.cfg.agent)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.name = name
        self.use_projection = self.cfg.env.use_projection
        self._class_idx: torch.Tensor | None = None

    def reset(self, env: ChargingEnv) -> None:
        self._class_idx = torch.as_tensor(env.fleet.class_index(), dtype=torch.long)

    @torch.no_grad()
    def act(self, env: ChargingEnv, obs: np.ndarray) -> np.ndarray:
        if self._class_idx is None:
            self.reset(env)
        mean, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32), self._class_idx)
        return mean.numpy() * env.active
