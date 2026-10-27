"""Deterministic seeding for numpy, Python and torch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed all RNGs and return a numpy Generator for explicit use.

    Torch is imported lazily so that non-learning code paths do not need it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return np.random.default_rng(seed)
