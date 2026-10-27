"""Structured logging helpers (key=value records on stderr)."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        root = logging.getLogger("solace")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("solace") else f"solace.{name}")


def kv(**fields: object) -> str:
    """Format fields as ``key=value`` pairs for log messages."""
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)
