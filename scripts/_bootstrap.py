"""Make ``src/`` importable when the package is not installed (``pip install -e .`` is preferred)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import solace  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
