"""Runs scripts/smoke_test.py end to end (marked ``smoke``; excluded from the default run)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.smoke
def test_smoke_script(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py"), "--workdir", str(tmp_path)],
                          cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[FAIL]" not in proc.stdout
    summary = json.loads((tmp_path / "smoke_summary.json").read_text(encoding="utf-8"))
    assert set(summary["evaluation"]) == {"uncontrolled", "tou", "bandits", "solver_grounded", "mpc", "solace"}
