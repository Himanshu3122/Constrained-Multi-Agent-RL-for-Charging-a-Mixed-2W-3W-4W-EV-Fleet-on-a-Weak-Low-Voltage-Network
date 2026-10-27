"""Linearized DistFlow (paper eqs. (3)-(4)).

For a radial network with slack voltage v_0 = 1 (squared magnitude), summing
eq. (4) along the path from the slack gives

    v_k = 1 - 2 * sum_m ( R[k, m] * p_m + X[k, m] * q_m ),

where p_m, q_m are the net loads at bus m (consumption positive) and
R[k, m] (X[k, m]) is the resistance (reactance) of the path segment shared by
the paths from the slack to k and to m. These "path impedance" matrices make
voltages linear in EV powers, which keeps the safety projection a QP.
"""

from __future__ import annotations

import numpy as np


def _ancestors(parent: np.ndarray, k: int) -> list[int]:
    """Buses on the path from the slack to ``k`` (excluding the slack)."""
    path = []
    while k > 0:
        path.append(k)
        k = int(parent[k])
    return path


def path_impedance_matrices(parent: np.ndarray, r: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, X) with R[k, m] = sum of r over lines common to paths 0->k and 0->m."""
    n = len(parent)
    paths = [set(_ancestors(parent, k)) for k in range(n)]
    R = np.zeros((n, n))
    X = np.zeros((n, n))
    for k in range(1, n):
        for m in range(k, n):
            common = paths[k] & paths[m]
            if common:
                idx = list(common)
                R[k, m] = R[m, k] = r[idx].sum()
                X[k, m] = X[m, k] = x[idx].sum()
    return R, X


def linear_voltage_sq(R: np.ndarray, X: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Squared voltage magnitudes under LinDistFlow for bus loads p, q (p.u.)."""
    return 1.0 - 2.0 * (R @ p + X @ q)
