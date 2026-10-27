"""AC power flow by backward/forward sweep on a radial network.

The paper computes AC power flow with pandapower and reports violations
"under full AC power flow". This module implements an equivalent exact
(non-linearized) radial power flow without the pandapower dependency, for the
single-phase-equivalent networks built in ``solace.grid.feeder``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PowerFlowResult:
    v_mag: np.ndarray  # |V| per bus (p.u.), bus 0 = slack
    s_slack: complex  # complex power drawn through the transformer (p.u.)
    converged: bool
    iterations: int

    @property
    def transformer_loading_pu(self) -> float:
        return abs(self.s_slack)


class PowerFlowError(RuntimeError):
    """Raised when the sweep does not converge (e.g., voltage collapse)."""


def ac_power_flow(
    parent: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    p_load: np.ndarray,
    q_load: np.ndarray,
    v_slack: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 100,
    raise_on_divergence: bool = False,
) -> PowerFlowResult:
    """Solve a radial network with constant-power loads (consumption positive)."""
    n = len(parent)
    z = r + 1j * x
    s = p_load + 1j * q_load
    v = np.full(n, complex(v_slack))
    children_order = list(range(n - 1, 0, -1))  # parent[k] < k, so reverse order is leaves-first
    converged = False
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        i_inj = np.conj(s / v)
        i_branch = i_inj.copy()
        for k in children_order:
            i_branch[parent[k]] += i_branch[k]
        v_new = v.copy()
        v_new[0] = v_slack
        for k in range(1, n):
            v_new[k] = v_new[parent[k]] - z[k] * i_branch[k]
        if not np.all(np.isfinite(v_new)) or np.min(np.abs(v_new)) < 0.3:
            break
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < tol:
            converged = True
            break
    if not converged and raise_on_divergence:
        raise PowerFlowError(f"backward/forward sweep did not converge after {iterations} iterations")
    # Power through the transformer = loads + line losses = V0 * conj(I0).
    i_branch = np.conj(s / v)
    for k in children_order:
        i_branch[parent[k]] += i_branch[k]
    s_slack = complex(v_slack * np.conj(i_branch[0]))
    return PowerFlowResult(np.abs(v), s_slack, converged, iterations)
