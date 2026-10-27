"""Differentiable safety projection (paper eq. (11) and Proposition 1).

    a_hat = argmin_a ||a - a_tilde||^2   s.t.   lo <= a <= hi,   A a <= b

* Box bounds encode per-EV power limits, the SoC capacity, the user's SoC
  floor, the remaining discharge allowance and V2G permission.
* Linear rows encode LinDistFlow bus voltages within [v_min, v_max] and a
  ``polygon_sides``-gon inner approximation of the transformer kVA limit.

The paper solves the projection with cvxpylayers during training and with a
warm-started OSQP solve at inference. This module uses OSQP for the forward
solve in both cases and computes gradients by differentiating the KKT
conditions of the active set (the same implicit-differentiation idea as
OptNet/cvxpylayers), avoiding the cvxpylayers/diffcp dependency.

Proposition 1 requires a = 0 to be feasible (non-EV load within limits). When
that assumption fails, rows violated at a = 0 are relaxed to "EVs may not make
it worse" (A_j a <= 0) and the problem is flagged ``relaxed`` (assumption A10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # OSQP is the paper's inference solver; a pure-numpy fallback keeps the code usable without it.
    import osqp
    import scipy.sparse as sp

    _HAVE_OSQP = True
except ImportError:  # pragma: no cover
    _HAVE_OSQP = False

_TOL = 1e-6


@dataclass
class ProjectionProblem:
    lo: np.ndarray
    hi: np.ndarray
    A: np.ndarray  # (m, n)
    b: np.ndarray  # (m,)
    relaxed: bool = False

    @property
    def n(self) -> int:
        return len(self.lo)


@dataclass
class ProjectionGroup:
    """Agents (indices into the joint action vector) sharing one transformer problem."""

    agent_idx: np.ndarray
    problem: ProjectionProblem


def build_transformer_problem(
    R_path: np.ndarray,
    bus: np.ndarray,
    p_rated_pu: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    base_p_pu: np.ndarray,
    base_q_pu: np.ndarray,
    X_path: np.ndarray,
    s_rated_pu: float,
    v_min: float,
    v_max: float,
    polygon_sides: int,
) -> ProjectionProblem:
    """Assemble eq. (11) for the EVs under one transformer.

    ``bus`` gives each EV's bus, ``p_rated_pu`` maps the normalized action
    a in [-1, 1] to power (p = a * p_rated, consumption positive). EVs are
    modeled at unity power factor (assumption A2).
    """
    lo = np.minimum(lo, 0.0)
    hi = np.maximum(hi, 0.0)
    v_base = 1.0 - 2.0 * (R_path @ base_p_pu + X_path @ base_q_pu)
    sens = 2.0 * R_path[1:, :][:, bus] * p_rated_pu[None, :]  # dv/da for buses 1..n
    vb = v_base[1:]
    A_rows = [sens, -sens]
    b_rows = [vb - v_min**2, v_max**2 - vb]
    p0 = base_p_pu.sum()
    q0 = base_q_pu.sum()
    phi = 2.0 * np.pi * np.arange(polygon_sides) / polygon_sides
    radius = s_rated_pu * np.cos(np.pi / polygon_sides)  # inscribed polygon -> inner approximation
    coef = np.cos(phi)[:, None] * p_rated_pu[None, :]
    keep = np.abs(np.cos(phi)) > 1e-9
    A_rows.append(coef[keep])
    b_rows.append((radius - np.cos(phi) * p0 - np.sin(phi) * q0)[keep])
    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    violated = b < 0.0
    relaxed = bool(violated.any())
    if relaxed:
        b = np.where(violated, 0.0, b)
    return ProjectionProblem(lo=lo, hi=hi, A=A, b=b, relaxed=relaxed)


def _solve_osqp(a_raw: np.ndarray, prob: ProjectionProblem, warm: np.ndarray | None) -> np.ndarray:
    n = prob.n
    P = sp.identity(n, format="csc")
    C = sp.vstack([sp.identity(n), sp.csc_matrix(prob.A)], format="csc")
    lower = np.concatenate([prob.lo, np.full(len(prob.b), -np.inf)])
    upper = np.concatenate([prob.hi, prob.b])
    solver = osqp.OSQP()
    solver.setup(P=P, q=-a_raw, A=C, l=lower, u=upper, verbose=False, eps_abs=1e-8, eps_rel=1e-8,
                 max_iter=50000, polishing=False)
    if warm is not None:
        solver.warm_start(x=warm)
    res = solver.solve(raise_error=False)
    status = str(res.info.status).lower()
    if "solved" not in status or res.x is None or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"OSQP failed: {res.info.status}")
    return np.clip(res.x, prob.lo, prob.hi)


def _solve_dual_gradient(a_raw: np.ndarray, prob: ProjectionProblem, iters: int = 20000) -> np.ndarray:
    """Projected gradient ascent on the dual (fallback when OSQP is unavailable)."""
    A, b = prob.A, prob.b
    mu = np.zeros(len(b))
    step = 1.0 / max(np.linalg.norm(A, 2) ** 2, 1e-12)
    a = np.clip(a_raw, prob.lo, prob.hi)
    for _ in range(iters):
        a = np.clip(a_raw - A.T @ mu, prob.lo, prob.hi)
        resid = A @ a - b
        if resid.max(initial=-np.inf) <= 1e-9 and np.all(mu * resid >= -1e-9):
            break
        mu = np.maximum(0.0, mu + step * resid)
    return a


def solve_projection(a_raw: np.ndarray, prob: ProjectionProblem, warm: np.ndarray | None = None) -> np.ndarray:
    """Solve eq. (11) for one transformer. Returns the projected action."""
    if prob.n == 0:
        return np.zeros(0)
    if prob.A.size == 0:
        return np.clip(a_raw, prob.lo, prob.hi)
    if _HAVE_OSQP:
        try:
            return _solve_osqp(a_raw, prob, warm)
        except RuntimeError:
            pass
    return _solve_dual_gradient(a_raw, prob)


def project_groups(a_raw: np.ndarray, groups: list[ProjectionGroup]) -> np.ndarray:
    """Project a joint action; agents not in any group are set to zero."""
    out = np.zeros_like(a_raw, dtype=float)
    for g in groups:
        out[g.agent_idx] = solve_projection(a_raw[g.agent_idx], g.problem)
    return out


def kkt_jacobian(a_hat: np.ndarray, prob: ProjectionProblem) -> np.ndarray:
    """Jacobian d a_hat / d a_tilde from the active-set KKT conditions.

    With active bounds fixed and active rows A_J, the projection is locally the
    orthogonal projection onto the null space of A_J restricted to free
    coordinates F:  J_FF = I - A_JF^T (A_JF A_JF^T)^+ A_JF,  zero elsewhere.
    """
    n = prob.n
    jac = np.zeros((n, n))
    free = (a_hat > prob.lo + _TOL) & (a_hat < prob.hi - _TOL)
    if not free.any():
        return jac
    active_rows = (prob.A @ a_hat) >= prob.b - 1e-6 if prob.A.size else np.zeros(0, dtype=bool)
    idx = np.flatnonzero(free)
    if active_rows.any():
        AJF = prob.A[active_rows][:, idx]
        proj = np.eye(len(idx)) - AJF.T @ np.linalg.pinv(AJF @ AJF.T) @ AJF
    else:
        proj = np.eye(len(idx))
    jac[np.ix_(idx, idx)] = proj
    return jac


def differentiable_project(a_raw, groups_per_sample: list[list[ProjectionGroup]]):
    """Torch autograd wrapper: a_raw (B, N) -> a_hat (B, N) with KKT gradients."""
    import torch

    class _Project(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            x_np = x.detach().cpu().double().numpy()
            out = np.zeros_like(x_np)
            jacs = []
            for bi, groups in enumerate(groups_per_sample):
                row = []
                for g in groups:
                    a_hat = solve_projection(x_np[bi, g.agent_idx], g.problem)
                    out[bi, g.agent_idx] = a_hat
                    row.append((g.agent_idx, kkt_jacobian(a_hat, g.problem)))
                jacs.append(row)
            ctx.jacs = jacs
            return torch.as_tensor(out, dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, grad_out):  # type: ignore[override]
            g_np = grad_out.detach().cpu().double().numpy()
            grad_in = np.zeros_like(g_np)
            for bi, row in enumerate(ctx.jacs):
                for idx, jac in row:
                    grad_in[bi, idx] = jac.T @ g_np[bi, idx]
            return torch.as_tensor(grad_in, dtype=grad_out.dtype, device=grad_out.device)

    return _Project.apply(a_raw)
