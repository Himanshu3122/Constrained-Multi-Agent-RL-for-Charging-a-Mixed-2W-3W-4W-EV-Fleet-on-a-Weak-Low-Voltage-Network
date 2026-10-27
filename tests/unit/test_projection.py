import numpy as np
import torch

from solace.grid.lindistflow import path_impedance_matrices
from solace.safety.projection import (
    ProjectionGroup,
    ProjectionProblem,
    build_transformer_problem,
    differentiable_project,
    kkt_jacobian,
    solve_projection,
)


def random_problem(rng, n=6, m=4):
    A = rng.normal(size=(m, n))
    lo = -rng.uniform(0, 1, n)
    hi = rng.uniform(0, 1, n)
    b = rng.uniform(0.05, 0.5, m)  # a = 0 feasible
    return ProjectionProblem(lo=lo, hi=hi, A=A, b=b)


def test_projection_is_feasible_and_optimal():
    rng = np.random.default_rng(1)
    for _ in range(20):
        prob = random_problem(rng)
        a_raw = rng.uniform(-2, 2, prob.n)
        a = solve_projection(a_raw, prob)
        assert np.all(a >= prob.lo - 1e-7) and np.all(a <= prob.hi + 1e-7)
        assert np.all(prob.A @ a <= prob.b + 1e-6)
        # Optimality: no feasible point along random directions is closer.
        for _ in range(10):
            cand = a + 1e-3 * rng.normal(size=prob.n)
            if np.all(cand >= prob.lo) and np.all(cand <= prob.hi) and np.all(prob.A @ cand <= prob.b):
                assert np.linalg.norm(cand - a_raw) >= np.linalg.norm(a - a_raw) - 1e-6


def test_projection_leaves_feasible_points_unchanged():
    prob = random_problem(np.random.default_rng(2))
    a_raw = np.zeros(prob.n)
    assert np.allclose(solve_projection(a_raw, prob), 0.0, atol=1e-7)


def test_kkt_jacobian_matches_finite_differences():
    prob = ProjectionProblem(lo=-np.ones(4), hi=np.ones(4), A=np.array([[1.0, 1.0, 1.0, 1.0]]), b=np.array([1.0]))
    a_raw = np.array([0.6, 0.5, 0.4, -0.2])  # sum > 1 -> constraint active, no bounds active
    a = solve_projection(a_raw, prob)
    jac = kkt_jacobian(a, prob)
    eps = 1e-5
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        fd = (solve_projection(a_raw + d, prob) - solve_projection(a_raw - d, prob)) / (2 * eps)
        assert np.allclose(jac[:, i], fd, atol=1e-4)


def test_torch_backward_uses_kkt_jacobian():
    prob = ProjectionProblem(lo=-np.ones(3), hi=np.ones(3), A=np.array([[1.0, 1.0, 1.0]]), b=np.array([0.5]))
    groups = [[ProjectionGroup(agent_idx=np.arange(3), problem=prob)]]
    x = torch.tensor([[0.6, 0.4, 0.2]], dtype=torch.float64, requires_grad=True)
    y = differentiable_project(x, groups)
    assert torch.all(y.sum() <= 0.5 + 1e-6)
    y.sum().backward()
    # Moving along the active constraint normal is fully absorbed: d(sum y)/dx = 0.
    assert torch.allclose(x.grad, torch.zeros_like(x.grad), atol=1e-6)


def test_transformer_problem_limits_and_relaxation():
    parent = np.array([-1, 0, 1, 2])
    r = np.array([0.0, 0.05, 0.05, 0.05])
    x = r * 0.3
    R, X = path_impedance_matrices(parent, r, x)
    base_p = np.array([0.0, 0.05, 0.05, 0.05])  # v^2 at bus 3 = 1 - 2*(0.05*0.15+0.05*0.1+0.05*0.05) - q-term > 0.95^2
    kw = dict(R_path=R, X_path=X, bus=np.array([3, 3]), p_rated_pu=np.array([0.3, 0.3]),
              lo=np.array([-1.0, 0.0]), hi=np.array([1.0, 1.0]), base_q_pu=0.1 * base_p,
              s_rated_pu=1.0, v_min=0.95, v_max=1.05, polygon_sides=16)
    prob = build_transformer_problem(base_p_pu=base_p, **kw)
    assert not prob.relaxed
    a = solve_projection(np.array([1.0, 1.0]), prob)
    v = 1.0 - 2.0 * (R @ (base_p + np.array([0, 0, 0, 0.3 * a.sum()])) + X @ (0.1 * base_p))
    assert v[1:].min() >= 0.95**2 - 1e-6
    heavy = build_transformer_problem(base_p_pu=base_p * 6, **kw)
    assert heavy.relaxed  # base load alone violates -> rows relaxed to "do not make it worse"
    a_heavy = solve_projection(np.array([1.0, 1.0]), heavy)
    assert np.all(heavy.A @ a_heavy <= heavy.b + 1e-6)
