"""Receding-horizon convex QP baselines (paper baselines (ix) and (x)).

(x) Centralized MPC with perfect 24-h foresight: knows future arrivals, base
    load and prices; minimizes energy and degradation cost under the same hard
    LinDistFlow / kVA-polygon / battery constraints. No fairness term (the paper
    notes a Jain-index constraint is non-convex).
(ix) Solver-grounded single agent (EVAgent-style): calls a convex solver for
    the whole currently plugged-in fleet at each step, without knowledge of
    future arrivals and with a day-ahead persistence forecast of base load
    (assumption A13).

Both re-solve at every step and apply the first-step decision. Battery wear is
priced with a linear throughput cost derived from the degradation model at the
configured battery age (assumption A13), because eq. (7) is not convex in the
power trajectory. A small quadratic regularizer keeps the problem a strictly
convex QP.
"""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from solace.config import ExperimentConfig
from solace.data.scenarios import ScenarioSet
from solace.degradation import GAS_CONSTANT, REFERENCE_CELL_AH
from solace.env.charging_env import ChargingEnv

_REG = 1e-4


def linear_wear_cost(cfg: ExperimentConfig, chemistry: str, c_rate: float = 0.3) -> float:
    """Marginal degradation cost in INR per kWh of throughput at the configured battery age."""
    d = cfg.degradation
    p = d.chemistries[chemistry]
    arr = np.exp(-(p.Ea_j_per_mol + p.alpha_j_per_mol_per_c * c_rate) / (GAS_CONSTANT * (d.temperature_c + 273.15)))
    a = REFERENCE_CELL_AH * max(d.initial_equivalent_cycles, 1.0)
    dloss_dcycle = p.B * arr * p.z * a ** (p.z - 1.0) * REFERENCE_CELL_AH / 100.0
    # kappa_i * loss per kWh = (cost_per_kwh * cap / eol) * dloss_dcycle / cap
    return d.battery_cost_inr_per_kwh / d.end_of_life_loss * dloss_dcycle


class _HorizonQP:
    """Builds and solves the multi-step QP for each transformer at the current step."""

    perfect_foresight = True
    name = "horizon_qp"
    eps = 1e-4  # OSQP tolerance on the normalized problem
    fairness_weight = 0.0  # INR per kWh-equivalent of minimum delivered fraction (fair MPC only)

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.wear = {c: linear_wear_cost(cfg, c) for c in cfg.degradation.chemistries}

    def reset(self, env: ChargingEnv) -> None:
        return None

    def _base_forecast(self, env: ChargingEnv, tx: int) -> tuple[np.ndarray, np.ndarray]:
        return env.day.base_p_kw[tx], env.day.base_q_kvar[tx]

    def _sessions(self, env: ChargingEnv, members: np.ndarray) -> list[dict]:
        """Current sessions (with live state) plus, with perfect foresight, future ones."""
        t = env.t
        out = []
        for v in members[env.active[members]]:
            out.append({"v": int(v), "t0": t, "t1": int(env.departure[v]), "e0": float(env.energy[v]),
                        "deadline": int(env.deadline[v]), "min_soc": float(env.min_soc[v]),
                        "v2g": bool(env.v2g_allowed[v] and env.v2g_capable[v]), "floor": float(env.soc_floor[v]),
                        "dis_budget": float(max(0.0, env.max_discharge[v] - env.discharged[v])), "current": True})
        if self.perfect_foresight:
            mset = set(int(m) for m in members)
            for s in env.day.sessions:
                v = int(s["vehicle"])
                if v in mset and int(s["arrival"]) > t:
                    out.append({"v": v, "t0": int(s["arrival"]), "t1": int(s["departure"]),
                                "e0": float(s["soc_arrival"]) * env.cap[v], "deadline": int(s["deadline"]),
                                "min_soc": float(s["min_soc"]), "v2g": bool(s["v2g_allowed"] and env.v2g_capable[v]),
                                "floor": float(s["soc_floor"]), "dis_budget": float(s["max_discharge_kwh"]),
                                "current": False})
        return out

    def _solve_transformer(self, env: ChargingEnv, tx: int) -> dict[int, float]:
        cfg = self.cfg
        dt = env.dt
        eta_c, eta_d = cfg.env.eta_charge, cfg.env.eta_discharge
        net = env.feeder.networks[tx]
        members = env._tx_members[tx]
        sessions = [s for s in self._sessions(env, members) if s["t1"] > s["t0"]]
        if not sessions:
            return {}
        base_p, base_q = self._base_forecast(env, tx)
        S = env.s_base
        # Variable layout: per session charge (L), discharge (L if v2g), slack (1).
        offsets = []
        n_var = 0
        for s in sessions:
            L = s["t1"] - s["t0"]
            c0 = n_var
            n_var += L
            d0 = None
            if s["v2g"]:
                d0 = n_var
                n_var += L
            sl = n_var
            n_var += 1
            offsets.append((c0, d0, sl, L))
        phi_idx = None  # max-min delivered-fraction variable (fairness-aware MPC only)
        if self.fairness_weight > 0:
            phi_idx = n_var
            n_var += 1
        q = np.zeros(n_var)
        lb = np.zeros(n_var)
        ub = np.zeros(n_var)
        R_, C_, V_, LO, HI = [], [], [], [], []
        r = 0

        def add_block(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray | float, lo: np.ndarray,
                      hi: np.ndarray) -> None:
            """Append constraint rows; ``rows`` are local indices 0..len(lo)-1."""
            nonlocal r
            rows = np.asarray(rows)
            R_.append(rows + r)
            C_.append(np.asarray(cols))
            V_.append(np.broadcast_to(np.asarray(vals, dtype=float), rows.shape))
            LO.append(np.atleast_1d(np.asarray(lo, dtype=float)))
            HI.append(np.atleast_1d(np.asarray(hi, dtype=float)))
            r += len(LO[-1])

        price = env.day.tariff
        export = cfg.data.tariff.v2g_export_multiplier
        g_step, g_var, g_bus, g_sign = [], [], [], []  # grid-coupling terms
        fair_req: list[float] = []
        for s, (c0, d0, sl, L) in zip(sessions, offsets, strict=True):
            v = s["v"]
            P = float(env.p_rated[v])
            cap = float(env.cap[v])
            wear = self.wear[str(env.fleet.chemistry[v])]
            steps = np.arange(s["t0"], s["t1"])
            q[c0:c0 + L] = (price[steps] + wear) * dt
            ub[c0:c0 + L] = P
            g_step.append(steps)
            g_var.append(c0 + np.arange(L))
            g_bus.append(np.full(L, int(env.fleet.bus[v])))
            g_sign.append(np.ones(L))
            if d0 is not None:
                q[d0:d0 + L] = (-export * price[steps] + wear) * dt
                ub[d0:d0 + L] = P
                g_step.append(steps)
                g_var.append(d0 + np.arange(L))
                g_bus.append(np.full(L, int(env.fleet.bus[v])))
                g_sign.append(-np.ones(L))
            q[sl] = cfg.env.unmet_penalty_inr_per_kwh
            ub[sl] = np.inf
            # Energy after each step j: e0 + dt * sum_{u<=j} (eta_c c_u - d_u / eta_d) within [low_e, cap].
            jj, uu = np.tril_indices(L)
            rows, cols, vals = [jj], [c0 + uu], [np.full(len(jj), dt * eta_c)]
            if d0 is not None:
                rows.append(jj)
                cols.append(d0 + uu)
                vals.append(np.full(len(jj), -dt / eta_d))
            low_e = min(s["e0"], s["floor"] * cap) if s["v2g"] else -np.inf
            add_block(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals),
                      np.full(L, low_e - s["e0"]), np.full(L, cap - s["e0"]))
            k = s["deadline"] - s["t0"]
            if 0 < k <= L:  # soft deadline: delivered + slack >= requested
                cols_k = [c0 + np.arange(k), np.array([sl])]
                vals_k = [np.full(k, dt * eta_c), np.array([1.0])]
                if d0 is not None:
                    cols_k.insert(1, d0 + np.arange(k))
                    vals_k.insert(1, np.full(k, -dt / eta_d))
                cc = np.concatenate(cols_k)
                add_block(np.zeros(len(cc), dtype=int), cc, np.concatenate(vals_k),
                          [s["min_soc"] * cap - s["e0"]], [np.inf])
                req = s["min_soc"] * cap - s["e0"]
                if phi_idx is not None and req > 1e-6:
                    # delivered_by_deadline - phi * remaining_request >= 0 (no slack: phi is the min fraction)
                    cf = np.concatenate([cols_k[0]] + ([cols_k[1]] if d0 is not None else []) + [np.array([phi_idx])])
                    vf = np.concatenate([vals_k[0]] + ([vals_k[1]] if d0 is not None else []) + [np.array([-req])])
                    add_block(np.zeros(len(cf), dtype=int), cf, vf, [0.0], [np.inf])
                    fair_req.append(req)
            if d0 is not None:
                add_block(np.zeros(L, dtype=int), d0 + np.arange(L), dt / eta_d, [-np.inf], [s["dis_budget"]])
        # Grid constraints per step (same LinDistFlow + polygon rows as Layer 3).
        K = env.feeder.polygon_sides
        phi = 2.0 * np.pi * np.arange(K) / K
        phi = phi[np.abs(np.cos(phi)) > 1e-9]
        radius = net.s_rated_pu * np.cos(np.pi / K)
        all_step, all_var = np.concatenate(g_step), np.concatenate(g_var)
        all_bus, all_sign = np.concatenate(g_bus), np.concatenate(g_sign)
        nb = net.n_buses - 1
        for step in np.unique(all_step):
            sel = all_step == step
            var, bus, sign = all_var[sel], all_bus[sel], all_sign[sel]
            bp = base_p[step] / S
            bq = base_q[step] / S
            v_base = (1.0 - 2.0 * (net.R_path @ bp + net.X_path @ bq))[1:]
            coef = 2.0 * net.R_path[1:, :][:, bus] * sign[None, :] / S
            add_block(np.repeat(np.arange(nb), len(var)), np.tile(var, nb), coef.ravel(),
                      np.minimum(v_base - env.feeder.v_max**2, 0.0), np.maximum(v_base - env.feeder.v_min**2, 0.0))
            rhs = radius - np.cos(phi) * bp.sum() - np.sin(phi) * bq.sum()
            coef_p = np.cos(phi)[:, None] * sign[None, :] / S
            add_block(np.repeat(np.arange(len(phi)), len(var)), np.tile(var, len(phi)), coef_p.ravel(),
                      np.full(len(phi), -np.inf), np.maximum(rhs, 0.0))
        if phi_idx is not None:
            # Maximize the minimum delivered fraction; weight = fairness_weight (INR/kWh) x total request.
            ub[phi_idx] = 1.0 if fair_req else 0.0
            q[phi_idx] = -self.fairness_weight * float(sum(fair_req))
        A = sp.csr_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))), shape=(r, n_var))
        lo_c = np.concatenate(LO)
        hi_c = np.concatenate(HI)
        # Row equilibration: voltage rows have ~1e-5 coefficients, energy rows ~0.2. Without scaling
        # OSQP needs >10^4 iterations; scaling rows (and their bounds) leaves the feasible set unchanged.
        row_max = np.maximum(abs(A).max(axis=1).toarray().ravel(), 1e-12)
        D = sp.diags(1.0 / row_max)
        A = (D @ A).tocsc()
        lo_c = lo_c / row_max
        hi_c = hi_c / row_max
        A_full = sp.vstack([sp.identity(n_var, format="csc"), A], format="csc")
        l_full = np.concatenate([lb, lo_c])
        u_full = np.concatenate([ub, hi_c])
        # Normalize the objective to O(1) so the small regularizer acts on a well-scaled problem;
        # the minimizer of (q/s)'x + reg/2 ||x||^2 equals that of q'x + (s*reg)/2 ||x||^2.
        q_scale = max(float(np.abs(q[np.isfinite(q)]).max()), 1e-9)
        solver = osqp.OSQP()
        solver.setup(P=sp.identity(n_var, format="csc") * _REG, q=q / q_scale, A=A_full, l=l_full, u=u_full,
                     verbose=False, eps_abs=self.eps, eps_rel=self.eps, max_iter=20000, polishing=False,
                     adaptive_rho=True)
        res = solver.solve(raise_error=False)
        x = res.x if res.x is not None and np.all(np.isfinite(res.x)) else np.zeros(n_var)
        actions: dict[int, float] = {}
        for s, (c0, d0, _, _) in zip(sessions, offsets, strict=True):
            if s["current"]:
                p = x[c0] - (x[d0] if d0 is not None else 0.0)
                actions[s["v"]] = float(np.clip(p / env.p_rated[s["v"]], -1.0, 1.0))
        return actions

    def act(self, env: ChargingEnv, obs: np.ndarray) -> np.ndarray:
        a = np.zeros(env.n)
        for tx in range(env.feeder.n_transformers):
            for v, val in self._solve_transformer(env, tx).items():
                a[v] = val
        return a


class CentralizedMPCPolicy(_HorizonQP):
    """(x) Perfect-foresight centralized MPC (upper bound in the paper)."""

    perfect_foresight = True
    name = "mpc"


class FairMPCPolicy(CentralizedMPCPolicy):
    """(x') Perfect-foresight MPC with a convex fairness term (max-min delivered fraction).

    Added in revision so that the MPC bound is also fairness-aware. The Jain
    index itself is non-convex; maximizing the minimum delivered fraction of the
    sessions in the horizon is a standard convex surrogate.
    """

    name = "mpc_fair"

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        self.fairness_weight = cfg.env.unmet_penalty_inr_per_kwh


class SolverGroundedPolicy(_HorizonQP):
    """(ix) Solver-grounded single agent: current EVs only, day-ahead persistence forecast.

    Revision: on the first evaluated day there is no previous day, so the forecast
    falls back to the mean training-split base-load profile (previously it fell
    back to the true current day, which leaked future load; audit check C4).
    """

    perfect_foresight = False
    name = "solver_grounded"

    def __init__(self, cfg: ExperimentConfig, scen: ScenarioSet | None = None):
        super().__init__(cfg)
        self._prev_day: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        self._fallback: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        if scen is not None and scen.days["train"]:
            train = scen.days["train"]
            n_tx = len(train[0].base_p_kw)
            self._fallback = ([np.mean([d.base_p_kw[t] for d in train], axis=0) for t in range(n_tx)],
                              [np.mean([d.base_q_kvar[t] for d in train], axis=0) for t in range(n_tx)])
        self._current_day: tuple[list[np.ndarray], list[np.ndarray]] | None = None

    def reset(self, env: ChargingEnv) -> None:
        self._prev_day = self._current_day
        self._current_day = (env.day.base_p_kw, env.day.base_q_kvar)

    def _base_forecast(self, env: ChargingEnv, tx: int) -> tuple[np.ndarray, np.ndarray]:
        src = self._prev_day or self._fallback
        if src is None:
            raise ValueError("solver-grounded agent needs a previous day or training scenarios for its forecast")
        p = src[0][tx].copy()
        q = src[1][tx].copy()
        p[env.t] = env.day.base_p_kw[tx][env.t]  # the current step is measured, not forecast
        q[env.t] = env.day.base_q_kvar[tx][env.t]
        return p, q
