"""Evaluate methods on held-out days (paper Section VI-F metrics).

Metrics per episode (``EpisodeMetrics.summary``): peak transformer loading
(max over transformers, % of rating), voltage violations (bus-steps outside
[v_min, v_max] under AC power flow), unmet energy (%), Jain index (eq. (8)),
net energy cost and degradation cost (kINR/day), intent satisfaction rate,
projection distance (kW/step), per-class unmet energy, and decision latency
(ms/step, measured on the machine running the evaluation).

Aggregation (revised): the unit of analysis is the test day. Seed-dependent
methods (learned policies, bandits) are run once per seed and averaged per
day; deterministic methods are run once. Mean, standard deviation and a
normal-approximation 95% CI are taken over days, so deterministic methods are
no longer counted once per seed. ``paired_comparison`` adds a Wilcoxon
signed-rank test and Cohen's d_z on the per-day differences.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solace.baselines import (
    CentralizedMPCPolicy,
    FairMPCPolicy,
    SolverGroundedPolicy,
    ThompsonBanditPolicy,
    ToUPolicy,
    UncontrolledPolicy,
)
from solace.config import ExperimentConfig
from solace.data.scenarios import DayScenario, ScenarioSet
from solace.env.charging_env import ChargingEnv
from solace.training.checkpoint import LearnedPolicy
from solace.utils.logging import get_logger, kv

log = get_logger("evaluation")

METRIC_ORDER = [
    "peak_loading_pct", "voltage_violations", "unmet_energy_pct", "jain", "cost_kinr", "degradation_kinr",
    "intent_satisfaction_pct", "projection_distance_kw", "latency_ms", "relaxed_steps",
    "unmet_energy_pct_2W", "unmet_energy_pct_3W", "unmet_energy_pct_4W",
]


@dataclass
class MethodSpec:
    factory: Callable[[ExperimentConfig, ScenarioSet, int, str | None], object]
    use_projection: bool | None  # None -> taken from the learned checkpoint's config
    needs_checkpoint: bool = False
    seed_dependent: bool = False  # run once per seed and average per day


def _learned(cfg: ExperimentConfig, scen: ScenarioSet, seed: int, ckpt: str | None) -> object:
    if not ckpt:
        raise ValueError("learned methods need --checkpoint")
    return LearnedPolicy(ckpt.format(seed=seed))


METHODS: dict[str, MethodSpec] = {
    "uncontrolled": MethodSpec(lambda c, sc, s, k: UncontrolledPolicy(), use_projection=False),
    "tou": MethodSpec(lambda c, sc, s, k: ToUPolicy(), use_projection=False),
    "bandits": MethodSpec(lambda c, sc, s, k: ThompsonBanditPolicy(len(sc.fleet), seed=s), use_projection=False,
                          seed_dependent=True),
    "solver_grounded": MethodSpec(lambda c, sc, s, k: SolverGroundedPolicy(c, sc), use_projection=True),
    "mpc": MethodSpec(lambda c, sc, s, k: CentralizedMPCPolicy(c), use_projection=True),
    "mpc_fair": MethodSpec(lambda c, sc, s, k: FairMPCPolicy(c), use_projection=True),
    "solace": MethodSpec(_learned, use_projection=None, needs_checkpoint=True, seed_dependent=True),
    "learned": MethodSpec(_learned, use_projection=None, needs_checkpoint=True, seed_dependent=True),
}


def register_learned(name: str) -> None:
    """Register an additional learned method (checkpoint passed at evaluation time)."""
    METHODS[name] = MethodSpec(_learned, use_projection=None, needs_checkpoint=True, seed_dependent=True)


def evaluate_method(cfg: ExperimentConfig, scen: ScenarioSet, method: str, days: list[DayScenario], seed: int,
                    checkpoint: str | None = None, keep_traces: bool = False) -> list[dict]:
    if method not in METHODS:
        raise KeyError(f"unknown method '{method}'; choose from {sorted(METHODS)}")
    spec = METHODS[method]
    np.random.seed(seed)
    policy = spec.factory(cfg, scen, seed, checkpoint)
    use_projection = spec.use_projection
    if use_projection is None:
        use_projection = policy.use_projection  # type: ignore[attr-defined]
    env = ChargingEnv(cfg, scen, use_projection=use_projection)
    out = []
    for day in days:
        obs = env.reset(day)
        policy.reset(env)  # type: ignore[attr-defined]
        done = False
        act_times = []
        while not done:
            tic = time.perf_counter()
            action = policy.act(env, obs)  # type: ignore[attr-defined]
            act_times.append(time.perf_counter() - tic)
            res = env.step(action)
            obs, done = res.obs, res.done
        summary: dict = env.metrics.summary()
        # Decision latency = policy computation + safety projection (paper: "latency per 15-min step").
        proj = np.asarray(env.metrics.projection_latency_s)
        summary["latency_ms"] = 1e3 * float(np.mean(np.asarray(act_times) + proj))
        summary["day"] = float(day.day)
        summary["seed"] = float(seed)
        if keep_traces:
            summary["loading_trace"] = [list(map(float, row)) for row in env.metrics.loading_trace]
        out.append(summary)
    return out


def _per_day_mean(per_seed: list[list[dict]]) -> list[dict[str, float]]:
    """Average seed runs day by day; returns one row per day."""
    rows = []
    for d in range(len(per_seed[0])):
        row = {k: float(np.mean([runs[d][k] for runs in per_seed])) for k in METRIC_ORDER}
        row["day"] = per_seed[0][d]["day"]
        row["n_seeds"] = float(len(per_seed))
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean, std and normal-approximation 95% CI over rows (one row per test day)."""
    agg: dict[str, dict[str, float]] = {}
    for key in METRIC_ORDER:
        vals = np.array([r[key] for r in rows], dtype=float)
        n = len(vals)
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        agg[key] = {"mean": float(vals.mean()), "std": std, "ci95": 1.96 * std / math.sqrt(n) if n > 1 else 0.0,
                    "n": float(n)}
    return agg


def seed_spread(per_seed: list[list[dict]]) -> dict[str, dict[str, float]]:
    """Across-seed spread of the day-averaged metrics (mean and std over seeds)."""
    out = {}
    for key in METRIC_ORDER:
        per = np.array([np.mean([r[key] for r in runs]) for runs in per_seed])
        out[key] = {"mean": float(per.mean()), "std": float(per.std(ddof=1)) if len(per) > 1 else 0.0,
                    "values": per.tolist()}
    return out


def paired_comparison(results: dict[str, dict], reference: str, metric: str) -> dict[str, dict[str, float]]:
    """Wilcoxon signed-rank test and Cohen's d_z of (method - reference) over test days."""
    from scipy.stats import wilcoxon

    ref = np.array([r[metric] for r in results[reference]["days"]])
    out = {}
    for method, res in results.items():
        if method == reference:
            continue
        x = np.array([r[metric] for r in res["days"]])
        diff = x - ref
        sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
        p = 1.0 if np.allclose(diff, 0.0) else float(wilcoxon(x, ref, zero_method="zsplit").pvalue)
        out[method] = {"mean_diff": float(diff.mean()), "cohens_dz": float(diff.mean() / sd) if sd > 0 else 0.0,
                       "wilcoxon_p": p, "n_days": float(len(diff))}
    return out


def run_evaluation(cfg: ExperimentConfig, scen: ScenarioSet, methods: list[str], seeds: list[int],
                   checkpoints: dict[str, str] | None = None, split: str = "test",
                   max_days: int | None = None, keep_traces: bool = False) -> dict[str, dict]:
    checkpoints = checkpoints or {}
    days = scen.days[split][: max_days or None]
    if not days:
        raise ValueError(f"split '{split}' has no days")
    results: dict[str, dict] = {}
    for method in methods:
        if method not in METHODS:
            raise KeyError(f"unknown method '{method}'; choose from {sorted(METHODS)}")
        method_seeds = seeds if METHODS[method].seed_dependent else seeds[:1]
        per_seed = [evaluate_method(cfg, scen, method, days, s, checkpoints.get(method), keep_traces)
                    for s in method_seeds]
        day_rows = _per_day_mean(per_seed)
        results[method] = {"aggregate": aggregate(day_rows), "days": day_rows, "episodes": per_seed,
                           "seeds": [float(s) for s in method_seeds], "seed_spread": seed_spread(per_seed)}
        a = results[method]["aggregate"]
        log.info(kv(event="evaluated", method=method, days=len(day_rows), seeds=len(method_seeds),
                    peak=a["peak_loading_pct"]["mean"], viol=a["voltage_violations"]["mean"],
                    unmet=a["unmet_energy_pct"]["mean"], jain=a["jain"]["mean"]))
    return results


def write_report(results: dict[str, dict], out_dir: str | Path, title: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    header = ["Method", "Peak (%)", "V-viol./day", "Unmet (%)", "Jain", "Cost (kINR/d)", "Degr. (kINR/d)",
              "ISR (%)", "PD (kW)", "Latency (ms)"]
    lines = [f"# {title}", "",
             "Locally computed results: mean ± 95% CI over test days (seed-dependent methods averaged over "
             "seeds per day). See docs/evaluation.md.", "",
             "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method, res in results.items():
        a = res["aggregate"]
        cells = [method] + [f"{a[k]['mean']:.3g} ± {a[k]['ci95']:.2g}" for k in METRIC_ORDER[:9]]
        lines.append("| " + " | ".join(cells) + " |")
    path = out / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
