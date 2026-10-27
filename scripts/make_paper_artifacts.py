"""Generate the revised paper's tables (LaTeX) and figure data (.dat) from logged results.

Every number in the revised manuscript's results section is produced by this
script from files written by evaluate.py, train.py, density_study.py and
linearization_error.py. Nothing is typed by hand.

    python scripts/make_paper_artifacts.py \
        --eval results/study_eval_baselines/study/metrics.json results/study_eval_learned/study/metrics.json \
        --histories checkpoints --density results/density_study.json \
        --linearization results/linearization_error.json --scenarios data/processed/study --out ../generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from solace.evaluation.runner import paired_comparison

LABELS = {
    "uncontrolled": "Uncontrolled", "tou": "ToU rule", "bandits": "Thompson bandits",
    "solver_grounded": "Online QP", "mpc": "MPC (perfect foresight)",
    "mpc_fair": "MPC-fair (perfect foresight)", "mappo": "MAPPO + penalties",
    "ppo_proj": "PPO + projection", "nofair": r"SOLACE, $\lambda_F{=}0$", "solace": "SOLACE",
}
ORDER = ["uncontrolled", "tou", "bandits", "mappo", "ppo_proj", "solver_grounded", "nofair", "solace", "mpc",
         "mpc_fair"]
HIST_DIRS = {"solace": "study", "nofair": "study_nofair", "mappo": "study_mappo", "ppo_proj": "study_ppo_proj"}


def fmt(a: dict, key: str, digits: int) -> str:
    m, c = a[key]["mean"], a[key]["ci95"]
    return f"{m:.{digits}f} $\\pm$ {c:.{digits}f}"


def main_table(results: dict, out: Path) -> None:
    rows = []
    for m in ORDER:
        if m not in results:
            continue
        a = results[m]["aggregate"]
        seeds = len(results[m]["seeds"])
        rows.append(" & ".join([LABELS[m] + (f" ({seeds})" if seeds > 1 else ""), fmt(a, "peak_loading_pct", 1),
                                fmt(a, "voltage_violations", 1), fmt(a, "unmet_energy_pct", 1), fmt(a, "jain", 3),
                                fmt(a, "cost_kinr", 2), fmt(a, "degradation_kinr", 2),
                                f"{a['latency_ms']['mean']:.1f}"]) + r" \\")
    (out / "tab_main_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def paired_table(results: dict, out: Path) -> dict:
    metrics = ["unmet_energy_pct", "jain", "peak_loading_pct", "voltage_violations"]
    stats = {k: paired_comparison(results, "solace", k) for k in metrics}
    rows = []
    for m in ORDER:
        if m == "solace" or m not in results:
            continue
        cells = [LABELS[m]]
        for k in metrics:
            s = stats[k][m]
            p = s["wilcoxon_p"]
            digits = 3 if k == "jain" else 2
            cells.append(f"{s['mean_diff']:+.{digits}f} ({'$<$0.01' if p < 0.01 else f'{p:.2f}'})")
        rows.append(" & ".join(cells) + r" \\")
    (out / "tab_paired_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return stats


def class_table(results: dict, out: Path) -> None:
    rows = []
    for m in ORDER:
        if m not in results:
            continue
        a = results[m]["aggregate"]
        rows.append(" & ".join([LABELS[m]] + [f"{a[f'unmet_energy_pct_{c}']['mean']:.1f}" for c in ("2W", "3W", "4W")])
                    + r" \\")
    (out / "tab_class_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def training_curves(hist_root: Path, out: Path, window: int = 5) -> dict:
    summary = {}
    for key, name in HIST_DIRS.items():
        runs = sorted((hist_root / name).glob("seed*/history.json"))
        if not runs:
            continue
        hists = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
        n = min(len(h) for h in hists)
        unmet = np.array([[r["train_unmet_energy_pct"] for r in h[:n]] for h in hists])
        lam = np.array([[r["lambda_F"] for r in h[:n]] for h in hists])
        kernel = np.ones(window) / window
        smooth = np.array([np.convolve(u, kernel, mode="valid") for u in unmet])
        x = np.arange(window, n + 1)
        with (out / f"curve_{key}.dat").open("w", encoding="utf-8") as fh:
            fh.write("round mean lo hi lambdaF\n")
            for i, r in enumerate(x):
                mu, sd = smooth[:, i].mean(), smooth[:, i].std(ddof=1) if len(runs) > 1 else 0.0
                fh.write(f"{r} {mu:.3f} {mu - sd:.3f} {mu + sd:.3f} {lam[:, r - 1].mean():.4f}\n")
        summary[key] = {"seeds": len(runs), "rounds": n, "final_lambda_F": lam[:, -1].tolist(),
                        "first10_unmet": float(unmet[:, :10].mean()), "last10_unmet": float(unmet[:, -10:].mean()),
                        "wallclock_min": [h[n - 1].get("elapsed_s", float("nan")) / 60 for h in hists],
                        "env_steps": hists[0][n - 1].get("env_steps")}
    return summary


def loading_traces(results: dict, out: Path, start_hour: int, dt_min: int) -> None:
    for m in ("uncontrolled", "mpc_fair", "solace"):
        if m not in results:
            continue
        trace = results[m]["episodes"][0][0].get("loading_trace")  # first seed, first test day
        if trace is None:
            continue
        with (out / f"trace_{m}.dat").open("w", encoding="utf-8") as fh:
            fh.write("hour loading\n")
            for t, row in enumerate(trace):
                fh.write(f"{start_hour + t * dt_min / 60:.2f} {100 * max(row):.2f}\n")


def density_table(path: Path, out: Path) -> None:
    rows = []
    for d in json.loads(path.read_text(encoding="utf-8")):
        w, p = d["without_projection"], d["with_projection"]
        rows.append(" & ".join([str(d["n_ev"]), f"{w['peak_loading_pct']:.1f}", f"{p['peak_loading_pct']:.1f}",
                                f"{w['voltage_violations']:.1f}", f"{p['voltage_violations']:.1f}",
                                f"{w['unmet_energy_pct']:.1f}", f"{p['unmet_energy_pct']:.1f}",
                                f"{p['projection_latency_ms']:.1f}"]) + r" \\")
    (out / "tab_density_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def fleet_shares(scen_dir: Path) -> dict:
    meta = json.loads((scen_dir / "sessions.json").read_text(encoding="utf-8"))
    arrays = np.load(scen_dir / "scenarios.npz")
    cls = arrays["fleet_vehicle_class"]
    n = {c: 0 for c in ("2W", "3W", "4W")}
    e = {c: 0.0 for c in n}
    for split in meta["splits"].values():
        for day in split:
            for s in day["sessions"]:
                c = str(cls[s["vehicle"]])
                n[c] += 1
                e[c] += s["energy_kwh"]
    tn, te = sum(n.values()), sum(e.values())
    return {"session_share": {c: n[c] / tn for c in n}, "energy_share": {c: e[c] / te for c in e}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--histories", default="checkpoints")
    ap.add_argument("--density", default="results/density_study.json")
    ap.add_argument("--linearization", default="results/linearization_error.json")
    ap.add_argument("--scenarios", default="data/processed/study")
    ap.add_argument("--start-hour", type=int, default=8)
    ap.add_argument("--dt-min", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for p in args.eval:
        results.update(json.loads(Path(p).read_text(encoding="utf-8")))
    rename = {"study_nofair": "nofair", "study_mappo": "mappo", "study_ppo_proj": "ppo_proj"}
    results = {rename.get(k, k): v for k, v in results.items()}
    main_table(results, out)
    class_table(results, out)
    summary: dict = {"methods": {}}
    for m, r in results.items():
        summary["methods"][m] = {"aggregate": r["aggregate"], "seed_spread": r.get("seed_spread"),
                                 "seeds": r.get("seeds")}
    if "solace" in results:
        summary["paired_vs_solace"] = paired_table(results, out)
    loading_traces(results, out, args.start_hour, args.dt_min)
    summary["training"] = training_curves(Path(args.histories), out)
    if Path(args.density).is_file():
        density_table(Path(args.density), out)
    if Path(args.linearization).is_file():
        summary["linearization"] = json.loads(Path(args.linearization).read_text(encoding="utf-8"))
    summary["fleet_shares"] = fleet_shares(Path(args.scenarios))
    (out / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote artifacts to {out}")


if __name__ == "__main__":
    main()
