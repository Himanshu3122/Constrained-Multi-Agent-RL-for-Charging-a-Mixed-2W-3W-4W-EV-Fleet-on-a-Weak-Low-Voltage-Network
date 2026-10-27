# Changelog

## 0.2.0 (2026-10-27): manuscript-revision release

Fixes from `audit/REPO_FORENSIC_AUDIT.md` and additions requested in the peer review:
- Evaluation: per-day aggregation (deterministic methods no longer duplicated across seeds); paired Wilcoxon tests and Cohen's d_z (`paired_comparison`); per-class unmet energy; optional per-step loading traces (`--keep-traces`).
- Solver-grounded baseline: no future-load leakage on the first day (training-split mean forecast).
- New baseline `mpc_fair` (perfect-foresight MPC with a max-min delivered-fraction term).
- Normalized, capped dual update (`dual_normalize`, `lambda_max`); history logs wall-clock time and environment steps.
- Optional subsampled projection-gradient term (`projection_grad_batch`).
- Reduced-scale study configs (`configs/study*.yaml`) and scripts `run_study.py`, `density_study.py`, `linearization_error.py`, `make_paper_artifacts.py`.
- New tests for each fix.
- Study executed on 2026-10-27 (CPU only): checkpoints in `checkpoints/study*/seed{0,1,2}/`, evaluation in `results/study_eval_{baselines,learned}/`, density and linearization results in `results/`, paper tables and figure data in `../generated/` (from `make_paper_artifacts.py`).

## 0.1.0 (2026-10-27)

First research release, derived from the SOLACE manuscript (`main.tex`) and `PROJECT_REPORT.md`.

- Layer 1: constraint schema, validator with retries and a default fallback, per-EV feasibility LP, counter-offers, rule-based demo parser, optional Hugging Face LLM parser, LoRA fine-tuning script (not executed), demo trilingual corpus and intent metrics.
- Layer 2: per-class actors, GAT + mean-field critic, constrained MAPPO with a Lagrangian advantage, federated averaging (Algorithm 1), checkpoints.
- Layer 3: LinDistFlow + 16-gon projection QP (OSQP) with KKT-based gradients and handling of Proposition 1's assumption.
- Environment: synthetic Indian feeder, Table II fleet, sessions and constraints, AC backward/forward sweep, degradation model, paper metrics.
- Baselines: uncontrolled, ToU, Thompson bandits, solver-grounded single agent, perfect-foresight MPC, MAPPO and IPPO configs. Ablation configs.
- FastAPI service, Docker image, CI, documentation, tests.
