# Implementation status

Sources: manuscript `main.tex` (science) and `PROJECT_REPORT.md` (engineering specification). Details per component: [docs/paper_implementation_mapping.md](docs/paper_implementation_mapping.md). Assumption IDs (A1–A19): [docs/assumptions.md](docs/assumptions.md).

## Fully implemented

Directly from the paper's specification:

- Constraint schema with the paper's six fields and the safe V2G default (`solace/intent/schema.py`).
- Validator with ≤ 2 re-prompts and a default constraint plus operator flag (`solace/intent/pipeline.py`).
- LinDistFlow (eqs. 3–4), voltage bounds and the 16-gon inner kVA approximation (eq. 5) (`solace/grid`, `solace/safety`).
- Projection QP (eq. 11) with the Proposition 1 check (`solace/safety/projection.py`).
- Battery dynamics (eq. 2), per-step cost (eq. 6) and Jain index (eq. 8) (`solace/env`).
- CMDP with Lagrangian and projected dual ascent (eqs. 9–10), clipped PPO on the Lagrangian advantage with GAE (eq. 14), FedAvg (eq. 15) (`solace/training`).
- Per-class shared actors (2×128 tanh) and the GAT + mean-field critic (eqs. 12–13) (`solace/agents`).
- Table III hyperparameters and the 300/35/30 day split (`configs/default.yaml`).
- Rule-based baselines: uncontrolled and ToU.
- Evaluation metrics of Section VI-F, 5-seed protocol, ablation configs (except "w/o intent layer").

## Partially implemented

Runs, but relies on assumptions or approximations:

| Component | What is approximated | Ref |
|---|---|---|
| Synthetic Indian feeder | Topology and impedances; balanced single-phase LV; no MV drop | A1, A2 |
| Fleet and sessions | Sampling of Table II ranges; generator form; the 78%/67% statistics cannot be matched with Table II | A3, A4 |
| Base load, PV, tariff | Synthetic profiles and tariff values | A6 |
| Degradation (eq. 7) | Coefficients from the cited model family, battery age, cost | A8 |
| Reward | Soft deadline penalty and reward scale | A9 |
| Differentiable projection | KKT active-set gradient instead of cvxpylayers; relaxation when Proposition 1 fails | A10 |
| Algorithm 1 | Sequential aggregators, joint rollouts, gradient through the projection via a mean-correction term | A11 |
| Intent layer with LLM | `HFParser` implemented (no grammar-constrained decoding) but not executed; `RuleBasedParser` is a non-paper stand-in | A15 |
| Feasibility LP headroom | Caller-supplied forecast | A14 |
| Bandits, MPC, solver-grounded baselines | Bandit reward; linear wear cost; persistence forecast | A12, A13 |
| MAPPO / IPPO baselines | Penalty value | A19 |
| AC power flow | Own radial sweep instead of pandapower | – |
| Environment API | PettingZoo-like arrays; ideal charger power tracking instead of EV2Gym dynamics | – |
| Audit logging | Per-step projection distance and relaxation flags only; no persistent store | – |

## Not implemented

| Component | Reason |
|---|---|
| MADDPG baseline | An off-policy MARL learner with replay was out of scope for this release |
| QMIX baseline | Requires a discretized action space the paper does not specify |
| LLM-only scheduler baseline | Prompting and output format unspecified; requires running the 8B LLM as a scheduler |
| "w/o intent layer" ablation | Needs the environment to separate true user intents from registered constraints |
| Robustness study (PV error, tariff noise, message dropout, corrupted constraints) | Perturbation hooks not built |
| Communication-volume measurement for FedAvg | Not instrumented (parameter counts are available from checkpoints) |
| IEEE 34-bus feeder data | Not bundled. It must be converted from the official source, not reconstructed from memory. A JSON loader exists |
| Paper's trilingual corpus and LoRA adapter | Not released; only a demo corpus and the fine-tuning script are provided |
| Grammar-constrained decoding | Would need an extra constrained-decoding dependency; replaced by validation + retry |
| Hardware-in-the-loop, charger protocols (OCPP / ISO 15118) | Outside the paper's evaluated scope (listed there as future work) |
| Secure aggregation / differential privacy | Not part of the paper's method (noted there as a limitation) |

## Technical debt

- **Scalability.** The environment, the horizon-QP baselines and the projection's Python loop over transformers are CPU-bound. Critic attention is O(n²) per transformer. Full-scale training (1,000 agents, 300 rounds, 32 environments) has not been run, and its runtime is unknown.
- **Horizon-QP baselines are slow.** They take about 0.3–0.7 s per step on the 12-EV smoke problem because OSQP converges slowly on this nearly linear program. A dedicated LP/QP solver or warm starts across steps would help.
- **Projection gradient** assumes strict complementarity at the active set; degenerate active sets use a pseudo-inverse approximation.
- **Degradation coefficients** must be verified against the original publications (A8).
- **No pretrained weights**, and no LLM run.
- **Docker image** is written but was not built here (Docker not installed).
- PowerShell users see garbled `±` in `results.md` in the console; the file itself is UTF-8.

## Recommended next steps

1. Verify the degradation coefficients and replace the synthetic feeder and profiles with discom or real data (A1, A6, A8).
2. Run the full training on a GPU machine for 5 seeds and report *measured* results with CIs. Do not reuse the paper's illustrative numbers.
3. Add the robustness perturbations and the "w/o intent layer" ablation (environment hooks).
4. Implement MADDPG/QMIX, or drop them from the paper's comparison.
5. Run `HFParser` and LoRA fine-tuning on a real annotated trilingual corpus. Add grammar-constrained decoding.
6. Profile and vectorize the environment step and the per-transformer projection, for example by batching QPs.

## Research reproducibility

| Can be reproduced | Cannot be reproduced |
|---|---|
| The full software pipeline on synthetic data, deterministically per seed on one machine | The paper's Tables IV–VIII (they are illustrative targets, not measurements) |
| The experimental protocol (splits, seeds, baselines, ablations, metrics) under documented assumptions | Results on the unpublished Indian feeder, corpus or LoRA adapter |
| Unit-level properties: LinDistFlow vs AC, projection feasibility and KKT gradients, LP feasibility | MADDPG / QMIX / LLM-only comparisons and the robustness study |

## Production readiness assessment

Scores out of 10, for a research codebase. Not inflated.

| Aspect | Score | Justification |
|---|---|---|
| Architecture | 7 | Clean layer separation that mirrors the paper; configuration-driven; baselines share the environment and trainer |
| Code quality | 7 | Typed, documented, ruff-clean, real tests including numerical checks; some long functions (horizon QP, env step) |
| Maintainability | 7 | Small dependency set, explicit assumptions, mapping docs; no type checker in CI |
| Scalability | 3 | Only tiny workloads validated; Python loops in hot paths; O(n²) attention; slow QP baselines |
| Reliability | 5 | 53 tests passing, input validation, QP fallback solver; no long-run or failure-injection testing |
| Security | 3 | Research API without authentication, rate limiting or persistent audit; prompt-injection resistance untested |
| Reproducibility | 5 | Seeds, configs, metadata, pinned versions; exact scientific reproduction impossible (missing data, illustrative paper results) |
| Documentation | 8 | README, nine docs, traceability mapping, assumption register, honest status and smoke report |
