# Paper-to-repository mapping

Source of truth for the science: the manuscript `main.tex` (one directory above this repository). Status legend:

- **Implemented**: follows the paper's specification;
- **Implemented (assumption)**: runs, but needed decisions listed in [assumptions.md](assumptions.md);
- **Partial**: some of the paper's scope is missing;
- **Not implemented**: absent, with the reason.

## Architecture (Sections III–IV, Fig. 2)

| Paper concept | Module | Status | Notes |
|---|---|---|---|
| Layer 1: LLM intent translation | `solace/intent/` | Implemented (assumption) | Schema, validator, ≤ 2 retries, default + operator flag, per-EV feasibility LP, counter-offers (A14). LLM parser `HFParser` (Llama-3.1-8B-Instruct + LoRA) is implemented but was not executed here (weights gated, no GPU). Grammar-constrained decoding replaced by JSON extraction + validation (A15) |
| Constraint object fields (deadline, min_soc, v2g_allowed, soc_floor, max_discharge_kwh, priority) | `solace/intent/schema.py` | Implemented | `v2g_allowed=false` default when no limit is stated |
| Layer 2: CTDE MARL, shared per-class actors | `solace/agents/networks.py::ClassActors` | Implemented | 3 actors (2W/3W/4W), MLP 2×128 tanh |
| GAT + mean-field critic (eqs. 12–13) | `GATMeanFieldCritic` | Implemented | Attention over EVs on the same transformer; mean projected action of all other EVs; three value heads (reward, c_F, c_D) |
| Layer 3: differentiable convex projection (eq. 11) | `solace/safety/projection.py` | Implemented (assumption) | OSQP forward solve; gradients from the KKT active set (OptNet-style) instead of cvxpylayers; relaxation when Proposition 1's assumption fails (A10) |
| Federated averaging over 8 aggregators (eq. 15) | `solace/training/federated.py` | Implemented (assumption) | Sequential simulation of aggregators in one process (A11) |
| Audit logging of constraints/corrections | `EpisodeMetrics`, API responses | Partial | Projection distance and relaxed steps are recorded per step; no persistent audit store |
| Deployment split (actors on chargers, projection + Layer 1 on edge server) | `solace/api/` (`/v1/intent`, `/v1/dispatch`), `LearnedPolicy` | Partial | Services are provided as one research API; no charger protocol integration |

## Mathematical model (Section V)

| Equation | Module | Status |
|---|---|---|
| (1) Constrained Dec-POMDP | `solace/env/charging_env.py` | Implemented |
| (2) Battery energy dynamics with η_c, η_d | `ChargingEnv.step` | Implemented (η values: assumption) |
| (3)–(4) LinDistFlow | `solace/grid/lindistflow.py` | Implemented (path-impedance form) |
| (5) Voltage bounds and kVA limit with 16-gon inner approximation | `build_transformer_problem` | Implemented |
| (6) Per-step cost | `ChargingEnv.step` | Implemented + soft deadline term (A9) |
| (7) Semi-empirical degradation | `solace/degradation.py` | Implemented (parameters: A8) |
| (8) Jain's index | `jain_index`, `EpisodeMetrics.jain` | Implemented |
| (9)–(10) CMDP, Lagrangian, projected dual ascent | `FederatedTrainer.train`, `AggregatorLearner` | Implemented |
| (11) Projection QP | `solve_projection`, `differentiable_project` | Implemented |
| Proposition 1 | `build_transformer_problem` (+ unit test) | Implemented; violated assumption handled by A10 |
| (12)–(13) GAT attention, mean field, critic | `GATMeanFieldCritic` | Implemented |
| (14) Clipped PPO on Lagrangian advantage with GAE | `AggregatorLearner.update`, `rollout.gae` | Implemented (ratio convention: A11) |
| (15) FedAvg | `fed_average` | Implemented |
| Algorithm 1 | `FederatedTrainer.train` | Implemented (A11) |

## Data and experimental setup (Section VI)

| Paper item | Module / config | Status | Notes |
|---|---|---|---|
| Synthetic Indian feeder (11 kV, 4 × 630 kVA, 248 buses, 1,340 households) | `solace/grid/feeder.py`, `configs/default.yaml` | Implemented (assumption) | Counts match; topology/impedances assumed (A1, A2) |
| IEEE 34-bus feeder | `load_feeder_json` | Not implemented (data) | Loader exists; the IEEE 34-bus data must be converted by the user (not bundled, not reconstructed from memory) |
| pandapower AC power flow | `solace/grid/powerflow.py` | Implemented (substitute) | Own backward/forward sweep for radial networks; no pandapower dependency |
| PettingZoo parallel API, EV2Gym charger dynamics | `ChargingEnv` | Partial | Array-based reset/step equivalent to the parallel API; charger dynamics are ideal power tracking, not EV2Gym's |
| Fleet Table II | `configs/default.yaml::fleet`, `solace/data/fleet.py` | Implemented (A3) | |
| ACN-Data for 4W sessions | `solace/data/sessions.py::AcnSampler` | Implemented (data not bundled) | Falls back to a labelled synthetic generator |
| 2W/3W generator calibrated to Bolt.Earth statistics | `generate_day_sessions` | Implemented (assumption) | A4 |
| West Bengal irradiance, ToD tariff | `solace/data/profiles.py` | Implemented (assumption) | Synthetic defaults; real GHI via `data.irradiance_path` (A6) |
| 365 days, 300/35/30 split | `build_scenarios` | Implemented | Split by day |
| Trilingual request corpus (14,400 requests) | `solace/intent/corpus.py` | Not implemented (data) | Only a small demo corpus is generated; the paper's corpus is not released |
| LoRA fine-tuning (rank 16, α 32, lr 2e-4, 3 epochs) | `scripts/finetune_intent_lora.py` | Implemented, not executed | Requires `.[llm]` extra, weights and a GPU |
| Hyperparameters Table III | `configs/default.yaml` | Implemented | Batch size and LoRA dropout from the earlier Table III version are script defaults |
| 5 seeds, 30 test days | `evaluation.seeds`, test split | Implemented | |

## Baselines (Section VI-D)

| Baseline | Module | Status |
|---|---|---|
| (i) Uncontrolled | `baselines/rule_based.py::UncontrolledPolicy` | Implemented |
| (ii) ToU rule | `ToUPolicy` | Implemented |
| (iii) Thompson-sampling bandits | `baselines/bandits.py` | Implemented (A12) |
| (iv) IPPO | `configs/baselines/ippo.yaml` + trainer | Implemented (A19) |
| (v) MADDPG | – | Not implemented: off-policy MARL learner not built in this release |
| (vi) QMIX | – | Not implemented: needs a discretized action space not specified in the paper |
| (vii) MAPPO (concatenated critic, penalties) | `configs/baselines/mappo.yaml` + `ConcatCritic` | Implemented (A19) |
| (viii) LLM-only scheduler | – | Not implemented: requires running the LLM as a scheduler; no specification of its prompt/output |
| (ix) Solver-grounded single agent | `baselines/horizon_qp.py::SolverGroundedPolicy` | Implemented (A13) |
| (x) Centralized MPC, perfect foresight | `CentralizedMPCPolicy` | Implemented (A13) |

## Evaluation (Section VII)

| Paper item | Module | Status |
|---|---|---|
| Peak loading, AC voltage violations, unmet energy, Jain, cost, degradation cost, latency | `EpisodeMetrics`, `evaluation/runner.py` | Implemented (definitions: A17) |
| Intent satisfaction rate, projection distance | same | Implemented |
| Intent metrics: field F1, exact match, infeasible-passed, confusion matrix | `intent/metrics.py`, `scripts/intent_corpus.py` | Implemented (A16) |
| Training curves (return, projection distance) | `history.json` from `scripts/train.py` | Implemented (data only, no plotting) |
| Ablations: w/o projection, execution-only projection, w/o fairness, w/o degradation, w/o mean field, centralized | `configs/ablations/*.yaml` | Implemented |
| Ablation: w/o intent layer | – | Not implemented: needs separate "true" and "registered" constraints in the environment |
| Scalability (100–2,000 EVs, scaled ratings) | config overrides | Supported, not run |
| Robustness (PV forecast error, tariff noise, message dropout, corrupted constraints) | – | Not implemented |
| Communication volume of FedAvg | – | Not implemented (parameter count is available from the checkpoint) |
| Statistical reporting | `aggregate` | Mean, std and 95% CI over days × seeds |
| Paper's numerical results | – | **Not reproduced.** The paper's numbers are illustrative targets (HR-01) |
