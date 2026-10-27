# Smoke test report

Every entry below was executed on 2026-10-27 while building the repository. Nothing here reproduces the paper's results, which are illustrative targets in the manuscript.

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU / RAM | Intel Core Ultra 5 225U / 15.5 GB, no GPU used |
| Python | 3.12.15 (uv-managed venv). The system default Python 3.15.1 had no wheels for the stack, so it was not used |
| Packages | numpy 2.5.4, scipy 1.18.1, torch 2.14.1+cpu, osqp 1.1.3, pydantic 2.14.1, pyyaml 6.0.3, fastapi 0.162.0, uvicorn 0.54.0, httpx 0.28.1, pytest 9.1.1, ruff 0.17.2 |
| Install | `uv pip install -e ".[dev]"` (torch from the CPU index): **PASS** |

## Results

| Test | Status | Command | Observed result | Notes |
|---|---|---|---|---|
| Import validation | PASS | `python -c "import solace, solace.api, solace.training, solace.evaluation, solace.intent"` | `import ok 0.1.0` | |
| Lint | PASS | `ruff check src scripts tests` | `All checks passed!` | 15 findings fixed during development |
| Config validation | PASS | `pytest tests/unit/test_config.py` | 4 passed | default.yaml paper values, `extends` merge, validation errors, round trip |
| Unit tests | PASS | `pytest tests/unit` | 43 passed (in the full run) | Grid (LinDistFlow vs AC), projection (feasibility, optimality, KKT vs finite differences, torch backward), degradation, env, networks (GAT equivariance, masking), GAE, intent (trilingual parsing, LP closed form, counter-offers, retries), baselines |
| Integration tests | PASS | `pytest tests/integration` | 9 passed | Federated training → checkpoint → policy; centralized ablation; evaluation report; API via TestClient |
| Smoke test (pytest) | PASS | `pytest -o addopts="" tests/smoke` | 1 passed (86 s) | Runs `scripts/smoke_test.py` in a temp dir |
| **Full suite** | PASS | `pytest -o addopts="" -q -rA` | **53 passed in 214 s** | |
| Preprocess CLI | PASS | `python scripts/preprocess.py --config configs/smoke.yaml` | 12 vehicles, 8 buses, split 2/1/1 saved to `data/processed/smoke` (13.5 s incl. interpreter start) | |
| Training CLI (smoke) | PASS | `python scripts/train.py --config configs/smoke.yaml --seed 0` | 2 rounds, `final.pt`, `latest.pt`, `history.json`, `config.yaml`, `metadata.json` (20.2 s) | Round 2: λ_F = 1.3e-4, λ_D = 0, training unmet energy 37.8% |
| Evaluation CLI | PASS | `python scripts/evaluate.py --config configs/smoke.yaml --checkpoint "checkpoints/smoke/seed{seed}/final.pt"` | `results/smoke/results.md` for 6 methods (46.7 s) | Table below |
| Inference CLI (intent) | PASS | `python scripts/infer.py intent --text "Need 80% by 7 am, you can discharge up to 3 kWh but not below 40%" --context ...` | `accepted`, deadline 07:00, min_soc 0.8, V2G true, floor 0.4, 3 kWh | |
| Inference CLI (dispatch) | PASS | `python scripts/infer.py dispatch --config configs/smoke.yaml --checkpoint checkpoints/smoke/seed0/final.pt --step 48` | exit 0; raw vs projected actions and set-points printed | One earlier run reported exit -1 only because the output pipe was truncated by `Select-Object -First`; re-run exit 0 |
| Corpus CLI | PASS | `python scripts/intent_corpus.py generate --n-per-cell 5` then `evaluate --parser rule` | 60 demo requests; field F1 1.0, exact match 1.0, intent accuracy 1.0 | **Circular:** the rule parser was written against these templates. Not a measure of intent quality |
| API server startup | PASS | `uvicorn --factory solace.api.app:create_app --port 8765` + HTTP requests | `GET /health` 200; `POST /v1/intent` 200 (cost priority, 08:00); empty text → 422 | |
| Docker build | BLOCKED | `docker build -t solace-api .` | Docker is not installed on this machine | Dockerfile not validated |
| LLM parser (`HFParser`) | NOT RUN | – | – | Needs the `[llm]` extra, gated Llama weights and a GPU |
| LoRA fine-tuning | NOT RUN | `scripts/finetune_intent_lora.py` | – | Same reason |
| Full-scale training (`configs/default.yaml`) | NOT RUN | – | – | Multi-day workload; deliberately not launched |
| Full-scale environment step | PASS (manual check) | one-day episode, 1,000 EVs, synthetic feeder | Uncontrolled full-power charging: peak loading 163.5%, 727 voltage-violation bus-steps. With Layer 3: peak 103.6%, 305 bus-steps, max deviation 0.001 p.u., 0 relaxed steps; ~66 ms projection per step | Residual violations are LinDistFlow linearization error (losses). Synthetic data, demonstration only |

### Smoke evaluation table (tiny synthetic workload, 1 test day, seed 0)

Demonstrated by the smoke test. **Not** reproduced paper results.

| Method | Peak (%) | V-viol./day | Unmet (%) | Jain | Cost (kINR/d) | Degr. (kINR/d) | ISR (%) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| uncontrolled | 15.8 | 0 | 0 | 1.00 | 0.593 | 0.418 | 100 | 0.012 |
| tou | 13.6 | 0 | 0 | 1.00 | 0.482 | 0.406 | 100 | 0.034 |
| bandits | 14.8 | 0 | 0 | 1.00 | 0.506 | 0.396 | 100 | 0.094 |
| solver_grounded | 13.1 | 0 | 0.02 | 1.00 | 0.443 | 0.365 | 100 | 102 |
| mpc | 13.1 | 0 | 0.0006 | 1.00 | 0.442 | 0.365 | 100 | 301 |
| solace (2 smoke rounds) | 12.6 | 0 | 94.0 | 0.34 | 0.034 | 0.131 | 0 | 3.5 |

The SOLACE row comes from an essentially untrained policy (2 rounds × 1 episode). It charges almost nothing, hence 94% unmet energy. This is expected and shows only that the pipeline runs. The 12-EV feeder is lightly loaded, so no method causes violations.

## Failures encountered and fixes applied

| Problem | Fix |
|---|---|
| System Python 3.15 had no installed packages and no torch wheels | Created a Python 3.12 venv with uv |
| OSQP "Polishing not needed" messages printed from C code | Disabled polishing (tight tolerances kept) |
| Degradation cost ~950 kINR/day at full scale, because every episode started from a new battery and A^z is concave | Added `degradation.initial_equivalent_cycles` (A8) |
| PyTorch warning when converting loss tensors to float | Use `.item()` |
| Evaluation of the horizon-QP baselines took 172 s on the smoke problem (Python-built O(L²) rows; OSQP needed about 16k iterations) | Vectorized matrix assembly; row equilibration; objective normalization. About 5.3k iterations; evaluation now 47 s |
| Reported latency covered only the projection | Runner now measures policy computation + projection |
| `test_transformer_problem_limits_and_relaxation` failed | The test's "normal" base load already violated the voltage limit, so the code correctly relaxed the problem. Test data fixed; code unchanged |
| OSQP `PendingDeprecationWarning` about `raise_error` | Pass `raise_error=False` explicitly |

## Remaining issues and concerns

- The Docker image was never built (BLOCKED). The LLM path and LoRA fine-tuning were not run (NOT RUN).
- Horizon-QP baselines still take 0.1–0.3 s per step at smoke scale. They will be slow at 1,000 EVs.
- `fastapi.testclient` emits a Starlette deprecation warning about `httpx` (filtered in pytest config). It is harmless with the pinned versions.
- Full test suite runtime is about 3.5 minutes on this CPU, dominated by the smoke test and the QP baselines.
