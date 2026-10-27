# Reproducibility

Two different claims are involved, and they need to be kept apart:

| Claim | Status |
|---|---|
| **Software pipeline reproducibility**: running the same commands with the same config and seed gives the same scenarios and the same training trajectory | Supported (see below) |
| **Scientific result reproduction**: obtaining the paper's Tables IV–VIII | **Not possible.** The paper's numbers are illustrative targets, and the feeder, corpus and several model parameters are unpublished |

## What is controlled

- **Seeds.** `seed_everything` seeds Python, NumPy and torch, and enables torch deterministic algorithms (`warn_only`). Scenario generation uses an explicit NumPy `Generator`. Training uses `cfg.seed` (or `--seed`) for action sampling, minibatch shuffling and network initialization.
- **Configs.** Each run writes the resolved `config.yaml` next to its outputs. Configs are versioned in `configs/` and support `extends:`.
- **Metadata.** `metadata.json` records the Python, NumPy, SciPy, torch and OSQP versions, the platform, the git commit (when available) and the seed.
- **Splits.** Days are split 300/35/30 *before* any learning, so test days never enter training.
- **Pinned versions.** `requirements.txt` holds the versions used for the validation run.

Sources of nondeterminism that remain: OSQP's iterative tolerances, multithreaded BLAS reductions, and different CPU/GPU kernels across machines. Results should match closely on one machine and approximately elsewhere.

## Reproducing the intended experiments

```bash
python scripts/preprocess.py --config configs/default.yaml
for s in 0 1 2 3 4; do
  python scripts/train.py --config configs/default.yaml --seed $s
  python scripts/train.py --config configs/baselines/mappo.yaml --seed $s
  python scripts/train.py --config configs/baselines/ippo.yaml --seed $s
  for a in no_projection exec_projection_only no_fairness no_degradation no_mean_field centralized; do
    python scripts/train.py --config configs/ablations/$a.yaml --seed $s
  done
done
python scripts/evaluate.py --config configs/default.yaml \
  --checkpoint "checkpoints/full/seed{seed}/final.pt" \
  --learned "mappo=checkpoints/mappo/seed{seed}/final.pt" "ippo=checkpoints/ippo/seed{seed}/final.pt"
```

This executes the paper's protocol (5 seeds, 30 test days, baselines and ablations) under the assumptions in [assumptions.md](assumptions.md). It is a heavy workload and was not run while building the repository.

## What blocks exact reproduction

| Missing artifact | Effect |
|---|---|
| Measured results (paper shows targets only) | Nothing to compare against |
| Feeder topology/impedances, Indian load profiles | A1, A2, A6 approximations |
| 2W/3W session data or generator parameters | A4 approximation |
| ACN-Data subset, irradiance source, tariff values | Loaders provided, defaults synthetic |
| Trilingual request corpus (14,400 requests) | Demo corpus only |
| Llama-3.1-8B LoRA adapter | Fine-tuning script provided, not run |
| Degradation parameters, reward weights for the deadline term | A8, A9 |
| MADDPG, QMIX, LLM-only scheduler details | Not implemented |
