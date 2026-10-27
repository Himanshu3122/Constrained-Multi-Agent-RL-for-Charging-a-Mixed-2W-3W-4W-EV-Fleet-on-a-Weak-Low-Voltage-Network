# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pip install torch` fails / no matching distribution | Python too new (e.g. 3.14+) or unsupported platform | Use Python 3.10–3.13; install from `https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: solace` when running a script | Package not installed | `pip install -e .` (scripts also fall back to `src/` automatically) |
| `FileNotFoundError: .../scenarios.npz not found` | Scenarios not preprocessed for this config | `python scripts/preprocess.py --config <config>`; baseline/ablation configs reuse `scenario_set: full` |
| `ValueError: split ... needs ... days` | `data.split` sums to more than `data.n_days` | Fix the split or `n_days` |
| `learned methods need --checkpoint` | Evaluating `solace` without a checkpoint | Pass `--checkpoint "checkpoints/<name>/seed{seed}/final.pt"` |
| Evaluation is slow | `mpc` and `solver_grounded` re-solve a horizon QP every step | Limit `--max-days`, drop these methods, or run them on fewer seeds |
| Training is slow at full scale | Projection QPs in every forward and backward pass, O(n²) attention per transformer | Use a GPU (`training.device: cuda`), fewer `n_envs`, or start with smaller fleets |
| `relaxed_steps > 0` in results | Base load alone violates a limit (Proposition 1's assumption fails) | Expected on weak feeders. EVs are then only prevented from worsening the violation (A10) |
| Residual AC voltage violations with projection on | LinDistFlow ignores losses, so AC voltages are slightly lower than the linear model's | Expected (the paper reports the same effect); add margins to `v_min` if needed |
| `HFParser needs the optional dependencies` | `[llm]` extra not installed | `pip install -e ".[llm]"` and set `HF_TOKEN` for gated models |
| OSQP `PendingDeprecationWarning` about `raise_error` | OSQP 1.x API change | Already handled (`solve(raise_error=False)`); update OSQP if it reappears |
| Garbled `±` in `results.md` on Windows consoles | Console code page is not UTF-8 | The file is UTF-8; open it in an editor or run `chcp 65001` |
