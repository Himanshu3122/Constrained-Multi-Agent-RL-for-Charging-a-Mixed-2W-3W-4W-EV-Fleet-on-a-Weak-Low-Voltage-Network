# Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint "checkpoints/full/seed{seed}/final.pt" \
    --learned "mappo=checkpoints/mappo/seed{seed}/final.pt" "ippo=checkpoints/ippo/seed{seed}/final.pt"
```

- `--methods` overrides `evaluation.methods`. Available: `uncontrolled`, `tou`, `bandits`, `solver_grounded`, `mpc`, `solace`, plus any `--learned NAME=PATH`.
- `{seed}` in a checkpoint path is replaced by each seed in `evaluation.seeds` (paper: 5 seeds).
- `--split test` (default) uses the held-out test days; `--max-days N` limits them.
- Outputs go to `results/<name>/`: `results.md` (table), `metrics.json` (per-episode rows and aggregates), `config.yaml`, `metadata.json`.

## Metrics (per episode, `EpisodeMetrics.summary`)

| Metric | Definition |
|---|---|
| `peak_loading_pct` | max over transformers and steps of AC apparent power / rating, in % |
| `voltage_violations` | AC bus-steps with \|V\| outside [0.95, 1.05] p.u. by more than 1e-4 |
| `unmet_energy_pct` | Σ (requested − delivered)⁺ / Σ requested, measured at each session's deadline |
| `jain` | Jain index of delivered/requested ratios (clipped to [0, 1]) |
| `cost_kinr` | net energy cost (V2G discharge credited at the tariff), kINR/day |
| `degradation_kinr` | κ · capacity loss for plugged-in vehicles, kINR/day |
| `intent_satisfaction_pct` | sessions with SoC ≥ min_soc − 0.01 at the deadline and discharge cap respected |
| `projection_distance_kw` | ‖(â − ã) ⊙ P_rated‖₂ per step, averaged |
| `latency_ms` | policy computation + projection per step, measured on the evaluation machine |
| `relaxed_steps` | transformer-steps where Proposition 1's assumption failed (A10) |

Aggregation: mean, sample standard deviation and normal-approximation 95% CI over (test days × seeds). No significance tests are computed. Use `metrics.json` for paired tests.

## Baselines

See [paper_implementation_mapping.md](paper_implementation_mapping.md#baselines-section-vi-d). MADDPG, QMIX and the LLM-only scheduler are not implemented.

## Ablations

Train each config in `configs/ablations/`, then evaluate it with `--learned NAME=checkpoints/ablation_<x>/seed{seed}/final.pt`. The "w/o intent layer" ablation is not implemented.

## Relation to the paper's tables

The paper's Tables IV–VIII contain **illustrative target values, not measurements** (marked in `main.tex`). This repository has not reproduced them. Any table it produces is a local measurement under the assumptions in [assumptions.md](assumptions.md) and should be reported as such.

## Intent layer

```bash
python scripts/intent_corpus.py generate --n-per-cell 5 --out data/corpus/demo.jsonl
python scripts/intent_corpus.py evaluate --corpus data/corpus/demo.jsonl --parser rule
```

This reports field F1, exact match, per-field accuracy, intent-type accuracy, the confusion matrix and the counter-offer rate. The demo corpus is template-generated, and the rule parser was written against the same templates, so its near-perfect score is **not** evidence of intent-extraction quality. Use a real annotated corpus and `--parser hf`.
