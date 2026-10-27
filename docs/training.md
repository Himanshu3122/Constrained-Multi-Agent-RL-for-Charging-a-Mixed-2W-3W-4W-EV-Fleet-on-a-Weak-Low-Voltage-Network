# Training

## Commands

```bash
# 1. scenarios (feeder, fleet, 365 days split 300/35/30 for the full config)
python scripts/preprocess.py --config configs/default.yaml          # -> data/processed/full/
# 2. SOLACE, one seed
python scripts/train.py --config configs/default.yaml --seed 0      # -> checkpoints/full/seed0/
# learned baselines and ablations reuse the "full" scenarios (scenario_set: full)
python scripts/train.py --config configs/baselines/mappo.yaml --seed 0
python scripts/train.py --config configs/ablations/no_projection.yaml --seed 0
```

For a quick check, use `configs/smoke.yaml`: 12 EVs, 1 transformer, 4 days, 2 rounds.

## What happens

`FederatedTrainer` (Algorithm 1):

1. Runs an uncontrolled, no-V2G reference on up to 3 training days. Its per-agent capacity loss normalizes the degradation cost c_D, so the budget `d_degradation_rel = 1.1` means "1.1× the no-V2G degradation".
2. For each round: copies the global actor and critic to every aggregator, then repeats `local_iters` times:
   - run `n_envs` joint episodes on random training days, with each EV acting with its aggregator's actor and Layer 3 in the loop;
   - have each aggregator run `ppo_epochs` × `minibatches` updates on its own agents' transitions: clipped PPO on the Lagrangian advantage, entropy bonus, the projection term β‖Proj(μ) − μ‖² (when `projection_gradient` and `use_projection`), and value regression on three heads.
3. Runs FedAvg (weighted by transitions), then projected dual ascent on λ_F (fairness budget 0.06) and λ_D.

## Outputs

`checkpoints/<name>/seed<seed>/`:

| File | Content |
|---|---|
| `latest.pt` | checkpoint after each round |
| `final.pt` | checkpoint after the last round |
| `history.json` | per round: λ_F, λ_D, Ĵ_F, Ĵ_D, losses, mean training-episode metrics |
| `config.yaml` | resolved configuration |
| `metadata.json` | Python / library versions, platform, git commit, seed |

Checkpoint contents: global actor and critic `state_dict`s, configuration, multipliers, round, degradation reference and history (`solace/training/checkpoint.py`). **No pretrained checkpoints are distributed.**

## Cost

The full configuration (1,000 agents, 300 rounds × 5 local iterations × 32 environments × 96 steps, 8 aggregators) is a large workload. Each step solves 4 projection QPs and 4 AC power flows; each update solves projection QPs again for the gradient. The smoke configuration finishes 2 rounds in a few seconds on a laptop CPU (see SMOKE_TEST_REPORT.md). The full run was **not** executed while building this repository.

## Key configuration

| Key | Default | Source |
|---|---|---|
| `training.rounds`, `local_iters`, `n_envs` | 300, 5, 32 | paper |
| `training.lr_actor`, `lr_critic` | 3e-4, 5e-4 | paper |
| `training.ppo_epochs`, `minibatches`, `clip_eps`, `entropy_coef` | 10, 4, 0.2, 0.01 | paper |
| `training.d_fairness`, `d_degradation_rel`, `dual_lr` | 0.06, 1.1, 1e-3 | paper |
| `training.federated`, `aggregators_per_transformer` | true, 2 | paper |
| `training.projection_gradient` | true | paper |
| `env.beta_projection` | 0.05 | paper |
| `env.unmet_penalty_inr_per_kwh`, `reward_scale` | 30, 1e-3 | assumption A9 |
| `agent.critic_type` | `gat_mf` (`concat` = MAPPO baseline, `local` = IPPO) | paper |
