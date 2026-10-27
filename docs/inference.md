# Inference

## Layer 1: request → constraint (CLI)

```bash
python scripts/infer.py intent \
  --text "Need 80% by 7 am, you can discharge up to 3 kWh but not below 40%" \
  --context '{"vehicle_class": "4W", "battery_kwh": 40, "soc": 0.3, "charger_kw": 7.4, "plug_in_time": "20:00", "departure_time": "09:00", "v2g_capable": true}'
```

The output contains `status` (`accepted` or `counter_offer`), the registered `constraint`, the parser's `requested` constraint, `max_reachable_soc`, the number of parse attempts and whether the request was flagged for operator review. `--parser hf --hf-model <id> --hf-adapter <dir>` uses the LLM (optional extra).

## Layers 2 + 3: one dispatch step from a trained policy (CLI)

```bash
python scripts/infer.py dispatch --config configs/smoke.yaml \
  --checkpoint checkpoints/smoke/seed0/final.pt --step 48
```

This replays the first test day up to `--step` with the learned actors and the projection, then prints raw actions, projected actions and set-points (kW) for the plugged-in EVs, transformer loading and AC violations at that step.

## Programmatic use

```python
from solace.training import LearnedPolicy
from solace.env import ChargingEnv

policy = LearnedPolicy("checkpoints/smoke/seed0/final.pt")
env = ChargingEnv(policy.cfg, scenarios, use_projection=True)
obs = env.reset(day); policy.reset(env)
a_raw = policy.act(env, obs)          # deterministic per-class actors (decentralized)
result = env.step(a_raw)              # Layer 3 projection + AC power flow inside
```

## API

```bash
uvicorn --factory solace.api.app:create_app --host 127.0.0.1 --port 8000
```

| Route | Body | Returns |
|---|---|---|
| `GET /health` | – | `{"status": "ok", "version": ...}` |
| `POST /v1/intent` | `{"text", "context": SessionContext, "headroom_kw": [..]?}` | constraint or counter-offer |
| `POST /v1/dispatch` | `{"network": {parent, r_pu, x_pu, s_base_kva, ...}, "base_p_kw", "base_q_kvar", "evs": [{bus, rated_kw, lo, hi, raw_action}]}` | projected actions, set-points, projection distance, `relaxed`, latency |

`SOLACE_INTENT_PARSER=hf` (with `SOLACE_HF_MODEL`, `SOLACE_HF_ADAPTER`) switches `/v1/intent` to the LLM. Interactive docs are at `/docs`.
