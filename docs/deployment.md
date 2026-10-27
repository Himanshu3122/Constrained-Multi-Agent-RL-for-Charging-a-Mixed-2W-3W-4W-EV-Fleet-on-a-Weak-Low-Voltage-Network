# Deployment (research service)

The repository ships one container: the FastAPI research service (`/v1/intent`, `/v1/dispatch`). It is **not** production-hardened. It has no authentication, rate limiting, persistent audit log, charger-protocol (OCPP / ISO 15118) integration or hardware-in-the-loop validation. The paper itself lists HIL validation as future work.

## Docker

```bash
docker build -t solace-api .
docker run --rm -p 8000:8000 --env-file .env solace-api
curl http://127.0.0.1:8000/health
```

The image installs CPU-only PyTorch and the package, and runs `uvicorn --factory solace.api.app:create_app`. With `SOLACE_INTENT_PARSER=hf` the container would also need the `[llm]` extra, the model weights and (realistically) a GPU. The default image does not include them.

## Mapping to the paper's deployment

| Paper | Here |
|---|---|
| Per-class actors on charger controllers | `LearnedPolicy` (small MLPs; export to ONNX is straightforward but not provided) |
| Projection + Layer 1 on the aggregator edge server | `/v1/dispatch` (one transformer per request) and `/v1/intent` |
| Audit log of constraints, counter-offers and corrections | Returned in API responses; persistence is up to the deployer |

Before any real-world use, add authentication, input provenance, persistent tamper-evident logging and fail-safe behavior when the QP fails or telemetry is missing, and validate against AC power flow with real feeder data.
