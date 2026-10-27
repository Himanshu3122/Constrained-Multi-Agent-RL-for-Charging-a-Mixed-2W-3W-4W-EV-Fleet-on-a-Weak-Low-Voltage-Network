"""FastAPI application.

Routes
------
GET  /health        liveness
POST /v1/intent     Layer 1: request text + context -> constraint or counter-offer
POST /v1/dispatch   Layer 3: raw EV actions for one transformer -> projected set-points

The parser is selected with ``SOLACE_INTENT_PARSER`` (``rule`` by default;
``hf`` loads ``SOLACE_HF_MODEL`` and optionally ``SOLACE_HF_ADAPTER``).
This is a research service; it has no authentication (see SECURITY.md).
Run with ``uvicorn --factory solace.api.app:create_app``.
"""

from __future__ import annotations

import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException

from solace import __version__
from solace.api.schemas import DispatchRequest, DispatchResponse, IntentRequest, IntentResponse
from solace.grid.feeder import LVNetwork
from solace.intent import FeasibilityChecker, IntentPipeline, RuleBasedParser
from solace.intent.parsers import IntentParser
from solace.safety.projection import build_transformer_problem, solve_projection


def _make_parser() -> IntentParser:
    kind = os.environ.get("SOLACE_INTENT_PARSER", "rule").lower()
    if kind == "rule":
        return RuleBasedParser()
    if kind == "hf":
        from solace.intent.parsers import HFParser

        return HFParser(os.environ.get("SOLACE_HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                        os.environ.get("SOLACE_HF_ADAPTER") or None)
    raise ValueError(f"unknown SOLACE_INTENT_PARSER '{kind}'")


def create_app(parser: IntentParser | None = None) -> FastAPI:
    app = FastAPI(title="SOLACE research API", version=__version__)
    pipeline = IntentPipeline(parser or _make_parser(), FeasibilityChecker())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/intent", response_model=IntentResponse)
    def intent(req: IntentRequest) -> IntentResponse:
        headroom = np.asarray(req.headroom_kw) if req.headroom_kw is not None else None
        res = pipeline.process(req.text, req.context, headroom)
        return IntentResponse(status=res.status, constraint=res.constraint, requested=res.requested,
                              max_reachable_soc=res.max_reachable_soc, attempts=res.attempts,
                              flagged_for_review=res.flagged_for_review, validation_errors=res.validation_errors,
                              reason=res.reason)

    @app.post("/v1/dispatch", response_model=DispatchResponse)
    def dispatch(req: DispatchRequest) -> DispatchResponse:
        net_spec = req.network
        try:
            net = LVNetwork(transformer_id=0, parent=np.asarray(net_spec.parent), r_pu=np.asarray(net_spec.r_pu),
                            x_pu=np.asarray(net_spec.x_pu), households=np.zeros(len(net_spec.parent), dtype=int),
                            s_rated_pu=net_spec.s_rated_pu)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        s_base = net_spec.s_base_kva
        bus = np.array([ev.bus for ev in req.evs])
        rated = np.array([ev.rated_kw for ev in req.evs])
        raw = np.array([ev.raw_action for ev in req.evs])
        tic = time.perf_counter()
        prob = build_transformer_problem(
            R_path=net.R_path, X_path=net.X_path, bus=bus, p_rated_pu=rated / s_base,
            lo=np.array([ev.lo for ev in req.evs]), hi=np.array([ev.hi for ev in req.evs]),
            base_p_pu=np.asarray(req.base_p_kw) / s_base, base_q_pu=np.asarray(req.base_q_kvar) / s_base,
            s_rated_pu=net_spec.s_rated_pu, v_min=net_spec.v_min, v_max=net_spec.v_max,
            polygon_sides=net_spec.polygon_sides)
        a_hat = solve_projection(raw, prob)
        latency = (time.perf_counter() - tic) * 1e3
        return DispatchResponse(projected_action=a_hat.tolist(), setpoint_kw=(a_hat * rated).tolist(),
                                projection_distance_kw=float(np.linalg.norm((a_hat - raw) * rated)),
                                relaxed=prob.relaxed, latency_ms=latency)

    return app
