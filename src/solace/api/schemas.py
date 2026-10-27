"""Request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from solace.intent.schema import ChargingConstraint, SessionContext


class IntentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    context: SessionContext
    headroom_kw: list[float] | None = Field(default=None, max_length=192)


class IntentResponse(BaseModel):
    status: str
    constraint: ChargingConstraint
    requested: ChargingConstraint
    max_reachable_soc: float
    attempts: int
    flagged_for_review: bool
    validation_errors: list[str]
    reason: str


class NetworkSpec(BaseModel):
    """Radial LV network below one transformer (bus 0 = transformer secondary)."""

    parent: list[int] = Field(min_length=2, max_length=2000)
    r_pu: list[float]
    x_pu: list[float]
    s_base_kva: float = Field(gt=0)
    s_rated_pu: float = Field(default=1.0, gt=0)
    v_min: float = 0.95
    v_max: float = 1.05
    polygon_sides: int = Field(default=16, ge=4, le=64)

    @model_validator(mode="after")
    def _lengths(self) -> NetworkSpec:
        if not (len(self.parent) == len(self.r_pu) == len(self.x_pu)):
            raise ValueError("parent, r_pu and x_pu must have equal length")
        return self


class EVDispatch(BaseModel):
    bus: int = Field(ge=1)
    rated_kw: float = Field(gt=0)
    lo: float = Field(ge=-1.0, le=0.0, description="lower bound on normalized action (discharge)")
    hi: float = Field(ge=0.0, le=1.0)
    raw_action: float = Field(ge=-1.0, le=1.0)


class DispatchRequest(BaseModel):
    network: NetworkSpec
    base_p_kw: list[float]
    base_q_kvar: list[float]
    evs: list[EVDispatch] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def _check(self) -> DispatchRequest:
        n = len(self.network.parent)
        if len(self.base_p_kw) != n or len(self.base_q_kvar) != n:
            raise ValueError("base_p_kw and base_q_kvar need one value per bus")
        if any(ev.bus >= n for ev in self.evs):
            raise ValueError("EV bus index out of range")
        return self


class DispatchResponse(BaseModel):
    projected_action: list[float]
    setpoint_kw: list[float]
    projection_distance_kw: float
    relaxed: bool
    latency_ms: float
