"""Typed constraint object produced by Layer 1 (paper Section III-B).

Fields: deadline, min_soc, v2g_allowed, soc_floor, max_discharge_kwh, priority.
The safe default is ``v2g_allowed = False`` when no discharge limit is stated.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class Priority(str, Enum):
    normal = "normal"
    cost = "cost"
    urgent = "urgent"


def parse_hhmm(value: str) -> int:
    """'HH:MM' -> minutes after midnight."""
    m = _TIME.match(value)
    if not m:
        raise ValueError(f"time must be HH:MM (24 h), got '{value}'")
    return int(m.group(1)) * 60 + int(m.group(2))


class ChargingConstraint(BaseModel):
    deadline: str = Field(description="Local time HH:MM by which min_soc is required")
    min_soc: float = Field(ge=0.0, le=1.0)
    v2g_allowed: bool = False
    soc_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    max_discharge_kwh: float = Field(default=0.0, ge=0.0)
    priority: Priority = Priority.normal

    model_config = {"extra": "forbid"}

    @field_validator("deadline")
    @classmethod
    def _deadline(cls, v: str) -> str:
        parse_hhmm(v)
        return v

    @model_validator(mode="after")
    def _consistency(self) -> ChargingConstraint:
        if not self.v2g_allowed and self.max_discharge_kwh > 0:
            raise ValueError("max_discharge_kwh > 0 requires v2g_allowed = true")
        if self.soc_floor > self.min_soc:
            raise ValueError("soc_floor must not exceed min_soc")
        return self


class SessionContext(BaseModel):
    """Session context passed to Layer 1 with the request (paper Section III-B)."""

    vehicle_class: str = Field(pattern=r"^(2W|3W|4W)$")
    battery_kwh: float = Field(gt=0)
    soc: float = Field(ge=0.0, le=1.0)  # plug-in SoC
    charger_kw: float = Field(gt=0)
    plug_in_time: str = "18:00"
    departure_time: str | None = None
    v2g_capable: bool = False

    @field_validator("plug_in_time", "departure_time")
    @classmethod
    def _times(cls, v: str | None) -> str | None:
        if v is not None:
            parse_hhmm(v)
        return v


def default_constraint(ctx: SessionContext) -> ChargingConstraint:
    """Fallback used after failed validation (flagged for operator review)."""
    deadline = ctx.departure_time or f"{(parse_hhmm(ctx.plug_in_time) // 60 + 8) % 24:02d}:00"
    target = max(0.8, ctx.soc)
    return ChargingConstraint(deadline=deadline, min_soc=round(target, 2), v2g_allowed=False,
                              soc_floor=min(0.2, target), max_discharge_kwh=0.0, priority=Priority.normal)
