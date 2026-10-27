"""Schema validator (paper: rejects out-of-range or contradictory values)."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from solace.intent.schema import ChargingConstraint, SessionContext

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse the first JSON object in ``raw`` (LLM output may carry extra text)."""
    if isinstance(raw, dict):
        return raw
    m = _JSON_OBJECT.search(raw)
    if not m:
        raise ValueError("no JSON object found in model output")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def validate_constraint(raw: str | dict[str, Any], ctx: SessionContext) -> tuple[ChargingConstraint | None, list[str]]:
    """Return (constraint, []) if valid, else (None, error messages for the re-prompt)."""
    try:
        data = extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, [f"invalid JSON: {exc}"]
    try:
        con = ChargingConstraint.model_validate(data)
    except ValidationError as exc:
        return None, [f"{'.'.join(str(p) for p in e['loc']) or 'object'}: {e['msg']}" for e in exc.errors()]
    errors = []
    if con.v2g_allowed and not ctx.v2g_capable:
        errors.append("v2g_allowed: this vehicle/charger is not V2G capable")
    if con.max_discharge_kwh > ctx.battery_kwh:
        errors.append("max_discharge_kwh exceeds battery capacity")
    return (None, errors) if errors else (con, [])
