"""Intent translation with solver check (paper Fig. 1).

parse -> validate (<= max_retries re-prompts with the validator message) ->
default constraint + operator flag if still invalid -> per-EV feasibility LP ->
register, or return a counter-offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from solace.intent.feasibility import FeasibilityChecker
from solace.intent.parsers import IntentParser
from solace.intent.schema import ChargingConstraint, SessionContext, default_constraint
from solace.intent.validator import validate_constraint


@dataclass
class IntentResult:
    status: Literal["accepted", "counter_offer"]
    constraint: ChargingConstraint  # what is registered if the user accepts (or on timeout)
    requested: ChargingConstraint  # what the parser produced (or the default)
    max_reachable_soc: float
    attempts: int
    flagged_for_review: bool = False
    validation_errors: list[str] = field(default_factory=list)
    reason: str = ""


class IntentPipeline:
    def __init__(self, parser: IntentParser, checker: FeasibilityChecker | None = None, max_retries: int = 2):
        self.parser = parser
        self.checker = checker or FeasibilityChecker()
        self.max_retries = max_retries

    def process(self, text: str, ctx: SessionContext, headroom_kw: np.ndarray | None = None) -> IntentResult:
        feedback: list[str] | None = None
        errors: list[str] = []
        con: ChargingConstraint | None = None
        attempts = 0
        while attempts <= self.max_retries:  # first try + max_retries re-prompts
            attempts += 1
            raw = self.parser.parse(text, ctx, feedback)
            con, errors = validate_constraint(raw, ctx)
            if con is not None:
                break
            feedback = errors
        flagged = con is None
        if con is None:
            con = default_constraint(ctx)
        result = self.checker.check(con, ctx, headroom_kw)
        if result.feasible:
            return IntentResult("accepted", con, con, result.max_reachable_soc, attempts, flagged, errors)
        assert result.counter_offer is not None
        return IntentResult("counter_offer", result.counter_offer, con, result.max_reachable_soc, attempts, flagged,
                            errors, result.reason)
