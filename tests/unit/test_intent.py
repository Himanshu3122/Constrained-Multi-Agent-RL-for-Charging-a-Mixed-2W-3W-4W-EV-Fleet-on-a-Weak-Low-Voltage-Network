import json

import numpy as np
import pytest
from pydantic import ValidationError

from solace.intent import (
    ChargingConstraint,
    FeasibilityChecker,
    IntentPipeline,
    RuleBasedParser,
    SessionContext,
    validate_constraint,
)
from solace.intent.corpus import generate_demo_corpus, intent_type
from solace.intent.metrics import intent_metrics

CTX = SessionContext(vehicle_class="4W", battery_kwh=40.0, soc=0.3, charger_kw=7.4, plug_in_time="20:00",
                     departure_time="09:00", v2g_capable=True)


def parse(text, ctx=CTX):
    return json.loads(RuleBasedParser().parse(text, ctx))


def test_schema_rejects_contradictions():
    with pytest.raises(ValidationError):
        ChargingConstraint(deadline="08:00", min_soc=0.8, v2g_allowed=False, max_discharge_kwh=3.0)
    with pytest.raises(ValidationError):
        ChargingConstraint(deadline="25:00", min_soc=0.8)
    with pytest.raises(ValidationError):
        ChargingConstraint(deadline="08:00", min_soc=0.5, soc_floor=0.6)


def test_validator_reports_errors_for_reprompt():
    con, errs = validate_constraint("no json here", CTX)
    assert con is None and "invalid JSON" in errs[0]
    ctx_no_v2g = CTX.model_copy(update={"v2g_capable": False})
    con, errs = validate_constraint({"deadline": "07:00", "min_soc": 0.8, "v2g_allowed": True,
                                     "soc_floor": 0.3, "max_discharge_kwh": 2, "priority": "normal"}, ctx_no_v2g)
    assert con is None and "V2G" in errs[0]


@pytest.mark.parametrize("text,deadline,min_soc,v2g,floor,kwh", [
    ("Need 80% by 7 am, you can discharge up to 3 kWh but not below 40%.", "07:00", 0.8, True, 0.4, 3.0),
    ("I need 90% by 6 am, no V2G please.", "06:00", 0.9, False, 0.2, 0.0),
    ("सुबह 7 बजे तक 70% चाहिए, 2 यूनिट तक डिस्चार्ज कर सकते हो पर 50% से नीचे नहीं।", "07:00", 0.7, True, 0.5, 2.0),
    ("সকাল ৮টার মধ্যে ৬০% চার্জ চাই, বিক্রি করবেন না।", "08:00", 0.6, False, 0.2, 0.0),
])
def test_rule_parser_trilingual(text, deadline, min_soc, v2g, floor, kwh):
    out = parse(text)
    assert out["deadline"] == deadline
    assert out["min_soc"] == pytest.approx(min_soc)
    assert out["v2g_allowed"] is v2g
    assert out["soc_floor"] == pytest.approx(floor)
    assert out["max_discharge_kwh"] == pytest.approx(kwh)


def test_feasibility_lp_matches_closed_form():
    chk = FeasibilityChecker(dt_minutes=15, eta_charge=0.95)
    reach = chk.max_reachable_soc(CTX, steps=8)  # 2 h at 7.4 kW
    expected = (0.3 * 40 + 0.95 * 7.4 * 2.0) / 40
    assert reach == pytest.approx(expected, rel=1e-6)
    capped = chk.max_reachable_soc(CTX, steps=8, headroom_kw=np.full(8, 2.0))
    assert capped == pytest.approx((0.3 * 40 + 0.95 * 2.0 * 2.0) / 40, rel=1e-6)


def test_counter_offer_moves_deadline_or_soc():
    chk = FeasibilityChecker()
    con = ChargingConstraint(deadline="21:00", min_soc=0.9)  # 1 h after plug-in: infeasible
    res = chk.check(con, CTX)
    assert not res.feasible and res.counter_offer is not None
    assert res.counter_offer.deadline != "21:00"  # later deadline inside the plug-in window
    short = CTX.model_copy(update={"departure_time": "21:00"})
    res2 = chk.check(con, short)
    assert res2.counter_offer.min_soc < 0.9 and res2.counter_offer.deadline == "21:00"


def test_pipeline_retries_then_defaults_and_flags():
    class Broken:
        calls = 0

        def parse(self, text, ctx, feedback=None):
            Broken.calls += 1
            return '{"deadline": "99:99"}'

    pipe = IntentPipeline(Broken(), FeasibilityChecker(), max_retries=2)
    res = pipe.process("anything", CTX)
    assert Broken.calls == 3 and res.flagged_for_review and res.attempts == 3
    assert res.requested.v2g_allowed is False  # safe default


def test_metrics_on_demo_corpus():
    recs = generate_demo_corpus(1)
    golds = [ChargingConstraint.model_validate(r["constraint"]) for r in recs]
    assert {intent_type(g) for g in golds} == {"deadline", "v2g", "cost", "urgent"}
    preds = [g.model_copy(update={"min_soc": 0.1, "soc_floor": 0.1}) if i == 0 else g for i, g in enumerate(golds)]
    m = intent_metrics(preds, golds)
    assert m["exact_match"] == pytest.approx(1 - 1 / len(golds))
    assert sum(sum(row) for row in m["confusion_matrix"]["matrix"]) == len(golds)
