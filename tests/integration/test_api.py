import numpy as np
import pytest
from fastapi.testclient import TestClient

from solace.api import create_app
from solace.intent import RuleBasedParser

CTX = {"vehicle_class": "4W", "battery_kwh": 40, "soc": 0.3, "charger_kw": 7.4, "plug_in_time": "20:00",
       "departure_time": "09:00", "v2g_capable": True}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(RuleBasedParser()))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_intent_accept_and_counter_offer(client):
    ok = client.post("/v1/intent", json={"text": "Need 80% by 7 am, no V2G", "context": CTX}).json()
    assert ok["status"] == "accepted" and ok["constraint"]["deadline"] == "07:00"
    co = client.post("/v1/intent", json={"text": "Need 90% by 9 pm", "context": CTX}).json()
    assert co["status"] == "counter_offer" and co["constraint"] != co["requested"]


def test_intent_validation_error(client):
    r = client.post("/v1/intent", json={"text": "", "context": CTX})
    assert r.status_code == 422


def dispatch_body(raw=1.0, base_kw=5.0):
    n = 5
    return {"network": {"parent": list(range(-1, n - 1)), "r_pu": [0.0] + [0.03] * (n - 1),
                        "x_pu": [0.0] + [0.01] * (n - 1), "s_base_kva": 100.0},
            "base_p_kw": [0.0] + [base_kw] * (n - 1), "base_q_kvar": [0.0] + [0.3 * base_kw] * (n - 1),
            "evs": [{"bus": 4, "rated_kw": 30.0, "lo": -1.0, "hi": 1.0, "raw_action": raw},
                    {"bus": 2, "rated_kw": 30.0, "lo": 0.0, "hi": 1.0, "raw_action": raw}]}


def test_dispatch_projects_onto_limits(client):
    r = client.post("/v1/dispatch", json=dispatch_body())
    assert r.status_code == 200, r.text
    out = r.json()
    assert np.all(np.array(out["projected_action"]) <= 1.0 + 1e-9)
    assert out["projection_distance_kw"] > 0  # 60 kW on a 100 kVA feeder with 20 kW base load must be curtailed
    assert sum(out["setpoint_kw"]) < 60.0


def test_dispatch_rejects_bad_bus(client):
    body = dispatch_body()
    body["evs"][0]["bus"] = 99
    assert client.post("/v1/dispatch", json=body).status_code == 422
