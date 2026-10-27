import json

import numpy as np
import pytest

from solace.config import FeederConfig
from solace.grid import ac_power_flow, build_feeder, linear_voltage_sq, path_impedance_matrices
from solace.grid.feeder import LVNetwork, load_feeder_json


def chain(n=5, r=0.01, x=0.003):
    parent = np.arange(-1, n)
    rr = np.full(n + 1, r)
    xx = np.full(n + 1, x)
    rr[0] = xx[0] = 0.0
    return parent, rr, xx


def test_path_matrix_for_chain():
    parent, r, x = chain(3, r=0.1)
    R, X = path_impedance_matrices(parent, r, x)
    assert np.allclose(R, R.T)
    assert R[3, 3] == pytest.approx(0.3)
    assert R[1, 3] == pytest.approx(0.1)  # shared segment 0->1
    assert R[0].sum() == 0.0


def test_no_load_flat_voltage():
    parent, r, x = chain()
    res = ac_power_flow(parent, r, x, np.zeros(6), np.zeros(6))
    assert res.converged
    assert np.allclose(res.v_mag, 1.0)
    assert abs(res.s_slack) < 1e-12


def test_lindistflow_close_to_ac_at_light_load():
    parent, r, x = chain()
    p = np.r_[0.0, np.full(5, 0.02)]
    q = 0.3 * p
    ac = ac_power_flow(parent, r, x, p, q)
    R, X = path_impedance_matrices(parent, r, x)
    lin = np.sqrt(linear_voltage_sq(R, X, p, q))
    assert np.max(np.abs(lin - ac.v_mag)) < 2e-3
    assert np.all(ac.v_mag[1:] <= lin[1:] + 1e-9)  # LinDistFlow ignores losses -> optimistic voltages
    assert ac.s_slack.real > p.sum()  # transformer supplies losses as well


def test_synthetic_feeder_matches_config():
    feeder = build_feeder(FeederConfig(), np.random.default_rng(0))
    assert feeder.n_transformers == 4
    assert feeder.total_buses == 248
    assert sum(n.households.sum() for n in feeder.networks) == 1340


def test_invalid_network_rejected():
    with pytest.raises(ValueError):
        LVNetwork(0, np.array([-1, 2, 0]), np.zeros(3), np.zeros(3), np.zeros(3, dtype=int))


def test_feeder_json_loader(tmp_path):
    parent, r, x = chain(3)
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"s_base_kva": 100, "networks": [
        {"parent": parent.tolist(), "r_pu": r.tolist(), "x_pu": x.tolist(), "households": [0, 1, 1, 1]}]}))
    feeder = load_feeder_json(path, FeederConfig())
    assert feeder.s_base_kva == 100 and feeder.networks[0].n_buses == 4
