import numpy as np

from solace.config import DegradationConfig
from solace.degradation import DegradationModel


def model(n=2):
    return DegradationModel(DegradationConfig(), np.array(["LFP", "NMC"])[:n], np.array([3.0, 40.0])[:n])


def test_idle_loss_is_calendar_only():
    m = model()
    loss = m.step(np.zeros(2), np.array([0.5, 0.5]), 0.25, np.ones(2, dtype=bool))
    cfg = DegradationConfig()
    expected = np.array([cfg.chemistries["LFP"].calendar_per_day, cfg.chemistries["NMC"].calendar_per_day]) * 0.25 / 24
    assert np.allclose(loss, expected)


def test_loss_increases_with_power_and_age_effect():
    m1, m2 = model(), model()
    low = m1.step(np.array([1.0, 3.0]), np.full(2, 0.5), 0.25, np.ones(2, dtype=bool))
    high = m2.step(np.array([3.0, 30.0]), np.full(2, 0.5), 0.25, np.ones(2, dtype=bool))
    assert np.all(high > low)
    # Concave throughput term: the same energy costs less on an older battery.
    m3 = model()
    first = m3.step(np.array([3.0, 30.0]), np.full(2, 0.5), 0.25, np.ones(2, dtype=bool))
    m3.equivalent_cycles += 1000
    later = m3.step(np.array([3.0, 30.0]), np.full(2, 0.5), 0.25, np.ones(2, dtype=bool))
    assert np.all(later < first)


def test_cost_uses_replacement_value():
    m = model()
    cfg = DegradationConfig()
    assert np.allclose(m.cost_inr(np.array([0.01, 0.01])),
                       0.01 * cfg.battery_cost_inr_per_kwh * np.array([3.0, 40.0]) / cfg.end_of_life_loss)
