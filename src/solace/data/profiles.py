"""Time-series inputs: residential base load, PV and the time-of-day tariff.

The paper uses a residential profile with an evening peak between 19:00 and
22:00, hourly West Bengal irradiance and a three-band ToD tariff (peak at
1.2x normal). Neither the load data nor the irradiance source is published,
so the defaults below are SYNTHETIC profiles (assumption A6). Real irradiance
can be supplied through ``load_irradiance_csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solace.config import DataConfig, TariffConfig


def step_hours(cfg: DataConfig) -> np.ndarray:
    """Clock hour (0-24, float) at the start of every step of an episode."""
    dt = cfg.dt_minutes / 60.0
    return (cfg.episode_start_hour + dt * np.arange(cfg.steps_per_day)) % 24.0


def household_shape(hours: np.ndarray) -> np.ndarray:
    """Normalized household demand (peak = 1) with a 19:00-22:00 evening peak."""
    night = 0.35
    morning = 0.30 * np.exp(-0.5 * ((hours - 8.0) / 1.2) ** 2)
    midday = 0.15 * np.exp(-0.5 * ((hours - 13.5) / 2.5) ** 2)
    evening = 0.65 * np.exp(-0.5 * ((hours - 20.5) / 1.4) ** 2)
    shape = night + morning + midday + evening
    return shape / shape.max()


def pv_shape(hours: np.ndarray) -> np.ndarray:
    """Normalized clear-sky PV output between 06:00 and 18:00 (synthetic)."""
    out = np.sin(np.pi * (hours - 6.0) / 12.0)
    return np.clip(out, 0.0, None)


def tariff_profile(hours: np.ndarray, cfg: TariffConfig) -> np.ndarray:
    """INR/kWh per step for the three-band ToD tariff."""

    def in_band(h: np.ndarray, band: tuple[int, int]) -> np.ndarray:
        start, end = band
        return (h >= start) & (h < end) if start < end else (h >= start) | (h < end)

    price = np.full(hours.shape, cfg.normal_inr_per_kwh)
    price[in_band(hours, cfg.offpeak_hours)] = cfg.normal_inr_per_kwh * cfg.offpeak_multiplier
    price[in_band(hours, cfg.peak_hours)] = cfg.normal_inr_per_kwh * cfg.peak_multiplier
    return price


def load_irradiance_csv(path: str | Path) -> np.ndarray:
    """Load hourly global horizontal irradiance (W/m^2) as an array of shape (days, 24).

    Expected CSV: a header row and one value per line in a column named
    ``ghi`` (any other columns are ignored), ordered chronologically from 00:00.
    """
    import csv

    with Path(path).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "ghi" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a 'ghi' column")
        values = np.array([float(row["ghi"]) for row in reader])
    n_days = len(values) // 24
    if n_days == 0:
        raise ValueError(f"{path}: fewer than 24 hourly values")
    return values[: n_days * 24].reshape(n_days, 24)


def pv_from_irradiance(daily_ghi: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Interpolate one day of hourly GHI to step hours, normalized by 1000 W/m^2."""
    ext = np.append(daily_ghi, daily_ghi[0])
    return np.clip(np.interp(hours, np.arange(25), ext) / 1000.0, 0.0, None)
