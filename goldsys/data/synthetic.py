"""Synthetic gold-like minute data for tests and offline development.

Not a substitute for real ticks. It reproduces: 23h/5d trading week, session
volatility seasonality (London/NY high, Asia low), vol clustering, fat tails,
occasional news spikes, and a time-of-day spread profile.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .store import BAR_COLS


def _intraday_vol_profile(hours: np.ndarray) -> np.ndarray:
    # relative vol by UTC hour (1.0 = average)
    prof = np.array([0.6, 0.55, 0.6, 0.65, 0.65, 0.7, 0.8, 1.0, 1.2, 1.2, 1.1, 1.0, 1.3, 1.7, 1.8, 1.6, 1.3, 1.1, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6])
    return prof[hours]


def spread_profile(hours: np.ndarray) -> np.ndarray:
    # $ spread by UTC hour (typical raw-spread broker)
    prof = np.array([0.45, 0.35, 0.3, 0.3, 0.3, 0.3, 0.28, 0.2, 0.18, 0.18, 0.18, 0.18, 0.17, 0.17, 0.17, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.6, 1.5, 0.8])
    return prof[hours]


def make_m1(start: dt.date, end: dt.date, seed: int = 7, s0: float = 2400.0, annual_vol: float = 0.16, drift: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="1min", inclusive="left")
    # trading week: Sunday 22:00 -> Friday 21:00 UTC, with a daily 21:00-22:00 break
    wd = idx.dayofweek.to_numpy()
    hr = idx.hour.to_numpy()
    open_mask = ~((wd == 5) | ((wd == 4) & (hr >= 21)) | ((wd == 6) & (hr < 22)) | (hr == 21))
    idx = idx[open_mask]
    n = len(idx)
    hours = idx.hour.to_numpy()
    minutes_per_year = 252 * 23 * 60
    base_sigma = annual_vol / np.sqrt(minutes_per_year)
    # GARCH-like vol clustering
    z = rng.standard_t(df=4, size=n) / np.sqrt(2)  # fat tails, unit-ish variance
    sig = np.empty(n)
    v = 1.0
    for i in range(n):
        v = 0.02 + 0.9 * v + 0.08 * (z[i - 1] ** 2 if i else 1.0)
        sig[i] = np.sqrt(v)
    sig = sig / sig.mean()
    r = base_sigma * sig * _intraday_vol_profile(hours) * z + drift / minutes_per_year
    # occasional news spikes at 12:30 / 13:30 / 18:00 UTC
    spike_times = ((idx.hour == 12) & (idx.minute == 30)) | ((idx.hour == 13) & (idx.minute == 30)) | ((idx.hour == 18) & (idx.minute == 0))
    spike_mask = np.asarray(spike_times) & (rng.random(n) < 0.15)
    r = r + spike_mask * rng.normal(0, 6 * base_sigma, n)
    close = s0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[s0], close[:-1]])
    wiggle = np.abs(rng.normal(0, 0.6, n)) * base_sigma * sig * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    spread = spread_profile(hours) * (1 + 0.5 * (sig - 1).clip(0, 3))
    ticks = (50 * sig * _intraday_vol_profile(hours)).astype(int) + 5
    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "spread": spread, "ticks": ticks, "volume": ticks * 0.1},
        index=idx,
    )[BAR_COLS]
    bars.index.name = "ts"
    return bars
