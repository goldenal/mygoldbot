"""Every feature must be causal: computing on a truncated series gives the same values as the full series."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from goldsys.data.synthetic import make_m1
from goldsys.data.store import resample_bars
from goldsys.features.registry import base_features, ensure_feature, available_feature_names

FEATURES = ["atr_14", "ema_20", "sma_50", "rsi_14", "rsi_2", "adx_14", "donchian_high_20", "donchian_low_20", "bbz_20", "rvol_20", "ret_5",
            "slope_close_10", "atr_pct_rank_50", "asian_width_pct_rank_20", "vwap", "vwap_z_30", "asian_high", "asian_low", "in_london", "minute_of_day"]


@pytest.fixture(scope="module")
def df():
    bars = resample_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 3, 1), seed=3), "M15")
    return base_features(bars)


@pytest.mark.parametrize("name", FEATURES)
def test_feature_is_causal(df, name):
    full = ensure_feature(df.copy(), name)
    cut = len(df) - 200
    part = ensure_feature(df.iloc[:cut].copy(), name)
    a, b = full.iloc[:cut].to_numpy(dtype=float), part.to_numpy(dtype=float)
    mask = ~(np.isnan(a) & np.isnan(b))
    assert np.allclose(a[mask], b[mask], equal_nan=True), f"{name} uses future data"


def test_asian_range_unknown_before_window_end(df):
    before = df[(df.index.hour < 7)]
    assert before["asian_high"].isna().all()
