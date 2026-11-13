"""Feature registry: builds the full feature frame used by the DSL.

Every feature is a column; the DSL refers to columns by name. Parameterised
features use the naming convention  name_param  e.g.  atr_14, ema_50, rsi_2.
Unknown names are built lazily by `ensure_feature`, so the search engines can
request e.g. ema_37 without a hand-written list.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data.sessions import asian_range, session_flags
from . import indicators as I

BASE_COLS = ("open", "high", "low", "close", "spread", "ticks", "volume")

_PARAM_FEATURES = {
    "atr": lambda df, n: I.atr(df["high"], df["low"], df["close"], n),
    "ema": lambda df, n: I.ema(df["close"], n),
    "sma": lambda df, n: I.sma(df["close"], n),
    "rsi": lambda df, n: I.rsi(df["close"], n),
    "adx": lambda df, n: I.adx(df["high"], df["low"], df["close"], n),
    "donchian_high": lambda df, n: I.donchian_high(df["high"], n),
    "donchian_low": lambda df, n: I.donchian_low(df["low"], n),
    "bbz": lambda df, n: I.bollinger_z(df["close"], n),
    "rvol": lambda df, n: I.realised_vol(df["close"], n),
    "ret": lambda df, n: df["close"].pct_change(n),
    "slope_close": lambda df, n: I.slope(df["close"], n),
    "atr_pct_rank": lambda df, n: I.rolling_pct_rank(I.atr(df["high"], df["low"], df["close"], 14), n),
    "asian_width_pct_rank": lambda df, n: I.rolling_pct_rank(df["asian_width"].ffill(), n),
    "vwap_z": lambda df, n: _vwap_z(df, n),
}

_SIMPLE_FEATURES = {
    "vwap": lambda df: I.session_vwap(df["close"], df["volume"], df.index.normalize()),
    "hour": lambda df: pd.Series(df.index.hour, index=df.index),
    "minute_of_day": lambda df: pd.Series(df.index.hour * 60 + df.index.minute, index=df.index),
    "bar_range": lambda df: df["high"] - df["low"],
    "body": lambda df: (df["close"] - df["open"]),
}

_NAME_RE = re.compile(r"^([a-z_]+?)_(\d+)$")


def _vwap_z(df: pd.DataFrame, n: int) -> pd.Series:
    vwap = ensure_feature(df, "vwap")
    dev = df["close"] - vwap
    sd = dev.rolling(n, min_periods=n).std(ddof=0)
    return dev / sd.replace(0, np.nan)


def base_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars + session flags + Asian range. Cheap, always computed."""
    df = bars.copy()
    sf = session_flags(df.index)
    for c in sf.columns:
        df[c] = sf[c].to_numpy()
    ar = asian_range(df)
    for c in ar.columns:
        df[c] = ar[c].to_numpy()
    return df


def ensure_feature(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name], computing and caching it if needed."""
    if name in df.columns:
        return df[name]
    if name in _SIMPLE_FEATURES:
        df[name] = _SIMPLE_FEATURES[name](df)
        return df[name]
    m = _NAME_RE.match(name)
    if m and m.group(1) in _PARAM_FEATURES:
        df[name] = _PARAM_FEATURES[m.group(1)](df, int(m.group(2)))
        return df[name]
    raise KeyError(f"unknown feature '{name}'. Known: {sorted(_PARAM_FEATURES)} (with _N suffix), {sorted(_SIMPLE_FEATURES)}, columns {list(df.columns)}")


def available_feature_names() -> dict:
    return {"parametric": sorted(_PARAM_FEATURES), "simple": sorted(_SIMPLE_FEATURES), "base": list(BASE_COLS) + ["in_asia", "in_london", "in_newyork", "in_overlap", "weekday", "asian_high", "asian_low", "asian_width"]}
