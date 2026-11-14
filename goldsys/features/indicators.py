"""Vectorised indicators. Every function takes/returns pandas Series aligned to the bar index
and must be CAUSAL: value at t depends only on data <= t. tests/test_lookahead.py enforces this."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(au.notna())


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    mdi = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def donchian_high(high: pd.Series, n: int) -> pd.Series:
    """Highest high of the PREVIOUS n bars (excludes current bar so a breakout can be detected)."""
    return high.shift(1).rolling(n, min_periods=n).max()


def donchian_low(low: pd.Series, n: int) -> pd.Series:
    return low.shift(1).rolling(n, min_periods=n).min()


def bollinger_z(close: pd.Series, n: int = 20) -> pd.Series:
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std(ddof=0)
    return (close - m) / s.replace(0, np.nan)


def realised_vol(close: pd.Series, n: int = 20) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(n, min_periods=n).std(ddof=0)


@njit(cache=True)
def _pct_rank(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    bad = 0                                    # NaNs inside the current window
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            bad += 1
        if i >= n and np.isnan(a[i - n]):
            bad -= 1
        if i >= n - 1 and bad == 0:
            cur, below = a[i], 0
            for j in range(i - n + 1, i):
                if a[j] < cur:
                    below += 1
            out[i] = below * 100.0 / (n - 1) if n > 1 else np.nan
    return out


def rolling_pct_rank(x: pd.Series, n: int) -> pd.Series:
    """Percentile rank (0-100) of the current value among the n-1 values before it; NaN until a full
    window of n valid values exists. (Compiled: the pandas rolling.apply version of this was the
    single slowest step of every M5/M15 backtest.)"""
    return pd.Series(_pct_rank(np.ascontiguousarray(x.to_numpy(dtype=np.float64)), int(n)), index=x.index)


def session_vwap(close: pd.Series, volume: pd.Series, day_key: pd.Series) -> pd.Series:
    """VWAP anchored at the start of each UTC day."""
    pv = (close * volume).groupby(day_key).cumsum()
    vv = volume.groupby(day_key).cumsum()
    return pv / vv.replace(0, np.nan)


def slope(x: pd.Series, n: int) -> pd.Series:
    """Simple n-bar change (positive = rising)."""
    return x - x.shift(n)


def cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger bands: middle (SMA), upper and lower at k standard deviations (population sd, as bollinger_z)."""
    m = close.rolling(n, min_periods=n).mean()
    s = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": m, "upper": m + k * s, "lower": m - k * s}, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=close.index)
