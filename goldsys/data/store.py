"""Parquet tick/bar lake and resampling.

Layout:
  data/ticks/{SYMBOL}/{YYYY-MM-DD}.parquet       raw ticks for one UTC day
  data/bars/{SYMBOL}/M1/{YYYY}.parquet            minute bars per year
Bars are built from the BID price (what you sell at) with the average spread
stored per bar so the cost model can add it back on the ask side.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BAR_DIR, SYMBOL, TICK_DIR

BAR_COLS = ["open", "high", "low", "close", "spread", "ticks", "volume"]

PANDAS_FREQ = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}


def tick_path(symbol: str, day: dt.date) -> Path:
    return TICK_DIR / symbol / f"{day.isoformat()}.parquet"


def write_ticks(df: pd.DataFrame, symbol: str, day: dt.date) -> Path:
    p = tick_path(symbol, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def read_ticks(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    frames = []
    day = start
    while day < end:
        p = tick_path(symbol, day)
        if p.exists():
            frames.append(pd.read_parquet(p))
        day += dt.timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["ts", "ask", "bid", "ask_vol", "bid_vol"])
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values("ts").reset_index(drop=True)


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ticks to 1-minute bars (bid OHLC, mean spread, tick count, volume)."""
    if ticks.empty:
        return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    t = ticks.set_index("ts")
    spread = t["ask"] - t["bid"]
    g = t["bid"].resample("1min")
    bars = pd.DataFrame(
        {
            "open": g.first(),
            "high": g.max(),
            "low": g.min(),
            "close": g.last(),
            "spread": spread.resample("1min").mean(),
            "ticks": g.count(),
            "volume": (t["ask_vol"] + t["bid_vol"]).resample("1min").sum(),
        }
    )
    bars = bars.dropna(subset=["open"])
    bars.index.name = "ts"
    return bars


def resample_bars(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample M1 (or finer) bars to a higher timeframe. Bars are labelled by their OPEN time."""
    freq = PANDAS_FREQ[timeframe]
    if timeframe == "M1":
        return bars
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "spread": "mean",
        "ticks": "sum",
        "volume": "sum",
    }
    out = bars.resample(freq, label="left", closed="left").agg(agg).dropna(subset=["open"])
    out.index.name = "ts"
    return out


def bar_path(symbol: str, timeframe: str, year: int) -> Path:
    return BAR_DIR / symbol / timeframe / f"{year}.parquet"


def write_bars(bars: pd.DataFrame, symbol: str = SYMBOL, timeframe: str = "M1") -> list[Path]:
    paths = []
    for year, chunk in bars.groupby(bars.index.year):
        p = bar_path(symbol, timeframe, int(year))
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            old = pd.read_parquet(p)
            chunk = pd.concat([old, chunk])
            chunk = chunk[~chunk.index.duplicated(keep="last")].sort_index()
        chunk.to_parquet(p)
        paths.append(p)
    return paths


def read_bars(symbol: str = SYMBOL, timeframe: str = "M1", start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
    d = BAR_DIR / symbol / timeframe
    if not d.exists():
        return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    frames = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    if start is not None:
        bars = bars[bars.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        bars = bars[bars.index < pd.Timestamp(end, tz="UTC")]
    return bars


def build_m1_from_ticks(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Read the tick lake day by day, build M1 bars, and persist them."""
    all_bars = []
    day = start
    while day < end:
        p = tick_path(symbol, day)
        if p.exists():
            ticks = pd.read_parquet(p)
            ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)
            b = ticks_to_m1(ticks)
            if not b.empty:
                all_bars.append(b)
        day += dt.timedelta(days=1)
    if not all_bars:
        return pd.DataFrame(columns=BAR_COLS)
    bars = pd.concat(all_bars).sort_index()
    write_bars(bars, symbol, "M1")
    return bars
