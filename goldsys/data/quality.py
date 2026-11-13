"""Data-quality checks run on every ingest. Returns a dict of findings; raise on fatal ones if strict."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_ticks(ticks: pd.DataFrame) -> dict:
    out: dict = {"n": int(len(ticks))}
    if ticks.empty:
        out["empty"] = True
        return out
    ts = pd.to_datetime(ticks["ts"], utc=True)
    out["unsorted"] = int((ts.diff().dt.total_seconds() < 0).sum())
    out["duplicates"] = int(ts.duplicated().sum())
    spread = ticks["ask"] - ticks["bid"]
    out["negative_spread"] = int((spread < 0).sum())
    out["zero_price"] = int(((ticks["bid"] <= 0) | (ticks["ask"] <= 0)).sum())
    out["spread_median"] = float(spread.median())
    out["spread_p99"] = float(spread.quantile(0.99))
    # price outliers: tick-to-tick log return beyond 50 sigma of the day
    r = np.log(ticks["bid"]).diff().dropna()
    if len(r) > 10 and r.std() > 0:
        out["outlier_ticks"] = int((r.abs() > 50 * r.std()).sum())
    gaps = ts.diff().dt.total_seconds()
    out["max_gap_seconds"] = float(gaps.max()) if len(gaps) else 0.0
    return out


def check_bars(bars: pd.DataFrame, timeframe_minutes: int = 1) -> dict:
    out: dict = {"n": int(len(bars))}
    if bars.empty:
        out["empty"] = True
        return out
    bad_ohlc = ((bars["high"] < bars["low"]) | (bars["high"] < bars[["open", "close"]].max(axis=1)) | (bars["low"] > bars[["open", "close"]].min(axis=1))).sum()
    out["bad_ohlc"] = int(bad_ohlc)
    out["duplicates"] = int(bars.index.duplicated().sum())
    out["nonpositive"] = int((bars[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    out["weekend_bars"] = int(((bars.index.dayofweek == 5)).sum())
    gaps = bars.index.to_series().diff().dt.total_seconds() / 60
    # gaps longer than 3 hours that are not the weekend
    wk = bars.index.dayofweek
    big = (gaps > 180) & ~((wk == 6) | (wk == 0))
    out["intraweek_gaps_gt_3h"] = int(big.sum())
    out["first"] = str(bars.index[0])
    out["last"] = str(bars.index[-1])
    return out
