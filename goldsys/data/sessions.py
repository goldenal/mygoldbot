"""Trading-session calendar for gold, DST-aware.

Sessions are defined in LOCAL time of each centre and converted to UTC per bar,
so London/New York DST shifts are handled by the zoneinfo database.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS = {
    # name: (tz, start_local, end_local)   end exclusive
    "asia": ("Asia/Tokyo", "09:00", "15:00"),
    "london": ("Europe/London", "08:00", "16:30"),
    "newyork": ("America/New_York", "08:00", "17:00"),
}


def _local_minutes(idx: pd.DatetimeIndex, tz: str) -> np.ndarray:
    loc = idx.tz_convert(tz)
    return (loc.hour * 60 + loc.minute).to_numpy()


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def session_flags(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean columns in_asia / in_london / in_newyork plus minutes-since-open for each."""
    out = {}
    for name, (tz, start, end) in SESSIONS.items():
        mins = _local_minutes(idx, tz)
        s, e = _hm(start), _hm(end)
        inside = (mins >= s) & (mins < e)
        out[f"in_{name}"] = inside
        since = np.where(inside, mins - s, -1)
        out[f"mins_since_{name}_open"] = since
    df = pd.DataFrame(out, index=idx)
    df["in_overlap"] = df["in_london"] & df["in_newyork"]
    wd = idx.dayofweek.to_numpy()
    df["weekday"] = wd
    return df


def asian_range(bars: pd.DataFrame, start_utc: str = "00:00", end_utc: str = "07:00") -> pd.DataFrame:
    """Rolling 'Asian range': high/low between start_utc and end_utc of the same UTC day,
    frozen after end_utc and carried forward through the rest of the day.
    Values before end_utc each day are NaN (range not yet known) -> no look-ahead."""
    idx = bars.index
    mins = idx.hour * 60 + idx.minute
    s, e = _hm(start_utc), _hm(end_utc)
    in_win = (mins >= s) & (mins < e)
    day = idx.normalize()
    hi = bars["high"].where(in_win)
    lo = bars["low"].where(in_win)
    # cumulative within the day while in window, then frozen
    cum_hi = hi.groupby(day).cummax()
    cum_lo = lo.groupby(day).cummin()
    # only reveal after the window has ended
    known = mins >= e
    rng_hi = cum_hi.groupby(day).ffill().where(known)
    rng_lo = cum_lo.groupby(day).ffill().where(known)
    return pd.DataFrame({"asian_high": rng_hi, "asian_low": rng_lo, "asian_width": rng_hi - rng_lo}, index=idx)
