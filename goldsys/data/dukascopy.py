"""Dukascopy free tick feed downloader.

URL layout:  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5
Each file is LZMA-compressed; each tick is 20 bytes big-endian:
    int32  ms offset from the start of the hour
    int32  ask * scale
    int32  bid * scale
    float32 ask volume (millions of units)
    float32 bid volume
An empty (0-byte) file means no ticks in that hour (weekend / holiday).
The feed is aggressively rate limited; keep concurrency low and back off on 429/503.
"""
from __future__ import annotations

import datetime as dt
import lzma
import struct
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from ..config import PRICE_SCALE, RAW_DIR, SYMBOL

BASE = "https://datafeed.dukascopy.com/datafeed"
TICK_STRUCT = struct.Struct(">iiiff")
HEADERS = {"User-Agent": "goldsys-research/0.1"}


def hour_url(symbol: str, when: dt.datetime) -> str:
    return f"{BASE}/{symbol}/{when.year}/{when.month - 1:02d}/{when.day:02d}/{when.hour:02d}h_ticks.bi5"


def raw_path(symbol: str, when: dt.datetime) -> Path:
    return RAW_DIR / symbol / f"{when.year}" / f"{when.month:02d}" / f"{when.day:02d}" / f"{when.hour:02d}h_ticks.bi5"


def decode_bi5(blob: bytes, hour_start: dt.datetime, scale: float = PRICE_SCALE) -> pd.DataFrame:
    """Decode one hour of ticks into a DataFrame (UTC timestamps)."""
    if not blob:
        return _empty_ticks()
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError:
        # Some hours are served uncompressed
        raw = blob
    n = len(raw) // TICK_STRUCT.size
    if n == 0:
        return _empty_ticks()
    arr = np.frombuffer(raw[: n * TICK_STRUCT.size], dtype=np.dtype(">i4,>i4,>i4,>f4,>f4"))
    base_ns = np.datetime64(hour_start.replace(tzinfo=None), "ns")
    ts = base_ns + arr["f0"].astype("timedelta64[ms]").astype("timedelta64[ns]")
    df = pd.DataFrame(
        {
            "ts": pd.DatetimeIndex(ts, tz="UTC"),
            "ask": arr["f1"].astype(np.float64) / scale,
            "bid": arr["f2"].astype(np.float64) / scale,
            "ask_vol": arr["f3"].astype(np.float64),
            "bid_vol": arr["f4"].astype(np.float64),
        }
    )
    return df


def _empty_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.DatetimeIndex([], tz="UTC"),
            "ask": np.array([], dtype=np.float64),
            "bid": np.array([], dtype=np.float64),
            "ask_vol": np.array([], dtype=np.float64),
            "bid_vol": np.array([], dtype=np.float64),
        }
    )


def fetch_hour(
    symbol: str,
    when: dt.datetime,
    session: requests.Session | None = None,
    max_retries: int = 6,
    timeout: int = 60,
    use_cache: bool = True,
) -> bytes:
    """Download one hour file, caching the raw bytes on disk. Returns raw (compressed) bytes."""
    p = raw_path(symbol, when)
    if use_cache and p.exists():
        return p.read_bytes()
    sess = session or requests.Session()
    url = hour_url(symbol, when)
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            r = sess.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r.content)
                return r.content
            if r.status_code == 404:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")
                return b""
            if r.status_code in (429, 503):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            r.raise_for_status()
        except (requests.RequestException, ConnectionError) as e:  # noqa: PERF203
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def iter_hours(start: dt.datetime, end: dt.datetime) -> Iterable[dt.datetime]:
    cur = start.replace(minute=0, second=0, microsecond=0)
    while cur < end:
        yield cur
        cur += dt.timedelta(hours=1)


def download_day(symbol: str, day: dt.date, session: requests.Session | None = None, pause: float = 0.25) -> pd.DataFrame:
    """Download all 24 hour-files for a UTC day and return the concatenated ticks."""
    sess = session or requests.Session()
    frames = []
    for h in range(24):
        when = dt.datetime(day.year, day.month, day.day, h, tzinfo=dt.timezone.utc)
        blob = fetch_hour(symbol, when, sess)
        frames.append(decode_bi5(blob, when))
        if pause:
            time.sleep(pause)
    df = pd.concat(frames, ignore_index=True) if frames else _empty_ticks()
    return df


def download_range(
    start: dt.date,
    end: dt.date,
    symbol: str = SYMBOL,
    pause: float = 0.25,
    on_day=None,
) -> None:
    """Download [start, end) day by day and write each day to the tick lake via store.write_ticks."""
    from .store import write_ticks

    sess = requests.Session()
    day = start
    while day < end:
        if day.weekday() == 5:  # Saturday: market closed all day
            day += dt.timedelta(days=1)
            continue
        df = download_day(symbol, day, sess, pause)
        write_ticks(df, symbol, day)
        if on_day:
            on_day(day, len(df))
        day += dt.timedelta(days=1)
