"""Import MetaTrader 5 bar exports (Symbols -> Bars -> Export Bars, or copy_rates_* dumps) into the lake.

MT5 writes BID bars in the BROKER'S SERVER TIME, with the spread in points. The lake wants UTC bars with the
spread in $/oz, so this module (1) parses defensively, (2) works out which clock the file is on from the weekly
market-open pattern (the open is anchored to New York, so only the right interpretation gives a constant offset
across the DST changes), (3) converts to UTC, (4) cleans and reports what it changed.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import store
from .quality import check_bars
from .store import BAR_COLS

NY = "America/New_York"

# Clock rules a broker file may be on. "ny+7" is the common "GMT+2 winter / GMT+3 summer, switching with US DST" server.
CANDIDATE_RULES = ["utc", "+1", "+2", "+3", "ny+7", "Europe/Athens", "Europe/London"]


class ImportErrorMT5(ValueError):
    pass


# ---------------------------------------------------------------- parsing

_ALIASES = {
    "date": "date", "time": "time", "datetime": "datetime", "timestamp": "datetime", "open": "open", "high": "high", "low": "low", "close": "close",
    "tickvol": "tickvol", "tick_volume": "tickvol", "tickvolume": "tickvol", "vol": "vol", "volume": "vol", "real_volume": "vol", "spread": "spread",
}


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z_]", "", c.strip().strip("<>").lower())


def parse_mt5_csv(src: str | Path | io.StringIO) -> pd.DataFrame:
    """Parse an MT5 bar export into a frame indexed by NAIVE broker time with columns open, high, low, close, tickvol, vol, spread(points)."""
    text = src.getvalue() if isinstance(src, io.StringIO) else Path(src).read_text(encoding="utf-8-sig", errors="replace")
    first = text.split("\n", 1)[0]
    sep = max(["\t", ";", ","], key=first.count)
    if first.count(sep) == 0:
        raise ImportErrorMT5("could not find a delimiter in the header line; expected a tab-, semicolon- or comma-separated MT5 export")
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    cols = {c: _ALIASES.get(_norm_col(c)) for c in df.columns}
    df = df.rename(columns={c: v for c, v in cols.items() if v})
    df = df[[c for c in df.columns if c in set(_ALIASES.values())]]
    for need in ("open", "high", "low", "close"):
        if need not in df:
            raise ImportErrorMT5(f"missing column {need!r}; found {list(cols)}")
    if "datetime" in df:
        raw = df["datetime"].str.strip()
        if raw.str.fullmatch(r"\d{9,11}").all():
            ts = pd.to_datetime(raw.astype("int64"), unit="s")
        else:
            ts = pd.to_datetime(raw.str.replace(".", "-", regex=False), errors="coerce")
    elif "date" in df and "time" in df:
        ts = pd.to_datetime(df["date"].str.strip().str.replace(".", "-", regex=False) + " " + df["time"].str.strip(), errors="coerce")
    elif "date" in df:
        ts = pd.to_datetime(df["date"].str.strip().str.replace(".", "-", regex=False), errors="coerce")
    elif "time" in df:                                               # copy_rates dumps: a single 'time' column (epoch seconds or ISO text)
        raw = df["time"].str.strip()
        ts = pd.to_datetime(raw.astype("int64"), unit="s") if raw.str.fullmatch(r"\d{9,11}").all() else pd.to_datetime(raw, errors="coerce")
    else:
        raise ImportErrorMT5("no date/time columns found")
    bad = int(ts.isna().sum())
    if bad > max(5, 0.001 * len(df)):
        raise ImportErrorMT5(f"{bad} rows have an unreadable date/time; is this an MT5 bar export?")
    out = pd.DataFrame({"open": pd.to_numeric(df["open"], errors="coerce"), "high": pd.to_numeric(df["high"], errors="coerce"),
                        "low": pd.to_numeric(df["low"], errors="coerce"), "close": pd.to_numeric(df["close"], errors="coerce")})
    for c, default in (("tickvol", 0.0), ("vol", 0.0), ("spread", np.nan)):
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else default
    out.index = pd.DatetimeIndex(ts, name="ts")
    out = out[~out.index.isna()]
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


# ---------------------------------------------------------------- timezone

def to_utc(naive: pd.DatetimeIndex, rule: str) -> tuple[pd.DatetimeIndex, int]:
    """Interpret naive broker timestamps under `rule` and return (UTC index, rows that could not be placed: DST gap / repeated hour)."""
    r = rule.strip()
    if r.lower() == "utc":
        return naive.tz_localize("UTC"), 0
    if re.fullmatch(r"[+-]\d{1,2}(\.\d+)?", r):                      # fixed offset: file clock = UTC + N hours
        return (naive - pd.Timedelta(hours=float(r))).tz_localize("UTC"), 0
    if r.lower() == "ny+7":                                          # file clock = New York + 7 h (follows US DST)
        loc = (naive - pd.Timedelta(hours=7)).tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    else:
        try:
            loc = naive.tz_localize(r, ambiguous="NaT", nonexistent="NaT")      # any IANA zone name
        except Exception as e:  # noqa: BLE001
            raise ImportErrorMT5(f"unknown timezone rule {rule!r}: use 'utc', '+2', 'ny+7' or an IANA name like 'Europe/Athens'") from e
    return loc.tz_convert("UTC"), int(loc.isna().sum())


def detect_timezone(naive: pd.DatetimeIndex) -> list[dict]:
    """Rank clock rules by where the last bar of each trading week falls in New York time.

    FX and gold close for the weekend at 17:00 New York on Friday, whatever the broker's server clock says. Under the TRUE rule the last
    bar of (nearly) every week lands on a Friday between 16:30 and 17:15 New York; a rule that is even one hour off, or that ignores a DST
    switch, misses that window. (The weekly OPEN cannot do this job: a fixed offset shifts every open equally, so it cannot tell UTC from UTC+1.)"""
    gaps = pd.Series(naive, index=naive).diff()
    pos = np.flatnonzero((gaps > pd.Timedelta(hours=24)).to_numpy())
    if len(pos) < 12:
        return []
    last, first = naive[pos - 1], naive[pos]
    out = []
    for rule in CANDIDATE_RULES:
        ul, _ = to_utc(last, rule)
        ny = ul.tz_convert(NY)
        mins = np.asarray(ny.hour * 60 + ny.minute)
        fri = np.asarray(ny.dayofweek) == 4
        score = float(((mins >= 990) & (mins <= 1035) & fri).sum() / len(last))
        uf, _ = to_utc(first, rule)
        nf = uf.tz_convert(NY)
        mo = int(pd.Series(np.asarray(nf.hour * 60 + nf.minute)).mode().iloc[0])
        out.append({"rule": rule, "score": round(score, 3), "close_ny": f"{int(pd.Series(mins).mode().iloc[0] // 60):02d}:{int(pd.Series(mins).mode().iloc[0] % 60):02d}",
                    "open_ny": f"{mo // 60:02d}:{mo % 60:02d}", "weeks": int(len(last))})
    return sorted(out, key=lambda x: (-x["score"], CANDIDATE_RULES.index(x["rule"])))


def choose_timezone(naive: pd.DatetimeIndex, min_score: float = 0.9, min_margin: float = 0.1) -> tuple[str, list[dict]]:
    rank = detect_timezone(naive)
    if not rank:
        raise ImportErrorMT5("too few weekly gaps (need ~12 weeks of data) to detect the timezone; pass --tz explicitly")
    best, second = rank[0], (rank[1] if len(rank) > 1 else {"score": 0.0})
    if best["score"] < min_score or best["score"] - second["score"] < min_margin:
        raise ImportErrorMT5("could not determine the file's timezone with confidence; pass --tz explicitly. Candidates: "
                             + ", ".join(f"{r['rule']}={r['score']}" for r in rank[:4]))
    return best["rule"], rank


# ---------------------------------------------------------------- cleaning and conversion

@dataclass
class ImportReport:
    source_file: str = ""
    sha256: str = ""
    rows_read: int = 0
    rows_written: int = 0
    timezone_rule: str = ""
    timezone_ranking: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    spread_usd: dict = field(default_factory=dict)
    spread_by_hour_median: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def convert(raw: pd.DataFrame, tz: str = "auto", point: float = 0.01) -> tuple[pd.DataFrame, ImportReport]:
    """Naive-broker-time frame from parse_mt5_csv -> (lake-format UTC M1 bars, report)."""
    rep = ImportReport(rows_read=len(raw))
    df = raw.sort_index()
    dup = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")]
    rep.dropped["duplicate_timestamps"] = dup
    px = ["open", "high", "low", "close"]
    badpx = df[px].isna().any(axis=1) | (df[px] <= 0).any(axis=1)
    rep.dropped["missing_or_nonpositive_price"] = int(badpx.sum())
    df = df[~badpx]
    # OHLC consistency: widen high/low to cover open/close rather than discard the bar
    hi = df[["open", "close", "high"]].max(axis=1); lo = df[["open", "close", "low"]].min(axis=1)
    rep.fixed["ohlc_envelope_widened"] = int(((hi != df["high"]) | (lo != df["low"])).sum())
    df = df.assign(high=hi, low=lo)

    if tz == "auto":
        tz, rank = choose_timezone(df.index)
        rep.timezone_ranking = rank
    rep.timezone_rule = tz
    utc, unplaced = to_utc(df.index, tz)
    rep.dropped["unplaceable_dst_rows"] = unplaced
    if unplaced > max(5, 0.001 * len(df)):
        raise ImportErrorMT5(f"{unplaced} rows fall in a DST gap or repeated hour under rule {tz!r}; the rule is probably wrong")
    df.index = utc
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "ts"

    spread = df["spread"].astype(float) * point
    zero = (spread <= 0) | spread.isna()
    rep.fixed["spread_unknown_set_to_nan"] = int(zero.sum())            # 0 points means "not recorded"; the cost model then uses its table
    spread = spread.where(~zero, np.nan)
    tick = df["tickvol"].fillna(0).astype(float)
    vol = df["vol"].fillna(0).astype(float)
    out = pd.DataFrame({"open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"], "spread": spread,
                        "ticks": tick, "volume": np.where(vol > 0, vol, tick)}, index=df.index)[BAR_COLS]
    rep.rows_written = len(out)
    rep.quality = check_bars(out)
    pos = spread.dropna()
    if len(pos):
        rep.spread_usd = {"median": round(float(pos.median()), 4), "p95": round(float(pos.quantile(.95)), 4), "p99": round(float(pos.quantile(.99)), 4), "max": round(float(pos.max()), 2)}
        by_h = pos.groupby(pos.index.hour).median().reindex(range(24))
        rep.spread_by_hour_median = [None if pd.isna(v) else round(float(v), 3) for v in by_h]
    return out, rep


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- provenance

def source_path(symbol: str) -> Path:
    return store.BAR_DIR / symbol / "SOURCE.json"


def read_source(symbol: str) -> dict | None:
    p = source_path(symbol)
    return json.loads(p.read_text()) if p.exists() else None


def has_bars(symbol: str) -> bool:
    d = store.BAR_DIR / symbol / "M1"
    return d.exists() and any(d.glob("*.parquet"))
