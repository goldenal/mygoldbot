"""Build data/calendar/events.csv (the economic calendar behind the news blackout, the chart's event markers and the replay's news proximity).

Only the SCHEDULE is stored: when, what, how important. Never a released value. Sources, in order of preference:

  * FOMC statements and minutes: the Federal Reserve's own calendar page (no key). Every meeting 2021-2027. Each past meeting row also carries a link
    whose file name holds the statement date, and the parser checks the table's dates against it.
  * FRED release dates (needs your free API key in FRED_API_KEY): complete HISTORICAL and scheduled dates for the jobs report, CPI, PPI, PCE, GDP and retail
    sales, including every month the government delayed or moved a release. Dates only, so the time of day comes from a fixed table (08:30 New York).
  * An .ics or .csv file you downloaded yourself (the BLS publishes bls.ics; the BLS blocks automated downloads, so this one is manual).

Nothing is guessed from rules like "first Friday of the month": a rule that is right 11 months out of 12 is exactly the calendar that silently misses a rescheduled
release. If a source is missing the importer says what is missing instead of filling the gap.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

from ..config import CALENDAR_FILE

NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "goldsys-research/0.1"}
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FRED_URL = "https://api.stlouisfed.org/fred"

# canonical event -> (tier, default release time in New York)
EVENTS = {
    "FOMC Statement": (1, "14:00"), "FOMC Minutes": (2, "14:00"),
    "US Non-Farm Payrolls": (1, "08:30"), "US CPI": (1, "08:30"), "US PPI": (2, "08:30"), "US PCE": (2, "08:30"), "US GDP": (2, "08:30"),
    "US Retail Sales": (2, "08:30"), "US JOLTS": (3, "10:00"),
}
# FRED release id -> (expected words in its name, canonical event). The importer asks FRED for the name and refuses on a mismatch.
FRED_RELEASES = {
    50: ("employment situation", "US Non-Farm Payrolls"), 10: ("consumer price index", "US CPI"), 46: ("producer price index", "US PPI"),
    54: ("personal income and outlays", "US PCE"), 53: ("gross domestic product", "US GDP"), 9: ("advance monthly sales for retail", "US Retail Sales"),
    192: ("job openings", "US JOLTS"),
}
BLS_ICS_NAMES = [("employment situation", "US Non-Farm Payrolls"), ("consumer price index", "US CPI"), ("producer price index", "US PPI"),
                 ("job openings", "US JOLTS")]


class CalendarImportError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    ts: pd.Timestamp          # UTC
    name: str
    tier: int
    source: str


def ny_to_utc(day: date, hhmm: str) -> pd.Timestamp:
    h, m = hhmm.split(":")
    loc = datetime(day.year, day.month, day.day, int(h), int(m), tzinfo=NY)
    return pd.Timestamp(loc).tz_convert("UTC")


def make(name: str, day: date, source: str, hhmm: str | None = None) -> Event:
    tier, default = EVENTS[name]
    return Event(ny_to_utc(day, hhmm or default), name, tier, source)


# ---------------------------------------------------------------- FOMC (Federal Reserve calendar page)

_MONTHS = {m: i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_PANEL = re.compile(r'<div class="panel panel-default">\s*<div class="panel-heading"><h4><a id="[^"]*">(\d{4}) FOMC Meetings</a></h4></div>(.*?)(?=<div class="panel panel-default">|\Z)', re.S)
_ROW_START = r'<div class="[^"]*\brow fomc-meeting"[^>]*>'                  # a meeting row (inner cells are "fomc-meeting__month" and the like)
_ROW = re.compile(_ROW_START + r'(.*?)(?=' + _ROW_START + r'|\Z)', re.S)
_MONTH = re.compile(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong>')
_DATE = re.compile(r'fomc-meeting__date[^>]*>([^<]+)<')
_STMT_LINK = re.compile(r'monetary(\d{8})a\.htm')
_MINUTES = re.compile(r'Released\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})')


def parse_fomc(html: str) -> tuple[list[Event], dict]:
    """Statement (14:00 New York, day 2 of the meeting) and minutes (14:00) from the Fed's calendar page. Returns (events, diagnostics)."""
    events, diag = [], {"meetings": 0, "link_checked": 0, "link_mismatch": [], "skipped": []}
    for year, body in _PANEL.findall(html):
        year = int(year)
        for row in _ROW.findall(body):
            mm, dd = _MONTH.search(row), _DATE.search(row)
            if not mm or not dd:
                continue
            month_txt, date_txt = mm.group(1).strip(), re.sub(r"[*\s]", "", dd.group(1))
            if re.search(r"unscheduled|notation|cancel", row, re.I):
                diag["skipped"].append(f"{year} {month_txt} {date_txt}")
                continue
            days = [int(x) for x in re.findall(r"\d+", date_txt)]
            if not days:
                continue
            months = [m.strip() for m in month_txt.split("/")]
            try:
                mo = _MONTHS[months[-1]]                                  # a meeting that crosses a month ("Apr/May 30-1") announces in the second month
            except KeyError:
                diag["skipped"].append(f"{year} {month_txt} {date_txt}")
                continue
            day = date(year, mo, days[-1])
            link = _STMT_LINK.search(row)
            if link:                                                       # the statement's own file name carries its date: trust it, and report any disagreement
                ld = datetime.strptime(link.group(1), "%Y%m%d").date()
                diag["link_checked"] += 1
                if ld != day:
                    diag["link_mismatch"].append((str(day), str(ld)))
                    day = ld
            events.append(make("FOMC Statement", day, "federalreserve.gov"))
            diag["meetings"] += 1
            rel = _MINUTES.search(row)
            if rel and rel.group(1) in _MONTHS:
                events.append(make("FOMC Minutes", date(int(rel.group(3)), _MONTHS[rel.group(1)], int(rel.group(2))), "federalreserve.gov"))
    if not events:
        raise CalendarImportError("no FOMC meetings found: the Fed page layout may have changed")
    return events, diag


def fetch_fomc(get: Callable = None) -> tuple[list[Event], dict]:
    import requests
    r = (get or requests.get)(FOMC_URL, headers=UA, timeout=30)
    if r.status_code != 200:
        raise CalendarImportError(f"Fed calendar returned HTTP {r.status_code}")
    return parse_fomc(r.text)


# ---------------------------------------------------------------- FRED release dates (needs FRED_API_KEY)

def fetch_fred(api_key: str, start: str = "2015-01-01", end: str = "2027-12-31", releases: dict | None = None, get: Callable = None) -> tuple[list[Event], dict]:
    import requests
    if not re.fullmatch(r"[a-z0-9]{32}", api_key or ""):
        raise CalendarImportError("FRED_API_KEY must be your 32-character FRED key (free: https://fred.stlouisfed.org/docs/api/api_key.html)")
    g = get or requests.get
    out, diag = [], {}
    for rid, (words, name) in (releases or FRED_RELEASES).items():
        meta = g(f"{FRED_URL}/release", params={"release_id": rid, "api_key": api_key, "file_type": "json"}, headers=UA, timeout=30)
        if meta.status_code != 200:
            raise CalendarImportError(f"FRED release {rid}: HTTP {meta.status_code} ({meta.text[:120]})")
        rname = (meta.json().get("releases") or [{}])[0].get("name", "")
        if words not in rname.lower():
            raise CalendarImportError(f"FRED release {rid} is called {rname!r}, expected something with {words!r}: refusing to label it {name!r}")
        r = g(f"{FRED_URL}/release/dates", params={"release_id": rid, "api_key": api_key, "file_type": "json", "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                                                     "include_release_dates_with_no_data": "true", "sort_order": "asc", "limit": 10000}, headers=UA, timeout=60)
        if r.status_code != 200:
            raise CalendarImportError(f"FRED release dates {rid}: HTTP {r.status_code}")
        days = [d["date"] for d in r.json().get("release_dates", []) if start <= d["date"] <= end]
        out += [make(name, date.fromisoformat(x), "fred.stlouisfed.org") for x in days]
        diag[name] = len(days)
    return out, diag


# ---------------------------------------------------------------- .ics (BLS publishes bls.ics; download it in a browser) and .csv

def parse_ics(text: str) -> tuple[list[Event], dict]:
    text = re.sub(r"\r?\n[ \t]", "", text)                                  # unfold continuation lines
    events, diag = [], {"vevents": 0, "ignored": 0}
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.S):
        diag["vevents"] += 1
        summ = re.search(r"^SUMMARY[^:]*:(.*)$", block, re.M)
        dts = re.search(r"^DTSTART([^:\r\n]*):(\S+)", block, re.M)
        name = next((n for k, n in BLS_ICS_NAMES if summ and k in summ.group(1).lower()), None)
        if not (summ and dts) or name is None:
            diag["ignored"] += 1
            continue
        params, val = dts.group(1), dts.group(2)
        if val.endswith("Z"):
            ts = pd.Timestamp(datetime.strptime(val, "%Y%m%dT%H%M%SZ"), tz="UTC")
        elif "T" in val:
            tz = re.search(r"TZID=([^;:]+)", params)
            loc = datetime.strptime(val, "%Y%m%dT%H%M%S").replace(tzinfo=ZoneInfo(tz.group(1)) if tz else NY)
            ts = pd.Timestamp(loc).tz_convert("UTC")
        else:                                                               # an all-day entry carries no time: use the release's usual one
            events.append(make(name, datetime.strptime(val, "%Y%m%d").date(), "ics"))
            continue
        events.append(Event(ts, name, EVENTS[name][0], "ics"))
    return events, diag


def read_csv(path: str | Path) -> tuple[list[Event], dict]:
    """A CSV with columns ts_utc,name,tier[,source]."""
    df = pd.read_csv(path)
    if not {"ts_utc", "name", "tier"} <= set(df.columns):
        raise CalendarImportError("CSV needs the columns ts_utc,name,tier")
    out = []
    for r in df.itertuples():
        tier = int(r.tier)
        if tier not in (1, 2, 3) or not str(r.name).strip():
            raise CalendarImportError(f"bad row: {r}")
        out.append(Event(pd.Timestamp(r.ts_utc, tz="UTC") if pd.Timestamp(r.ts_utc).tzinfo is None else pd.Timestamp(r.ts_utc).tz_convert("UTC"), str(r.name).strip(), tier, "csv"))
    return out, {"rows": len(out)}


# ---------------------------------------------------------------- merge and write

def validate(events: list[Event]) -> None:
    for e in events:
        if not (2000 <= e.ts.year <= 2035):
            raise CalendarImportError(f"implausible event date {e.ts} for {e.name}")
        if e.tier not in (1, 2, 3):
            raise CalendarImportError(f"bad tier for {e.name}")


def load_existing(path: str | Path = CALENDAR_FILE) -> list[Event]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    src = df["source"] if "source" in df else pd.Series([""] * len(df))
    return [Event(pd.Timestamp(r[0], tz="UTC"), str(r[1]), int(r[2]), str(r[3]) if isinstance(r[3], str) else "") for r in zip(df["ts_utc"], df["name"], df["tier"], src)]


def merge(old: list[Event], new: list[Event]) -> list[Event]:
    """Union, one row per (time, name); a newly imported row replaces an old one for the same event."""
    by = {(e.ts, e.name): e for e in old}
    by.update({(e.ts, e.name): e for e in new})
    return sorted(by.values(), key=lambda e: (e.ts, e.name))


def write(events: list[Event], path: str | Path = CALENDAR_FILE) -> int:
    validate(events)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_utc", "name", "tier", "source"])
        for e in events:
            w.writerow([e.ts.strftime("%Y-%m-%d %H:%M"), e.name, e.tier, e.source])
    os.replace(tmp, p)
    return len(events)


def coverage(events: list[Event]) -> pd.DataFrame:
    """Events per name per year: the gaps are visible instead of silent."""
    df = pd.DataFrame({"name": [e.name for e in events], "year": [e.ts.year for e in events]})
    return df.groupby(["name", "year"]).size().unstack(fill_value=0) if len(df) else df
