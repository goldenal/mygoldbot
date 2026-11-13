#!/usr/bin/env python3
"""Download XAUUSD ticks from Dukascopy into the lake and build M1 bars.
Usage: python scripts/download_data.py 2019-01-01 2026-11-01 [--pause 0.3]
Rate-limited feed: expect ~1-2 s per hour-file; a year is ~6000 files. Resume is automatic (raw files are cached)."""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.data.dukascopy import download_range
from goldsys.data.store import build_m1_from_ticks
from goldsys.data.quality import check_bars

p = argparse.ArgumentParser()
p.add_argument("start")
p.add_argument("end")
p.add_argument("--pause", type=float, default=0.3)
p.add_argument("--symbol", default="XAUUSD")
a = p.parse_args()
s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
download_range(s, e, a.symbol, a.pause, on_day=lambda d, n: print(f"{d} ticks={n}", flush=True))
bars = build_m1_from_ticks(a.symbol, s, e)
print("M1 bars:", len(bars))
print(check_bars(bars))
