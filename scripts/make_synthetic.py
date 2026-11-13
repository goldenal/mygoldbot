#!/usr/bin/env python3
"""Generate synthetic gold-like M1 bars into the lake (for development without real data)."""
import argparse, datetime as dt, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.data.synthetic import make_m1
from goldsys.data.store import write_bars
p = argparse.ArgumentParser(); p.add_argument("start"); p.add_argument("end"); p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
bars = make_m1(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), seed=a.seed)
print("wrote", len(bars), "bars to", write_bars(bars, "XAUUSD_SYN", "M1"))
