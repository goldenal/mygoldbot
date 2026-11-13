"""Project-wide paths and constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TICK_DIR = DATA_DIR / "ticks"
BAR_DIR = DATA_DIR / "bars"
MACRO_DIR = DATA_DIR / "macro"
REPORT_DIR = ROOT / "reports"
REGISTRY_DB = ROOT / "registry.sqlite"

SYMBOL = "XAUUSD"
OZ_PER_LOT = 100.0          # 1 standard lot = 100 troy oz
PRICE_SCALE = 1000.0        # Dukascopy XAUUSD prices are stored as int / 1000

for _d in (RAW_DIR, TICK_DIR, BAR_DIR, MACRO_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
