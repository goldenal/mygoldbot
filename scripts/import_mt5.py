#!/usr/bin/env python3
"""Import an MT5 bar export (Symbols -> Bars -> Export Bars) into the lake as UTC M1 bars.

  python scripts/import_mt5.py ~/Documents/XAUUSD_M1_....csv --check          # profile + timezone detection, writes nothing
  python scripts/import_mt5.py ~/Documents/XAUUSD_M1_....csv                 # import as XAUUSD (timezone auto-detected)
  python scripts/import_mt5.py FILE --tz ny+7                                # force a clock rule: utc | +2 | ny+7 | Europe/Athens ...
  python scripts/import_mt5.py FILE --symbol XAUUSD_DERIV --source-name "Deriv MT5"

The lake keeps one source per symbol: importing into a symbol that holds other data is refused unless --replace.
Re-importing the same source is safe (bars are merged by timestamp)."""
import argparse, datetime as dt, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import REPORT_DIR
from goldsys.data import mt5csv, store

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("file")
p.add_argument("--symbol", default="XAUUSD")
p.add_argument("--tz", default="auto", help="auto | utc | +N | ny+7 | IANA zone (default auto: detected from the weekly-open pattern)")
p.add_argument("--point", type=float, default=0.01, help="price value of one MT5 spread point (gold, 2 digits: 0.01)")
p.add_argument("--source-name", default="MT5 export")
p.add_argument("--check", action="store_true", help="analyse only; write nothing")
p.add_argument("--replace", action="store_true", help="delete the symbol's existing M1 bars first")
a = p.parse_args()

f = Path(a.file).expanduser()
if not f.is_file():
    sys.exit(f"no such file: {f}")
try:
    raw = mt5csv.parse_mt5_csv(f)
    bars, rep = mt5csv.convert(raw, a.tz, a.point)
except mt5csv.ImportErrorMT5 as e:
    sys.exit(f"import refused: {e}")
rep.source_file, rep.sha256 = str(f), mt5csv.sha256_of(f)

print(f"file        {f.name}  ({rep.rows_read:,} rows)")
print(f"clock       {rep.timezone_rule}" + ("  (auto-detected)" if a.tz == "auto" else "  (given)"))
for r in rep.timezone_ranking[:4]:
    print(f"            candidate {r['rule']:14s} score {r['score']:.3f}  (weekly close {r['close_ny']} / open {r['open_ny']} New York)")
print(f"range       {bars.index[0]} -> {bars.index[-1]} UTC   ({len(bars):,} bars)")
print(f"dropped     {rep.dropped}")
print(f"fixed       {rep.fixed}")
print(f"quality     {rep.quality}")
print(f"spread $/oz {rep.spread_usd}")
print(f"spread by UTC hour (median $): {rep.spread_by_hour_median}")
if a.check:
    print("\n--check: nothing written.")
    sys.exit(0)

existing = mt5csv.has_bars(a.symbol)
src = mt5csv.read_source(a.symbol)
if existing and a.replace:
    n = 0
    for q in (store.BAR_DIR / a.symbol / "M1").glob("*.parquet"):
        q.unlink(); n += 1
    print(f"replaced    removed {n} existing yearly files for {a.symbol}")
elif existing and (src is None or src.get("source_name") != a.source_name):
    sys.exit(f"import refused: {a.symbol} already holds bars from {'an unrecorded source' if src is None else repr(src.get('source_name'))}. "
             f"Use --symbol <other>, --source-name to match, or --replace to overwrite.")
paths = store.write_bars(bars, a.symbol, "M1")
meta = {"source_name": a.source_name, "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "file": f.name, "sha256": rep.sha256,
        "timezone_rule": rep.timezone_rule, "point": a.point, "first": str(bars.index[0]), "last": str(bars.index[-1]), "rows": rep.rows_written,
        "price_basis": "BID", "spread_unit": "USD per oz; NaN where the export had 0 points"}
mt5csv.source_path(a.symbol).write_text(json.dumps(meta, indent=2))
out = REPORT_DIR / f"import_{a.symbol}_{dt.date.today().isoformat()}.json"
out.write_text(json.dumps(rep.to_dict(), indent=2, default=str))
print(f"\nwrote {len(paths)} yearly files to {store.BAR_DIR / a.symbol / 'M1'}; provenance in SOURCE.json; report {out}")
