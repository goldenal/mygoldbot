#!/usr/bin/env python3
"""Build data/calendar/events.csv: the schedule of high-impact US releases (never the released numbers).

  python scripts/import_calendar.py --fomc                       # Federal Reserve calendar page: FOMC statements and minutes, 2021-2027 (no key)
  FRED_API_KEY=... python scripts/import_calendar.py --fred      # jobs report, CPI, PPI, PCE, GDP, retail sales: full history + schedule (free key)
  python scripts/import_calendar.py --ics ~/Downloads/bls.ics    # a file you downloaded yourself (the BLS blocks automated downloads)
  python scripts/import_calendar.py --csv my_events.csv          # columns ts_utc,name,tier
  add --check to see what would be written without writing it.

The new rows are merged into the existing file (one row per time and event). Nothing is guessed: if a source is not imported, its events are missing and the
coverage table at the end says so."""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import CALENDAR_FILE, ROOT
from goldsys.data import calendar_import as ci

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--fomc", action="store_true"); p.add_argument("--fred", action="store_true"); p.add_argument("--ics"); p.add_argument("--csv")
p.add_argument("--start", default="2015-01-01"); p.add_argument("--end", default="2027-12-31"); p.add_argument("--check", action="store_true")
a = p.parse_args()
if not (a.fomc or a.fred or a.ics or a.csv):
    p.error("choose at least one source: --fomc, --fred, --ics FILE or --csv FILE")


def key() -> str:
    k = os.environ.get("FRED_API_KEY", "")
    env = ROOT / ".env"
    if not k and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("FRED_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("\"'")
    return k


new = []
try:
    if a.fomc:
        ev, d = ci.fetch_fomc()
        print(f"FOMC: {d['meetings']} meetings, {len(ev)} events; statement dates cross-checked against the statement links for {d['link_checked']} meetings; "
              f"{len(d['link_mismatch'])} disagreed {d['link_mismatch'][:3]}; skipped {d['skipped']}")
        new += ev
    if a.fred:
        ev, d = ci.fetch_fred(key(), a.start, a.end)
        print(f"FRED: {d}")
        new += ev
    if a.ics:
        ev, d = ci.parse_ics(Path(a.ics).expanduser().read_text(errors="replace"))
        print(f"ICS: {d['vevents']} entries, {len(ev)} recognised, {d['ignored']} ignored")
        new += ev
    if a.csv:
        ev, d = ci.read_csv(Path(a.csv).expanduser())
        print(f"CSV: {d['rows']} rows")
        new += ev
except ci.CalendarImportError as e:
    sys.exit(f"import refused: {e}")
allev = ci.merge(ci.load_existing(CALENDAR_FILE), new)
print("\nevents per year:\n" + ci.coverage(allev).to_string())
missing = [n for n in ci.EVENTS if n not in {e.name for e in allev} and ci.EVENTS[n][0] <= 2]
if missing:
    print(f"\nNOT in the calendar (so the news blackout and news proximity cannot see them): {', '.join(missing)}")
if a.check:
    print("\n--check: nothing written.")
else:
    n = ci.write(allev, CALENDAR_FILE)
    print(f"\nwrote {n} events to {CALENDAR_FILE}")
