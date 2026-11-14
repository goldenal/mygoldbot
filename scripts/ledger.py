#!/usr/bin/env python3
"""Your own verified track record.
  python scripts/ledger.py verify [--db live.sqlite]          # recompute the hash chain; any edited/deleted trade fails
  python scripts/ledger.py statement [--stage live]            # monthly table + headline stats, also written to reports/
  python scripts/ledger.py export trades.csv"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, REPORT_DIR
from goldsys.live.ledger import Ledger
from goldsys.live.statement import render_statement, statement

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--db", default=str(LIVE_DB))
sub = p.add_subparsers(dest="cmd", required=True)
sub.add_parser("verify")
s = sub.add_parser("statement"); s.add_argument("--stage")
e = sub.add_parser("export"); e.add_argument("path")
a = p.parse_args()
if not Path(a.db).exists():
    sys.exit(f"no ledger at {a.db}")
led = Ledger(a.db)
if a.cmd == "verify":
    v = led.verify_chain()
    print(v)
    sys.exit(0 if v["ok"] else 1)
elif a.cmd == "statement":
    st = statement(led, a.stage)
    text = render_statement(st)
    print(text)
    out = REPORT_DIR / f"statement_{a.stage or 'all'}.md"
    out.write_text(text); print("written:", out)
else:
    led.trades().to_csv(a.path, index=False); print("wrote", a.path)
