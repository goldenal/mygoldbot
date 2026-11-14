#!/usr/bin/env python3
"""Chart viewer: scroll back through history, add indicators, overlay a strategy's trades.
  python scripts/chart.py [--port 8787] [--db live.sqlite]
Then open http://127.0.0.1:8787/chart . With --db (or an existing live.sqlite) the operations dashboard is served at / as well."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, PORTFOLIO_FILE
from goldsys.dashboard.server import make_dashboard
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--port", type=int, default=8787); p.add_argument("--db", default=None)
a = p.parse_args()
db = a.db or (str(LIVE_DB) if LIVE_DB.exists() else None)
srv = make_dashboard(db, str(PORTFOLIO_FILE), port=a.port, read_only=True)
print(f"chart on http://127.0.0.1:{a.port}/chart" + (f"   operations (read-only) on http://127.0.0.1:{a.port}/  ledger={db}" if db else "") + "   Ctrl-C to stop", flush=True)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
