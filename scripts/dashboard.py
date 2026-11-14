#!/usr/bin/env python3
"""Operations dashboard: equity, risk state, approval queue, per-strategy drift, fills, alerts.
  python scripts/dashboard.py [--db live.sqlite] [--port 8787] [--read-only]
Then open http://127.0.0.1:8787 . Approve / veto / risk-off / kill act on the running runner through the ledger."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, PORTFOLIO_FILE
from goldsys.dashboard.server import make_dashboard
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--db", default=str(LIVE_DB)); p.add_argument("--portfolio", default=str(PORTFOLIO_FILE)); p.add_argument("--port", type=int, default=8787); p.add_argument("--read-only", action="store_true")
a = p.parse_args()
if not Path(a.db).exists():
    sys.exit(f"no ledger at {a.db} (run scripts/run_live.py first)")
srv = make_dashboard(a.db, a.portfolio, port=a.port, read_only=a.read_only)
print(f"dashboard on http://127.0.0.1:{a.port}  ledger={a.db}{'  (read-only)' if a.read_only else ''}  Ctrl-C to stop", flush=True)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
