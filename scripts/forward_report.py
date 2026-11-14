#!/usr/bin/env python3
"""Forward-test reconciliation: live/paper trades vs the screening engine on the same bars (Phase 4 exit check).
  python scripts/forward_report.py [--db live.sqlite] [--stage demo] [--min-weeks 8] [--min-trades 30] [--warmup-symbol XAUUSD]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, PORTFOLIO_FILE
from goldsys.data.store import read_bars
from goldsys.engine.costs import CostModel
from goldsys.live.ledger import Ledger
from goldsys.live.reconcile import ForwardConfig, forward_report, render_markdown
from goldsys.portfolio.config import Portfolio

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--db", default=str(LIVE_DB)); p.add_argument("--portfolio", default=str(PORTFOLIO_FILE)); p.add_argument("--stage")
p.add_argument("--min-weeks", type=float, default=8.0); p.add_argument("--min-trades", type=int, default=30)
p.add_argument("--warmup-symbol", help="lake symbol whose earlier M1 bars warm the indicators up (e.g. XAUUSD)")
a = p.parse_args()
pf = Portfolio.load(a.portfolio)
hist = read_bars(a.warmup_symbol, "M1") if a.warmup_symbol else None
rep = forward_report(Ledger(a.db), [s.load_spec() for s in pf.strategies if s.status != "retired"], CostModel(),
                     ForwardConfig(min_weeks=a.min_weeks, min_trades=a.min_trades), history=hist, stage=a.stage)
print(render_markdown(rep))
print("report:", rep.get("report_path"))
sys.exit(0 if rep["passed"] else 1)
