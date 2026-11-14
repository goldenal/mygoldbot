#!/usr/bin/env python3
"""Phase 5 monthly routine: refresh the walk-forward on new data, retire what has drifted, let the
formulation engine propose replacements, re-allocate.
  python scripts/monthly_review.py --symbol XAUUSD [--db live.sqlite] [--gp-generations 5] [--quick] [--apply]
Without --apply nothing is changed: the proposed portfolio is printed and the report is written to reports/.
Download last month's ticks first (scripts/download_data.py)."""
import argparse, datetime as dt, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, PORTFOLIO_FILE, REPORT_DIR
from goldsys.data.store import read_bars
from goldsys.engine.costs import CostModel
from goldsys.live.ledger import Ledger
from goldsys.portfolio.config import Portfolio
from goldsys.research.review import ReviewConfig, monthly_review, render_markdown
from goldsys.search.registry import Registry
from goldsys.validation.gate import GateConfig

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--symbol", default="XAUUSD"); p.add_argument("--start"); p.add_argument("--end")
p.add_argument("--portfolio", default=str(PORTFOLIO_FILE)); p.add_argument("--db", default=str(LIVE_DB))
p.add_argument("--tier1-trials", type=int, default=40); p.add_argument("--gp-generations", type=int, default=0); p.add_argument("--gp-pop", type=int, default=20)
p.add_argument("--no-replacements", action="store_true"); p.add_argument("--quick", action="store_true", help="smaller search grids (plumbing check, not a real review)")
p.add_argument("--apply", action="store_true")
a = p.parse_args()
bars = read_bars(a.symbol, "M1", dt.date.fromisoformat(a.start) if a.start else None, dt.date.fromisoformat(a.end) if a.end else None)
if bars.empty:
    sys.exit(f"no M1 bars for {a.symbol}")
pf = Portfolio.load(a.portfolio)
cfg = ReviewConfig(tier1_trials=a.tier1_trials, gp_generations=a.gp_generations, gp_pop=a.gp_pop, propose_replacements=not a.no_replacements)
if a.quick:
    cfg.wfa_max_points = 10
    cfg.gate = GateConfig(wfa_max_points=10, cpcv_max_points=6, pbo_trials=10, run_plateau=False)
ledger = Ledger(a.db) if Path(a.db).exists() else None
new, rep = monthly_review(pf, bars, ledger, Registry(), CostModel(), cfg)
text = render_markdown(rep)
print(text)
tag = f"monthly_review_{dt.date.today():%Y-%m}"
(REPORT_DIR / f"{tag}.md").write_text(text); (REPORT_DIR / f"{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
print(f"report: reports/{tag}.md")
if a.apply:
    new.save(a.portfolio); print(f"applied to {a.portfolio} (restart the runner)")
else:
    print("dry run: portfolio unchanged. Re-run with --apply to write it.")
