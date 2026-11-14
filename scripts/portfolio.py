#!/usr/bin/env python3
"""Portfolio step and its lifecycle.
  python scripts/portfolio.py build strategies/candidates/*.yaml --symbol XAUUSD [--allow-unvalidated] [--method risk_parity]
  python scripts/portfolio.py show
  python scripts/portfolio.py promote [--confirm]        # research -> demo -> tiny_live -> live, gated by the forward-test report
  python scripts/portfolio.py scale [--apply]            # size recommendation from REALISED live trades
  python scripts/portfolio.py mode STRATEGY auto|approve|off
Only strategies with a passing validation-gate report (scripts/run_gate.py) are eligible unless --allow-unvalidated,
and unvalidated strategies can never reach a real-money stage."""
import argparse, datetime as dt, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, PORTFOLIO_FILE, REPORT_DIR
from goldsys.data.store import read_bars
from goldsys.engine.costs import CostModel
from goldsys.portfolio.build import build_portfolio, render_markdown
from goldsys.portfolio.config import Portfolio

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--file", default=str(PORTFOLIO_FILE))
sub = p.add_subparsers(dest="cmd", required=True)
b = sub.add_parser("build"); b.add_argument("specs", nargs="+"); b.add_argument("--symbol", default="XAUUSD"); b.add_argument("--start"); b.add_argument("--end")
b.add_argument("--allow-unvalidated", action="store_true"); b.add_argument("--method", choices=["inverse_vol", "risk_parity"], default="inverse_vol")
b.add_argument("--corr-cap", type=float, default=0.7); b.add_argument("--target-vol", type=float, default=0.10); b.add_argument("--equity", type=float, default=10_000.0)
b.add_argument("--stage", choices=["research", "demo"], default="research"); b.add_argument("--mode", choices=["auto", "approve", "off"], default="approve")
sub.add_parser("show")
pr = sub.add_parser("promote"); pr.add_argument("--confirm", action="store_true"); pr.add_argument("--db", default=str(LIVE_DB))
sc = sub.add_parser("scale"); sc.add_argument("--apply", action="store_true"); sc.add_argument("--db", default=str(LIVE_DB))
md = sub.add_parser("mode"); md.add_argument("strategy"); md.add_argument("mode", choices=["auto", "approve", "off"])
a = p.parse_args()

if a.cmd == "build":
    bars = read_bars(a.symbol, "M1", dt.date.fromisoformat(a.start) if a.start else None, dt.date.fromisoformat(a.end) if a.end else None)
    if bars.empty:
        sys.exit(f"no M1 bars for {a.symbol}")
    pf, rep = build_portfolio(a.specs, bars, CostModel(), a.allow_unvalidated, a.method, a.corr_cap, a.target_vol, a.equity, a.stage, a.mode, symbol=a.symbol.replace("_SYN", ""))
    text = render_markdown(pf, rep)
    print(text)
    if not pf.strategies:
        sys.exit("nothing to write: no strategy is eligible (run scripts/run_gate.py on the candidates, or pass --allow-unvalidated for a plumbing test)")
    pf.save(a.file)
    (REPORT_DIR / "portfolio.md").write_text(text)
    (REPORT_DIR / "portfolio.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"wrote {a.file} and reports/portfolio.md")
    sys.exit(0)

pf = Portfolio.load(a.file)
if a.cmd == "show":
    print(f"stage={pf.stage} scale={pf.scale} equity_ref={pf.account_equity:,.0f} target_vol={pf.target_annual_vol:.0%} updated={pf.updated}")
    for s in pf.strategies:
        print(f"  {s.name:28s} {s.status:9s} mode={s.mode:8s} weight={s.weight:5.2f} validated={s.validated} drift_floor={s.drift.get('p05_r')}")
    for h in pf.history[-8:]:
        print(f"  [{h['ts']}] {h['action']}: {h['detail']}")
elif a.cmd == "mode":
    e = pf.get(a.strategy) or sys.exit(f"no strategy {a.strategy}")
    e.mode = a.mode; pf.log("mode", f"{a.strategy} -> {a.mode}"); pf.save(a.file); print("ok (restart the runner to pick it up; for an immediate change use scripts/ops.py mode)")
elif a.cmd == "promote":
    from goldsys.live.ledger import Ledger
    from goldsys.live.reconcile import forward_report
    from goldsys.live.scaling import REAL_MONEY, STAGE_FORWARD, ScalingConfig, stage_gate
    fwd = None
    if pf.stage in STAGE_FORWARD and Path(a.db).exists():
        fwd = forward_report(Ledger(a.db), [s.load_spec() for s in pf.active()], CostModel(), STAGE_FORWARD[pf.stage], stage=pf.stage)
    g = stage_gate(pf, fwd)
    print(json.dumps(g, indent=2))
    if not g["eligible"]:
        sys.exit(f"NOT eligible for '{g['next']}'")
    if not a.confirm:
        sys.exit(f"eligible for '{g['next']}'. Re-run with --confirm to apply.")
    old = pf.stage; pf.stage = g["next"]
    if pf.stage == "tiny_live":
        pf.scale = ScalingConfig().tiny_live_scale
    pf.log("promote", f"{old} -> {pf.stage} (scale {pf.scale})"); pf.save(a.file)
    print(f"stage is now {pf.stage}, scale {pf.scale}" + (". REAL MONEY: run_live.py additionally needs --confirm-live and GOLDSYS_ALLOW_LIVE=1" if pf.stage in REAL_MONEY else ""))
elif a.cmd == "scale":
    from goldsys.live.ledger import Ledger
    from goldsys.live.scaling import recommend_scale
    r = recommend_scale(Ledger(a.db), pf)
    print(json.dumps(r, indent=2))
    if a.apply and r["recommended"] != r["current"]:
        pf.scale = r["recommended"]; pf.log("scale", f"{r['current']} -> {r['recommended']}: {'; '.join(r['reasons'])}"); pf.save(a.file); print("applied")
