#!/usr/bin/env python3
"""Operator console for the running system (the hybrid / human layer).
  python scripts/ops.py status | pending | events
  python scripts/ops.py approve 12        python scripts/ops.py veto 12 --reason "FOMC in 20 min"
  python scripts/ops.py mode vol_momentum approve|auto|off
  python scripts/ops.py risk-off --reason "geopolitical headline"     python scripts/ops.py risk-on
  python scripts/ops.py kill --reason "..."                           python scripts/ops.py reset-kill
Add --db live_paper.sqlite (or live_replay.sqlite) to look at another ledger. Every action is logged with $USER."""
import argparse, getpass, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.config import LIVE_DB, PORTFOLIO_FILE
from goldsys.live import control
from goldsys.live.ledger import Ledger
from goldsys.portfolio.config import Portfolio

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--db", default=str(LIVE_DB)); p.add_argument("--portfolio", default=str(PORTFOLIO_FILE))
sub = p.add_subparsers(dest="cmd", required=True)
for c in ("status", "pending", "events", "risk-on", "reset-kill"):
    sub.add_parser(c)
for c in ("approve", "veto"):
    s = sub.add_parser(c); s.add_argument("id", type=int); s.add_argument("--reason", default="")
m = sub.add_parser("mode"); m.add_argument("strategy"); m.add_argument("mode", choices=control.MODES)
for c in ("risk-off", "kill"):
    s = sub.add_parser(c); s.add_argument("--reason", default="manual")
a = p.parse_args()
if not Path(a.db).exists():
    sys.exit(f"no ledger at {a.db}")
led, who = Ledger(a.db), getpass.getuser()
pf = Portfolio.load(a.portfolio) if Path(a.portfolio).exists() else None
side = lambda d: "BUY " if d == 1 else "SELL"  # noqa: E731

if a.cmd == "status":
    st = control.state(led, pf)
    ac = st["account"]
    print(f"stage={st['stage']} kill={'ACTIVE: ' + str(st['kill'].get('reason')) if st['kill'].get('active') else 'off'} risk_off={'ON' if st['risk_off'].get('active') else 'off'}")
    if ac:
        f = lambda v, pct=False: "n/a" if v is None else (f"{v:.2%}" if pct else f"{v:+,.0f}")  # noqa: E731
        print(f"equity={ac['equity']:,.0f} balance={ac['balance']:,.0f} day={f(ac['day_pnl'])} week={f(ac['week_pnl'])} drawdown={f(ac['drawdown'], True)} open_lots={ac['open_lots']:.2f}  (as of {ac['ts']})")
    for s in st["strategies"]:
        d = s["drift"]
        print(f"  {s['name']:28s} mode={s['mode']:8s} trades={s['trades']:4d} pnl={s['pnl']:+10,.0f} meanR={'n/a' if s['mean_r'] is None else format(s['mean_r'], '+.3f')} drift={d['state']}"
              + (f" ({d['mean_r']:+.3f} vs floor {d['p05_r']:+.3f})" if d.get("mean_r") is not None and d.get("p05_r") is not None else ""))
    for t in st["positions"]:
        print(f"  OPEN {t['strategy']} {side(t['dir'])} {t['lots']:.2f} @ {t['entry_px']:.2f} stop {t['stop_px']} since {t['entry_ts']}")
    print(f"pending approvals: {len(st['pending'])}")
elif a.cmd == "pending":
    rows = led.intents("pending")
    for r in rows:
        print(f"#{r['id']} {r['ts']} {r['strategy']} {side(r['dir'])} {r['lots']:.2f} lots @ ~{r['ref_px']:.2f} stop_dist={r['stop_dist']:.2f} expires {r['expires_ts']}")
    print(f"{len(rows)} pending")
elif a.cmd == "events":
    for e in reversed(led.events(40)):
        print(f"{e['ts'] or '':32s} {e['level']:8s} {e['kind']:18s} {e['message']}")
elif a.cmd == "approve":
    print("approved" if control.approve(led, a.id, who) else "nothing to approve (not pending)")
elif a.cmd == "veto":
    print("vetoed" if control.veto(led, a.id, who, a.reason) else "nothing to veto (not pending)")
elif a.cmd == "mode":
    control.set_mode(led, a.strategy, a.mode, who); print("ok")
elif a.cmd == "risk-off":
    control.risk_off(led, True, who, a.reason); print("risk-off ON: no new entries until risk-on")
elif a.cmd == "risk-on":
    control.risk_off(led, False, who); print("risk-off OFF")
elif a.cmd == "kill":
    control.kill(led, who, a.reason); print("kill switch ACTIVE: positions are flattened on the runner's next step; no entries until reset-kill")
elif a.cmd == "reset-kill":
    control.reset_kill(led, who); print("kill switch reset")
