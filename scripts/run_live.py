#!/usr/bin/env python3
"""Run the portfolio through the live stack (signals -> risk -> approval queue -> broker -> ledger).

  replay   historical M1 bars from the lake through the paper broker, as fast as possible.
           python scripts/run_live.py replay --symbol XAUUSD --start 2026-08-01 --end 2026-11-01
  paper    live bars from the MT5 bridge, simulated execution. No orders leave this machine.
           python scripts/run_live.py paper --bridge http://127.0.0.1:8765
  broker   live bars AND real orders through the bridge (demo or real account, per the portfolio's stage).
           python scripts/run_live.py broker --bridge http://127.0.0.1:8765 [--confirm-live]

The bridge token is read from GOLDSYS_BRIDGE_TOKEN (environment or .env). Each mode keeps its own
ledger file so a replay can never be mistaken for a track record. Stop with Ctrl-C or SIGTERM: open positions
stay open (their stops are at the broker) unless --flatten-on-exit."""
import argparse, datetime as dt, os, signal, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from goldsys.config import LIVE_DB, PORTFOLIO_FILE, ROOT
from goldsys.data.store import read_bars
from goldsys.engine.costs import CostModel
from goldsys.live.alerts import load_dotenv
from goldsys.live.ledger import Ledger
from goldsys.live.session import StageError, build_runner, check_stage, paper_broker
from goldsys.portfolio.config import Portfolio

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("mode", choices=["replay", "paper", "broker"])
p.add_argument("--portfolio", default=str(PORTFOLIO_FILE)); p.add_argument("--db")
p.add_argument("--symbol", default="XAUUSD", help="replay: lake symbol"); p.add_argument("--start"); p.add_argument("--end")
p.add_argument("--fresh", action="store_true", help="replay: delete the replay ledger first")
p.add_argument("--no-slippage", action="store_true"); p.add_argument("--seed", type=int, default=0)
p.add_argument("--bridge", help="URL of scripts/mt5_bridge_server.py"); p.add_argument("--poll", type=float, default=5.0)
p.add_argument("--confirm-live", action="store_true"); p.add_argument("--flatten-on-exit", action="store_true")
p.add_argument("--approval-ttl", type=int, default=900, help="seconds a signal waits for approval before it expires")
a = p.parse_args()
load_dotenv(str(ROOT / ".env"))
pf = Portfolio.load(a.portfolio)
cost = CostModel()
db = Path(a.db) if a.db else {"replay": ROOT / "live_replay.sqlite", "paper": ROOT / "live_paper.sqlite", "broker": LIVE_DB}[a.mode]

if a.mode == "replay":
    if a.fresh:
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
    bars = read_bars(a.symbol, "M1", None, dt.date.fromisoformat(a.end) if a.end else None)
    if bars.empty:
        sys.exit(f"no M1 bars for {a.symbol}")
    ledger = Ledger(db)
    broker = paper_broker(pf, cost, None, slippage=not a.no_slippage, seed=a.seed)
    runner = build_runner(pf, broker, ledger, cost, stage="replay", bars_m1=bars, console=False, snapshot_minutes=60)
    ex = bars[bars.index >= pd.Timestamp(a.start, tz="UTC")] if a.start else bars
    ex = ex.copy(); ex["spread"] = cost.spread_at(ex.index, ex["spread"])
    t0 = time.time(); runner.replay(ex); el = time.time() - t0
    t = ledger.trades(closed_only=True); acct = broker.account()
    print(f"replayed {len(ex):,} M1 bars ({ex.index[0].date()}..{ex.index[-1].date()}) in {el:.1f}s -> {db.name}")
    print(f"trades={len(t)} net={t['pnl'].sum():,.0f} equity={acct.equity:,.0f} intents={ {r['status']: r['n'] for r in map(dict, ledger.conn.execute('SELECT status, COUNT(*) n FROM intents GROUP BY status'))} }")
    print("kill switch:", ledger.get("kill"), "| chain:", ledger.verify_chain()["ok"])
    sys.exit(0)

from goldsys.live.bridge import RemoteBroker
token = os.environ.get("GOLDSYS_BRIDGE_TOKEN") or sys.exit("set GOLDSYS_BRIDGE_TOKEN (environment or .env)")
if not a.bridge:
    sys.exit("--bridge URL is required for paper and broker modes")
remote = RemoteBroker(a.bridge, token)
if not remote.connected():
    sys.exit(f"bridge at {a.bridge} is not reachable or the terminal is not connected")
ledger = Ledger(db)
def now():
    """The venue's clock (wall time for MT5; its own for a simulated venue). Wall time if the venue cannot be reached."""
    try:
        return remote.now()
    except Exception:  # noqa: BLE001
        return pd.Timestamp.now(tz="UTC")


stop = {"flag": False}
for sig in (signal.SIGINT, signal.SIGTERM):          # SIGTERM too: service managers and `kill` send that, not Ctrl-C
    signal.signal(sig, lambda *_: stop.update(flag=True))
try:
    if a.mode == "paper":
        broker = paper_broker(pf, cost, ledger, slippage=not a.no_slippage, seed=a.seed)
        runner = build_runner(pf, broker, ledger, cost, stage="paper", feed=remote, snapshot_minutes=5, realtime=True, approval_ttl_s=a.approval_ttl)
    else:
        check_stage(pf, remote, a.confirm_live)
        broker = remote
        runner = build_runner(pf, broker, ledger, cost, feed=remote, snapshot_minutes=5, realtime=True, approval_ttl_s=a.approval_ttl)
except (StageError, RuntimeError) as e:
    sys.exit(f"refused: {e}")
last = remote.bars("M1", 3)
runner.last_bar_ns = int(last.index.as_unit("ns").asi8[-1]) if len(last) else None      # start from the next bar: never act on history
if a.mode == "paper" and len(last):
    broker.set_quote(last.index[-1], float(last["close"].iloc[-1]), float(cost.spread_at(last.index[-1:], last["spread"].iloc[-1:])[0]))
runner.adopt_positions(now())
runner.notify("info", "start", f"runner started: mode={a.mode} stage={runner.stage} strategies={[s.name for s in runner.slots]} account={remote.account()}")
while not stop["flag"]:
    try:
        if a.mode == "broker":
            runner.live_step(now())
        else:
            bars = remote.bars("M1", 300)
            if len(bars):
                ns = bars.index.as_unit("ns").asi8
                sp = cost.spread_at(bars.index, bars["spread"])
                for i in range(len(bars)):
                    if runner.last_bar_ns is None or ns[i] > runner.last_bar_ns:
                        r = bars.iloc[i]
                        runner.feed_bar(int(ns[i]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(sp[i]), float(r["ticks"]), float(r["volume"]))
                runner._process_approvals(now())
    except Exception as e:  # noqa: BLE001  keep the loop alive; the kill switch handles persistent faults
        runner.notify("error", "loop", f"{type(e).__name__}: {e}", None, now())
    time.sleep(a.poll)
if a.flatten_on_exit:
    runner.flatten_all("manual", now())
runner.risk.flush(); ledger.conn.commit()
runner.release_lock()
runner.notify("warning", "stop", "runner stopped by operator" + (" (flattened)" if a.flatten_on_exit else f" ({sum(1 for s in runner.slots if s.position_id)} positions left open)"))
