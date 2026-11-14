#!/usr/bin/env python3
"""Run ON THE WINDOWS VPS next to a logged-in MT5 terminal: exposes the broker to the research box.
  set GOLDSYS_BRIDGE_TOKEN=<long random string>      (same value on the research box; never commit it)
  python scripts/mt5_bridge_server.py --symbol XAUUSD [--port 8765] [--utc-offset 3]
  python scripts/mt5_bridge_server.py --selftest      # DEMO ONLY: opens and closes one minimum-size position, prints what came back
  python scripts/mt5_bridge_server.py --paper --replay-symbol XAUUSD --replay-start 2026-08-03 --speed 120
                                                      # any OS, no MT5: a simulated venue playing lake history forward as if live
Binds to 127.0.0.1. Reach it from the research box with an SSH tunnel:  ssh -L 8765:127.0.0.1:8765 user@vps"""
import argparse, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.live.alerts import load_dotenv
from goldsys.live.bridge import make_server

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--symbol", default="XAUUSD"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
p.add_argument("--utc-offset", type=float, help="broker server time minus UTC, in hours (auto-detected from the latest tick if omitted)")
p.add_argument("--selftest", action="store_true"); p.add_argument("--paper", action="store_true")
p.add_argument("--replay-symbol", help="with --paper: lake symbol to play forward as a live venue"); p.add_argument("--replay-start"); p.add_argument("--speed", type=float, default=60.0)
a = p.parse_args()
load_dotenv()
if a.paper:
    import pandas as pd
    if a.replay_symbol:                       # a simulated venue playing lake history forward: rehearses the whole live loop
        from goldsys.data.store import read_bars
        from goldsys.live.simvenue import SimVenue
        m1 = read_bars(a.replay_symbol, "M1")
        if m1.empty:
            sys.exit(f"no M1 bars for {a.replay_symbol}")
        broker = SimVenue(m1, start=pd.Timestamp(a.replay_start, tz="UTC") if a.replay_start else None, speed=a.speed, symbol=a.symbol)
        print(f"simulated venue: {a.replay_symbol} from {broker.now()} at {a.speed:g}x")
    else:
        from goldsys.live.broker import PaperBroker
        broker = PaperBroker(symbol=a.symbol)
        broker.set_quote(pd.Timestamp.now(tz="UTC"), 2400.0, 0.2)
else:
    from goldsys.live.mt5 import MT5Broker
    broker = MT5Broker(a.symbol, server_utc_offset_hours=a.utc_offset)
    print(f"MT5 connected: {broker.account()} demo={broker.is_demo} server-UTC offset={broker.offset_s / 3600:+g}h point={broker.point} min lot={broker.vol_min}")
if a.selftest:
    if not a.paper and not broker.is_demo:
        sys.exit("selftest refuses to run on a real-money account")
    q = broker.quote(); print("quote:", q)
    print("last M1 bars:\n", broker.bars("M1", 3))
    lots = 0.01 if a.paper else broker.vol_min
    f = broker.open_position("selftest", 1, lots, stop_dist=20.0, target_dist=40.0); print("entry fill:", f)
    print("positions:", broker.positions())
    time.sleep(2)
    print("exit fill:", broker.close_position(f.position_id, "manual"))
    print("positions after close:", broker.positions(), "| poll_fills:", broker.poll_fills())
    sys.exit(0)
token = os.environ.get("GOLDSYS_BRIDGE_TOKEN") or sys.exit("set GOLDSYS_BRIDGE_TOKEN first")
srv = make_server(broker, token, a.host, a.port)
print(f"bridge serving {broker.name} on http://{a.host}:{a.port}  (Ctrl-C to stop)")
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
