#!/usr/bin/env python3
"""Backtest one strategy YAML on the lake. Usage: python scripts/run_backtest.py strategies/asian_breakout.yaml [--symbol XAUUSD] [--start 2019-01-01] [--end 2026-01-01] [--spread-mult 1.0]"""
import argparse, datetime as dt, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.data.store import read_bars
from goldsys.dsl.spec import StrategySpec
from goldsys.dsl.compile import backtest_spec
from goldsys.engine.costs import CostModel
from goldsys.engine.metrics import summarize
p = argparse.ArgumentParser(); p.add_argument("yaml"); p.add_argument("--symbol", default="XAUUSD"); p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--spread-mult", type=float, default=1.0); p.add_argument("--trades", type=int, default=10)
a = p.parse_args()
bars = read_bars(a.symbol, "M1", dt.date.fromisoformat(a.start) if a.start else None, dt.date.fromisoformat(a.end) if a.end else None)
if bars.empty:
    sys.exit(f"no M1 bars for {a.symbol}; run scripts/download_data.py or scripts/make_synthetic.py")
spec = StrategySpec.from_yaml(a.yaml)
cost = CostModel(spread_multiplier=a.spread_mult)
t0 = time.time(); res = backtest_spec(spec, bars, cost); dt_ = time.time() - t0
m = summarize(res.equity, res.trades)
print(f"{spec.name} on {a.symbol} {bars.index[0].date()}..{bars.index[-1].date()} ({len(bars):,} M1 bars) in {dt_:.2f}s")
for k, v in m.items():
    print(f"  {k:22s} {v:,.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
print(res.trades.tail(a.trades).to_string())
