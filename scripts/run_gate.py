#!/usr/bin/env python3
"""Run the full validation gate on a strategy YAML. Usage: python scripts/run_gate.py strategies/vol_momentum.yaml [--symbol XAUUSD] [--quick]"""
import argparse, datetime as dt, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.data.store import read_bars
from goldsys.dsl.spec import StrategySpec
from goldsys.engine.costs import CostModel
from goldsys.validation.gate import GateConfig, run_gate
from goldsys.search.registry import Registry
p = argparse.ArgumentParser(); p.add_argument("yaml"); p.add_argument("--symbol", default="XAUUSD"); p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--quick", action="store_true")
a = p.parse_args()
bars = read_bars(a.symbol, "M1", dt.date.fromisoformat(a.start) if a.start else None, dt.date.fromisoformat(a.end) if a.end else None)
if bars.empty:
    sys.exit(f"no M1 bars for {a.symbol}")
spec = StrategySpec.from_yaml(a.yaml)
cfg = GateConfig(wfa_max_points=12, cpcv_max_points=8, pbo_trials=12, run_plateau=False) if a.quick else GateConfig()
reg = Registry()
rep = run_gate(spec, bars, CostModel(), cfg, n_trials_so_far=reg.n_trials() + 1)
reg.log_trial("gate", spec, rep.get("in_sample", {}), bars.index[0], bars.index[-1], "v1-default", oos_sharpe=rep.get("walk_forward", {}).get("oos_sharpe"), passed=rep.get("passed"))
print("report:", rep.get("report_path"))
