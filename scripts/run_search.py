#!/usr/bin/env python3
"""Strategy formulation. Usage:
  python scripts/run_search.py tier1 strategies/asian_breakout.yaml --trials 50
  python scripts/run_search.py tier2 --generations 5 --pop 20 --timeframe M15
  python scripts/run_search.py tier3 --iterations 3 [--dry-run]
Surviving candidates are written to strategies/candidates/ and must then pass scripts/run_gate.py."""
import argparse, datetime as dt, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goldsys.data.store import read_bars
from goldsys.dsl.spec import StrategySpec
from goldsys.engine.costs import CostModel
from goldsys.search.registry import Registry
p = argparse.ArgumentParser(); p.add_argument("tier", choices=["tier1", "tier2", "tier3"]); p.add_argument("yaml", nargs="?")
p.add_argument("--symbol", default="XAUUSD"); p.add_argument("--start"); p.add_argument("--end")
p.add_argument("--trials", type=int, default=50); p.add_argument("--generations", type=int, default=5); p.add_argument("--pop", type=int, default=20); p.add_argument("--timeframe", default="M15")
p.add_argument("--iterations", type=int, default=3); p.add_argument("--dry-run", action="store_true"); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
bars = read_bars(a.symbol, "M1", dt.date.fromisoformat(a.start) if a.start else None, dt.date.fromisoformat(a.end) if a.end else None)
if bars.empty:
    sys.exit(f"no M1 bars for {a.symbol}")
cost, reg = CostModel(), Registry()
out_dir = Path("strategies/candidates"); out_dir.mkdir(exist_ok=True)
if a.tier == "tier1":
    from goldsys.search.parametric import search_template
    spec = StrategySpec.from_yaml(a.yaml)
    r = search_template(spec, bars, cost, a.trials, reg, a.seed)
    best = spec.with_params(**r["best_params"]); best.name = spec.name + "_t1"
    (out_dir / f"{best.name}.yaml").write_text(best.to_yaml())
    print(json.dumps({k: v for k, v in r.items() if k != "top"}, indent=2, default=str)); print("top 5:"); [print(" ", h["score"], h["params"]) for h in r["top"][:5]]
elif a.tier == "tier2":
    from goldsys.search.genetic import evolve
    r = evolve(bars, cost, a.generations, a.pop, a.timeframe, a.seed, reg)
    for i, e in enumerate(r["elites"][:5]):
        (out_dir / f"gp_elite_{i}.yaml").write_text(e["spec_yaml"])
        print(f"elite {i}: niche={e['niche']} score={e['score']:.3f} trades={e['trades']} sharpe={e['sharpe']:.2f}\n   {json.dumps(e['expr'])}")
    print("evaluations:", r["n_evaluations"], "registry trials:", reg.n_trials())
else:
    from goldsys.search.llm import hypothesis_loop
    reps = hypothesis_loop(bars, cost, a.iterations, reg, dry_run=a.dry_run)
    for r in reps:
        print(r["strategy"], "PASS" if r.get("passed") else "FAIL", r.get("reasons"))
