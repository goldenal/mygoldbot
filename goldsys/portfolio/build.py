"""Build portfolio.yaml from strategy specs: check each has a passing gate report, backtest it for
its return stream and drift reference, then allocate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import REPORT_DIR, ROOT
from ..dsl.compile import backtest_spec
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..engine.metrics import summarize
from ..live.drift import drift_reference
from ..risk.limits import RiskConfig
from .allocate import allocate
from .config import Portfolio, StrategyEntry


def gate_passed(spec: StrategySpec) -> bool | None:
    """True/False from the validation report for exactly this spec (name + hash); None if there is no report."""
    p = REPORT_DIR / f"{spec.name}_{spec.hash()}.json"
    if not p.exists():
        return None
    return bool(json.loads(p.read_text()).get("passed"))


def daily_pnl_returns(equity: pd.Series, base: float) -> pd.Series:
    """Daily P&L as a fraction of the (fixed) sizing equity: additive across strategies, unlike compounded returns."""
    eq = equity.resample("1D").last().dropna()
    return (eq.diff() / base).dropna()


def build_portfolio(spec_paths: list[str | Path], bars_m1: pd.DataFrame, cost: CostModel | None = None, allow_unvalidated: bool = False,
                    method: str = "inverse_vol", corr_cap: float = 0.7, target_annual_vol: float = 0.10, account_equity: float = 10_000.0,
                    stage: str = "demo", mode: str = "approve", drift_window: int = 30, risk: RiskConfig | None = None, symbol: str = "XAUUSD") -> tuple[Portfolio, dict]:
    cost = cost or CostModel()
    daily, scores, info, rejected = {}, {}, {}, []
    for path in spec_paths:
        spec = StrategySpec.from_yaml(path)
        passed = gate_passed(spec)
        if not passed and not allow_unvalidated:
            rejected.append({"name": spec.name, "reason": "no validation report for this spec" if passed is None else "failed the validation gate"})
            continue
        res = backtest_spec(spec, bars_m1, cost, cache_key=None)
        m = summarize(res.equity, res.trades)
        daily[spec.name] = daily_pnl_returns(res.equity, spec.sizing.equity)
        scores[spec.name] = m["sharpe"]
        rel = Path(path).resolve()
        info[spec.name] = {"spec": spec, "path": str(rel.relative_to(ROOT)) if rel.is_relative_to(ROOT) else str(rel), "validated": bool(passed), "metrics": m,
                           "drift": drift_reference(res.trades, drift_window)}
    alloc = allocate(daily, scores, method, corr_cap, target_annual_vol)
    pf = Portfolio(stage=stage, symbol=symbol, account_equity=account_equity, target_annual_vol=target_annual_vol, risk=(risk or RiskConfig()).to_dict())
    for name, w in alloc["weights"].items():
        i = info[name]
        m = i["metrics"]
        pf.strategies.append(StrategyEntry(
            name=name, spec_path=i["path"], spec_hash=i["spec"].hash(), weight=w, mode=mode, validated=i["validated"], drift=i["drift"],
            reference={"trades": m["trades"], "sharpe": round(m["sharpe"], 3), "expectancy": round(m["expectancy"], 2), "max_drawdown": round(m["max_drawdown"], 4),
                       "win_rate": round(m["win_rate"], 3), "data_start": str(bars_m1.index[0].date()), "data_end": str(bars_m1.index[-1].date()), "cost_version": cost.version}))
    pf.log("build", f"{len(pf.strategies)} strategies, method={method}, corr_cap={corr_cap}, allow_unvalidated={allow_unvalidated}")
    report = {k: v for k, v in alloc.items() if k != "daily"}
    report.update(rejected=rejected, strategies={n: {"validated": i["validated"], **{k: i["metrics"][k] for k in ("trades", "sharpe", "net_profit", "max_drawdown")}} for n, i in info.items()})
    return pf, report


def render_markdown(pf: Portfolio, rep: dict) -> str:
    L = [f"# Portfolio report ({pf.stage})", "", f"Target annual vol {pf.target_annual_vol:.0%}, reference equity {pf.account_equity:,.0f}.", ""]
    if not pf.strategies:
        L += ["**No strategy qualified.**", ""]
    else:
        p = rep["portfolio"]
        L += [f"Combined (backtest): Sharpe {p['sharpe']:.2f}, annual vol {p['annual_vol']:.1%}, annual return {p['annual_return']:.1%}, max drawdown {p['max_drawdown']:.1%}, gross weight {p['gross_weight']:.2f}"
              + ("" if p["vol_target_reached"] else " (vol target NOT reached: weight caps bind)"), "",
              "| strategy | weight | risk share | validated | trades | Sharpe | drift floor (R, 30 trades) |", "|---|---|---|---|---|---|---|"]
        for s in pf.strategies:
            L.append(f"| {s.name} | {s.weight:.2f} | {rep['risk_contribution'].get(s.name, 0):.0%} | {'yes' if s.validated else '**NO**'} | {s.reference['trades']} | {s.reference['sharpe']:.2f} | {s.drift.get('p05_r', float('nan')):.3f} |")
        L.append("")
    if rep.get("dropped"):
        L += ["Dropped by the correlation cap:"] + [f"- {d['name']} (corr {d['corr']:.2f} with {d['correlated_with']})" for d in rep["dropped"]] + [""]
    if rep.get("rejected"):
        L += ["Not eligible:"] + [f"- {d['name']}: {d['reason']}" for d in rep["rejected"]] + [""]
    names = list(rep.get("correlation", {}))
    if len(names) > 1:
        L += ["## Daily-return correlation", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for a in names:
            L.append(f"| {a} | " + " | ".join(f"{rep['correlation'][b][a]:.2f}" for b in names) + " |")
        L.append("")
    return "\n".join(L)
