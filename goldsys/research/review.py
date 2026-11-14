"""Phase 5: the monthly review. For every strategy in the portfolio
  1. re-run the walk-forward on data that now includes last month,
  2. read its realised drift from the ledger,
  3. retire it if either says the edge is gone,
then let the formulation engine propose replacements (re-fit of the retired template, optional GP
run), push those through the full validation gate, and re-allocate the weights of what remains.

Replacements enter as status `candidate` with weight 0: they have passed the gate but not a forward
test, so a human decides when they start trading on demo. Every search trial is counted in the
registry, so next month's deflated Sharpe is harder to clear, as it should be.
The function returns a proposed portfolio and a report; nothing is written unless the caller saves it."""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from ..dsl.compile import backtest_spec
from ..engine.costs import CostModel
from ..engine.metrics import summarize
from ..live.drift import drift_reference, drift_status
from ..live.ledger import Ledger
from ..portfolio.allocate import allocate
from ..portfolio.build import daily_pnl_returns
from ..portfolio.config import Portfolio, StrategyEntry
from ..search.registry import Registry
from ..validation.gate import GateConfig, run_gate
from ..validation.walkforward import walk_forward


@dataclass
class ReviewConfig:
    wfa_is_months: int = 18
    wfa_oos_months: int = 6
    wfa_max_points: int = 30
    retire_if_oos_sharpe_below: float = 0.0       # stitched walk-forward OOS Sharpe on refreshed data
    retire_if_last_windows_negative: int = 2      # ... or this many most-recent OOS windows all lost money
    retire_on_drift: bool = True
    propose_replacements: bool = True
    tier1_trials: int = 40
    gp_generations: int = 0                       # 0 = skip the genetic search
    gp_pop: int = 20
    gate: GateConfig = field(default_factory=GateConfig)
    corr_cap: float = 0.7
    method: str = "inverse_vol"
    drift_window: int = 30
    candidates_dir: Path | None = None            # where proposed specs are written (default strategies/candidates)


def review_strategy(entry: StrategyEntry, bars_m1: pd.DataFrame, cost: CostModel, ledger: Ledger | None, cfg: ReviewConfig) -> dict:
    spec = entry.load_spec()
    wf = walk_forward(spec, bars_m1, cost, cfg.wfa_is_months, cfg.wfa_oos_months, max_points=cfg.wfa_max_points)
    last = [w["oos_net"] for w in wf["windows"][-cfg.retire_if_last_windows_negative:]] if wf["n_windows"] else []
    live_r = []
    if ledger is not None:
        t = ledger.trades(closed_only=True, strategy=entry.name)
        live_r = [float(x) for x in t["r_multiple"].dropna()]
    drift = drift_status(live_r, entry.drift)
    override = ledger.get(f"mode:{entry.name}") if ledger is not None else None
    reasons = []
    if wf["n_windows"] == 0:
        reasons.append("not enough data for a walk-forward")
    else:
        if not np.isfinite(wf["oos_sharpe"]) or wf["oos_sharpe"] <= cfg.retire_if_oos_sharpe_below:
            reasons.append(f"refreshed walk-forward OOS Sharpe {wf['oos_sharpe']:.2f} <= {cfg.retire_if_oos_sharpe_below}")
        if cfg.retire_if_last_windows_negative and len(last) >= cfg.retire_if_last_windows_negative and all((x or 0) <= 0 for x in last):
            reasons.append(f"last {len(last)} out-of-sample windows all lost money")
    if cfg.retire_on_drift and (drift["state"] == "breach" or (override or {}).get("by") == "drift_monitor"):
        reasons.append("live drift breach: rolling expectancy below the backtest 5th percentile")
    return {"name": entry.name, "decision": "retire" if reasons else "keep", "reasons": reasons, "drift": drift, "live_trades": len(live_r),
            "wf": {"windows": wf["n_windows"], "oos_sharpe": wf.get("oos_sharpe"), "wfe": wf.get("wfe"), "oos_windows_positive": wf.get("oos_windows_positive"), "last_oos_net": last}}


def propose_replacements(retired: list[StrategyEntry], bars_m1: pd.DataFrame, cost: CostModel, registry: Registry, cfg: ReviewConfig, verbose: bool = True) -> list[dict]:
    """Re-fit each retired template (Tier 1) and optionally run the GP (Tier 2); gate every proposal."""
    from ..search.parametric import search_template
    out, stamp = [], dt.date.today().strftime("%Y%m")
    cand_dir = Path(cfg.candidates_dir) if cfg.candidates_dir else ROOT / "strategies" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    proposals = []
    for e in retired:
        spec = e.load_spec()
        if not spec.params_space:
            continue
        r = search_template(spec, bars_m1, cost, cfg.tier1_trials, registry, verbose=verbose)
        new = spec.with_params(**r["best_params"])
        new.name = f"{spec.name.split('_r20')[0]}_r{stamp}"
        if new.with_params().params != spec.params:
            proposals.append((new, f"tier1 re-fit of {e.name}"))
    if cfg.gp_generations > 0:
        from ..dsl.spec import StrategySpec
        from ..search.genetic import evolve
        g = evolve(bars_m1, cost, cfg.gp_generations, cfg.gp_pop, registry=registry, verbose=verbose)
        import yaml
        for i, el in enumerate(g["elites"][:3]):
            s = StrategySpec.from_dict(yaml.safe_load(el["spec_yaml"]))
            s.name = f"gp_r{stamp}_{i}"
            proposals.append((s, "tier2 genetic search"))
    for spec, origin in proposals:
        rep = run_gate(spec, bars_m1, cost, cfg.gate, n_trials_so_far=registry.n_trials() + 1, save=cfg.candidates_dir is None, verbose=verbose)
        registry.log_trial("review", spec, rep.get("in_sample", {}), bars_m1.index[0], bars_m1.index[-1], cost.version, oos_sharpe=rep.get("walk_forward", {}).get("oos_sharpe"), passed=rep.get("passed"))
        path = cand_dir / f"{spec.name}.yaml"
        path.write_text(spec.to_yaml())
        out.append({"name": spec.name, "origin": origin, "passed": bool(rep.get("passed")), "reasons": rep.get("reasons", []), "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "spec": spec})
    return out


def monthly_review(pf: Portfolio, bars_m1: pd.DataFrame, ledger: Ledger | None = None, registry: Registry | None = None, cost: CostModel | None = None,
                   cfg: ReviewConfig | None = None, verbose: bool = True) -> tuple[Portfolio, dict]:
    cost, cfg = cost or CostModel(), cfg or ReviewConfig()
    registry = registry or Registry()
    new = copy.deepcopy(pf)
    rep: dict = {"date": str(dt.date.today()), "data": {"start": str(bars_m1.index[0]), "end": str(bars_m1.index[-1])}, "reviews": [], "replacements": [], "changes": []}
    retired = []
    for e in new.strategies:
        if e.status != "active":
            continue
        r = review_strategy(e, bars_m1, cost, ledger, cfg)
        rep["reviews"].append(r)
        if verbose:
            print(f"[review] {e.name}: {r['decision']}" + (f" ({'; '.join(r['reasons'])})" if r["reasons"] else ""), flush=True)
        if r["decision"] == "retire":
            e.status, e.mode, e.weight = "retired", "off", 0.0
            e.notes = f"retired {rep['date']}: {'; '.join(r['reasons'])}"
            retired.append(e)
            rep["changes"].append(f"retire {e.name}: {'; '.join(r['reasons'])}")
    if cfg.propose_replacements and retired:
        for c in propose_replacements(retired, bars_m1, cost, registry, cfg, verbose):
            spec = c.pop("spec")
            rep["replacements"].append(c)
            if c["passed"] and new.get(c["name"]) is None:
                res = backtest_spec(spec, bars_m1, cost, cache_key=None)
                m = summarize(res.equity, res.trades)
                new.strategies.append(StrategyEntry(name=c["name"], spec_path=c["path"], spec_hash=spec.hash(), weight=0.0, mode="off", status="candidate", validated=True,
                                                    drift=drift_reference(res.trades, cfg.drift_window), reference={"trades": m["trades"], "sharpe": round(m["sharpe"], 3)},
                                                    notes=f"proposed {rep['date']} ({c['origin']}); passed the gate, needs a demo forward test"))
                rep["changes"].append(f"add candidate {c['name']} ({c['origin']})")
    # re-allocate what is still active, and refresh each survivor's drift reference on the new data
    daily, scores = {}, {}
    for e in new.active():
        spec = e.load_spec()
        res = backtest_spec(spec, bars_m1, cost, cache_key=None)
        daily[e.name] = daily_pnl_returns(res.equity, spec.sizing.equity)
        scores[e.name] = summarize(res.equity, res.trades)["sharpe"]
        e.drift = drift_reference(res.trades, cfg.drift_window) or e.drift
    alloc = allocate(daily, scores, cfg.method, cfg.corr_cap, new.target_annual_vol)
    for e in new.active():
        w = alloc["weights"].get(e.name)
        if w is None:
            rep["changes"].append(f"{e.name}: weight {e.weight:.2f} -> 0.00 (dropped by the correlation cap)")
            e.weight = 0.0
        elif abs(w - e.weight) > 1e-9:
            rep["changes"].append(f"{e.name}: weight {e.weight:.2f} -> {w:.2f}")
            e.weight = w
    rep["portfolio"] = alloc.get("portfolio", {})
    rep["registry_trials"] = registry.n_trials()
    new.log("monthly_review", "; ".join(rep["changes"]) or "no changes")
    return new, rep


def render_markdown(rep: dict) -> str:
    L = [f"# Monthly review {rep['date']}", "", f"Data {rep['data']['start'][:10]} to {rep['data']['end'][:10]}. Registry now holds {rep.get('registry_trials', 0)} trials.", "",
         "| strategy | decision | WF windows | OOS Sharpe | OOS windows positive | live trades | drift |", "|---|---|---|---|---|---|---|"]
    for r in rep["reviews"]:
        w = r["wf"]
        sh = "n/a" if w["oos_sharpe"] is None or not np.isfinite(w["oos_sharpe"]) else f"{w['oos_sharpe']:.2f}"
        pos = "n/a" if w.get("oos_windows_positive") is None else f"{w['oos_windows_positive']:.0%}"
        L.append(f"| {r['name']} | **{r['decision']}** | {w['windows']} | {sh} | {pos} | {r['live_trades']} | {r['drift']['state']} |")
    L.append("")
    for r in rep["reviews"]:
        if r["reasons"]:
            L += [f"{r['name']}:"] + [f"- {x}" for x in r["reasons"]] + [""]
    if rep["replacements"]:
        L += ["## Proposed replacements", "", "| name | origin | gate | why not |", "|---|---|---|---|"]
        L += [f"| {c['name']} | {c['origin']} | {'PASS' if c['passed'] else 'fail'} | {'; '.join(c['reasons'])[:160]} |" for c in rep["replacements"]] + [""]
    L += ["## Changes", ""] + ([f"- {c}" for c in rep["changes"]] or ["- none"]) + [""]
    p = rep.get("portfolio") or {}
    if p:
        L += [f"Re-allocated book (backtest): Sharpe {p['sharpe']:.2f}, annual vol {p['annual_vol']:.1%}, max drawdown {p['max_drawdown']:.1%}.", ""]
    return "\n".join(L)
