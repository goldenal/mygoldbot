"""The validation gate: every candidate strategy goes through these steps in order.
Produces a dict report (JSON-serialisable) and a pass/fail with reasons."""
from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import REPORT_DIR
from ..dsl.compile import backtest_spec
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..engine.metrics import summarize
from .montecarlo import monte_carlo
from .sensitivity import cost_sensitivity, parameter_plateau
from .stats import deflated_sharpe_ratio, min_track_record_length, probability_of_backtest_overfitting
from .walkforward import cpcv, walk_forward, param_grid


@dataclass
class GateConfig:
    min_trades: int = 200
    max_single_trade_share: float = 0.10
    max_best_month_share: float = 0.25
    min_wfe: float = 0.5
    min_oos_windows_positive: float = 0.5
    min_plateau_ratio: float = 0.6
    max_pbo: float = 0.20
    min_dsr: float = 0.95
    max_mc_dd_p95: float = -0.30
    cost_mult_must_survive: float = 1.5
    wfa_is_months: int = 18
    wfa_oos_months: int = 6
    wfa_max_points: int = 40
    cpcv_groups: int = 6
    cpcv_test: int = 2
    cpcv_max_points: int = 25
    run_cpcv: bool = True
    run_plateau: bool = True
    pbo_trials: int = 30        # how many param combos to use for the PBO matrix


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items() if k not in ("oos_daily", "paths")}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o


def run_gate(spec: StrategySpec, bars_m1: pd.DataFrame, cost: CostModel | None = None, cfg: GateConfig | None = None, n_trials_so_far: int = 1, save: bool = True, verbose: bool = True) -> dict:
    cost = cost or CostModel()
    cfg = cfg or GateConfig()
    t_start = time.time()
    reasons: list[str] = []
    report: dict = {"strategy": spec.name, "hash": spec.hash(), "params": dict(spec.params), "cost_version": cost.version,
                    "data": {"start": str(bars_m1.index[0]), "end": str(bars_m1.index[-1]), "m1_bars": int(len(bars_m1))}, "complexity": spec.complexity()}

    def log(msg):
        if verbose:
            print(f"[gate:{spec.name}] {msg}", flush=True)

    # 1. sanity on the full in-sample run
    res = backtest_spec(spec, bars_m1, cost)
    m = summarize(res.equity, res.trades)
    report["in_sample"] = m
    log(f"1 sanity: trades={m['trades']} net={m['net_profit']:.0f} sharpe={m['sharpe']:.2f} mdd={m['max_drawdown']:.2%}")
    if m["trades"] < cfg.min_trades:
        reasons.append(f"too few trades ({m['trades']} < {cfg.min_trades})")
    if m["largest_trade_share"] > cfg.max_single_trade_share:
        reasons.append(f"single trade is {m['largest_trade_share']:.0%} of gross profit")
    if m.get("best_month_share", 0) > cfg.max_best_month_share:
        reasons.append(f"best month is {m['best_month_share']:.0%} of profit")
    if m["net_profit"] <= 0:
        reasons.append("not profitable in-sample after costs")
        report["passed"] = False
        report["reasons"] = reasons
        report["elapsed_s"] = time.time() - t_start
        return _finish(report, save)

    # 2. cost sensitivity
    cs = cost_sensitivity(spec, bars_m1, cost)
    report["cost_sensitivity"] = cs
    surv = [r for r in cs if r["spread_mult"] == cfg.cost_mult_must_survive]
    log("2 cost sensitivity: " + ", ".join(f"x{r['spread_mult']}: {r['net_profit']:.0f}" for r in cs))
    if surv and surv[0]["net_profit"] <= 0:
        reasons.append(f"edge disappears at spread x{cfg.cost_mult_must_survive}")

    # 3. parameter plateau
    if cfg.run_plateau and spec.params:
        pl = parameter_plateau(spec, bars_m1, cost)
        report["plateau"] = pl
        log(f"3 plateau: base={pl['base']:.2f} neighbours={pl['neighbour_mean']:.2f} ratio={pl['plateau_ratio']:.2f}")
        if np.isfinite(pl["plateau_ratio"]) and pl["plateau_ratio"] < cfg.min_plateau_ratio:
            reasons.append(f"parameter peak is isolated (plateau ratio {pl['plateau_ratio']:.2f})")

    # 4. walk-forward
    trial_log: list = []
    wf = walk_forward(spec, bars_m1, cost, cfg.wfa_is_months, cfg.wfa_oos_months, max_points=cfg.wfa_max_points, trial_log=trial_log)
    report["walk_forward"] = _to_jsonable(wf)
    log(f"4 walk-forward: windows={wf['n_windows']} WFE={wf['wfe']:.2f} oos_sharpe={wf['oos_sharpe']:.2f} oos_pos={wf.get('oos_windows_positive', float('nan')):.0%}")
    if wf["n_windows"] == 0:
        reasons.append("not enough data for walk-forward")
    else:
        if not (np.isfinite(wf["wfe"]) and wf["wfe"] >= cfg.min_wfe):
            reasons.append(f"walk-forward efficiency {wf['wfe']:.2f} < {cfg.min_wfe}")
        if wf["oos_windows_positive"] < cfg.min_oos_windows_positive:
            reasons.append(f"only {wf['oos_windows_positive']:.0%} of OOS windows profitable")
        if wf["oos_sharpe"] <= 0:
            reasons.append("stitched OOS Sharpe <= 0")

    # 5. CPCV
    if cfg.run_cpcv and wf["n_windows"] > 0:
        cv = cpcv(spec, bars_m1, cost, cfg.cpcv_groups, cfg.cpcv_test, max_points=cfg.cpcv_max_points)
        report["cpcv"] = _to_jsonable(cv)
        log(f"5 CPCV: splits={cv['n_splits']} oos_sharpe mean={cv['oos_sharpe_mean']:.2f} p05={cv['oos_sharpe_p05']:.2f} positive={cv['oos_sharpe_positive_frac']:.0%}")
        if cv["oos_sharpe_positive_frac"] < 0.5:
            reasons.append("CPCV: fewer than half of OOS paths have positive Sharpe")

    # 6. DSR and PBO
    oos_daily = wf.get("oos_daily", pd.Series(dtype=float))
    total_trials = n_trials_so_far + len(trial_log)
    if len(oos_daily) > 30:
        sr_var = float(np.var([t["metrics"]["sharpe"] / np.sqrt(252) for t in trial_log])) if len(trial_log) > 2 else None
        dsr = deflated_sharpe_ratio(oos_daily.to_numpy(), total_trials, sr_var)
        dsr["min_track_record_days"] = min_track_record_length(oos_daily.to_numpy(), dsr["expected_max_sharpe"])
        report["dsr"] = _to_jsonable(dsr)
        log(f"6 DSR: trials={total_trials} E[maxSR]={dsr['expected_max_sharpe']:.3f}/period DSR={dsr['dsr']:.2f}")
        if dsr["dsr"] < cfg.min_dsr:
            reasons.append(f"deflated Sharpe confidence {dsr['dsr']:.2f} < {cfg.min_dsr}")
    # PBO from a matrix of param-combo daily returns over the full sample
    if spec.params_space:
        combos = param_grid(spec.params_space, cfg.pbo_trials, seed=1)
        cols = []
        for p in combos:
            r = backtest_spec(spec.with_params(**p), bars_m1, cost)
            cols.append(r.equity.resample("1D").last().dropna().pct_change().fillna(0.0))
        M = pd.concat(cols, axis=1).fillna(0.0).to_numpy()
        pbo = probability_of_backtest_overfitting(M, n_groups=8)
        report["pbo"] = _to_jsonable(pbo)
        log(f"6 PBO: {pbo['pbo']:.2f} over {pbo['n_combinations']} combinations / {len(combos)} trials")
        if np.isfinite(pbo["pbo"]) and pbo["pbo"] > cfg.max_pbo:
            reasons.append(f"probability of backtest overfitting {pbo['pbo']:.2f} > {cfg.max_pbo}")

    # 7. Monte Carlo
    mc = monte_carlo(res.trades, init_cash=spec.sizing.equity)
    report["monte_carlo"] = _to_jsonable(mc)
    if mc.get("n"):
        log(f"7 Monte Carlo: dd p95={mc['dd_boot_p95']:.1%} P(net<0)={mc['prob_net_negative']:.2f}")
        if mc["dd_boot_p95"] < cfg.max_mc_dd_p95:
            reasons.append(f"95th-percentile drawdown {mc['dd_boot_p95']:.0%} exceeds budget")

    # 8. stress windows (worst day / worst week in sample, plus named windows if present)
    daily = res.equity.resample("1D").last().dropna().pct_change().dropna()
    stress = {"worst_day": float(daily.min()) if len(daily) else 0.0, "worst_week": float(res.equity.resample("1W").last().dropna().pct_change().min()) if len(daily) > 7 else 0.0}
    for name, (s, e) in {"covid_2020_03": ("2020-03-01", "2020-04-01"), "aug_2024_unwind": ("2024-08-01", "2024-08-10"), "apr_2025": ("2025-04-01", "2025-05-01")}.items():
        seg = res.equity[(res.equity.index >= pd.Timestamp(s, tz="UTC")) & (res.equity.index < pd.Timestamp(e, tz="UTC"))]
        if len(seg) > 10:
            stress[name] = float(seg.iloc[-1] / seg.iloc[0] - 1)
    report["stress"] = stress
    log(f"8 stress: {stress}")

    report["passed"] = len(reasons) == 0
    report["reasons"] = reasons
    report["n_trials_total"] = total_trials
    report["elapsed_s"] = time.time() - t_start
    log(("PASSED" if report["passed"] else "FAILED: " + "; ".join(reasons)) + f"  ({report['elapsed_s']:.0f}s)")
    return _finish(report, save)


def _finish(report: dict, save: bool) -> dict:
    report = _to_jsonable(report)
    if save:
        p = REPORT_DIR / f"{report['strategy']}_{report['hash']}.json"
        p.write_text(json.dumps(report, indent=2, default=str))
        (REPORT_DIR / f"{report['strategy']}_{report['hash']}.md").write_text(render_markdown(report))
        report["report_path"] = str(p)
    return report


def render_markdown(r: dict) -> str:
    L = [f"# Validation report: {r['strategy']}  (`{r['hash']}`)", "",
         f"**Result: {'PASS' if r.get('passed') else 'FAIL'}**", ""]
    if r.get("reasons"):
        L += ["Reasons:"] + [f"- {x}" for x in r["reasons"]] + [""]
    L += [f"Data: {r['data']['start']} to {r['data']['end']} ({r['data']['m1_bars']:,} M1 bars). Cost model `{r['cost_version']}`. Complexity {r['complexity']}.", ""]
    L += [f"Params: `{r['params']}`", ""]
    m = r.get("in_sample", {})
    if m:
        L += ["## In-sample", "", "| metric | value |", "|---|---|"]
        for k in ("trades", "net_profit", "cagr", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "profit_factor", "expectancy", "avg_bars_held", "largest_trade_share", "best_month_share"):
            v = m.get(k)
            if isinstance(v, float):
                v = f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}"
            L.append(f"| {k} | {v} |")
        L.append("")
    if r.get("cost_sensitivity"):
        L += ["## Cost sensitivity", "", "| spread x | net | sharpe | PF |", "|---|---|---|---|"]
        for c in r["cost_sensitivity"]:
            L.append(f"| {c['spread_mult']} | {c['net_profit']:,.0f} | {c['sharpe']:.2f} | {c['profit_factor']:.2f} |")
        L.append("")
    if r.get("plateau"):
        p = r["plateau"]
        L += ["## Parameter plateau", "", f"Base {p['base']:.2f}, neighbour mean {p['neighbour_mean']:.2f}, ratio {p['plateau_ratio']}", ""]
    wf = r.get("walk_forward", {})
    if wf.get("n_windows"):
        L += ["## Walk-forward", "", f"Windows {wf['n_windows']}, WFE {wf['wfe']}, stitched OOS Sharpe {wf['oos_sharpe']:.2f}, OOS windows positive {wf['oos_windows_positive']:.0%}, OOS total return {wf['oos_total_return']:.1%}", "",
              "| IS | OOS | params | IS Sharpe | OOS Sharpe | OOS trades | OOS net |", "|---|---|---|---|---|---|---|"]
        for w in wf["windows"]:
            L.append(f"| {w['is']} | {w['oos']} | {w['params']} | {w['is_sharpe']:.2f} | {w['oos_sharpe'] if w['oos_sharpe'] is None else round(w['oos_sharpe'],2)} | {w['oos_trades']} | {w['oos_net'] if w['oos_net'] is None else round(w['oos_net'])} |")
        L.append("")
    if r.get("cpcv"):
        c = r["cpcv"]
        L += ["## CPCV", "", f"{c['n_splits']} splits. OOS Sharpe mean {c['oos_sharpe_mean']:.2f}, std {c['oos_sharpe_std']:.2f}, 5th pct {c['oos_sharpe_p05']:.2f}, positive {c['oos_sharpe_positive_frac']:.0%}", ""]
    if r.get("dsr"):
        d = r["dsr"]
        L += ["## Deflated Sharpe", "", f"Trials {d['n_trials']}, expected max SR/period {d['expected_max_sharpe']:.3f}, observed {d['sharpe_per_period']:.3f}, DSR {d['dsr']:.2f}, min track record {d['min_track_record_days']} days", ""]
    if r.get("pbo"):
        L += ["## PBO", "", f"Probability of backtest overfitting {r['pbo']['pbo']:.2f} ({r['pbo']['n_combinations']} combinations)", ""]
    if r.get("monte_carlo", {}).get("n"):
        mc = r["monte_carlo"]
        L += ["## Monte Carlo", "", f"Observed DD {mc['observed_dd']:.1%}; bootstrap DD p95 {mc['dd_boot_p95']:.1%}; P(net<0) {mc['prob_net_negative']:.2f}; net with 10% trades skipped p05 {mc['net_skip10_p05']:,.0f}", ""]
    if r.get("stress"):
        L += ["## Stress", "", "| window | return |", "|---|---|"] + [f"| {k} | {v:.2%} |" for k, v in r["stress"].items()] + [""]
    return "\n".join(L)
