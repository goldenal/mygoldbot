"""Forward-test reconciliation (PLAN Phase 4 exit criteria): compare what was traded with what the
screening engine says should have been traded on the very bars the live system saw.

For every strategy the spec is backtested over the bars stored in the ledger (the shadow), live and
shadow trades are paired by direction and entry time, and per pair we measure entry slippage, exit
slippage and P&L per lot. Shadow trades with no live partner are explained from the intent log
(vetoed / expired / blocked by risk / strategy off), which also prices the human override: the sum
of the shadow P&L of vetoed trades is what discretion cost or saved.

Caveat: once a signal is skipped the two paths can diverge (live is flat where the shadow holds a
position), so a few unpaired trades after a veto are expected and are reported, not hidden.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import REPORT_DIR
from ..dsl.compile import TF_MINUTES, backtest_spec
from ..engine.costs import CostModel
from .ledger import Ledger

SKIP_STATUSES = ("vetoed", "expired", "blocked", "off", "stale", "too_small", "failed", "pending", "approved")


@dataclass
class ForwardConfig:
    min_weeks: float = 8.0                 # PLAN: >= 8 weeks demo, then >= 4 weeks tiny live
    min_trades: int = 30
    match_tolerance_bars: float = 1.0      # entry-time tolerance, in bars of the strategy timeframe
    max_entry_slippage: float = 0.15       # $/oz, mean adverse difference live vs simulated entry
    max_exit_slippage: float = 0.25
    min_match_rate: float = 0.85           # paired / (paired + unexplained shadow trades)
    expectancy_sigmas: float = 2.0         # live per-lot expectancy may trail the shadow by at most this many standard errors
    max_errors: int = 0                    # order failures and position mismatches during the period


def shadow_trades(spec, bars_m1: pd.DataFrame, cost: CostModel, start: pd.Timestamp | None = None) -> pd.DataFrame:
    t = backtest_spec(spec, bars_m1, cost, cache_key=None).trades      # a trade still open at the end of data has reason "end"
    return t[t["entry_ts"] >= start].reset_index(drop=True) if start is not None and len(t) else t.reset_index(drop=True)


def match_trades(live: pd.DataFrame, sim: pd.DataFrame, tol: pd.Timedelta) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy nearest-entry pairing within one strategy. Returns (pairs of row positions, unmatched live, unmatched sim)."""
    pairs, used = [], set()
    for i in range(len(live)):
        lt, ld = live["entry_ts"].iloc[i], live["dir"].iloc[i]
        best, best_d = None, None
        for j in range(len(sim)):
            if j in used or sim["dir"].iloc[j] != ld:
                continue
            d = abs(sim["entry_ts"].iloc[j] - lt)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    paired_live = {i for i, _ in pairs}
    return pairs, [i for i in range(len(live)) if i not in paired_live], [j for j in range(len(sim)) if j not in used]


def forward_report(ledger: Ledger, specs: list, cost: CostModel | None = None, cfg: ForwardConfig | None = None, history: pd.DataFrame | None = None,
                   start: pd.Timestamp | None = None, stage: str | None = None, save: bool = True) -> dict:
    """specs: StrategySpec list. history: M1 bars from before the forward period, used only to warm indicators up."""
    cost, cfg = cost or CostModel(), cfg or ForwardConfig()
    bars = ledger.bars()
    reasons: list[str] = []
    rep: dict = {"stage": stage, "config": asdict(cfg), "strategies": {}}
    if bars.empty:
        rep.update(passed=False, reasons=["no bars recorded in the ledger"])
        return rep
    start = start or bars.index[0]
    bars = bars[bars.index >= start]
    full = pd.concat([history[history.index < bars.index[0]], bars]) if history is not None and len(history) else bars
    weeks = (bars.index[-1] - bars.index[0]).total_seconds() / (7 * 86400)
    rep["period"] = {"start": str(bars.index[0]), "end": str(bars.index[-1]), "weeks": round(weeks, 2), "m1_bars": int(len(bars))}
    live_all = ledger.trades(closed_only=True)
    live_all = live_all[live_all["entry_ts"] >= start]
    if stage:
        live_all = live_all[live_all["stage"] == stage]
    intents = ledger.intents_frame()
    rows, discretion = [], {}
    for spec in specs:
        tol = pd.Timedelta(minutes=TF_MINUTES[spec.timeframe] * cfg.match_tolerance_bars + 1)
        live = live_all[live_all["strategy"] == spec.name].reset_index(drop=True)
        sim = shadow_trades(spec, full, cost, start)
        pairs, extra, missing = match_trades(live, sim, tol)
        it = intents[(intents["strategy"] == spec.name) & intents["status"].isin(SKIP_STATUSES)]
        explained, unexplained = {}, 0
        for j in missing:
            s = sim.iloc[j]
            near = it[(it["dir"] == s["dir"]) & ((it["ts"] - s["entry_ts"]).abs() <= tol)]
            if len(near):
                st = near.iloc[0]["status"]
                e = explained.setdefault(st, {"n": 0, "shadow_pnl_per_lot": 0.0})
                e["n"] += 1
                e["shadow_pnl_per_lot"] += float(s["pnl"] / s["lots"]) if s["lots"] else 0.0
            else:
                unexplained += 1
        for st, e in explained.items():
            d = discretion.setdefault(st, {"n": 0, "shadow_pnl_per_lot": 0.0})
            d["n"] += e["n"]
            d["shadow_pnl_per_lot"] += e["shadow_pnl_per_lot"]
        for i, j in pairs:
            lv, sm = live.iloc[i], sim.iloc[j]
            row = {"strategy": spec.name, "dir": int(lv["dir"]), "entry_ts": lv["entry_ts"], "entry_slip": float((lv["entry_px"] - sm["entry_px"]) * lv["dir"])}
            if sm["reason"] != "end":      # the shadow's forced close at the end of the data is not an outcome to compare against
                row.update(exit_slip=float((sm["exit_px"] - lv["exit_px"]) * lv["dir"]), live_ppl=float(lv["pnl"] / lv["lots"]), sim_ppl=float(sm["pnl"] / sm["lots"]),
                           same_exit=bool(lv["reason"] == sm["reason"]))
            rows.append(row)
        denom = len(pairs) + unexplained
        rep["strategies"][spec.name] = {"live_trades": int(len(live)), "shadow_trades": int(len(sim)), "paired": len(pairs), "live_only": len(extra),
                                        "shadow_only_explained": explained, "shadow_only_unexplained": unexplained, "match_rate": float(len(pairs) / denom) if denom else None}
    P = pd.DataFrame(rows, columns=["strategy", "dir", "entry_ts", "entry_slip", "exit_slip", "live_ppl", "sim_ppl", "same_exit"])
    entry_slip_all = P["entry_slip"]
    P = P.dropna(subset=["live_ppl"])       # pairs with a complete outcome on both sides
    n = len(P)
    summ: dict = {"paired": n, "live_trades": int(len(live_all))}
    if n:
        diff = P["live_ppl"] - P["sim_ppl"]
        se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
        summ.update(entry_slip_mean=float(entry_slip_all.mean()), entry_slip_p95=float(entry_slip_all.quantile(0.95)), exit_slip_mean=float(P["exit_slip"].mean()),
                    live_expectancy_per_lot=float(P["live_ppl"].mean()), shadow_expectancy_per_lot=float(P["sim_ppl"].mean()), expectancy_diff=float(diff.mean()),
                    expectancy_diff_se=se, same_exit_reason=float(P["same_exit"].astype(float).mean()))
    tot_pair = sum(s["paired"] for s in rep["strategies"].values())
    tot_unexp = sum(s["shadow_only_unexplained"] for s in rep["strategies"].values())
    summ["match_rate"] = float(tot_pair / (tot_pair + tot_unexp)) if tot_pair + tot_unexp else None
    summ["live_only"] = sum(s["live_only"] for s in rep["strategies"].values())
    errs = [e for e in ledger.events(100000) if e["level"] in ("error", "critical") and e["kind"] in ("order_failed", "reconcile", "broker")
            and (e["ts"] is None or pd.Timestamp(e["ts"]) >= start)]
    summ["errors"] = len(errs)
    chain = ledger.verify_chain()
    summ["ledger_chain_ok"] = chain["ok"]
    rep["summary"], rep["discretion"] = summ, discretion

    if weeks < cfg.min_weeks:
        reasons.append(f"forward period {weeks:.1f} weeks < {cfg.min_weeks:g}")
    if n < cfg.min_trades:
        reasons.append(f"only {n} paired trades (< {cfg.min_trades})")
    if n:
        if summ["entry_slip_mean"] > cfg.max_entry_slippage:
            reasons.append(f"mean entry slippage ${summ['entry_slip_mean']:.3f}/oz > ${cfg.max_entry_slippage}")
        if summ["exit_slip_mean"] > cfg.max_exit_slippage:
            reasons.append(f"mean exit slippage ${summ['exit_slip_mean']:.3f}/oz > ${cfg.max_exit_slippage}")
        if np.isfinite(summ["expectancy_diff_se"]) and summ["expectancy_diff"] < -cfg.expectancy_sigmas * summ["expectancy_diff_se"] - 1e-9:
            reasons.append(f"live expectancy trails the shadow by ${-summ['expectancy_diff']:.2f}/lot ({-summ['expectancy_diff'] / max(summ['expectancy_diff_se'], 1e-12):.1f} standard errors)")
    if summ["match_rate"] is not None and summ["match_rate"] < cfg.min_match_rate:
        reasons.append(f"only {summ['match_rate']:.0%} of shadow trades were taken or explained")
    if summ["errors"] > cfg.max_errors:
        reasons.append(f"{summ['errors']} order / reconciliation errors during the period")
    if not chain["ok"]:
        reasons.append(f"ledger hash chain invalid: {chain.get('error')}")
    rep.update(passed=not reasons, reasons=reasons)
    if save:
        tag = f"forward_{stage or 'all'}_{bars.index[-1].date()}"
        (REPORT_DIR / f"{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
        (REPORT_DIR / f"{tag}.md").write_text(render_markdown(rep))
        rep["report_path"] = str(REPORT_DIR / f"{tag}.md")
    return rep


def render_markdown(r: dict) -> str:
    L = [f"# Forward-test reconciliation ({r.get('stage') or 'all stages'})", "", f"**Result: {'PASS' if r.get('passed') else 'FAIL'}**", ""]
    if r.get("reasons"):
        L += ["Reasons:"] + [f"- {x}" for x in r["reasons"]] + [""]
    if "period" not in r:
        return "\n".join(L)
    p, s = r["period"], r["summary"]
    L += [f"Period {p['start']} to {p['end']} ({p['weeks']} weeks, {p['m1_bars']:,} M1 bars). {s['live_trades']} live trades, {s['paired']} paired with the shadow backtest.", ""]
    if s["paired"]:
        L += ["| measure | value |", "|---|---|",
              f"| entry slippage, mean ($/oz, + = worse than simulated) | {s['entry_slip_mean']:.3f} |", f"| entry slippage, 95th pct | {s['entry_slip_p95']:.3f} |",
              f"| exit slippage, mean | {s['exit_slip_mean']:.3f} |", f"| live expectancy ($/lot) | {s['live_expectancy_per_lot']:.2f} |",
              f"| shadow expectancy ($/lot) | {s['shadow_expectancy_per_lot']:.2f} |", f"| difference (std error) | {s['expectancy_diff']:.2f} ({s['expectancy_diff_se']:.2f}) |",
              f"| same exit reason | {s['same_exit_reason']:.0%} |", f"| match rate | {s['match_rate'] if s['match_rate'] is None else format(s['match_rate'], '.0%')} |",
              f"| order / reconciliation errors | {s['errors']} |", f"| ledger hash chain | {'ok' if s['ledger_chain_ok'] else 'BROKEN'} |", ""]
    L += ["## Per strategy", "", "| strategy | live | shadow | paired | live only | shadow only, unexplained | match rate |", "|---|---|---|---|---|---|---|"]
    for name, v in r["strategies"].items():
        mr = "n/a" if v["match_rate"] is None else f"{v['match_rate']:.0%}"
        L.append(f"| {name} | {v['live_trades']} | {v['shadow_trades']} | {v['paired']} | {v['live_only']} | {v['shadow_only_unexplained']} | {mr} |")
    L.append("")
    if r.get("discretion"):
        L += ["## Signals not taken (what the shadow says they would have made)", "", "| reason | signals | shadow P&L, $/lot |", "|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['shadow_pnl_per_lot']:,.0f} |" for k, v in r["discretion"].items()]
        L += ["", "A negative total for `vetoed` means the vetoes saved money; a positive one means they cost it.", ""]
    return "\n".join(L)
