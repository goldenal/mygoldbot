"""Track-record statement from the ledger: monthly table and headline statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ledger import Ledger


def statement(ledger: Ledger, stage: str | None = None) -> dict:
    t = ledger.trades(closed_only=True)
    if stage:
        t = t[t["stage"] == stage]
    chain = ledger.verify_chain()
    out: dict = {"stage": stage or "all", "chain": chain, "trades": int(len(t)), "months": []}
    if t.empty:
        return out
    eq = ledger.equity_curve()
    start_eq = float(eq["equity"].iloc[0]) if len(eq) else None
    wins, losses = t[t["pnl"] > 0]["pnl"].sum(), -t[t["pnl"] <= 0]["pnl"].sum()
    r = t["r_multiple"].dropna()
    out.update(first=str(t["entry_ts"].min()), last=str(t["exit_ts"].max()), net=float(t["pnl"].sum()), win_rate=float((t["pnl"] > 0).mean()),
               profit_factor=float(wins / losses) if losses > 0 else None, mean_r=float(r.mean()) if len(r) else None,
               commission=float(t["commission"].sum()), swap=float(t["swap"].sum()), start_equity=start_eq,
               max_drawdown=float((eq["equity"] / eq["equity"].cummax() - 1).min()) if len(eq) else None)
    monthly = t.set_index("exit_ts").groupby(pd.Grouper(freq="1ME"))
    cum = 0.0
    for month, g in monthly:
        if g.empty:
            continue
        pnl = float(g["pnl"].sum())
        base = (start_eq + cum) if start_eq else None
        cum += pnl
        out["months"].append({"month": month.strftime("%Y-%m"), "trades": int(len(g)), "win_rate": float((g["pnl"] > 0).mean()), "pnl": pnl,
                              "return": pnl / base if base else None, "cumulative": cum})
    out["months_positive"] = float(np.mean([m["pnl"] > 0 for m in out["months"]]))
    return out


def render_statement(s: dict) -> str:
    c = s["chain"]
    L = [f"# Trading statement ({s['stage']})", "", f"Ledger integrity: {'verified, ' + str(c['rows']) + ' chained records, head `' + c.get('head', '')[:16] + '`' if c['ok'] else 'FAILED: ' + c.get('error', '')}", ""]
    if not s["trades"]:
        return "\n".join(L + ["No closed trades."])
    pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
    dd = "n/a" if s["max_drawdown"] is None else f"{s['max_drawdown']:.2%}"
    mr = "n/a" if s["mean_r"] is None else f"{s['mean_r']:+.3f}R"
    L += [f"{s['first'][:10]} to {s['last'][:10]}: {s['trades']} trades, net {s['net']:+,.2f}, win rate {s['win_rate']:.1%}, profit factor {pf}, expectancy {mr}, "
          f"max drawdown {dd}, {s['months_positive']:.0%} of months positive. Costs paid: commission {s['commission']:,.2f}, swap {s['swap']:,.2f}.", "",
          "| month | trades | win rate | P&L | return | cumulative |", "|---|---|---|---|---|---|"]
    for m in s["months"]:
        ret = "n/a" if m["return"] is None else f"{m['return']:+.2%}"
        L.append(f"| {m['month']} | {m['trades']} | {m['win_rate']:.0%} | {m['pnl']:+,.2f} | {ret} | {m['cumulative']:+,.2f} |")
    return "\n".join(L + [""])
