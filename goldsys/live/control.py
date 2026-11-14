"""Operator controls and the read-only state view, shared by the CLI (scripts/ops.py) and the dashboard.
This is the hybrid (human) layer: approve / veto queued signals, switch a strategy's mode, go
risk-off, pull the kill switch. Every action is written to the event log with who did it, so
discretion can be reviewed and measured later (live/reconcile.py prices the vetoes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .drift import drift_status, rolling_mean_r
from .ledger import Ledger

MODES = ("auto", "approve", "off")


def approve(ledger: Ledger, intent_id: int, by: str = "operator") -> bool:
    ok = ledger.set_intent(intent_id, "approved", by=by, only_if_status="pending")
    if ok:
        ledger.event("info", "override", f"intent #{intent_id} approved by {by}")
    return ok


def veto(ledger: Ledger, intent_id: int, by: str = "operator", reason: str = "") -> bool:
    ok = ledger.set_intent(intent_id, "vetoed", by=by, reasons=reason or None, only_if_status="pending")
    if ok:
        ledger.event("info", "override", f"intent #{intent_id} vetoed by {by}" + (f": {reason}" if reason else ""))
    return ok


def set_mode(ledger: Ledger, strategy: str, mode: str, by: str = "operator") -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    ledger.set(f"mode:{strategy}", {"mode": mode, "by": by, "ts": str(pd.Timestamp.now(tz="UTC"))})
    ledger.event("warning", "override", f"{strategy} mode set to '{mode}' by {by}")


def risk_off(ledger: Ledger, on: bool, by: str = "operator", reason: str = "") -> None:
    ledger.set("risk_off", {"active": bool(on), "by": by, "reason": reason, "ts": str(pd.Timestamp.now(tz="UTC"))})
    ledger.event("warning", "override", f"manual risk-off {'ON' if on else 'OFF'} by {by}" + (f": {reason}" if reason else ""))


def kill(ledger: Ledger, by: str = "operator", reason: str = "manual") -> None:
    ledger.set("kill", {"active": True, "reason": f"{reason} (by {by})", "ts": str(pd.Timestamp.now(tz="UTC"))})
    ledger.event("critical", "kill_switch", f"KILL SWITCH pulled by {by}: {reason}")


def reset_kill(ledger: Ledger, by: str = "operator") -> None:
    ledger.set("kill", {"active": False, "reset_by": by})
    ledger.event("warning", "kill_reset", f"kill switch reset by {by}")


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    if v is pd.NaT:
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def state(ledger: Ledger, pf=None, max_points: int = 400) -> dict:
    eq = ledger.equity_curve()
    rs = ledger.get("risk_state") or {}
    out: dict = {"kill": ledger.get("kill") or {"active": False}, "risk_off": ledger.get("risk_off") or {"active": False}, "risk_state": rs,
                 "stage": pf.stage if pf is not None else None, "scale": pf.scale if pf is not None else None}
    if len(eq):
        step = max(1, len(eq) // max_points)
        pts = eq.iloc[::step]
        if pts.index[-1] != eq.index[-1]:
            pts = pd.concat([pts, eq.iloc[[-1]]])
        last = eq.iloc[-1]
        out["equity_curve"] = [[t.isoformat(), float(v)] for t, v in zip(pts.index, pts["equity"])]
        out["account"] = {"ts": eq.index[-1].isoformat(), "equity": float(last["equity"]), "balance": float(last["balance"]), "open_lots": float(last["open_lots"]),
                          "day_pnl": float(last["equity"] - rs["day_start_equity"]) if rs.get("day_start_equity") else None,
                          "week_pnl": float(last["equity"] - rs["week_start_equity"]) if rs.get("week_start_equity") else None,
                          "drawdown": float(last["equity"] / rs["hwm"] - 1) if rs.get("hwm") else None, "start_equity": float(eq["equity"].iloc[0])}
    else:
        out["equity_curve"], out["account"] = [], None
    closed = ledger.trades(closed_only=True)
    names = [s.name for s in pf.strategies if s.status != "retired"] if pf is not None else sorted(set(ledger.trades()["strategy"]))
    strategies = []
    for n in names:
        t = closed[closed["strategy"] == n]
        r = [float(x) for x in t["r_multiple"].dropna()]
        e = pf.get(n) if pf is not None else None
        ov = ledger.get(f"mode:{n}")
        ref = e.drift if e is not None else {}
        strategies.append({"name": n, "mode": ov["mode"] if ov else (e.mode if e else "auto"), "mode_by": ov.get("by") if ov else None, "mode_reason": ov.get("reason") if ov else None,
                           "weight": e.weight if e else None, "status": e.status if e else None, "validated": e.validated if e else None,
                           "trades": int(len(t)), "pnl": float(t["pnl"].sum()) if len(t) else 0.0, "win_rate": float((t["pnl"] > 0).mean()) if len(t) else None,
                           "mean_r": float(np.mean(r)) if r else None, "drift": drift_status(r, ref), "rolling_r": rolling_mean_r(r, ref["window"]) if ref.get("window") else []})
    out["strategies"] = strategies
    out["positions"] = ledger.open_trades()
    out["pending"] = ledger.intents("pending")
    out["intents"] = ledger.intents(n=25)
    tail = closed.sort_values("exit_ts").tail(25).iloc[::-1]
    out["trades"] = [{k: r[k] for k in ("strategy", "dir", "lots", "entry_ts", "entry_px", "exit_ts", "exit_px", "reason", "pnl", "r_multiple", "stage")} for r in tail.to_dict("records")]
    out["events"] = [{k: e[k] for k in ("wall", "ts", "level", "kind", "message")} for e in ledger.events(40)]
    out["totals"] = {"trades": int(len(closed)), "pnl": float(closed["pnl"].sum()) if len(closed) else 0.0, "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else None}
    return _clean(out)
