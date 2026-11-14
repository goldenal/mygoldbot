"""Phase 5 rules for going live and for size: promotion between stages needs a passing forward-test
reconciliation, and the live size multiplier only ever moves on REALISED results.

Nothing here changes portfolio.yaml by itself: the functions return a recommendation with reasons
and the operator applies it (scripts/portfolio.py), so every size change has a human signature."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..portfolio.config import STAGES, Portfolio
from .ledger import Ledger
from .reconcile import ForwardConfig

STAGE_FORWARD = {"demo": ForwardConfig(min_weeks=8.0), "tiny_live": ForwardConfig(min_weeks=4.0, min_trades=15)}
REAL_MONEY = ("tiny_live", "live")


@dataclass
class ScalingConfig:
    min_trades: int = 40             # realised trades at the current size before any step up
    step_up: float = 1.25
    step_down: float = 0.5
    max_scale: float = 1.0           # ceiling; raise it deliberately in portfolio decisions, not here
    tiny_live_scale: float = 0.1     # size multiplier on first going live
    drawdown_step_down_pct: float = 5.0
    min_mean_r: float = 0.0
    confidence_sigmas: float = 1.0   # mean R must clear zero by this many standard errors to step up


def next_stage(stage: str) -> str | None:
    i = STAGES.index(stage)
    return STAGES[i + 1] if i + 1 < len(STAGES) else None


def stage_gate(pf: Portfolio, forward: dict | None) -> dict:
    """Can the portfolio be promoted to the next stage? `forward` = reconcile.forward_report for the CURRENT stage."""
    nxt = next_stage(pf.stage)
    why = []
    if nxt is None:
        return {"eligible": False, "next": None, "reasons": ["already at the final stage"]}
    if not pf.active():
        why.append("no active strategies")
    unval = [s.name for s in pf.active() if not s.validated]
    if unval and nxt in REAL_MONEY:
        why.append(f"strategies without a passing validation gate: {', '.join(unval)}")
    if pf.stage in STAGE_FORWARD:
        if forward is None:
            why.append(f"no forward-test report for stage '{pf.stage}'")
        elif forward.get("stage") != pf.stage:
            why.append(f"forward-test report is for stage '{forward.get('stage')}', not '{pf.stage}'")
        elif not forward.get("passed"):
            why.append("forward-test reconciliation failed: " + "; ".join(forward.get("reasons", [])))
    return {"eligible": not why, "next": nxt, "reasons": why}


def recommend_scale(ledger: Ledger, pf: Portfolio, cfg: ScalingConfig | None = None) -> dict:
    cfg = cfg or ScalingConfig()
    cur = pf.scale
    if pf.stage not in REAL_MONEY:
        return {"current": cur, "recommended": cur, "action": "hold", "reasons": [f"stage '{pf.stage}' is not real money: size is not scaled"]}
    since = None
    for h in reversed(pf.history):
        if h["action"] in ("scale", "promote"):
            since = pd.Timestamp(h["ts"])
            break
    t = ledger.trades(closed_only=True)
    t = t[t["stage"].isin(REAL_MONEY)]
    if since is not None:
        t = t[t["entry_ts"] >= since]
    r = t["r_multiple"].dropna().to_numpy(dtype=float)
    eq = ledger.equity_curve()
    if since is not None and len(eq):
        eq = eq[eq.index >= since]
    dd = float((eq["equity"] / eq["equity"].cummax() - 1).min()) if len(eq) else 0.0
    stats = {"trades": int(len(r)), "mean_r": float(r.mean()) if len(r) else None, "drawdown": dd, "since": str(since) if since is not None else None}
    if dd <= -cfg.drawdown_step_down_pct / 100:
        return {"current": cur, "recommended": round(cur * cfg.step_down, 4), "action": "step_down", "reasons": [f"realised drawdown {dd:.1%} at this size exceeds {cfg.drawdown_step_down_pct:g}%"], **stats}
    if len(r) < cfg.min_trades:
        return {"current": cur, "recommended": cur, "action": "hold", "reasons": [f"{len(r)} realised trades at this size (< {cfg.min_trades})"], **stats}
    se = float(r.std(ddof=1) / np.sqrt(len(r)))
    if r.mean() <= cfg.min_mean_r or r.mean() - cfg.confidence_sigmas * se <= 0:
        return {"current": cur, "recommended": cur, "action": "hold", "reasons": [f"realised expectancy {r.mean():.3f}R (std error {se:.3f}) is not clearly positive"], **stats}
    new = min(round(cur * cfg.step_up, 4), cfg.max_scale)
    if new <= cur:
        return {"current": cur, "recommended": cur, "action": "hold", "reasons": [f"already at the scale ceiling {cfg.max_scale:g}"], **stats}
    return {"current": cur, "recommended": new, "action": "step_up", "reasons": [f"{len(r)} realised trades, expectancy {r.mean():.3f}R (std error {se:.3f}), drawdown {dd:.1%}"], **stats}
