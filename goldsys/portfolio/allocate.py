"""Portfolio step (PLAN 5.7 output / 5.8): turn the surviving strategies into weights.

1. correlation cap   walk the strategies best-first and drop any whose daily returns correlate with
                     an already-kept one above the cap (two clones are one bet with double the size)
2. risk weights      inverse volatility, or equal risk contribution ("risk parity")
3. vol target        scale all weights so the combined book runs at the target annual volatility,
                     subject to a per-strategy and a gross cap

A weight multiplies the strategy's own lot sizing, so 1.0 means "trade it exactly as backtested"."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_matrix(daily: dict[str, pd.Series]) -> pd.DataFrame:
    """Align per-strategy daily returns on a common calendar (days without a value = flat)."""
    if not daily:
        return pd.DataFrame()
    return pd.concat(daily, axis=1).sort_index().fillna(0.0)


def correlation_filter(R: pd.DataFrame, scores: dict[str, float], cap: float = 0.7) -> tuple[list[str], list[dict]]:
    corr = R.corr().fillna(0.0)
    kept, dropped = [], []
    for name in sorted(R.columns, key=lambda n: scores.get(n, 0.0), reverse=True):
        clash = [k for k in kept if corr.loc[name, k] > cap]
        if clash:
            dropped.append({"name": name, "correlated_with": clash[0], "corr": float(corr.loc[name, clash[0]])})
        else:
            kept.append(name)
    return kept, dropped


def inverse_vol_weights(R: pd.DataFrame) -> pd.Series:
    vol = R.std(ddof=1).replace(0, np.nan)
    w = (1.0 / vol).fillna(0.0)
    return w / w.sum() if w.sum() > 0 else w


def risk_parity_weights(R: pd.DataFrame, iters: int = 500) -> pd.Series:
    """Equal risk contribution by fixed-point iteration. Falls back to inverse vol if it does not settle."""
    cov = R.cov().to_numpy()
    n = len(cov)
    w = inverse_vol_weights(R).to_numpy()
    if n < 2 or not np.isfinite(w).all() or w.sum() == 0:
        return pd.Series(w, index=R.columns)
    for _ in range(iters):
        rc = w * (cov @ w)
        if (rc <= 0).any():
            return inverse_vol_weights(R)
        new = w * np.sqrt(rc.mean() / rc)
        new /= new.sum()
        if np.abs(new - w).max() < 1e-10:
            w = new
            break
        w = new
    return pd.Series(w, index=R.columns)


def allocate(daily: dict[str, pd.Series], scores: dict[str, float], method: str = "inverse_vol", corr_cap: float = 0.7,
             target_annual_vol: float = 0.10, max_weight: float = 2.0, max_gross: float = 4.0) -> dict:
    R_all = returns_matrix(daily)
    if R_all.empty:
        return {"weights": {}, "dropped": [], "correlation": {}, "portfolio": {}}
    kept, dropped = correlation_filter(R_all, scores, corr_cap)
    R = R_all[kept]
    w = risk_parity_weights(R) if method == "risk_parity" else inverse_vol_weights(R)
    port = R.to_numpy() @ w.to_numpy()
    vol = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(port) > 1 else 0.0
    k = target_annual_vol / vol if vol > 0 else 0.0
    k = min(k, max_gross / w.sum() if w.sum() > 0 else 0.0, (max_weight / w.max()) if w.max() > 0 else 0.0)
    w = w * k
    port = pd.Series(R.to_numpy() @ w.to_numpy(), index=R.index)
    eq = (1 + port).cumprod()
    pvol = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(port) > 1 else 0.0
    cov = R.cov().to_numpy()
    rc = w.to_numpy() * (cov @ w.to_numpy())
    return {
        "weights": {n: float(round(w[n], 4)) for n in kept},
        "dropped": dropped,
        "correlation": R_all.corr().round(3).fillna(0.0).to_dict(),
        "risk_contribution": {n: float(rc[i] / rc.sum()) if rc.sum() > 0 else 0.0 for i, n in enumerate(kept)},
        "portfolio": {
            "annual_vol": pvol,
            "annual_return": float(port.mean() * TRADING_DAYS),
            "sharpe": float(port.mean() / port.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(port) > 1 and port.std(ddof=1) > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0,
            "gross_weight": float(w.sum()),
            "vol_target_reached": bool(pvol >= target_annual_vol * 0.99),
            "days": int(len(port)),
        },
        "daily": port,
    }
