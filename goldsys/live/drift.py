"""Live-vs-backtest drift (PLAN 5.8): a strategy is switched off when its rolling N-trade expectancy
falls below the 5th percentile of what the backtest says N trades can look like by chance.

Everything is in R-multiples (trade P&L / initial risk), so the comparison does not depend on size."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drift_reference(trades: pd.DataFrame, window: int = 30, n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap the mean R of `window` trades drawn from the reference (backtest) trades."""
    r = trades["r_multiple"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float) if len(trades) and "r_multiple" in trades else np.array([])
    if len(r) < window:
        return {}
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(n_boot, window), replace=True).mean(axis=1)
    return {"window": int(window), "p05_r": float(np.percentile(means, 5)), "p50_r": float(np.percentile(means, 50)), "mean_r": float(r.mean()), "std_r": float(r.std(ddof=1)), "n_ref": int(len(r))}


def drift_status(live_r: list[float], ref: dict) -> dict:
    w = ref.get("window")
    if not w or "p05_r" not in ref:
        return {"state": "no_reference", "n": len(live_r)}
    if len(live_r) < w:
        return {"state": "warming_up", "n": len(live_r), "window": w, "mean_r": float(np.mean(live_r)) if live_r else None, "p05_r": ref["p05_r"]}
    m = float(np.mean(live_r[-w:]))
    return {"state": "breach" if m < ref["p05_r"] else "ok", "n": len(live_r), "window": w, "mean_r": m, "p05_r": ref["p05_r"], "ref_mean_r": ref.get("mean_r")}


def rolling_mean_r(live_r: list[float], window: int) -> list[float]:
    s = pd.Series(live_r, dtype=float)
    return s.rolling(window, min_periods=window).mean().dropna().tolist()
