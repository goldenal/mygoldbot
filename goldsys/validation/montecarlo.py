"""Monte Carlo robustness: trade reshuffling, bootstrap, random trade skipping."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dd_of_path(pnl: np.ndarray, init: float) -> float:
    eq = init + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[init], eq]))[1:]
    return float((eq / peak - 1).min()) if len(eq) else 0.0


def monte_carlo(trades: pd.DataFrame, init_cash: float = 10_000.0, n: int = 2000, skip_frac: float = 0.10, seed: int = 0) -> dict:
    if len(trades) < 10:
        return {"n": 0}
    pnl = trades["pnl"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    dd_shuffle, dd_boot, net_boot, net_skip = [], [], [], []
    for _ in range(n):
        p = rng.permutation(pnl)
        dd_shuffle.append(_dd_of_path(p, init_cash))
        b = rng.choice(pnl, size=len(pnl), replace=True)
        dd_boot.append(_dd_of_path(b, init_cash))
        net_boot.append(b.sum())
        keep = rng.random(len(pnl)) > skip_frac
        net_skip.append(pnl[keep].sum())
    net_boot = np.array(net_boot)
    return {
        "n": n,
        "observed_net": float(pnl.sum()),
        "observed_dd": _dd_of_path(pnl, init_cash),
        "dd_shuffle_p50": float(np.percentile(dd_shuffle, 50)),
        "dd_shuffle_p95": float(np.percentile(dd_shuffle, 5)),       # 5th pct of a negative number = worst 5%
        "dd_boot_p95": float(np.percentile(dd_boot, 5)),
        "net_boot_p05": float(np.percentile(net_boot, 5)),
        "net_boot_p50": float(np.percentile(net_boot, 50)),
        "prob_net_negative": float((net_boot <= 0).mean()),
        "net_skip10_p05": float(np.percentile(net_skip, 5)),
        "ruin_prob_50pct": float((np.array(dd_boot) <= -0.5).mean()),
    }
