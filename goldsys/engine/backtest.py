"""Screening backtester: signals on the strategy timeframe, fills resolved on execution bars.

Conventions
- Signals are computed at bar CLOSE of the signal timeframe and acted on at the OPEN of the
  next execution bar (no look-ahead).
- Prices in bars are BID. Buys pay bid + spread; sells receive bid. Slippage is adverse.
- Inside a bar, if both stop and target are touched we assume the STOP was hit first (conservative).
- One position at a time per strategy (flat / long / short). Reversal = close then open.
- Position size in lots is given per signal bar by the sizing array.
The hot loop is numba-compiled; a pure-python fallback exists for debugging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import OZ_PER_LOT
from .costs import CostModel

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series          # $ equity per execution bar
    bar_returns: pd.Series     # simple returns per bar on equity
    params: dict
    cost_version: str

    @property
    def daily_returns(self) -> pd.Series:
        eq = self.equity.resample("1D").last().dropna()
        return eq.pct_change().dropna()


@njit(cache=True)
def _run(open_, high, low, close, spread, sig_long, sig_short, exit_long, exit_short,
         stop_dist, target_dist, lots, time_exit_flag, slip, commission_rt, init_cash, oz_per_lot,
         max_bars_in_trade):
    n = open_.shape[0]
    pos = 0            # -1, 0, 1
    entry_px = 0.0
    entry_i = -1
    stop_px = 0.0
    tgt_px = 0.0
    cur_lots = 0.0
    cash = init_cash
    equity = np.empty(n)
    # trade log buffers
    max_trades = n // 2 + 1
    t_entry_i = np.empty(max_trades, np.int64)
    t_exit_i = np.empty(max_trades, np.int64)
    t_dir = np.empty(max_trades, np.int64)
    t_entry_px = np.empty(max_trades)
    t_exit_px = np.empty(max_trades)
    t_lots = np.empty(max_trades)
    t_pnl = np.empty(max_trades)
    t_reason = np.empty(max_trades, np.int64)   # 0 stop,1 target,2 signal exit,3 time exit,4 reverse,5 end
    nt = 0

    for i in range(n):
        # ---- 1. manage open position on this bar using its high/low (stop first) ----
        if pos != 0:
            exit_px = -1.0
            reason = -1
            if pos == 1:
                if low[i] <= stop_px:
                    exit_px = min(stop_px, open_[i]) - slip[i]      # gap through stop fills at open
                    reason = 0
                elif high[i] >= tgt_px:
                    exit_px = tgt_px if open_[i] < tgt_px else open_[i]
                    reason = 1
            else:
                ask_high = high[i] + spread[i]
                ask_low = low[i] + spread[i]
                if ask_high >= stop_px:
                    exit_px = max(stop_px, open_[i] + spread[i]) + slip[i]
                    reason = 0
                elif ask_low <= tgt_px:
                    exit_px = tgt_px if open_[i] + spread[i] > tgt_px else open_[i] + spread[i]
                    reason = 1
            if reason < 0:
                # signal-based / time-based exits act at this bar's close -> next bar open; we approximate at close
                if pos == 1 and (exit_long[i] or time_exit_flag[i] or (i - entry_i) >= max_bars_in_trade):
                    exit_px = close[i] - slip[i]
                    reason = 3 if time_exit_flag[i] else 2
                elif pos == -1 and (exit_short[i] or time_exit_flag[i] or (i - entry_i) >= max_bars_in_trade):
                    exit_px = close[i] + spread[i] + slip[i]
                    reason = 3 if time_exit_flag[i] else 2
            if reason >= 0:
                pnl = (exit_px - entry_px) * pos * cur_lots * oz_per_lot - commission_rt * cur_lots
                cash += pnl
                t_entry_i[nt] = entry_i; t_exit_i[nt] = i; t_dir[nt] = pos
                t_entry_px[nt] = entry_px; t_exit_px[nt] = exit_px; t_lots[nt] = cur_lots
                t_pnl[nt] = pnl; t_reason[nt] = reason
                nt += 1
                pos = 0
        # ---- 2. new entries: signal from previous bar close, fill at this bar's open ----
        if i > 0:
            want = 0
            if sig_long[i - 1] and not sig_short[i - 1]:
                want = 1
            elif sig_short[i - 1] and not sig_long[i - 1]:
                want = -1
            if want != 0 and pos != want and lots[i - 1] > 0 and not time_exit_flag[i]:
                if pos != 0:
                    # reverse: close at open
                    if pos == 1:
                        exit_px = open_[i] - slip[i]
                    else:
                        exit_px = open_[i] + spread[i] + slip[i]
                    pnl = (exit_px - entry_px) * pos * cur_lots * oz_per_lot - commission_rt * cur_lots
                    cash += pnl
                    t_entry_i[nt] = entry_i; t_exit_i[nt] = i; t_dir[nt] = pos
                    t_entry_px[nt] = entry_px; t_exit_px[nt] = exit_px; t_lots[nt] = cur_lots
                    t_pnl[nt] = pnl; t_reason[nt] = 4
                    nt += 1
                    pos = 0
                pos = want
                cur_lots = lots[i - 1]
                entry_i = i
                if pos == 1:
                    entry_px = open_[i] + spread[i] + slip[i]
                    stop_px = entry_px - stop_dist[i - 1]
                    tgt_px = entry_px + target_dist[i - 1]
                else:
                    entry_px = open_[i] - slip[i]
                    stop_px = entry_px + stop_dist[i - 1]
                    tgt_px = entry_px - target_dist[i - 1]
        # ---- 3. mark to market ----
        if pos == 1:
            equity[i] = cash + (close[i] - entry_px) * cur_lots * oz_per_lot
        elif pos == -1:
            equity[i] = cash + (entry_px - (close[i] + spread[i])) * cur_lots * oz_per_lot
        else:
            equity[i] = cash
    # close any open position at the last close
    if pos != 0:
        i = n - 1
        exit_px = close[i] - slip[i] if pos == 1 else close[i] + spread[i] + slip[i]
        pnl = (exit_px - entry_px) * pos * cur_lots * oz_per_lot - commission_rt * cur_lots
        cash += pnl
        t_entry_i[nt] = entry_i; t_exit_i[nt] = i; t_dir[nt] = pos
        t_entry_px[nt] = entry_px; t_exit_px[nt] = exit_px; t_lots[nt] = cur_lots
        t_pnl[nt] = pnl; t_reason[nt] = 5
        nt += 1
        equity[i] = cash
    return (equity, t_entry_i[:nt], t_exit_i[:nt], t_dir[:nt], t_entry_px[:nt], t_exit_px[:nt], t_lots[:nt], t_pnl[:nt], t_reason[:nt])


REASONS = {0: "stop", 1: "target", 2: "signal", 3: "time", 4: "reverse", 5: "end"}


def run_backtest(
    bars: pd.DataFrame,
    sig_long: np.ndarray,
    sig_short: np.ndarray,
    exit_long: np.ndarray,
    exit_short: np.ndarray,
    stop_dist: np.ndarray,
    target_dist: np.ndarray,
    lots: np.ndarray,
    time_exit_flag: np.ndarray,
    cost: CostModel,
    init_cash: float = 10_000.0,
    max_bars_in_trade: int = 10**9,
    params: dict | None = None,
    seed: int = 0,
) -> BacktestResult:
    idx = bars.index
    n = len(bars)
    spread = cost.spread_at(idx, bars["spread"] if "spread" in bars else None)
    rng = np.random.default_rng(seed)
    rng_bar = (bars["high"] - bars["low"]).to_numpy()
    med = np.nanmedian(rng_bar) if n else 1.0
    vol_extra = cost.slippage_vol_scale * np.clip(rng_bar / max(med, 1e-9) - 1.0, 0, None) * cost.slippage_mean
    slip = np.clip(rng.normal(cost.slippage_mean, cost.slippage_std, n), 0, None) + vol_extra

    def b(x):
        return np.ascontiguousarray(np.asarray(x, dtype=np.bool_))

    def f(x):
        return np.ascontiguousarray(np.asarray(x, dtype=np.float64))

    out = _run(
        f(bars["open"]), f(bars["high"]), f(bars["low"]), f(bars["close"]), f(spread),
        b(sig_long), b(sig_short), b(exit_long), b(exit_short),
        f(np.nan_to_num(stop_dist, nan=np.inf)), f(np.nan_to_num(target_dist, nan=np.inf)),
        f(np.nan_to_num(lots, nan=0.0)), b(time_exit_flag), f(slip),
        float(cost.commission_per_lot_rt), float(init_cash), float(OZ_PER_LOT), int(max_bars_in_trade),
    )
    equity, ei, xi, d, epx, xpx, lt, pnl, reason = out
    trades = pd.DataFrame(
        {
            "entry_ts": idx[ei] if len(ei) else pd.DatetimeIndex([], tz="UTC"),
            "exit_ts": idx[xi] if len(xi) else pd.DatetimeIndex([], tz="UTC"),
            "dir": d,
            "entry_px": epx,
            "exit_px": xpx,
            "lots": lt,
            "pnl": pnl,
            "reason": [REASONS[int(r)] for r in reason],
        }
    )
    # swaps: applied post hoc per trade (cheap and keeps the hot loop simple)
    if len(trades):
        swaps = np.array([cost.swap_cost(int(r.dir), float(r.lots), r.entry_ts, r.exit_ts) for r in trades.itertuples()])
        trades["swap"] = swaps
        trades["pnl"] = trades["pnl"] + swaps
        # fold swaps into equity at exit bars
        eq = equity.copy()
        adj = np.zeros(n)
        np.add.at(adj, xi, swaps)
        eq = eq + np.cumsum(adj)
        equity = eq
        trades["bars_held"] = xi - ei
        trades["r_multiple"] = trades["pnl"] / (np.abs(np.nan_to_num(stop_dist[np.clip(ei - 1, 0, n - 1)], nan=np.inf)) * trades["lots"] * OZ_PER_LOT).replace(0, np.nan)
    eq_s = pd.Series(equity, index=idx, name="equity")
    rets = eq_s.pct_change().fillna(0.0)
    return BacktestResult(trades=trades, equity=eq_s, bar_returns=rets, params=params or {}, cost_version=cost.version)
