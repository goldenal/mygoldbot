"""Replay simulation core.

The clock of a replay is the M1 bar. A strategy still decides on ITS timeframe (a signal exists when one of its bars closes), but
once a trade is on, stops and targets are tested against every M1 bar, in the backtester's order of events:
  fill at the open of the first M1 bar after the signal's bar closed -> stop/target on each bar's range, stop first when both are touched
  -> signal / time / max-bars exits at the close of a strategy bar.
A trailing stop (strategy rule) ratchets from each bar's favourable extreme and applies from the NEXT bar. A trade can also be changed by hand
mid-way (`adjust`: move the stop, move or clear the target, close at the next open). Reversals are a forced close at the next open.

Every proposal, taken or skipped, is simulated by the same function, so a skipped signal's "shadow" outcome is exactly what taking it would
have produced. All of it is server-side: the page only ever receives what the clock has already reached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..chart import data as chartdata
from ..config import OZ_PER_LOT
from ..dsl.compile import TF_MINUTES, compile_signals, feature_frame
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel

CHUNK = 20000


@dataclass
class Market:
    symbol: str
    stamp: tuple
    ns: np.ndarray            # M1 bar open times, int64 ns (UTC)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray        # $/oz applied at each bar by the cost model
    slip: np.ndarray          # $/oz adverse slippage per fill at each bar (seeded, same formula as the backtester)
    cost: CostModel
    frame: pd.DataFrame       # the M1 bars (kept for aggregation)


_MARKETS: dict = {}
_CTX: dict = {}


def market(symbol: str, cost: CostModel | None = None, seed: int = 0) -> Market:
    cost = cost or CostModel()
    df = chartdata.m1(symbol)
    if df.empty:
        raise chartdata.ChartError(f"no bars for {symbol}")
    stamp = chartdata._M1[symbol][0]
    key = (symbol, stamp, cost.version, seed)
    if key not in _MARKETS:
        _MARKETS.clear()
        _CTX.clear()
        n = len(df)
        spread = cost.spread_at(df.index, df["spread"])
        rng_bar = (df["high"] - df["low"]).to_numpy()
        med = np.nanmedian(rng_bar)
        vol_extra = cost.slippage_vol_scale * np.clip(rng_bar / max(med, 1e-9) - 1.0, 0, None) * cost.slippage_mean
        slip = np.clip(np.random.default_rng(seed).normal(cost.slippage_mean, cost.slippage_std, n), 0, None) + vol_extra
        f = lambda c: np.ascontiguousarray(df[c].to_numpy(dtype=np.float64))  # noqa: E731
        _MARKETS[key] = Market(symbol, stamp, df.index.as_unit("ns").asi8, f("open"), f("high"), f("low"), f("close"), np.ascontiguousarray(spread, dtype=np.float64), slip, cost, df)
    return _MARKETS[key]


@dataclass
class StratCtx:
    name: str
    path: str
    tf: str
    tf_ns: int
    max_bars: int
    row: np.ndarray            # strategy-frame row of every M1 bar
    close_idx: np.ndarray      # M1 indices at which a strategy bar closes
    exit_long_idx: np.ndarray  # closes where a long should be exited (signal or time)
    exit_short_idx: np.ndarray
    tflag: np.ndarray          # per strategy row: time-exit flag
    cand: np.ndarray           # M1 indices at which a tradable entry signal exists
    cand_dir: np.ndarray
    cand_stop: np.ndarray
    cand_target: np.ndarray
    cand_lots: np.ndarray
    cand_trail: np.ndarray     # trailing distance at the signal bar (NaN = no trailing stop)
    cand_trail_act: np.ndarray # profit (in $/oz) the trade must show before the trail starts


def strategy_path(rel: str):
    return chartdata._strategy_path(rel)


def build_ctx(M: Market, rel: str) -> StratCtx:
    path = strategy_path(rel)
    key = (M.symbol, M.stamp, str(path), path.stat().st_mtime_ns)
    if key in _CTX:
        return _CTX[key]
    ctx = ctx_from_spec(M, StrategySpec.from_yaml(str(path)), rel)
    _CTX[key] = ctx
    return ctx


def ctx_from_spec(M: Market, spec: StrategySpec, rel: str) -> StratCtx:
    df = feature_frame(M.frame, spec.timeframe, None)
    sig = compile_signals(spec, df)
    tf_ns = TF_MINUTES[spec.timeframe] * 60 * 10**9
    frame_ns = df.index.as_unit("ns").asi8
    bucket = (M.ns // tf_ns) * tf_ns
    row = np.searchsorted(frame_ns, bucket).astype(np.int64)
    closes = np.empty(len(bucket), dtype=bool)
    closes[:-1] = bucket[1:] != bucket[:-1]
    closes[-1] = True
    close_idx = np.flatnonzero(closes)
    rows_c = row[close_idx]
    tflag = np.asarray(sig["time_exit_flag"], dtype=bool)
    xl = np.asarray(sig["exit_long"], dtype=bool)[rows_c] | tflag[rows_c]
    xs = np.asarray(sig["exit_short"], dtype=bool)[rows_c] | tflag[rows_c]
    sl, ss = np.asarray(sig["sig_long"], dtype=bool)[rows_c], np.asarray(sig["sig_short"], dtype=bool)[rows_c]
    lots, stop, tgt = sig["lots"][rows_c], sig["stop_dist"][rows_c], sig["target_dist"][rows_c]
    trail, tact = np.asarray(sig["trail_dist"], dtype=float)[rows_c], np.asarray(sig["trail_act"], dtype=float)[rows_c]
    nxt = tflag[np.minimum(rows_c + 1, len(tflag) - 1)]                 # the engine skips entries on a bar that is already past the time exit
    ok = (sl ^ ss) & (np.nan_to_num(lots, nan=0.0) > 0) & np.isfinite(stop) & ~nxt
    return StratCtx(spec.name, rel, spec.timeframe, tf_ns, int(spec.exit.max_bars or 10**9), row, close_idx, close_idx[xl], close_idx[xs], tflag,
                    close_idx[ok], np.where(sl[ok], 1, -1), stop[ok].astype(float), tgt[ok].astype(float), lots[ok].astype(float), trail[ok], tact[ok])


# ---------------------------------------------------------------- one trade

def _scan(M: Market, d: int, lo: int, hi: int, entry: float, stop: float, tgt: float, ext: float, trail_d: float | None, trail_a: float, trail_from: int):
    """First stop / target touch in bars [lo, hi] (inclusive), chunked, carrying the trailing extreme. The trail only applies from bar
    `trail_from` on (the bar after the fill: the engine ratchets at the END of a bar, so the fill bar still has the stop that was set).
    Returns (first_stop, first_tgt, stop_in_force_at_first_stop, ext_after).  Indices are absolute; n (= len) means 'none'."""
    n = len(M.ns)
    first_stop = first_tgt = n
    eff_at_stop = stop
    pos = lo
    while pos <= hi and first_stop == n and first_tgt == n:
        end = min(hi, pos + CHUNK - 1)
        sp = M.spread[pos:end + 1]
        if d > 0:
            H, L = M.high[pos:end + 1], M.low[pos:end + 1]
            if trail_d:
                run = np.maximum.accumulate(np.concatenate(([ext], H)))
                prev = run[:-1]
                eff = np.maximum(stop, np.where(prev - entry >= trail_a, prev - trail_d, -np.inf))
                ext_next = float(run[-1])
                if pos < trail_from:
                    eff[:trail_from - pos] = stop
            else:
                eff, ext_next = stop, ext
            hs, ht = L <= eff, H >= tgt
        else:
            AH, AL = M.high[pos:end + 1] + sp, M.low[pos:end + 1] + sp
            if trail_d:
                run = np.minimum.accumulate(np.concatenate(([ext], AL)))
                prev = run[:-1]
                eff = np.minimum(stop, np.where(entry - prev >= trail_a, prev + trail_d, np.inf))
                ext_next = float(run[-1])
                if pos < trail_from:
                    eff[:trail_from - pos] = stop
            else:
                eff, ext_next = stop, ext
            hs, ht = AH >= eff, AL <= tgt
        ext = ext_next
        if hs.any():
            k = int(np.argmax(hs))
            first_stop = pos + k
            eff_at_stop = float(eff[k]) if np.ndim(eff) else float(eff)
        if ht.any():
            first_tgt = pos + int(np.argmax(ht))
        pos = end + 1
    return first_stop, first_tgt, eff_at_stop, ext


def _exit_px(M: Market, d: int, x: int, kind: str, level: float = 0.0) -> float:
    o, c, sp, sl = M.open[x], M.close[x], M.spread[x], M.slip[x]
    if kind == "stop":
        return (min(level, o) - sl) if d > 0 else (max(level, o + sp) + sl)
    if kind == "target":
        return (level if o < level else o) if d > 0 else (level if o + sp > level else o + sp)
    if kind == "open":                                    # closed by hand / by a reversal at the next open
        return (o - sl) if d > 0 else (o + sp + sl)
    return (c - sl) if d > 0 else (c + sp + sl)           # at the close of a bar


def simulate(M: Market, C: StratCtx, fill_i: int, direction: int, lots: float, *, stop_dist: float | None = None, target_dist: float | None = None,
             stop_px: float | None = None, tgt_px: float | None = None, trail_dist: float | None = None, trail_act: float = 0.0,
             adjust: Sequence[dict] = (), until_i: int | None = None) -> dict | None:
    """The trade filled at the open of M1 bar `fill_i`.

    Levels: give `stop_dist` / `target_dist` (distances from the actual fill, like the backtester) or absolute `stop_px` / `tgt_px`; no target = None.
    `adjust`: manual changes, each {"i": bar at which it takes effect, "stop_px"?, "tgt_px"?, "clear_target"?, "close"?, "reason"?}.
       A new stop switches the trailing stop off (you are managing it now).
    `until_i`: do not find the exit; return the levels that will apply to bar until_i+1 (or the exit, if the trade is already over)."""
    n = len(M.ns)
    f = int(fill_i)
    if f >= n or f < 1:
        return None
    d = int(direction)
    entry = M.open[f] + M.spread[f] + M.slip[f] if d > 0 else M.open[f] - M.slip[f]
    stop = float(stop_px) if stop_px is not None else (entry - d * float(stop_dist))
    if tgt_px is not None:
        tgt = float(tgt_px)
    elif target_dist is not None and np.isfinite(target_dist):
        tgt = entry + d * float(target_dist)
    else:
        tgt = np.inf if d > 0 else -np.inf
    stop0, tgt0 = stop, tgt
    trail_on = bool(trail_dist) and np.isfinite(trail_dist) and trail_dist > 0
    trail_d = float(trail_dist) if trail_on else None
    erow = C.row[f]
    xs = C.exit_long_idx if d > 0 else C.exit_short_idx
    p = np.searchsorted(xs, f)
    sx = int(xs[p]) if p < len(xs) else n
    maxi = n
    if C.max_bars < 10**8:
        k = int(np.searchsorted(C.row, erow + C.max_bars, "left"))
        q = int(np.searchsorted(C.close_idx, k, "left"))
        maxi = int(C.close_idx[q]) if q < len(C.close_idx) else n
    hard = min(sx, maxi, n - 1)                           # the strategy's own exit (signal / time / max bars), at that bar's close
    adj = sorted((a for a in adjust if a["i"] > f), key=lambda a: a["i"])
    ext, seg, k = entry, f, 0
    while True:
        a = adj[k] if k < len(adj) else None
        seg_end = min(hard if until_i is None else min(hard, until_i), (a["i"] - 1) if a else n - 1)
        fs, ft, eff, ext = _scan(M, d, seg, seg_end, entry, stop, tgt, ext, trail_d if trail_on else None, trail_act, f + 1)
        x = min(fs, ft)
        if x <= seg_end:
            kind = "stop" if fs == x else "target"
            raised = trail_on and (eff > stop + 1e-9 if d > 0 else eff < stop - 1e-9)       # the stop in force is the trailed one, not the set one
            reason = "trail" if (kind == "stop" and raised) else kind
            exit_px = _exit_px(M, d, x, kind, eff if kind == "stop" else tgt)
            break
        if until_i is not None and seg_end >= until_i and hard > until_i:      # still open at until_i: report what applies next
            if trail_on:
                eff_next = max(stop, ext - trail_d) if d > 0 and ext - entry >= trail_act else (min(stop, ext + trail_d) if d < 0 and entry - ext >= trail_act else stop)
            else:
                eff_next = stop
            closing = False
            while a is not None and a["i"] == until_i + 1:                  # a change that takes effect at the very next bar is already "in force"
                if a.get("close"):
                    closing = True
                else:
                    if a.get("stop_px") is not None:
                        eff_next, trail_on = float(a["stop_px"]), False
                    if a.get("tgt_px") is not None:
                        tgt = float(a["tgt_px"])
                    if a.get("clear_target"):
                        tgt = np.inf if d > 0 else -np.inf
                k += 1
                a = adj[k] if k < len(adj) else None
            return {"open": True, "closing": closing, "stop_px": float(eff_next), "tgt_px": float(tgt) if np.isfinite(tgt) else None}
        if a is not None and seg_end == a["i"] - 1 and seg_end < hard:       # the strategy's own exit at a bar's close comes before a change at the next open
            if a.get("close"):
                x, exit_px, reason = a["i"], _exit_px(M, d, a["i"], "open"), a.get("reason", "manual")
                break
            if a.get("stop_px") is not None:
                stop, trail_on = float(a["stop_px"]), False
            if a.get("tgt_px") is not None:
                tgt = float(a["tgt_px"])
            if a.get("clear_target"):
                tgt = np.inf if d > 0 else -np.inf
            seg, k = a["i"], k + 1
            continue
        # reached the strategy's own exit (or the end of the data) without a touch
        x = hard
        reason = "end" if (x == n - 1 and sx >= n and maxi >= n) else ("time" if (x == sx and C.tflag[C.row[x]]) else "signal")
        exit_px = _exit_px(M, d, x, "close")
        break
    cost = M.cost
    ts = pd.Timestamp(int(M.ns[f]), unit="ns", tz="UTC"), pd.Timestamp(int(M.ns[x]), unit="ns", tz="UTC")
    pnl = (exit_px - entry) * d * lots * OZ_PER_LOT - cost.commission_per_lot_rt * lots + cost.swap_cost(d, lots, ts[0], ts[1])
    risk = abs(entry - stop0) * lots * OZ_PER_LOT
    return {"open": False, "fill_i": f, "exit_i": int(x), "entry_px": float(entry), "exit_px": float(exit_px), "stop_px": float(stop0),
            "tgt_px": float(tgt0) if np.isfinite(tgt0) else None, "reason": reason, "pnl": float(pnl), "r": float(pnl / risk) if risk > 0 else None}


def simulate_signal(M: Market, C: StratCtx, j: int, direction: int, stop_dist: float, target_dist: float | None, lots: float,
                    trail_dist: float | None = None, trail_act: float = 0.0, **kw) -> dict | None:
    """Convenience: the trade a strategy signal at strategy-bar close `j` produces with the strategy's own levels (fill at j + 1)."""
    return simulate(M, C, j + 1, direction, lots, stop_dist=stop_dist, target_dist=target_dist, trail_dist=trail_dist, trail_act=trail_act, **kw)
