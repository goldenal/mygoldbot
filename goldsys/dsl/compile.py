"""Compile a StrategySpec + bars into backtester inputs and run it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OZ_PER_LOT
from ..data.store import PANDAS_FREQ, resample_bars
from ..engine.backtest import BacktestResult, run_backtest
from ..engine.costs import CostModel
from ..features.registry import base_features, ensure_feature
from .expr import eval_expr
from .spec import StrategySpec, mirror, substitute_params

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

_FEATURE_CACHE: dict = {}


def feature_frame(bars_m1: pd.DataFrame, timeframe: str, cache_key: str | None = None) -> pd.DataFrame:
    """Resampled bars with base features. Cached per (cache_key, timeframe) because the search
    engines call this thousands of times on the same data."""
    key = (cache_key, timeframe, len(bars_m1), str(bars_m1.index[0]) if len(bars_m1) else "")
    if cache_key is not None and key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key].copy()
    bars = resample_bars(bars_m1, timeframe)
    df = base_features(bars)
    if cache_key is not None:
        _FEATURE_CACHE[key] = df.copy()
    return df


def _time_flag(idx: pd.DatetimeIndex, hhmm: str | None, tf_minutes: int) -> np.ndarray:
    if not hhmm:
        return np.zeros(len(idx), dtype=bool)
    h, m = hhmm.split(":")
    t = int(h) * 60 + int(m)
    mod = idx.hour * 60 + idx.minute
    # flat from the exit time until the next UTC day start (22:00 rollover => treat >= t OR < 1:00 as flat? keep simple: >= t)
    return np.asarray(mod >= t)


def compile_signals(spec: StrategySpec, df: pd.DataFrame) -> dict:
    p = spec.params
    filt = eval_expr(substitute_params(spec.filters, p), df).astype(bool) if spec.filters else pd.Series(True, index=df.index)
    el = substitute_params(spec.entry_long, p)
    es = spec.entry_short
    if es == "mirror":
        es = mirror(el)
    else:
        es = substitute_params(es, p)
    sig_long = (eval_expr(el, df).astype(bool) & filt) if el is not None else pd.Series(False, index=df.index)
    sig_short = (eval_expr(es, df).astype(bool) & filt) if es is not None else pd.Series(False, index=df.index)

    xl = substitute_params(spec.exit.exit_long, p)
    xs = spec.exit.exit_short
    xs = mirror(xl) if xs == "mirror" else substitute_params(xs, p)
    exit_long = eval_expr(xl, df).astype(bool) if xl is not None else pd.Series(False, index=df.index)
    exit_short = eval_expr(xs, df).astype(bool) if xs is not None else pd.Series(False, index=df.index)

    # stop / target distances in $ per oz
    st = substitute_params(spec.exit.stop, p)
    if "atr_mult" in st:
        atr = ensure_feature(df, f"atr_{int(st.get('atr_n', 14))}")
        stop_dist = atr * float(st["atr_mult"])
    elif "fixed" in st:
        stop_dist = pd.Series(float(st["fixed"]), index=df.index)
    elif "expr" in st:
        stop_dist = eval_expr(st["expr"], df)
    else:
        raise ValueError("stop must have atr_mult, fixed or expr")
    stop_dist = stop_dist.clip(lower=0.05)

    tg = substitute_params(spec.exit.target, p) if spec.exit.target else None
    if tg is None:
        target_dist = pd.Series(np.inf, index=df.index)
    elif "r_mult" in tg:
        target_dist = stop_dist * float(tg["r_mult"])
    elif "atr_mult" in tg:
        atr = ensure_feature(df, f"atr_{int(tg.get('atr_n', 14))}")
        target_dist = atr * float(tg["atr_mult"])
    elif "fixed" in tg:
        target_dist = pd.Series(float(tg["fixed"]), index=df.index)
    else:
        raise ValueError("target must have r_mult, atr_mult or fixed")

    # sizing
    sz = spec.sizing
    if sz.type == "fixed":
        lots = pd.Series(float(sz.lots), index=df.index)
    elif sz.type == "risk_pct":
        risk_dollars = sz.equity * sz.risk_pct / 100.0
        lots = risk_dollars / (stop_dist * OZ_PER_LOT)
    elif sz.type == "vol_target":
        tfm = TF_MINUTES[spec.timeframe]
        bars_per_day = 23 * 60 / tfm
        rv = ensure_feature(df, "rvol_50")
        daily_vol_dollars = df["close"] * rv * np.sqrt(bars_per_day) * OZ_PER_LOT  # per lot
        target_daily = sz.equity * sz.annual_vol / np.sqrt(252)
        lots = target_daily / daily_vol_dollars.replace(0, np.nan)
    else:
        raise ValueError(f"unknown sizing {sz.type}")
    lots = lots.clip(lower=0.0, upper=sz.max_lots).round(2)
    lots = lots.where(lots >= sz.min_lots, 0.0)

    time_flag = _time_flag(df.index, spec.exit.time_exit_utc, TF_MINUTES[spec.timeframe])
    tr = substitute_params(spec.exit.trail, p) if spec.exit.trail else None
    if tr is None:
        trail_dist = np.full(len(df), np.nan)
        trail_act = np.zeros(len(df))
    else:
        if "atr_mult" in tr:
            trail_dist = (ensure_feature(df, f"atr_{int(tr.get('atr_n', 14))}") * float(tr["atr_mult"])).to_numpy(dtype=float)
        elif "fixed" in tr:
            trail_dist = np.full(len(df), float(tr["fixed"]))
        else:
            raise ValueError("trail must have atr_mult or fixed")
        trail_dist = np.where(trail_dist > 0, trail_dist, np.nan)
        trail_act = float(tr.get("activate_r", 0.0)) * stop_dist.to_numpy(dtype=float)
    return {
        "sig_long": sig_long.to_numpy(), "sig_short": sig_short.to_numpy(),
        "exit_long": exit_long.to_numpy(), "exit_short": exit_short.to_numpy(),
        "stop_dist": stop_dist.to_numpy(dtype=float), "target_dist": target_dist.to_numpy(dtype=float),
        "lots": lots.to_numpy(dtype=float), "time_exit_flag": time_flag,
        "trail_dist": trail_dist, "trail_act": trail_act,
    }


def backtest_spec(spec: StrategySpec, bars_m1: pd.DataFrame, cost: CostModel | None = None, cache_key: str | None = "default", init_cash: float | None = None, seed: int = 0) -> BacktestResult:
    cost = cost or CostModel()
    df = feature_frame(bars_m1, spec.timeframe, cache_key)
    sig = compile_signals(spec, df)
    res = run_backtest(
        df[["open", "high", "low", "close", "spread"]], cost=cost,
        init_cash=init_cash or spec.sizing.equity,
        max_bars_in_trade=spec.exit.max_bars or 10**9,
        params=dict(spec.params), seed=seed, **sig,
    )
    return res
