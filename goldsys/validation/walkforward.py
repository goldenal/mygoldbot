"""Walk-forward analysis and combinatorial purged cross-validation over a strategy spec."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..dsl.compile import backtest_spec
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..engine.metrics import summarize, sharpe


@dataclass
class WindowResult:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    params: dict
    is_metrics: dict
    oos_metrics: dict
    oos_daily: pd.Series


def param_grid(space: dict, max_points: int | None = None, seed: int = 0) -> list[dict]:
    """Expand params_space into a list of param dicts (optionally random-subsampled)."""
    axes = {}
    for k, v in space.items():
        if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v) and v[2] > 0 and v[1] > v[0]:
            lo, hi, step = v
            vals = np.arange(lo, hi + step / 2, step)
            vals = [int(x) if float(x).is_integer() and isinstance(lo, int) and isinstance(step, int) else float(x) for x in vals]
            axes[k] = vals
        else:
            axes[k] = list(v)
    keys = list(axes)
    combos = [dict(zip(keys, c)) for c in itertools.product(*[axes[k] for k in keys])]
    if max_points and len(combos) > max_points:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_points, replace=False)]
    return combos


def _slice(bars: pd.DataFrame, start, end, warmup_days: int = 30) -> pd.DataFrame:
    s = pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    return bars[(bars.index >= s) & (bars.index < pd.Timestamp(end))]


def _score(metrics: dict, objective: str) -> float:
    v = metrics.get(objective, 0.0)
    if objective == "profit_factor" and not np.isfinite(v):
        v = 10.0
    # strategies with too few trades are not comparable
    if metrics.get("trades", 0) < 20:
        return -1e9
    return float(v)


def optimise_window(spec: StrategySpec, bars: pd.DataFrame, cost: CostModel, objective: str = "sharpe", max_points: int = 60, seed: int = 0, trial_log: list | None = None) -> tuple[dict, dict]:
    best, best_m, best_s = dict(spec.params), None, -np.inf
    for p in param_grid(spec.params_space, max_points, seed):
        s = spec.with_params(**p)
        res = backtest_spec(s, bars, cost, cache_key=None)
        m = summarize(res.equity, res.trades)
        sc = _score(m, objective)
        if trial_log is not None:
            trial_log.append({"params": p, "metrics": m})
        if sc > best_s:
            best, best_m, best_s = p, m, sc
    if best_m is None:
        res = backtest_spec(spec, bars, cost, cache_key=None)
        best_m = summarize(res.equity, res.trades)
    return best, best_m


def walk_forward(spec: StrategySpec, bars_m1: pd.DataFrame, cost: CostModel, is_months: int = 18, oos_months: int = 6, step_months: int | None = None, objective: str = "sharpe", max_points: int = 60, anchored: bool = False, seed: int = 0, trial_log: list | None = None) -> dict:
    step_months = step_months or oos_months
    t0, t1 = bars_m1.index[0], bars_m1.index[-1]
    windows: list[WindowResult] = []
    is_start = t0
    oos_start = t0 + pd.DateOffset(months=is_months)
    while oos_start + pd.DateOffset(months=oos_months) <= t1 + pd.Timedelta(days=1):
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        is_bars = _slice(bars_m1, is_start, oos_start, warmup_days=0)
        best, is_m = optimise_window(spec, is_bars, cost, objective, max_points, seed, trial_log)
        oos_bars = _slice(bars_m1, oos_start, oos_end, warmup_days=45)
        res = backtest_spec(spec.with_params(**best), oos_bars, cost, cache_key=None)
        eq = res.equity[res.equity.index >= oos_start]
        tr = res.trades[res.trades["entry_ts"] >= oos_start]
        oos_m = summarize(eq, tr) if len(eq) else {}
        daily = eq.resample("1D").last().dropna().pct_change().dropna()
        windows.append(WindowResult(is_start, oos_start, oos_start, oos_end, best, is_m, oos_m, daily))
        oos_start = oos_start + pd.DateOffset(months=step_months)
        if not anchored:
            is_start = oos_start - pd.DateOffset(months=is_months)
    if not windows:
        return {"windows": [], "n_windows": 0, "wfe": float("nan"), "oos_sharpe": float("nan"), "oos_daily": pd.Series(dtype=float)}
    oos_daily = pd.concat([w.oos_daily for w in windows]).sort_index()
    is_sh = np.array([w.is_metrics.get("sharpe", 0) for w in windows])
    oos_sh = np.array([w.oos_metrics.get("sharpe", 0) for w in windows])
    stitched_sharpe = sharpe(oos_daily)
    wfe = float(np.mean(oos_sh) / np.mean(is_sh)) if np.mean(is_sh) > 0 else float("nan")
    return {
        "windows": [
            {"is": f"{w.is_start.date()}..{w.is_end.date()}", "oos": f"{w.oos_start.date()}..{w.oos_end.date()}", "params": w.params,
             "is_sharpe": w.is_metrics.get("sharpe"), "oos_sharpe": w.oos_metrics.get("sharpe"), "oos_trades": w.oos_metrics.get("trades"), "oos_net": w.oos_metrics.get("net_profit")}
            for w in windows
        ],
        "n_windows": len(windows),
        "wfe": wfe,
        "oos_sharpe": float(stitched_sharpe),
        "oos_windows_positive": float((np.array([w.oos_metrics.get("net_profit", 0) for w in windows]) > 0).mean()),
        "oos_daily": oos_daily,
        "oos_total_return": float((1 + oos_daily).prod() - 1) if len(oos_daily) else 0.0,
        "param_stability": _param_stability(windows),
    }


def _param_stability(windows) -> dict:
    if not windows:
        return {}
    keys = windows[0].params.keys()
    out = {}
    for k in keys:
        vals = np.array([float(w.params[k]) for w in windows])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
    return out


def cpcv(spec: StrategySpec, bars_m1: pd.DataFrame, cost: CostModel, n_groups: int = 6, n_test: int = 2, embargo_days: int = 5, objective: str = "sharpe", max_points: int = 40, seed: int = 0, fixed_params: bool = False) -> dict:
    """Combinatorial purged cross-validation: split time into n_groups contiguous blocks; for each
    combination of n_test test blocks, fit params on the rest (purged + embargoed) and evaluate on
    the test blocks. Returns the distribution of OOS Sharpe and the stitched OOS paths."""
    idx = bars_m1.index
    edges = pd.DatetimeIndex([idx[int(i)] for i in np.linspace(0, len(idx) - 1, n_groups + 1)])
    blocks = [(edges[i], edges[i + 1]) for i in range(n_groups)]
    oos_sharpes, paths = [], []
    for test_combo in itertools.combinations(range(n_groups), n_test):
        # training = everything not in test, with embargo around each test block
        train_mask = pd.Series(True, index=idx)
        for g in test_combo:
            s, e = blocks[g]
            train_mask[(idx >= s - pd.Timedelta(days=embargo_days)) & (idx < e + pd.Timedelta(days=embargo_days))] = False
        train_bars = bars_m1[train_mask.to_numpy()]
        if fixed_params or not spec.params_space:
            best = dict(spec.params)
        else:
            best, _ = optimise_window(spec, train_bars, cost, objective, max_points, seed)
        dailies = []
        for g in test_combo:
            s, e = blocks[g]
            tb = _slice(bars_m1, s, e, warmup_days=45)
            res = backtest_spec(spec.with_params(**best), tb, cost, cache_key=None)
            eq = res.equity[res.equity.index >= s]
            dailies.append(eq.resample("1D").last().dropna().pct_change().dropna())
        d = pd.concat(dailies).sort_index() if dailies else pd.Series(dtype=float)
        oos_sharpes.append(sharpe(d))
        paths.append(d)
    arr = np.array(oos_sharpes)
    return {
        "n_splits": len(arr),
        "oos_sharpe_mean": float(arr.mean()) if len(arr) else float("nan"),
        "oos_sharpe_std": float(arr.std()) if len(arr) else float("nan"),
        "oos_sharpe_p05": float(np.percentile(arr, 5)) if len(arr) else float("nan"),
        "oos_sharpe_positive_frac": float((arr > 0).mean()) if len(arr) else float("nan"),
        "sharpes": arr.tolist(),
        "paths": paths,
    }
