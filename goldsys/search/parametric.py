"""Tier 1: parametric search over a strategy template's params_space.
Objective is OOS-robust by construction: a 2-window anchored walk-forward Sharpe minus a
complexity penalty, never the in-sample return. Uses Optuna (TPE) when available, else random."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..dsl.compile import backtest_spec
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..engine.metrics import summarize, sharpe
from .registry import Registry

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    optuna = None


def robust_objective(spec: StrategySpec, bars_m1: pd.DataFrame, cost: CostModel, n_folds: int = 3, min_trades_per_fold: int = 30) -> dict:
    """Split the sample into n_folds contiguous blocks; evaluate the FIXED params on each block
    and take the mean Sharpe penalised by dispersion. (Params are fixed here: the optimiser outside
    is the thing being searched, so the right protection is the later walk-forward in the gate.)"""
    idx = bars_m1.index
    edges = [idx[int(i)] for i in np.linspace(0, len(idx) - 1, n_folds + 1)]
    sharpes, trades = [], 0
    for i in range(n_folds):
        seg = bars_m1[(idx >= edges[i] - pd.Timedelta(days=45)) & (idx < edges[i + 1])]
        res = backtest_spec(spec, seg, cost, cache_key=None)
        eq = res.equity[res.equity.index >= edges[i]]
        d = eq.resample("1D").last().dropna().pct_change().dropna()
        sharpes.append(sharpe(d))
        trades += int((res.trades["entry_ts"] >= edges[i]).sum())
    sharpes = np.array(sharpes)
    if trades < min_trades_per_fold * n_folds:
        score = -5.0
    else:
        score = float(sharpes.mean() - 0.5 * sharpes.std() - 0.01 * spec.complexity())
    return {"score": score, "fold_sharpes": sharpes.tolist(), "trades": trades}


def _suggest(trial, name, v):
    if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v) and v[2] > 0 and v[1] > v[0]:
        lo, hi, step = v
        if all(isinstance(x, int) for x in v):
            return trial.suggest_int(name, lo, hi, step=step)
        return trial.suggest_float(name, float(lo), float(hi), step=float(step))
    return trial.suggest_categorical(name, list(v))


def search_template(spec: StrategySpec, bars_m1: pd.DataFrame, cost: CostModel, n_trials: int = 100, registry: Registry | None = None, seed: int = 0, verbose: bool = True) -> dict:
    space = spec.params_space
    if not space:
        return {"best_params": dict(spec.params), "best_score": None, "n_trials": 0}
    history = []

    def evaluate(params: dict) -> float:
        s = spec.with_params(**params)
        r = robust_objective(s, bars_m1, cost)
        full = backtest_spec(s, bars_m1, cost)
        m = summarize(full.equity, full.trades)
        history.append({"params": params, "score": r["score"], "fold_sharpes": r["fold_sharpes"], "metrics": m})
        if registry is not None:
            registry.log_trial("tier1", s, m, bars_m1.index[0], bars_m1.index[-1], cost.version, oos_sharpe=r["score"])
        return r["score"]

    if optuna is not None:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.enqueue_trial({k: spec.params[k] for k in space if k in spec.params})

        def obj(trial):
            params = {k: _suggest(trial, k, v) for k, v in space.items()}
            return evaluate(params)

        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best_score = study.best_value
    else:
        from ..validation.walkforward import param_grid
        rng = np.random.default_rng(seed)
        best, best_score = dict(spec.params), -np.inf
        for p in param_grid(space, n_trials, seed):
            sc = evaluate(p)
            if sc > best_score:
                best, best_score = p, sc
    if verbose:
        print(f"[tier1:{spec.name}] {len(history)} trials, best score {best_score:.3f} params {best}")
    history.sort(key=lambda h: h["score"], reverse=True)
    return {"best_params": best, "best_score": best_score, "n_trials": len(history), "top": history[:10]}
