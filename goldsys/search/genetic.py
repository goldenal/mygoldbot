"""Tier 2: genetic programming over the DSL with MAP-Elites niches.

Individuals are StrategySpecs whose entry_long expression is a typed tree:
  bool  := cmp(num, num) | and(bool, bool) | or(bool, bool) | cross(num, num) | bool_feature
  num   := feature | const | num +- num | num * const
Shorts are mirrored; filters, exits and sizing come from a fixed template so the search stays in a
small, hard-to-overfit grammar. Fitness = robust_objective (fold-mean Sharpe minus dispersion minus
complexity). Niches = (holding-time bucket, trade-frequency bucket) so survivors are diverse.
Every evaluation is logged to the registry; the count feeds DSR later."""
from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd

from ..dsl.compile import backtest_spec
from ..dsl.spec import ExitSpec, SizingSpec, StrategySpec
from ..engine.costs import CostModel
from ..engine.metrics import summarize
from .parametric import robust_objective
from .registry import Registry

PRICE_FEATURES = ["close", "open", "high", "low", "ema_10", "ema_20", "ema_50", "ema_100", "sma_200", "donchian_high_20", "donchian_low_20", "donchian_high_50", "donchian_low_50", "asian_high", "asian_low", "vwap"]
OSC_FEATURES = {"rsi_2": (0, 100), "rsi_14": (0, 100), "adx_14": (0, 60), "bbz_20": (-3, 3), "vwap_z_60": (-3, 3), "atr_pct_rank_100": (0, 100), "asian_width_pct_rank_60": (0, 100), "hour": (0, 23)}
BOOL_FEATURES = ["in_london", "in_newyork", "in_asia", "in_overlap"]
ATR = "atr_14"


def _rand_price(rng: random.Random):
    f = rng.choice(PRICE_FEATURES)
    if rng.random() < 0.4:
        k = rng.choice([0.25, 0.5, 1.0, 1.5, 2.0])
        op = rng.choice(["+", "-"])
        return {"op": op, "args": [f, {"op": "*", "args": [ATR, k]}]}
    return f


def _rand_cmp(rng: random.Random):
    r = rng.random()
    if r < 0.4:
        a, b = _rand_price(rng), _rand_price(rng)
        return {"op": rng.choice(["gt", "cross_above"]), "args": [a, b]}
    if r < 0.8:
        f, (lo, hi) = rng.choice(list(OSC_FEATURES.items()))
        thr = round(rng.uniform(lo, hi), 1)
        return {"op": rng.choice(["gt", "lt"]), "args": [f, thr]}
    return rng.choice(BOOL_FEATURES)


def random_bool(rng: random.Random, depth: int = 0, max_depth: int = 2):
    if depth >= max_depth or rng.random() < 0.5:
        return _rand_cmp(rng)
    return {"op": rng.choice(["and", "and", "or"]), "args": [random_bool(rng, depth + 1, max_depth), random_bool(rng, depth + 1, max_depth)]}


def _nodes(expr, path=()):
    """Yield (path, node) for every dict/str node."""
    yield path, expr
    if isinstance(expr, dict):
        for i, a in enumerate(expr.get("args", [])):
            yield from _nodes(a, path + (i,))


def _get(expr, path):
    for i in path:
        expr = expr["args"][i]
    return expr


def _set(expr, path, new):
    if not path:
        return new
    e = copy.deepcopy(expr)
    cur = e
    for i in path[:-1]:
        cur = cur["args"][i]
    cur["args"][path[-1]] = new
    return e


def mutate(expr, rng: random.Random):
    paths = [p for p, n in _nodes(expr)]
    path = rng.choice(paths)
    node = _get(expr, path)
    r = rng.random()
    if isinstance(node, dict) and r < 0.4 and node.get("op") in ("gt", "lt") and isinstance(node["args"][1], (int, float)):
        # jitter threshold
        f = node["args"][0]
        lo, hi = OSC_FEATURES.get(f, (0, 100))
        new = copy.deepcopy(node)
        new["args"][1] = float(np.clip(node["args"][1] + rng.gauss(0, (hi - lo) * 0.1), lo, hi).round(1))
        return _set(expr, path, new)
    if isinstance(node, dict) and r < 0.6 and node.get("op") in ("gt", "lt", "cross_above", "cross_below"):
        new = copy.deepcopy(node)
        new["op"] = {"gt": "lt", "lt": "gt", "cross_above": "cross_below", "cross_below": "cross_above"}[node["op"]]
        return _set(expr, path, new)
    # replace subtree
    return _set(expr, path, random_bool(rng, max_depth=1))


def crossover(a, b, rng: random.Random):
    pa = rng.choice([p for p, n in _nodes(a) if isinstance(n, (dict, str))])
    pb = rng.choice([p for p, n in _nodes(b) if isinstance(n, (dict, str))])
    return _set(a, pa, copy.deepcopy(_get(b, pb)))


def make_template(name: str, timeframe: str = "M15") -> StrategySpec:
    return StrategySpec(
        name=name, family="gp", timeframe=timeframe,
        filters=[{"op": "or", "args": ["in_london", "in_newyork"]}],
        entry_long=None, entry_short="mirror",
        exit=ExitSpec(stop={"atr_mult": 1.5, "atr_n": 14}, target={"r_mult": 2.0}, time_exit_utc="20:00", max_bars=64),
        sizing=SizingSpec(type="risk_pct", risk_pct=0.5, equity=10_000, max_lots=2.0),
        params={}, params_space={},
    )


def niche_of(metrics: dict) -> tuple[int, int]:
    hold = metrics.get("avg_bars_held", 0)
    hb = 0 if hold < 4 else 1 if hold < 16 else 2
    tr = metrics.get("trades", 0)
    tb = 0 if tr < 300 else 1 if tr < 1000 else 2
    return hb, tb


def evolve(bars_m1: pd.DataFrame, cost: CostModel, generations: int = 10, pop_size: int = 30, timeframe: str = "M15", seed: int = 0, registry: Registry | None = None, verbose: bool = True, template: StrategySpec | None = None) -> dict:
    rng = random.Random(seed)
    base = template or make_template("gp", timeframe)
    elites: dict[tuple, dict] = {}        # niche -> best individual
    n_eval = 0

    def evaluate(expr, gen: int) -> dict:
        nonlocal n_eval
        n_eval += 1
        s = copy.deepcopy(base)
        s.name = f"gp_g{gen}_{n_eval}"
        s.entry_long = expr
        try:
            full = backtest_spec(s, bars_m1, cost)
            m = summarize(full.equity, full.trades)
            if m["trades"] < 50:
                score = -5.0
                r = {"fold_sharpes": []}
            else:
                r = robust_objective(s, bars_m1, cost)
                score = r["score"]
        except Exception as e:  # malformed tree, unknown feature etc.
            m, score, r = {"trades": 0}, -9.0, {"fold_sharpes": [], "err": str(e)}
        if registry is not None and m.get("trades", 0) > 0:
            registry.log_trial("tier2", s, m, bars_m1.index[0], bars_m1.index[-1], cost.version, oos_sharpe=score)
        return {"expr": expr, "score": score, "metrics": m, "folds": r.get("fold_sharpes", []), "spec": s}

    def add(ind):
        nz = niche_of(ind["metrics"])
        if nz not in elites or ind["score"] > elites[nz]["score"]:
            elites[nz] = ind

    pop = [evaluate(random_bool(rng), 0) for _ in range(pop_size)]
    for ind in pop:
        add(ind)
    for g in range(1, generations + 1):
        parents = list(elites.values()) + sorted(pop, key=lambda x: x["score"], reverse=True)[: pop_size // 3]
        children = []
        for _ in range(pop_size):
            a = rng.choice(parents)["expr"]
            if rng.random() < 0.5:
                b = rng.choice(parents)["expr"]
                c = crossover(a, b, rng)
            else:
                c = copy.deepcopy(a)
            if rng.random() < 0.7:
                c = mutate(c, rng)
            children.append(evaluate(c, g))
        for ind in children:
            add(ind)
        pop = children
        if verbose:
            best = max(elites.values(), key=lambda x: x["score"])
            print(f"[tier2] gen {g}: evals={n_eval} niches={len(elites)} best={best['score']:.3f} trades={best['metrics'].get('trades')} folds={[round(x,2) for x in best['folds']]}", flush=True)
    ranked = sorted(elites.values(), key=lambda x: x["score"], reverse=True)
    return {"n_evaluations": n_eval, "elites": [{"niche": niche_of(e["metrics"]), "score": e["score"], "trades": e["metrics"].get("trades"), "sharpe": e["metrics"].get("sharpe"), "expr": e["expr"], "spec_yaml": e["spec"].to_yaml()} for e in ranked]}
