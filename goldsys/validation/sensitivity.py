"""Cost-sensitivity and parameter-plateau tests."""
from __future__ import annotations

import copy

import numpy as np

from ..dsl.compile import backtest_spec
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..engine.metrics import summarize


def cost_sensitivity(spec: StrategySpec, bars_m1, cost: CostModel, multipliers=(0.0, 1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    out = []
    for m in multipliers:
        c = copy.deepcopy(cost)
        c.spread_multiplier = m
        if m == 0.0:
            c.slippage_mean = 0.0
            c.slippage_std = 0.0
            c.commission_per_lot_rt = 0.0
        res = backtest_spec(spec, bars_m1, c)
        met = summarize(res.equity, res.trades)
        out.append({"spread_mult": m, "net_profit": met["net_profit"], "sharpe": met["sharpe"], "profit_factor": met["profit_factor"], "trades": met["trades"]})
    return out


def parameter_plateau(spec: StrategySpec, bars_m1, cost: CostModel, metric: str = "sharpe", rel_steps=(-0.2, -0.1, 0.0, 0.1, 0.2)) -> dict:
    """Perturb each numeric parameter by +/-10/20% (one at a time) and report the metric.
    A robust strategy degrades smoothly; a sharp peak at 0% is an overfit signature."""
    base = summarize(*(lambda r: (r.equity, r.trades))(backtest_spec(spec, bars_m1, cost)))[metric]
    rows = {}
    for k, v in spec.params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        vals = []
        for s in rel_steps:
            nv = v * (1 + s)
            if isinstance(v, int):
                nv = max(1, int(round(nv)))
            res = backtest_spec(spec.with_params(**{k: nv}), bars_m1, cost)
            vals.append(summarize(res.equity, res.trades)[metric])
        rows[k] = {"values": vals, "steps": list(rel_steps)}
    neighbours = [x for r in rows.values() for i, x in enumerate(r["values"]) if r["steps"][i] != 0.0]
    nb_mean = float(np.mean(neighbours)) if neighbours else base
    return {"base": float(base), "neighbour_mean": nb_mean, "plateau_ratio": float(nb_mean / base) if base not in (0, None) and base > 0 else float("nan"), "per_param": rows}
