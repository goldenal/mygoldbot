"""The replay simulator must reproduce the backtester trade for trade (M1 strategy, same bars, same cost model, same slippage seed),
and must agree with a slow bar-by-bar reference on trailing stops, your own levels and manual changes."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from goldsys.chart import data as chartdata
from goldsys.data import store
from goldsys.data.synthetic import make_m1
from goldsys.dsl.compile import backtest_spec
from goldsys.dsl.spec import StrategySpec
from goldsys.replay import sim


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    d = tmp_path_factory.mktemp("bars")
    mp = pytest.MonkeyPatch()
    mp.setattr(store, "BAR_DIR", d); mp.setattr(chartdata, "BAR_DIR", d)
    store.write_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 2, 5), seed=11), "SIM", "M1")
    for dct in (chartdata._M1, chartdata._TF, chartdata._IND, chartdata._BT, sim._MARKETS, sim._CTX):
        dct.clear()
    yield "SIM"
    mp.undo()
    for dct in (chartdata._M1, chartdata._TF, chartdata._IND, chartdata._BT, sim._MARKETS, sim._CTX):
        dct.clear()


def spec(**over):
    d = {"name": "t", "timeframe": "M1", "entry_long": {"op": "cross_above", "args": ["close", "ema_20"]}, "entry_short": None,
         "exit": {"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": {"r_mult": 1.5}, "max_bars": 45}, "sizing": {"type": "fixed", "lots": 0.01}}
    d.update(over)
    return StrategySpec.from_dict(d)


def trade_of(M, C, k):
    return dict(fill_i=int(C.cand[k]) + 1, direction=int(C.cand_dir[k]), lots=float(C.cand_lots[k]), stop_dist=float(C.cand_stop[k]),
                target_dist=float(C.cand_target[k]) if np.isfinite(C.cand_target[k]) else None,
                trail_dist=None if not np.isfinite(C.cand_trail[k]) else float(C.cand_trail[k]), trail_act=float(C.cand_trail_act[k]))


def replay_all(M, C, reversals=False):
    """Take every signal the strategy is flat for, one position at a time like the backtester; with reversals=True an opposite signal closes
    the open trade at the next open and opens the other way (the backtester's behaviour)."""
    out, cur = [], None            # cur: (params, result)
    for k, j in enumerate(C.cand):
        j = int(j)
        if cur is not None and cur[1]["exit_i"] <= j:
            cur = None
        t = trade_of(M, C, k)
        if cur is not None:
            if not reversals or t["direction"] == cur[0]["direction"]:
                continue
            old, _ = cur
            trunc = sim.simulate(M, C, old["fill_i"], old["direction"], old["lots"], stop_dist=old["stop_dist"], target_dist=old["target_dist"],
                                 trail_dist=old["trail_dist"], trail_act=old["trail_act"], adjust=[{"i": j + 1, "close": True, "reason": "reverse"}])
            out[-1] = trunc
        res = sim.simulate(M, C, **t)
        if res is None:
            continue
        out.append(res)
        cur = (t, res)
    return out


def compare(M, mine, ref):
    assert len(ref) > 15, "scenario produced too few trades to be a test"
    assert len(mine) == len(ref), (len(mine), len(ref))
    for k, (m, r) in enumerate(zip(mine, ref.itertuples())):
        assert pd.Timestamp(int(M.ns[m["fill_i"]]), unit="ns", tz="UTC") == r.entry_ts, (k, "entry time")
        assert pd.Timestamp(int(M.ns[m["exit_i"]]), unit="ns", tz="UTC") == r.exit_ts, (k, "exit time", m["reason"], r.reason)
        assert m["entry_px"] == pytest.approx(r.entry_px, abs=1e-9), k
        assert m["exit_px"] == pytest.approx(r.exit_px, abs=1e-9), (k, m["reason"], r.reason)
        assert m["reason"] == r.reason, (k, m["reason"], r.reason)
        assert m["pnl"] == pytest.approx(r.pnl, abs=1e-6), k


EXIT_STOP = {"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": {"r_mult": 1.5}, "max_bars": 45}


@pytest.mark.parametrize("name,over", [
    ("long_only_stop_target_maxbars", {}),
    ("no_target", {"exit": {"stop": {"atr_mult": 2.0, "atr_n": 14}, "target": None, "max_bars": 90}}),
    ("signal_exit", {"exit": {"stop": {"atr_mult": 1.0, "atr_n": 14}, "target": {"r_mult": 4.0}, "exit_long": {"op": "cross_below", "args": ["close", "ema_20"]}}}),
    ("time_exit", {"exit": {"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": {"r_mult": 3.0}, "time_exit_utc": "13:00"}}),
    ("short_only", {"entry_long": None, "entry_short": {"op": "cross_below", "args": ["close", "ema_20"]}, "exit": {"stop": {"atr_mult": 1.2, "atr_n": 14}, "target": {"r_mult": 1.2}, "max_bars": 60}}),
    ("trailing_atr_long", {"exit": {"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": None, "max_bars": 200, "trail": {"atr_mult": 1.0, "atr_n": 14}}}),
    ("trailing_activated_long", {"exit": {"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": {"r_mult": 6.0}, "max_bars": 200, "trail": {"atr_mult": 1.0, "atr_n": 14, "activate_r": 0.8}}}),
    ("trailing_fixed_short", {"entry_long": None, "entry_short": {"op": "cross_below", "args": ["close", "ema_20"]},
                              "exit": {"stop": {"fixed": 2.0}, "target": None, "max_bars": 200, "trail": {"fixed": 1.2, "activate_r": 0.3}}}),
])
def test_matches_backtester(lake, name, over):
    s = spec(**over)
    M = sim.market(lake)
    C = sim.ctx_from_spec(M, s, name)
    mine = replay_all(M, C)
    ref = backtest_spec(s, M.frame, cache_key=None, seed=0).trades
    if "trail" in str(over):
        assert (ref["reason"] == "trail").sum() > 3, "the trailing scenarios must actually trail"
    compare(M, mine, ref)


def test_reversals_match_the_backtester(lake):
    """Long and short signals on one strategy: an opposite signal while in a trade closes it at the next open and enters the other way."""
    s = spec(entry_short="mirror", exit={"stop": {"atr_mult": 3.0, "atr_n": 14}, "target": None, "max_bars": 400, "trail": {"atr_mult": 2.0, "atr_n": 14, "activate_r": 0.5}})
    M = sim.market(lake)
    C = sim.ctx_from_spec(M, s, "rev")
    ref = backtest_spec(s, M.frame, cache_key=None, seed=0).trades
    assert (ref["reason"] == "reverse").sum() > 5
    compare(M, replay_all(M, C, reversals=True), ref)


# ---------------------------------------------------------------- slow reference for what the engine cannot do (manual changes, own levels)

def reference(M, C, f, d, lots, stop, tgt, trail_d, trail_a, adjust):
    n = len(M.ns)
    entry = M.open[f] + M.spread[f] + M.slip[f] if d > 0 else M.open[f] - M.slip[f]
    stop_set = stop if stop is not None else None
    ext, trailed, adj = entry, False, {}
    for a in adjust:
        adj.setdefault(a["i"], []).append(a)
    xs = set((C.exit_long_idx if d > 0 else C.exit_short_idx).tolist())
    closes = set(C.close_idx.tolist())
    erow = C.row[f]
    stop = entry - d * stop_set if False else stop
    for i in range(f, n):
        for a in adj.get(i, []):
            if i == f:
                continue
            if a.get("close"):
                px = M.open[i] - M.slip[i] if d > 0 else M.open[i] + M.spread[i] + M.slip[i]
                return i, px, a.get("reason", "manual")
            if a.get("stop_px") is not None:
                stop, trail_d, trailed = a["stop_px"], None, False
            if a.get("tgt_px") is not None:
                tgt = a["tgt_px"]
            if a.get("clear_target"):
                tgt = None
        o, h, lo, c, sp, sl = M.open[i], M.high[i], M.low[i], M.close[i], M.spread[i], M.slip[i]
        if d > 0:
            if lo <= stop:
                return i, min(stop, o) - sl, "trail" if trailed else "stop"
            if tgt is not None and h >= tgt:
                return i, (tgt if o < tgt else o), "target"
        else:
            if h + sp >= stop:
                return i, max(stop, o + sp) + sl, "trail" if trailed else "stop"
            if tgt is not None and lo + sp <= tgt:
                return i, (tgt if o + sp > tgt else o + sp), "target"
        if i in closes and (i in xs or (C.max_bars < 10**8 and C.row[i] - erow >= C.max_bars)):
            return i, (c - sl if d > 0 else c + sp + sl), ("time" if (i in xs and C.tflag[C.row[i]]) else "signal")
        if trail_d:
            if d > 0:
                ext = max(ext, h)
                if ext - entry >= trail_a and ext - trail_d > stop:
                    stop, trailed = ext - trail_d, True
            else:
                ext = min(ext, lo + sp)
                if entry - ext >= trail_a and ext + trail_d < stop:
                    stop, trailed = ext + trail_d, True
    return n - 1, (M.close[n - 1] - M.slip[n - 1] if d > 0 else M.close[n - 1] + M.spread[n - 1] + M.slip[n - 1]), "end"


def test_simulator_agrees_with_a_slow_reference_on_random_trades(lake):
    rng = np.random.default_rng(7)
    s = spec(entry_short="mirror", exit={"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": {"r_mult": 2.0}, "max_bars": 120,
                                         "exit_long": {"op": "cross_below", "args": ["close", "ema_20"]}, "exit_short": {"op": "cross_above", "args": ["close", "ema_20"]}})
    M = sim.market(lake)
    C = sim.ctx_from_spec(M, s, "ref")
    checked = {"trail": 0, "adjusted": 0, "own_levels": 0, "manual_close": 0}
    for _ in range(400):
        k = int(rng.integers(0, len(C.cand) - 5))
        t = trade_of(M, C, k)
        f, d = t["fill_i"], t["direction"]
        entry = M.open[f] + M.spread[f] + M.slip[f] if d > 0 else M.open[f] - M.slip[f]
        stop_d = float(rng.uniform(0.6, 4.0))
        stop = entry - d * stop_d
        tgt = None if rng.random() < .35 else entry + d * float(rng.uniform(0.6, 6.0))
        trail_d = float(rng.uniform(0.5, 3.0)) if rng.random() < .55 else None
        trail_a = float(rng.uniform(0, 2.0)) if trail_d else 0.0
        adjust = []
        for _a in range(int(rng.integers(0, 3))):
            i = f + int(rng.integers(1, 35))
            kind = rng.choice(["stop", "target", "clear", "close", "both"])
            a = {"i": i}
            px = M.close[min(i, len(M.ns) - 1)]
            if kind in ("stop", "both"):
                a["stop_px"] = float(px - d * rng.uniform(0.5, 4.0))
            if kind in ("target", "both"):
                a["tgt_px"] = float(px + d * rng.uniform(0.5, 5.0))
            if kind == "clear":
                a["clear_target"] = True
            if kind == "close":
                a["close"] = True
            adjust.append(a)
        mine = sim.simulate(M, C, f, d, t["lots"], stop_px=stop, tgt_px=tgt, trail_dist=trail_d, trail_act=trail_a, adjust=adjust)
        ref_i, ref_px, ref_reason = reference(M, C, f, d, t["lots"], stop, tgt, trail_d, trail_a, adjust)
        assert (mine["exit_i"], mine["reason"]) == (ref_i, ref_reason), (mine["exit_i"], mine["reason"], ref_i, ref_reason, adjust, trail_d)
        assert mine["exit_px"] == pytest.approx(ref_px, abs=1e-9)
        checked["trail"] += int(mine["reason"] == "trail"); checked["adjusted"] += int(bool(adjust)); checked["own_levels"] += 1; checked["manual_close"] += int(mine["reason"] == "manual")
    assert checked["trail"] >= 5 and checked["manual_close"] >= 5 and checked["adjusted"] > 100, checked


def test_levels_query_reports_what_applies_to_the_next_bar(lake):
    M = sim.market(lake)
    C = sim.ctx_from_spec(M, spec(exit={"stop": {"atr_mult": 3.0, "atr_n": 14}, "target": None, "max_bars": 300, "trail": {"fixed": 1.0}}), "lv")
    for k in range(0, 60, 3):
        t = trade_of(M, C, k)
        full = sim.simulate(M, C, **t)
        if full["exit_i"] - t["fill_i"] < 8:
            continue
        mid = t["fill_i"] + (full["exit_i"] - t["fill_i"]) // 2
        lv = sim.simulate(M, C, **t, until_i=mid)
        assert lv["open"] is True
        d = t["direction"]
        assert (lv["stop_px"] >= full["stop_px"]) if d > 0 else (lv["stop_px"] <= full["stop_px"])      # the trail only ever tightens
        # moving the stop to exactly that level by hand, from the next bar, reproduces the same exit if the trail would have stayed there or tightened less
        manual = sim.simulate(M, C, **{**t, "trail_dist": None}, adjust=[{"i": mid + 1, "stop_px": lv["stop_px"]}])
        assert manual["exit_i"] > mid


def test_adjustment_at_or_before_the_fill_is_ignored_and_exit_after_a_natural_exit_does_nothing(lake):
    M = sim.market(lake)
    C = sim.ctx_from_spec(M, spec(), "adj")
    t = trade_of(M, C, 5)
    base = sim.simulate(M, C, **t)
    late = sim.simulate(M, C, **t, adjust=[{"i": base["exit_i"] + 10, "close": True}, {"i": t["fill_i"], "close": True}])
    assert late["exit_i"] == base["exit_i"] and late["reason"] == base["reason"] and late["pnl"] == pytest.approx(base["pnl"])
