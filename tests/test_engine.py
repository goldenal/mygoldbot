"""Hand-checked trades through the backtester."""
import numpy as np
import pandas as pd

from goldsys.engine.backtest import run_backtest
from goldsys.engine.costs import CostModel
from goldsys.config import OZ_PER_LOT


def _bars(closes, hl=0.5):
    idx = pd.date_range("2024-01-02 10:00", periods=len(closes), freq="15min", tz="UTC")
    c = np.array(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + hl, "low": np.minimum(o, c) - hl, "close": c, "spread": 0.2}, index=idx)


def _cost():
    return CostModel(slippage_mean=0.0, slippage_std=0.0, commission_per_lot_rt=0.0, use_bar_spread=True, swap_long_per_lot_day=0.0, swap_short_per_lot_day=0.0)


def test_long_target_hit():
    bars = _bars([100, 100, 101, 102, 104, 104, 104])
    n = len(bars)
    sig_long = np.zeros(n, bool); sig_long[1] = True        # signal at close of bar 1 -> fill at open of bar 2 (=100 close of bar1)
    z = np.zeros(n, bool)
    res = run_backtest(bars, sig_long, z, z, z, stop_dist=np.full(n, 1.0), target_dist=np.full(n, 2.0), lots=np.full(n, 1.0), time_exit_flag=z, cost=_cost())
    t = res.trades
    assert len(t) == 1
    assert t.iloc[0]["entry_px"] == 100.2                    # open 100 + spread 0.2
    assert t.iloc[0]["reason"] == "target"
    assert abs(t.iloc[0]["exit_px"] - 102.2) < 1e-9          # target = entry + 2
    assert abs(t.iloc[0]["pnl"] - 2.0 * OZ_PER_LOT) < 1e-6


def test_long_stop_first_when_both_touched():
    bars = _bars([100, 100, 100, 100], hl=5.0)               # every bar spans +-5: stop and target both touched
    n = len(bars)
    sig_long = np.zeros(n, bool); sig_long[0] = True
    z = np.zeros(n, bool)
    res = run_backtest(bars, sig_long, z, z, z, np.full(n, 1.0), np.full(n, 2.0), np.full(n, 1.0), z, _cost())
    assert res.trades.iloc[0]["reason"] == "stop"
    assert res.trades.iloc[0]["pnl"] < 0


def test_short_round_trip_costs_spread():
    bars = _bars([100, 100, 100, 100, 100, 100], hl=0.1)
    n = len(bars)
    z = np.zeros(n, bool)
    sig_short = z.copy(); sig_short[0] = True
    exit_short = z.copy(); exit_short[3] = True
    res = run_backtest(bars, z, sig_short, z, exit_short, np.full(n, 50.0), np.full(n, 50.0), np.full(n, 1.0), z, _cost())
    t = res.trades.iloc[0]
    assert t["dir"] == -1 and t["reason"] == "signal"
    assert abs(t["pnl"] + 0.2 * OZ_PER_LOT) < 1e-6           # flat market: lose exactly the spread


def test_equity_matches_trade_pnl():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, 500))
    bars = _bars(closes)
    n = len(bars)
    sig_long = rng.random(n) < 0.05; sig_short = rng.random(n) < 0.05
    z = np.zeros(n, bool)
    res = run_backtest(bars, sig_long, sig_short, z, z, np.full(n, 1.0), np.full(n, 1.5), np.full(n, 0.5), z, _cost(), init_cash=10_000)
    assert abs(res.equity.iloc[-1] - 10_000 - res.trades["pnl"].sum()) < 1e-6
