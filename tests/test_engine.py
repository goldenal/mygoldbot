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


def test_stop_checked_on_entry_bar_and_no_same_bar_reentry():
    # bar 2 opens at 100 and trades down to 95: a long filled at that open must be stopped on that bar,
    # and the still-true signal may only re-enter at the NEXT bar's open (never at the earlier open).
    idx = pd.date_range("2024-01-02 10:00", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": [100, 100, 100, 96, 96], "high": [100.1, 100.1, 100.1, 96.1, 96.1], "low": [99.9, 99.9, 95.0, 95.9, 95.9],
                         "close": [100, 100, 96, 96, 96], "spread": 0.2}, index=idx, dtype=float)
    n = len(bars)
    z = np.zeros(n, bool)
    sig_long = np.array([False, True, True, False, False])
    res = run_backtest(bars, sig_long, z, z, z, np.full(n, 1.0), np.full(n, 50.0), np.full(n, 1.0), z, _cost())
    t = res.trades
    assert list(t["reason"]) == ["stop", "end"]
    assert t.iloc[0]["entry_ts"] == idx[2] and t.iloc[0]["exit_ts"] == idx[2]
    assert abs(t.iloc[0]["exit_px"] - 99.2) < 1e-9           # entry 100.2 - stop distance 1.0
    assert t.iloc[1]["entry_ts"] == idx[3] and abs(t.iloc[1]["entry_px"] - 96.2) < 1e-9


def test_ruin_liquidates_and_stops_trading():
    closes = np.concatenate([[100, 100], np.linspace(99, 40, 30)])
    bars = _bars(closes, hl=0.1)
    n = len(bars)
    z = np.zeros(n, bool)
    sig_long = np.ones(n, bool)
    res = run_backtest(bars, sig_long, z, z, z, np.full(n, 500.0), np.full(n, 500.0), np.full(n, 5.0), z, _cost(), init_cash=10_000)
    assert res.trades.iloc[-1]["reason"] == "ruin"
    assert len(res.trades) == 1                               # nothing is opened after the account is gone
    assert res.equity.iloc[-1] == res.equity[res.equity <= 0].iloc[0]


# ---------------------------------------------------------------- trailing stop (hand-checked)

def _trail_run(closes, d, stop=2.0, trail=1.0, act=0.0, hl=0.1):
    bars = _bars(closes, hl=hl)
    n = len(bars)
    z = np.zeros(n, bool)
    s = z.copy(); s[0] = True
    sl, ss = (s, z) if d > 0 else (z, s)
    return run_backtest(bars, sl, ss, z, z, np.full(n, stop), np.full(n, 99.0), np.full(n, 1.0), z, _cost(), trail_dist=np.full(n, trail), trail_act=np.full(n, act)).trades


def test_trailing_stop_long_ratchets_from_each_bars_high_and_exits_as_trail():
    # fill 100.2 (open 100 + spread 0.2), stop 98.2. After bar 1 the stop is 99.2, after bar 2 101.1, bar 3 103.1, bar 4 105.1; bar 5's low 104.9 touches it.
    t = _trail_run([100, 100, 102, 104, 106, 105, 105], +1)
    assert len(t) == 1 and t.iloc[0]["reason"] == "trail"
    assert abs(t.iloc[0]["exit_px"] - 105.1) < 1e-9 and abs(t.iloc[0]["pnl"] - (105.1 - 100.2) * OZ_PER_LOT) < 1e-6


def test_trailing_stop_short_is_the_mirror_image_and_uses_the_ask():
    # fill 100.0 (sell at bid). The stop follows the lowest ASK (low + 0.2) plus 1.0: 101 -> 99.1 -> 97.1 -> 95.1; bar 5's ask high 95.3 touches it.
    t = _trail_run([100, 100, 98, 96, 94, 95, 95], -1)
    assert len(t) == 1 and t.iloc[0]["reason"] == "trail" and t.iloc[0]["dir"] == -1
    assert abs(t.iloc[0]["exit_px"] - 95.1) < 1e-9 and abs(t.iloc[0]["pnl"] - (100.0 - 95.1) * OZ_PER_LOT) < 1e-6


def test_trail_waits_for_its_activation_profit():
    # activation 3.0 above the fill: the best price only reaches +1.9 before the trade turns, so the ORIGINAL stop (98.2) ends it, and it is not called a trail
    t = _trail_run([100, 100, 101, 102, 99, 97, 97], +1, act=3.0)
    assert t.iloc[0]["reason"] == "stop" and abs(t.iloc[0]["exit_px"] - 98.2) < 1e-9


def test_a_bars_own_extreme_never_moves_the_stop_within_that_bar():
    # one huge bar spikes up then back down through where a same-bar trail would sit; the stop only moves for the NEXT bar, so this bar cannot stop out on it
    bars = _bars([100, 100, 100, 100], hl=0.1)
    bars.iloc[1, bars.columns.get_loc("high")] = 110.0                  # spike in bar 1 (the fill bar)
    n = len(bars)
    z = np.zeros(n, bool); s = z.copy(); s[0] = True
    t = run_backtest(bars, s, z, z, z, np.full(n, 2.0), np.full(n, 99.0), np.full(n, 1.0), z, _cost(), trail_dist=np.full(n, 1.0), trail_act=np.zeros(n)).trades
    # bar 1 low 99.9 > stop 98.2: no exit in bar 1; the spike then sets the stop to 109.0 for bar 2, whose low 99.9 is below it
    assert t.iloc[0]["reason"] == "trail" and abs(t.iloc[0]["exit_px"] - min(109.0, bars.iloc[2]["open"])) < 1e-9


def test_no_trail_inputs_means_the_old_behaviour():
    bars = _bars([100, 100, 101, 102, 104, 104, 104])
    n = len(bars)
    z = np.zeros(n, bool); s = z.copy(); s[1] = True
    a = run_backtest(bars, s, z, z, z, np.full(n, 1.0), np.full(n, 2.0), np.full(n, 1.0), z, _cost()).trades
    b = run_backtest(bars, s, z, z, z, np.full(n, 1.0), np.full(n, 2.0), np.full(n, 1.0), z, _cost(), trail_dist=np.full(n, np.nan)).trades
    pd.testing.assert_frame_equal(a, b)
