"""The event-driven live path (LiveRunner + PaperBroker) must reproduce the screening backtester.
This is the bar-level 'second engine' check: same DSL spec, same cost model, two independent
implementations of order sequencing and fills."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from goldsys.data.synthetic import make_m1
from goldsys.dsl.compile import TF_MINUTES, backtest_spec, feature_frame
from goldsys.dsl.spec import StrategySpec
from goldsys.engine.costs import CostModel
from goldsys.live.broker import PaperBroker
from goldsys.live.ledger import Ledger
from goldsys.live.runner import LiveRunner, Slot, make_slots
from goldsys.live.signals import PrecomputedSignals, RollingSignals
from goldsys.risk.limits import RiskConfig, RiskManager

NO_LIMITS = dict(max_daily_loss_pct=1e9, max_weekly_loss_pct=1e9, max_open_risk_pct=1e9, max_lots_per_trade=1e9, max_total_lots=1e9,
                 max_positions=99, consecutive_losses_pause=0, spread_guard_mult=0, news_blackout_minutes=0, equity_floor_pct=1e9)


def no_slip_cost():
    return CostModel(slippage_mean=0.0, slippage_std=0.0)


@pytest.fixture(scope="module")
def bars():
    return make_m1(dt.date(2023, 1, 1), dt.date(2023, 5, 1), seed=11)


def run_live(spec, bars_m1, cost, exec_tf=None):
    """exec_tf=None: execution bars are the strategy's own timeframe (exact parity mode); 'M1': minute bars."""
    df = feature_frame(bars_m1, spec.timeframe, None)
    if exec_tf == "M1":
        ex = bars_m1.copy()
        minutes = 1
    else:
        ex = df[["open", "high", "low", "close", "spread", "ticks", "volume"]].copy()
        minutes = TF_MINUTES[spec.timeframe]
    ex["spread"] = cost.spread_at(ex.index, ex["spread"])
    ledger = Ledger(":memory:")
    broker = PaperBroker(cost, init_cash=spec.sizing.equity, slippage=False)
    risk = RiskManager(RiskConfig(**NO_LIMITS), cost, store=ledger)
    runner = LiveRunner(broker, ledger, [Slot(spec.name, spec, PrecomputedSignals(spec, df))], risk, cost, exec_minutes=minutes, store_bars=False)
    runner.replay(ex)
    return ledger, broker


@pytest.mark.parametrize("name", ["asian_breakout", "vol_momentum", "vwap_reversion"])
def test_paper_engine_matches_backtester_trade_for_trade(bars, name):
    spec = StrategySpec.from_yaml(f"strategies/{name}.yaml")
    cost = no_slip_cost()
    bt = backtest_spec(spec, bars, cost, cache_key=None)
    ledger, broker = run_live(spec, bars, cost)
    live = ledger.trades()
    assert len(bt.trades) > 20
    assert len(live) == len(bt.trades)
    assert (live["entry_ts"].to_numpy() == bt.trades["entry_ts"].to_numpy()).all()
    assert (live["exit_ts"].to_numpy() == bt.trades["exit_ts"].to_numpy()).all()
    assert (live["dir"].to_numpy() == bt.trades["dir"].to_numpy()).all()
    assert list(live["reason"]) == list(bt.trades["reason"])
    for col in ("lots", "entry_px", "exit_px", "pnl"):
        assert np.allclose(live[col].to_numpy(), bt.trades[col].to_numpy(), atol=1e-6), col
    assert abs(broker.account().equity - bt.equity.iloc[-1]) < 1e-4
    assert ledger.verify_chain()["ok"]


def test_minute_execution_stays_close_to_the_screening_engine(bars):
    """Same signals, but stops/targets resolved on M1 bars instead of the strategy's H1 bars. Entries
    are identical; outcomes may differ only where the coarse bar hid the intrabar order of events."""
    spec = StrategySpec.from_yaml("strategies/vol_momentum.yaml")
    cost = no_slip_cost()
    bt = backtest_spec(spec, bars, cost, cache_key=None)
    ledger, _ = run_live(spec, bars, cost, exec_tf="M1")
    live = ledger.trades()
    assert abs(len(live) - len(bt.trades)) <= max(2, 0.1 * len(bt.trades))
    both = set(live["entry_ts"]) & set(bt.trades["entry_ts"])
    assert len(both) >= 0.85 * len(bt.trades)


def test_rolling_signals_equal_precomputed(bars):
    """The live signal path (re-evaluated on a rolling window each bar) gives the same decisions as the
    one-shot evaluation the backtester uses, once the window covers the indicators' warm-up."""
    spec = StrategySpec.from_yaml("strategies/vol_momentum.yaml")
    df = feature_frame(bars, spec.timeframe, None)
    pre = PrecomputedSignals(spec, df)
    cut = len(df) - 60
    cols = ["open", "high", "low", "close", "spread", "ticks", "volume"]
    roll = RollingSignals(spec, df[cols].iloc[:cut], window=5000)
    ns = df.index.as_unit("ns").asi8
    for i in range(cut, len(df)):
        a = roll.on_bar(int(ns[i]), df[cols].iloc[i].to_dict())
        b = pre.on_bar(int(ns[i]))
        for k in ("sig_long", "sig_short", "exit_long", "exit_short", "time_exit_flag"):
            assert a[k] == b[k], (i, k)
        for k in ("stop_dist", "target_dist", "lots"):
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), (i, k)
