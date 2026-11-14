"""Risk manager: every limit blocks what it should, persists, and never needs the engine to cooperate."""
from dataclasses import dataclass

import pandas as pd

from goldsys.live.ledger import Ledger
from goldsys.risk.calendar import EventCalendar
from goldsys.risk.limits import RiskConfig, RiskManager, market_open

T = pd.Timestamp("2026-03-04 09:00", tz="UTC")       # a Wednesday, London hours (table spread 0.18)


@dataclass
class Pos:
    lots: float
    entry_px: float
    stop_px: float


def entry(rm, ts=T, lots=0.5, stop=5.0, spread=0.2, equity=10_000.0, positions=()):
    return rm.check_entry(ts, "s", lots, stop, spread, equity, list(positions))


def test_allows_a_normal_entry():
    d = entry(RiskManager())
    assert d.allowed and d.lots == 0.4            # 2% open-risk budget = $200 / ($5 stop x 100 oz) = 0.4 lots
    assert entry(RiskManager(RiskConfig(max_open_risk_pct=100))).lots == 0.5


def test_open_risk_budget_counts_existing_positions():
    rm = RiskManager(RiskConfig(max_open_risk_pct=2.0))
    d = entry(rm, lots=1.0, stop=5.0, positions=[Pos(0.2, 2400, 2395)])       # $100 of the $200 budget already at risk
    assert d.allowed and d.lots == 0.2 and "size cut" in d.reasons[0]
    assert not entry(rm, positions=[Pos(0.4, 2400, 2395)]).allowed


def test_daily_loss_limit_blocks_until_the_next_rollover_and_flattens():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=1.5))
    assert rm.update(T, 10_000) == []
    assert rm.update(T + pd.Timedelta(hours=1), 9_840) == ["flatten"]
    assert rm.update(T + pd.Timedelta(hours=2), 9_800) == []                    # reported once
    d = entry(rm, ts=T + pd.Timedelta(hours=2))
    assert not d.allowed and "daily loss limit" in d.reasons[0]
    rm.update(pd.Timestamp("2026-03-04 22:00", tz="UTC"), 9_800)                # rollover = new trading day
    assert entry(rm, ts=pd.Timestamp("2026-03-05 09:00", tz="UTC"), equity=9_800).allowed


def test_weekly_loss_limit_survives_the_daily_reset():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=50, max_weekly_loss_pct=3.0, equity_floor_pct=50))
    rm.update(T, 10_000)
    rm.update(T + pd.Timedelta(days=1), 9_650)
    assert "weekly loss limit" in entry(rm, ts=T + pd.Timedelta(days=1, hours=1)).reasons[0]
    rm.update(pd.Timestamp("2026-03-08 22:00", tz="UTC"), 9_650)                # Sunday open = new week
    assert entry(rm, ts=pd.Timestamp("2026-03-09 09:00", tz="UTC"), equity=9_650).allowed


def test_loss_streak_pauses_entries():
    rm = RiskManager(RiskConfig(consecutive_losses_pause=3, pause_minutes=60))
    for i in range(2):
        rm.on_trade_closed(T, -10)
    rm.on_trade_closed(T, +10)                                                   # a win resets the streak
    for i in range(3):
        rm.on_trade_closed(T, -10)
    assert "loss-streak pause" in entry(rm, ts=T + pd.Timedelta(minutes=30)).reasons[0]
    assert entry(rm, ts=T + pd.Timedelta(minutes=61)).allowed


def test_spread_guard_and_news_blackout():
    cal = EventCalendar(pd.DataFrame({"ts": ["2026-03-04 13:30"], "name": ["US CPI"], "tier": [1]}))
    rm = RiskManager(RiskConfig(news_blackout_minutes=30), calendar=cal)
    assert "spread" in entry(rm, spread=0.40).reasons[0]                         # table 0.18 x 1.5 = 0.27
    assert entry(rm, spread=0.27).allowed
    assert "US CPI" in entry(rm, ts=pd.Timestamp("2026-03-04 13:05", tz="UTC")).reasons[0]
    assert "US CPI" in entry(rm, ts=pd.Timestamp("2026-03-04 13:59", tz="UTC")).reasons[0]
    assert entry(rm, ts=pd.Timestamp("2026-03-04 14:01", tz="UTC")).allowed
    assert cal.covers(T) and not cal.covers(pd.Timestamp("2027-01-01", tz="UTC"))


def test_equity_floor_trips_the_kill_switch_and_it_persists(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    rm = RiskManager(RiskConfig(max_daily_loss_pct=99, max_weekly_loss_pct=99, equity_floor_pct=10), store=led)
    rm.update(T, 10_000)
    rm.update(T, 11_000)
    assert rm.update(T + pd.Timedelta(hours=1), 9_890) == ["kill"]               # 10.1% below the 11,000 high-water mark
    rm2 = RiskManager(RiskConfig(), store=Ledger(tmp_path / "l.sqlite"))          # a restarted process
    assert "kill switch active" in entry(rm2).reasons[0]
    rm2.reset_kill("tester")
    assert entry(rm2).allowed


def test_manual_risk_off_blocks_entries():
    led = Ledger(":memory:")
    rm = RiskManager(store=led)
    led.set("risk_off", {"active": True})
    assert entry(rm).reasons == ["manual risk-off"]


def test_health_check_kills_on_stale_data_only_when_the_market_is_open():
    rm = RiskManager(RiskConfig(max_data_staleness_s=180))
    last = T - pd.Timedelta(minutes=10)
    assert rm.check_health(T, last, 60, True, None) == ["kill"]
    rm.reset_kill()
    sat = pd.Timestamp("2026-03-07 10:00", tz="UTC")
    assert rm.check_health(sat, pd.Timestamp("2026-03-06 20:59", tz="UTC"), 60, True, None) == []
    assert rm.check_health(T, T - pd.Timedelta(seconds=90), 60, True, None) == []
    assert rm.check_health(T, T, 60, False, T - pd.Timedelta(seconds=200)) == ["kill"]   # broker down too long
    assert market_open(T) and not market_open(sat) and not market_open(pd.Timestamp("2026-03-04 21:30", tz="UTC"))
