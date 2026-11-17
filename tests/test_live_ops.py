"""Hybrid layer, ledger, bridge, portfolio step, reconciliation and the Phase 5 rules."""
import datetime as dt
import threading

import numpy as np
import pandas as pd
import pytest

from goldsys.data.synthetic import make_m1
from goldsys.dsl.spec import ExitSpec, StrategySpec
from goldsys.engine.costs import CostModel
from goldsys.live import control
from goldsys.live.bridge import RemoteBroker, make_server
from goldsys.live.broker import BrokerError, PaperBroker
from goldsys.live.drift import drift_reference, drift_status
from goldsys.live.ledger import Ledger
from goldsys.live.reconcile import ForwardConfig, forward_report
from goldsys.live.runner import LiveRunner, Slot, make_slots
from goldsys.live.scaling import ScalingConfig, recommend_scale, stage_gate
from goldsys.live.statement import statement
from goldsys.portfolio.allocate import allocate
from goldsys.portfolio.config import Portfolio, StrategyEntry
from goldsys.risk.limits import RiskConfig, RiskManager

FLAT = {"sig_long": False, "sig_short": False, "exit_long": False, "exit_short": False, "stop_dist": 5.0, "target_dist": 10.0, "lots": 0.1, "time_exit_flag": False}
OPEN = dict(max_daily_loss_pct=1e9, max_weekly_loss_pct=1e9, max_open_risk_pct=1e9, max_lots_per_trade=1e9, max_total_lots=1e9, consecutive_losses_pause=0, equity_floor_pct=1e9)
T0 = pd.Timestamp("2026-03-04 08:00", tz="UTC")
M15 = 15 * 60_000_000_000


class Scripted:
    """Signal source driven by the test: {bar number: overrides}."""

    def __init__(self, script):
        self.script, self.t0 = script, T0.as_unit("ns").value

    def on_bar(self, ts_ns, bar):
        return dict(FLAT, **self.script.get((ts_ns - self.t0) // M15, {}))


class Rig:
    def __init__(self, script, mode="auto", drift=None, risk=None, db=":memory:", **kw):
        cost = CostModel(slippage_mean=0.0, slippage_std=0.0)
        self.ledger = Ledger(db)
        self.broker = PaperBroker(cost, slippage=False)
        spec = StrategySpec(name="s", timeframe="M15", exit=ExitSpec())
        self.slot = Slot("s", spec, Scripted(script), mode=mode, drift=drift or {})
        self.runner = LiveRunner(self.broker, self.ledger, [self.slot], RiskManager(RiskConfig(**(risk or OPEN)), cost, store=self.ledger), cost,
                                 exec_minutes=15, store_bars=False, **kw)
        self.i = 0

    def bar(self, price=2400.0, hl=0.5, n=1):
        for _ in range(n):
            ts = T0 + pd.Timedelta(minutes=15 * self.i)
            self.runner.feed_bar(ts.as_unit("ns").value, price, price + hl, price - hl, price, 0.2)
            self.i += 1

    def statuses(self):
        return [r["status"] for r in reversed(self.ledger.intents())]


# ---------------------------------------------------------------- approval queue / modes / controls
def test_auto_mode_executes_at_the_next_open():
    r = Rig({0: {"sig_long": True}})
    r.bar(n=2)
    assert r.statuses() == ["executed"] and len(r.broker.positions()) == 1
    t = r.ledger.trades().iloc[0]
    assert t["entry_ts"] == T0 + pd.Timedelta(minutes=15) and abs(t["entry_px"] - 2400.2) < 1e-9 and abs(t["stop_px"] - 2395.2) < 1e-9


def test_approve_mode_waits_for_the_operator_then_executes():
    r = Rig({0: {"sig_long": True}}, mode="approve", approval_ttl_s=3600)
    r.bar(n=2)
    assert r.statuses() == ["pending"] and not r.broker.positions()
    iid = r.ledger.intents("pending")[0]["id"]
    assert control.approve(r.ledger, iid, by="alice")
    assert not control.approve(r.ledger, iid, by="bob")                  # only once
    r.bar()
    assert r.statuses() == ["executed"] and len(r.broker.positions()) == 1
    assert r.ledger.intents()[0]["decided_by"] == "alice"


def test_veto_and_expiry_and_price_drift():
    r = Rig({0: {"sig_long": True}}, mode="approve", approval_ttl_s=3600)
    r.bar(n=2)
    control.veto(r.ledger, r.ledger.intents("pending")[0]["id"], by="alice", reason="FOMC")
    r.bar(n=3)
    assert r.statuses() == ["vetoed"] and not r.broker.positions()

    r = Rig({0: {"sig_long": True}}, mode="approve", approval_ttl_s=900)
    r.bar(n=3)
    assert r.statuses() == ["expired"]

    r = Rig({0: {"sig_long": True}}, mode="approve", approval_ttl_s=3600)
    r.bar(n=2)
    control.approve(r.ledger, r.ledger.intents("pending")[0]["id"])
    r.bar(price=2404.0)                                                  # moved 4.0 > 0.25 x stop distance 5.0
    assert r.statuses() == ["expired"] and "price moved" in r.ledger.intents()[0]["reasons"]


def test_mode_off_logs_the_signal_but_does_not_trade_and_can_be_switched_live():
    r = Rig({0: {"sig_long": True}, 3: {"sig_short": True}}, mode="off")
    r.bar(n=2)
    assert r.statuses() == ["off"] and not r.broker.positions()
    control.set_mode(r.ledger, "s", "auto", by="alice")                  # override from the CLI / dashboard
    r.bar(n=3)
    assert r.statuses() == ["off", "executed"] and r.broker.positions()[0].dir == -1


def test_risk_block_is_recorded_and_exits_are_never_blocked():
    r = Rig({0: {"sig_long": True}, 2: {"sig_long": True}, 4: {"exit_long": True}})
    r.bar(n=2)
    control.risk_off(r.ledger, True, by="alice")
    r.bar(n=4)                                                           # exit signal at bar 4 must still close the position
    assert not r.broker.positions()
    assert r.ledger.trades().iloc[0]["reason"] == "signal"
    r2 = Rig({0: {"sig_long": True}})
    control.risk_off(r2.ledger, True)
    r2.bar(n=2)
    assert r2.statuses() == ["blocked"] and "manual risk-off" in r2.ledger.intents()[0]["reasons"]


def test_operator_kill_flattens_and_blocks_until_reset():
    r = Rig({0: {"sig_long": True}, 3: {"sig_long": True}}, realtime=True)
    r.bar(n=2)
    assert r.broker.positions()
    control.kill(r.ledger, by="alice", reason="test")
    r.bar(n=3)
    assert not r.broker.positions() and r.ledger.trades().iloc[0]["reason"] == "kill"
    assert r.statuses() == ["executed", "blocked"]
    control.reset_kill(r.ledger)
    assert not r.runner.risk.kill_state()["active"]


def test_daily_loss_limit_flattens_the_book():
    r = Rig({0: {"sig_long": True, "lots": 1.0, "stop_dist": 50.0, "target_dist": 100.0}}, risk=dict(OPEN, max_daily_loss_pct=1.5))
    r.bar(n=2)
    r.bar(price=2397.0)                                                  # -3.2 x 100 oz = -320 = -3.2% of 10,000
    assert not r.broker.positions() and r.ledger.trades().iloc[0]["reason"] == "risk"
    assert any(e["kind"] == "daily_loss_limit" for e in r.ledger.events())


def test_drift_breach_switches_the_strategy_off():
    script = {i: {"sig_long": True, "stop_dist": 0.3, "target_dist": 50.0} for i in range(0, 40)}     # stop inside the bar range: every trade loses 1R
    r = Rig(script, drift={"window": 5, "p05_r": -0.5})
    r.bar(n=12)
    assert r.slot.drift_tripped and r.ledger.get("mode:s")["by"] == "drift_monitor"
    n = len(r.ledger.trades())
    r.bar(n=6)
    assert len(r.ledger.trades()) == n and r.statuses()[-1] == "off"
    assert any(e["kind"] == "drift_breach" and e["level"] == "critical" for e in r.ledger.events())
    assert control.state(r.ledger)["strategies"][0]["mode"] == "off"


def test_drift_reference_and_status():
    rng = np.random.default_rng(0)
    ref = drift_reference(pd.DataFrame({"r_multiple": rng.normal(0.1, 1.0, 400)}), window=30)
    assert ref["p05_r"] < 0 < ref["p50_r"] and ref["n_ref"] == 400
    assert drift_status([0.1] * 10, ref)["state"] == "warming_up"
    assert drift_status([0.1] * 30, ref)["state"] == "ok"
    assert drift_status([-1.0] * 30, ref)["state"] == "breach"
    assert drift_reference(pd.DataFrame({"r_multiple": [0.1] * 5}), window=30) == {}


# ---------------------------------------------------------------- ledger
def test_ledger_chain_detects_tampering(tmp_path):
    r = Rig({i: {"sig_long": True, "stop_dist": 0.3} for i in range(0, 12, 2)}, db=tmp_path / "l.sqlite")
    r.bar(n=14)
    r.runner.finish()
    assert r.ledger.verify_chain()["ok"] and len(r.ledger.trades()) >= 3
    st = statement(r.ledger)
    assert st["trades"] == len(r.ledger.trades()) and st["chain"]["ok"] and abs(st["net"] - r.ledger.trades()["pnl"].sum()) < 1e-9
    r.ledger.conn.execute("UPDATE trades SET pnl = pnl + 500 WHERE id = 2"); r.ledger.conn.commit()
    assert "exit differs" in r.ledger.verify_chain()["error"]                # a loss edited into a win
    r.ledger.conn.execute("UPDATE trades SET pnl = pnl - 500 WHERE id = 2")
    r.ledger.conn.execute("DELETE FROM trades WHERE id = 1"); r.ledger.conn.commit()
    assert "missing" in r.ledger.verify_chain()["error"]                     # a losing trade deleted
    r2 = Rig({0: {"sig_long": True, "stop_dist": 0.3}})
    r2.bar(n=3)
    r2.ledger.conn.execute("UPDATE chain SET payload = replace(payload, '\"pnl\":-', '\"pnl\":') WHERE kind='exit'"); r2.ledger.conn.commit()
    assert "chain broken" in r2.ledger.verify_chain()["error"]               # the chain itself rewritten


# ---------------------------------------------------------------- bridge
def test_bridge_round_trip_and_auth():
    cost = CostModel(slippage_mean=0.0, slippage_std=0.0)
    paper = PaperBroker(cost, slippage=False)
    paper.set_quote(T0, 2400.0, 0.2)
    token = "t" * 32
    srv = make_server(paper, token, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        rb = RemoteBroker(url, token)
        assert rb.connected() and rb.quote().ask == pytest.approx(2400.2) and rb.quote().ts == T0
        f = rb.open_position("s", 1, 0.1, 5.0, float("inf"))                 # inf target must survive JSON
        assert f.price == pytest.approx(2400.2) and f.tgt_px is None and f.stop_px == pytest.approx(2395.2)
        pos = rb.positions()
        assert len(pos) == 1 and pos[0].id == f.position_id and pos[0].entry_ts == T0
        paper.on_bar(T0 + pd.Timedelta(minutes=15), 2400, 2400, 2390, 2391, 0.2)      # stop hit at the venue
        fills = rb.poll_fills()
        assert len(fills) == 1 and fills[0].reason == "stop" and fills[0].pnl == pytest.approx(-50 - 0.6)   # 0.1 lot x 100 oz x $5 stop, plus commission
        assert rb.account().balance == pytest.approx(10_000 - 50.6)
        with pytest.raises(BrokerError):
            rb.close_position("nope")
        assert not RemoteBroker(url, "wrong" * 8).connected()
        with pytest.raises(BrokerError):
            RemoteBroker(url, "wrong" * 8).account()
        with pytest.raises(BrokerError):
            rb._call("set_quote", 1, 2, 3)                                    # not on the allow-list
    finally:
        srv.shutdown()
    with pytest.raises(ValueError):
        make_server(paper, "short")


# ---------------------------------------------------------------- portfolio step
def test_allocation_drops_clones_and_hits_the_vol_target():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=750, freq="1D", tz="UTC")
    a = pd.Series(rng.normal(0.0004, 0.004, 750), idx)
    b = pd.Series(rng.normal(0.0003, 0.012, 750), idx)
    clone = a * 1.1 + rng.normal(0, 0.0005, 750)
    out = allocate({"a": a, "b": b, "clone": clone}, {"a": 1.5, "b": 0.8, "clone": 1.0}, corr_cap=0.7, target_annual_vol=0.10, max_weight=5, max_gross=10)
    assert set(out["weights"]) == {"a", "b"} and out["dropped"][0]["name"] == "clone"
    assert out["portfolio"]["annual_vol"] == pytest.approx(0.10, rel=1e-6)
    assert out["weights"]["a"] > out["weights"]["b"] * 2.5                    # inverse vol: the calmer stream gets ~3x the weight
    rp = allocate({"a": a, "b": b}, {"a": 1, "b": 1}, method="risk_parity", max_weight=5, max_gross=10)
    assert rp["risk_contribution"]["a"] == pytest.approx(0.5, abs=0.01)
    capped = allocate({"a": a, "b": b}, {"a": 1, "b": 1}, max_weight=0.5)
    assert max(capped["weights"].values()) <= 0.5 + 1e-9 and not capped["portfolio"]["vol_target_reached"]


# ---------------------------------------------------------------- forward test + Phase 5 rules
@pytest.fixture(scope="module")
def forward(tmp_path_factory):
    bars = make_m1(dt.date(2023, 1, 1), dt.date(2023, 4, 1), seed=21)
    spec = StrategySpec.from_yaml("strategies/asian_breakout.yaml")
    cost = CostModel()
    ledger = Ledger(tmp_path_factory.mktemp("fwd") / "l.sqlite")
    start = pd.Timestamp("2023-02-01", tz="UTC")
    slots = make_slots([spec], bars)
    broker = PaperBroker(cost, slippage=True, seed=3)
    runner = LiveRunner(broker, ledger, slots, RiskManager(RiskConfig(**OPEN), cost, store=ledger), cost, stage="demo")
    ex = bars[bars.index >= start].copy()
    ex["spread"] = cost.spread_at(ex.index, ex["spread"])
    runner.replay(ex)
    return ledger, spec, cost, bars


def test_forward_report_pairs_live_trades_with_the_shadow(forward):
    ledger, spec, cost, bars = forward
    rep = forward_report(ledger, [spec], cost, ForwardConfig(min_weeks=8, min_trades=20), history=bars, stage="demo", save=False)
    s = rep["summary"]
    assert rep["passed"], rep["reasons"]
    assert s["paired"] >= 20 and s["match_rate"] >= 0.95 and abs(s["entry_slip_mean"]) < 0.1 and s["errors"] == 0 and s["ledger_chain_ok"]
    short = forward_report(ledger, [spec], cost, ForwardConfig(min_weeks=12, min_trades=20), history=bars, stage="demo", save=False)
    assert not short["passed"] and "weeks" in short["reasons"][0]


def test_forward_report_fails_when_live_fills_are_much_worse(forward, tmp_path):
    ledger, spec, cost, bars = forward
    import shutil
    shutil.copy(ledger.path, tmp_path / "bad.sqlite")
    bad = Ledger(tmp_path / "bad.sqlite")
    bad.conn.execute("UPDATE trades SET entry_px = entry_px + 0.6 * dir, pnl = pnl - 0.6 * lots * 100"); bad.conn.commit()
    rep = forward_report(bad, [spec], cost, ForwardConfig(min_weeks=8, min_trades=20), history=bars, stage="demo", save=False)
    assert not rep["passed"]
    text = " ".join(rep["reasons"])
    assert "entry slippage" in text and "trails the shadow" in text and "hash chain" in text


def _pf(stage, validated=True):
    return Portfolio(stage=stage, strategies=[StrategyEntry(name="asian_breakout", spec_path="strategies/asian_breakout.yaml", validated=validated)])


def test_stage_gate_needs_a_passing_forward_test_and_validated_strategies():
    assert stage_gate(_pf("research"), None)["eligible"]                                           # research -> demo needs no forward test
    assert "no forward-test report" in stage_gate(_pf("demo"), None)["reasons"][0]
    assert not stage_gate(_pf("demo"), {"stage": "demo", "passed": False, "reasons": ["x"]})["eligible"]
    assert stage_gate(_pf("demo"), {"stage": "demo", "passed": True})["next"] == "tiny_live"
    g = stage_gate(_pf("demo", validated=False), {"stage": "demo", "passed": True})
    assert not g["eligible"] and "validation gate" in g["reasons"][0]
    assert not stage_gate(_pf("tiny_live"), {"stage": "demo", "passed": True})["eligible"]         # a demo report does not promote tiny_live
    assert not stage_gate(_pf("live"), None)["eligible"]


def _real_money_ledger(pnls, risk=100.0):
    led = Ledger(":memory:")
    t = pd.Timestamp("2026-06-01 09:00", tz="UTC")
    eq = 10_000.0
    led.snapshot_equity(t, eq, eq, 0)
    for i, p in enumerate(pnls):
        a = t + pd.Timedelta(hours=i + 1)
        led.open_trade(f"P{i}", "s", "XAUUSD", "tiny_live", 1, 0.1, a, 2400.0, 2400.0 - risk / 10, None)
        led.close_trade(f"P{i}", a + pd.Timedelta(minutes=30), 2400.0 + p / 10, "target" if p > 0 else "stop", p)
        eq += p
        led.snapshot_equity(a + pd.Timedelta(minutes=30), eq, eq, 0)
    return led


def test_size_only_steps_up_on_realised_results():
    pf = _pf("tiny_live"); pf.scale = 0.1
    cfg = ScalingConfig(min_trades=40, max_scale=1.0)
    assert recommend_scale(_real_money_ledger([150, -100] * 10), pf, cfg)["action"] == "hold"       # 20 trades: too few
    up = recommend_scale(_real_money_ledger([150, -100] * 25), pf, cfg)
    assert up["action"] == "step_up" and up["recommended"] == pytest.approx(0.125)
    assert recommend_scale(_real_money_ledger([100, -100] * 25), pf, cfg)["action"] == "hold"       # no realised edge
    down = recommend_scale(_real_money_ledger([-100] * 6), pf, cfg)
    assert down["action"] == "step_down" and down["recommended"] == pytest.approx(0.05)
    assert recommend_scale(_real_money_ledger([150, -100] * 25), _pf("demo"), cfg)["action"] == "hold"   # demo results never size real money


# ---------------------------------------------------------------- the real-broker loop, against a simulated venue
def test_live_step_against_a_simulated_venue_matches_the_bar_replay(forward):
    """live_step is the code path a real broker uses: poll closed M1 bars, read the venue's own bars for
    signals, send orders at the quote, pick up stops the venue executed by itself. Driven minute by minute
    against SimVenue it must take the same trades as the bar replay of the same period."""
    from goldsys.live.signals import BrokerBarSignals
    from goldsys.live.simvenue import SimVenue
    _, spec, _, bars = forward
    cost = CostModel(slippage_mean=0.0, slippage_std=0.0)
    start, end = pd.Timestamp("2023-03-01", tz="UTC"), pd.Timestamp("2023-03-11", tz="UTC")

    venue = SimVenue(bars, cost, start=start, slippage=False, manual_clock=True)
    led = Ledger(":memory:")
    slot = Slot(spec.name, spec, BrokerBarSignals(spec, venue.bars, window=3000))
    live = LiveRunner(venue, led, [slot], RiskManager(RiskConfig(**dict(OPEN, max_data_staleness_s=10**9)), cost, store=led), cost, stage="demo", realtime=True)
    live.last_bar_ns = int(venue.bars("M1", 1).index.as_unit("ns").asi8[-1])
    while venue.now() < end:
        venue.advance(60)
        live.live_step(venue.now())
    live.flatten_all("end", venue.now())

    led2 = Ledger(":memory:")
    paper = PaperBroker(cost, slippage=False)
    rep = LiveRunner(paper, led2, make_slots([spec], bars), RiskManager(RiskConfig(**OPEN), cost, store=led2), cost, stage="demo")
    ex = bars[(bars.index >= start) & (bars.index < end)].copy()
    ex["spread"] = cost.spread_at(ex.index, ex["spread"])
    rep.replay(ex)

    a, b = led.trades(), led2.trades()
    assert len(a) >= 5 and len(a) == len(b)
    assert (a["dir"].to_numpy() == b["dir"].to_numpy()).all()
    assert (abs(a["entry_ts"] - b["entry_ts"]) <= pd.Timedelta(minutes=1)).all()
    assert np.allclose(a["entry_px"], b["entry_px"], atol=0.35)           # quote at the close vs next bar's open: only the spread table differs
    assert (a["reason"].to_numpy()[:-1] == b["reason"].to_numpy()[:-1]).all()
    assert abs(a["pnl"].sum() - b["pnl"].sum()) < 0.02 * max(abs(b["pnl"]).sum(), 1.0)
    assert not [e for e in led.events() if e["level"] in ("error", "critical")]
    assert led.verify_chain()["ok"]


def test_lost_exit_is_recovered_from_the_venue_and_second_runner_is_refused(forward, tmp_path):
    """Failure modes seen in rehearsal: an exit fill consumed by someone else, and two runners on one ledger."""
    from goldsys.live.simvenue import SimVenue
    _, spec, _, bars = forward
    cost = CostModel(slippage_mean=0.0, slippage_std=0.0)
    venue = SimVenue(bars, cost, start=pd.Timestamp("2023-03-01 09:00", tz="UTC"), slippage=False, manual_clock=True)
    led = Ledger(tmp_path / "l.sqlite")
    slot = Slot("s", spec, Scripted({}))
    run = LiveRunner(venue, led, [slot], RiskManager(RiskConfig(**dict(OPEN, max_data_staleness_s=10**9)), cost, store=led), cost, stage="demo", realtime=True)
    with pytest.raises(RuntimeError, match="another runner"):
        other = Ledger(tmp_path / "l.sqlite")
        other.set("runner_lock", {"pid": 1, "wall": __import__("time").time()})          # pretend a different live process holds it
        LiveRunner(venue, other, [Slot("s", spec, Scripted({}))], cost=cost, realtime=True)
    run._beat()
    q = venue.quote()
    run._execute(slot, led.add_intent(q.ts, "s", 1, 0.1, 0.1, q.ask, 0.05, None, "accepted"), 1, 0.1, 0.05, float("inf"), q.ts)   # stop 5 cents away: hit on the next bar
    pid = slot.position_id
    venue.advance(180)
    assert venue.poll_fills()[0].position_id == pid                                        # ... and the fill is swallowed by another consumer
    run.live_step(venue.now())
    assert slot.position_id is None and not led.open_trades()
    t = led.trades().iloc[0]
    assert t["reason"] == "stop" and t["pnl"] < 0
    assert any(e["kind"] == "reconcile" and "recovered" in e["message"] for e in led.events())
    assert not run.risk.kill_state().get("active")

    # a position with no trace at all: closed in the ledger as 'lost' (flagged estimate) and the kill switch trips
    run._execute(slot, led.add_intent(q.ts, "s", 1, 0.1, 0.1, q.ask, 500.0, None, "accepted"), 1, 0.1, 500.0, float("inf"), venue.now())
    del venue._pos[slot.position_id]
    run.live_step(venue.now())
    assert led.trades().iloc[1]["reason"] == "lost" and run.risk.kill_state()["active"]
    n = len(led.events(1000))
    for _ in range(5):
        venue.advance(60); run.live_step(venue.now())
    assert len([e for e in led.events(1000) if e["level"] in ("error", "critical")]) <= 2   # no error flood


# ================================================================ trailing stops in the live runner
TRAIL = {"sig_long": True, "stop_dist": 5.0, "target_dist": float("inf"), "trail_dist": 1.0, "trail_act": 0.0}


def test_trailing_stop_in_the_runner_is_hand_checked_and_the_exit_is_labelled_trail():
    r = Rig({0: TRAIL})
    r.bar(n=1)                                   # bar 0: the signal
    for px in (2400.0, 2402.0, 2404.0):          # bar 1 is the entry bar (fill 2400.2, stop 2395.2); each bar's high then pulls the stop up for the NEXT bar
        r.bar(px)
        pos = r.broker.positions()
        assert len(pos) == 1
    # after bar 3: ext = 2404.5, stop = 2403.5. Bar 4 opens at 2401 (gap through the stop): fills at the open, labelled trail.
    assert pos[0].stop_px == pytest.approx(2403.5)
    r.bar(2401.0)
    t = r.ledger.trades().iloc[0]
    assert t["reason"] == "trail" and t["exit_px"] == pytest.approx(2401.0) and t["pnl"] == pytest.approx((2401.0 - 2400.2) * 0.1 * 100 - 0.6)
    assert t["stop_px"] == pytest.approx(2395.2) and t["r_multiple"] == pytest.approx(t["pnl"] / (5.0 * 0.1 * 100))     # R is measured against the stop SET at entry
    assert r.ledger.get(f"trail:{t['position_id']}") is None and r.slot.trail is None


def test_the_trail_waits_for_its_activation_profit_and_never_loosens():
    r = Rig({0: dict(TRAIL, trail_act=3.0)})
    r.bar(n=1)
    for px in (2400.0, 2401.0, 2399.0):                # best price stays under entry + 3: the original stop is untouched
        r.bar(px)
    assert r.broker.positions()[0].stop_px == pytest.approx(2395.2) and not r.slot.trail["trailed"]
    r.bar(2404.5)                                      # high 2405.0 is +4.8 over the fill: trail starts, stop = 2404.0
    assert r.broker.positions()[0].stop_px == pytest.approx(2404.0)
    r.bar(2403.0)                                      # a lower high must not loosen it (and this bar's low 2402.5 stops the trade out at the gap open)
    assert r.ledger.trades().iloc[0]["reason"] == "trail"
    r2 = Rig({0: dict(TRAIL, trail_act=0.0)})
    r2.bar(n=1); r2.bar(2410.0)
    top = r2.broker.positions()[0].stop_px
    r2.bar(2410.0, hl=0.1)
    assert r2.broker.positions()[0].stop_px >= top


def test_short_trade_trails_down_from_the_ask():
    r = Rig({0: dict(TRAIL, sig_long=False, sig_short=True)})
    r.bar(n=1)
    r.bar(2400.0); r.bar(2398.0)                       # fill 2400.0 (bid); ask lows 2399.7 then 2397.7; stop = ask low + 1.0
    assert r.broker.positions()[0].stop_px == pytest.approx(2398.7, abs=1e-9)
    r.bar(2400.0)                                      # ask high 2400.7 touches 2398.7: exits at the gap open ask (2400.2)
    t = r.ledger.trades().iloc[0]
    assert t["reason"] == "trail" and t["dir"] == -1 and t["exit_px"] == pytest.approx(2400.2)


def test_a_trailing_strategy_keeps_its_trail_through_the_approval_queue():
    r = Rig({0: TRAIL}, mode="approve", approval_ttl_s=3600)
    r.bar(n=2)
    iid = r.ledger.intents("pending")[0]["id"]
    control.approve(r.ledger, iid, "me")
    r.bar(2400.0)
    assert r.slot.position_id and r.slot.trail and r.slot.trail["d"] == 1.0
    saved = r.ledger.get(f"trail:{r.slot.position_id}")
    assert saved["stop0"] == pytest.approx(saved["entry"] - 5.0) and "dirty" not in saved


def test_a_strategy_without_a_trail_is_untouched_by_all_this():
    r = Rig({0: dict(sig_long=True, target_dist=float("inf"))})
    r.bar(n=2)
    assert r.slot.trail is None and r.broker.positions()[0].stop_px == pytest.approx(2395.2)
    r.bar(2420.0)
    assert r.broker.positions()[0].stop_px == pytest.approx(2395.2)


# ---------------------------------------------------------------- the real-broker path (live_step) against a simulated venue
def ramp_bars(t0, segments, hl=0.05):
    from goldsys.data.store import BAR_COLS
    rows, px = [], 100.0
    for minutes, step in segments:
        for _ in range(minutes):
            o, c = px, px + step
            rows.append((o, max(o, c) + hl, min(o, c) - hl, c, 0.2, 1.0, 1.0))
            px = c
    return pd.DataFrame(rows, columns=BAR_COLS, index=pd.date_range(t0, periods=len(rows), freq="1min", tz="UTC"))


def venue_rig(bars, *, trail=(1.0, 0.0), stop_dist=3.0, start_after=30, tmp_path=None, venue_cls=None, simulated=True):
    from goldsys.live.simvenue import SimVenue
    cost = CostModel(slippage_mean=0.0, slippage_std=0.0, commission_per_lot_rt=0.0)
    t0 = bars.index[0]
    venue = (venue_cls or SimVenue)(bars, cost, start=t0 + pd.Timedelta(minutes=start_after), slippage=False, manual_clock=True)
    venue.simulated = simulated
    led = Ledger(tmp_path / "l.sqlite" if tmp_path else ":memory:")
    spec = StrategySpec(name="s", timeframe="M15", exit=ExitSpec(trail={"fixed": trail[0]}))
    slot = Slot("s", spec, Scripted({}))
    run = LiveRunner(venue, led, [slot], RiskManager(RiskConfig(**dict(OPEN, max_data_staleness_s=10**9)), cost, store=led), cost, stage="demo")
    run.last_bar_ns = int(venue.bars("M1", 1).index.as_unit("ns").asi8[-1])
    q = venue.quote()
    run._execute(slot, led.add_intent(q.ts, "s", 1, 0.1, 0.1, q.ask, stop_dist, None, "accepted"), 1, 0.1, stop_dist, float("inf"), q.ts, trail=trail)
    return venue, led, slot, run


def step(venue, run, n=1):
    for _ in range(n):
        venue.advance(60)
        run.live_step(venue.now())


T_START = pd.Timestamp("2024-01-02 09:00", tz="UTC")
RAMP = [(30, 0.0), (60, 0.1), (40, -0.5)]


def test_live_step_moves_the_venues_stop_up_the_ramp_and_the_exit_is_a_trail():
    venue, led, slot, run = venue_rig(ramp_bars(T_START, RAMP))
    pid = slot.position_id
    stops = []
    for _ in range(90):
        step(venue, run)
        if venue._pos:
            stops.append(venue._pos[pid].stop_px)
    assert stops[0] == pytest.approx(97.2) or stops[0] > 97.2
    assert all(b >= a - 1e-12 for a, b in zip(stops, stops[1:])) and stops[-1] == pytest.approx(105.05, abs=1e-9)         # follows each new high by 1.0 and never loosens
    assert any(b > a for a, b in zip(stops, stops[1:]))
    step(venue, run, 10)                                                                                           # the drop: the venue stops it out on its own
    t = led.trades().iloc[0]
    assert not venue._pos and t["reason"] == "trail" and t["exit_px"] == pytest.approx(105.05)
    assert t["pnl"] == pytest.approx((105.05 - 100.2) * 0.1 * 100) and t["r_multiple"] == pytest.approx(t["pnl"] / 30.0)
    assert led.get(f"trail:{pid}") is None and slot.trail is None and led.verify_chain()["ok"]


def test_restart_adopts_the_position_with_its_trailing_state_and_keeps_trailing(tmp_path):
    bars = ramp_bars(T_START, RAMP)
    venue, led, slot, run = venue_rig(bars, tmp_path=tmp_path)
    pid = slot.position_id
    step(venue, run, 45)
    before = led.get(f"trail:{pid}")
    assert before and before["trailed"] and before["stop"] > 99.0
    slot2 = Slot("s", slot.spec, Scripted({}))
    run2 = LiveRunner(venue, led, [slot2], RiskManager(RiskConfig(**dict(OPEN, max_data_staleness_s=10**9)), venue.cost, store=led), venue.cost, stage="demo")
    run2.last_bar_ns = run.last_bar_ns
    run2.adopt_positions(venue.now())
    assert slot2.position_id == pid and slot2.trail["stop"] == pytest.approx(before["stop"]) and slot2.trail["ext"] == pytest.approx(before["ext"])
    step(venue, run2, 80)
    t = led.trades().iloc[0]
    assert t["reason"] == "trail" and t["exit_px"] == pytest.approx(105.05)


def test_restart_without_saved_state_says_so_and_leaves_the_venues_stop_alone(tmp_path):
    bars = ramp_bars(T_START, RAMP)
    venue, led, slot, run = venue_rig(bars, tmp_path=tmp_path)
    pid = slot.position_id
    step(venue, run, 45)
    frozen = venue._pos[pid].stop_px
    led.set(f"trail:{pid}", None)                                      # the state is gone
    slot2 = Slot("s", slot.spec, Scripted({}))
    run2 = LiveRunner(venue, led, [slot2], RiskManager(RiskConfig(**dict(OPEN, max_data_staleness_s=10**9)), venue.cost, store=led), venue.cost, stage="demo")
    run2.last_bar_ns = run.last_bar_ns
    run2.adopt_positions(venue.now())
    assert slot2.trail is None and any(e["level"] == "critical" and "no saved trailing state" in e["message"] for e in led.events(50))
    step(venue, run2, 90)
    assert not venue._pos
    t = led.trades().iloc[0]
    assert t["reason"] == "stop" and t["exit_px"] == pytest.approx(frozen)             # it exits at the stop it had: not relabelled, not moved


def test_a_rejected_stop_move_keeps_the_old_stop_says_so_and_is_retried():
    from goldsys.live.simvenue import SimVenue

    class Flaky(SimVenue):
        fail = 3

        def modify_position(self, *a, **k):
            if Flaky.fail > 0:
                Flaky.fail -= 1
                raise BrokerError("retcode 10016 invalid stops")
            return super().modify_position(*a, **k)

    Flaky.fail = 3
    venue, led, slot, run = venue_rig(ramp_bars(T_START, RAMP), venue_cls=Flaky)
    pid = slot.position_id
    step(venue, run, 3)
    assert venue._pos[pid].stop_px == pytest.approx(97.2)                              # still the entry stop: the protection never lapsed
    assert any(e["kind"] == "trail" and "did not accept" in e["message"] for e in led.events(50))
    step(venue, run, 10)
    assert venue._pos[pid].stop_px > 99.0 and slot.trail["dirty"] is False             # retried and applied with the latest value, not a stale one


def test_on_a_real_venue_a_trailed_stop_the_market_is_already_through_closes_at_market():
    """A venue refuses a stop on the wrong side of the price, so when a spike sets the trail above where price has already fallen back to, that IS the exit."""
    from goldsys.data.store import BAR_COLS
    bars = ramp_bars(T_START, [(60, 0.0)])
    bars.iloc[40] = (100.0, 110.0, 99.9, 100.0, 0.2, 1.0, 1.0)                         # spike up and fully back within one bar
    venue, led, slot, run = venue_rig(bars, stop_dist=15.0, simulated=False)
    step(venue, run, 11)
    t = led.trades().iloc[0]
    assert not venue._pos and t["reason"] == "trail" and t["exit_px"] == pytest.approx(100.0)     # at the market (bid 100), not at the 109 the trail computed
    assert any("already through the trailed stop" in e["message"] for e in led.events(50))


def test_mt5_modify_sends_the_right_order_and_skips_a_no_op():
    from types import SimpleNamespace
    from goldsys.live.mt5 import MT5Broker

    class FakeMT5:
        TRADE_ACTION_SLTP, POSITION_TYPE_BUY, TRADE_RETCODE_DONE = 6, 0, 10009

        def __init__(self):
            self.sent, self.retcode = [], 10009
            self.pos = [SimpleNamespace(ticket=77, comment="s", symbol="XAUUSD", type=0, volume=0.1, price_open=2000.0, time=1_700_000_000, sl=1990.0, tp=2040.0)]

        def positions_get(self, ticket=None, symbol=None):
            return [p for p in self.pos if p.ticket == ticket] if ticket is not None else self.pos

        def order_send(self, req):
            self.sent.append(req)
            return SimpleNamespace(retcode=self.retcode, comment="done" if self.retcode == 10009 else "invalid stops")

        def last_error(self):
            return (1, "x")

    fake = FakeMT5()
    b = object.__new__(MT5Broker)
    b.mt5, b.symbol, b.digits, b.point, b.magic, b.offset_s, b._known = fake, "XAUUSD", 2, 0.01, 26111300, 0, {}
    pos = b.modify_position("77", stop_px=1995.123)
    assert fake.sent == [{"action": 6, "symbol": "XAUUSD", "position": 77, "sl": 1995.12, "tp": 2040.0, "magic": 26111300}]      # tp kept, sl rounded to the broker's digits
    assert pos.stop_px == 1995.12 and pos.tgt_px == 2040.0
    b.modify_position("77", stop_px=1990.0)                                                    # same as now: nothing sent (MT5 would answer 'no changes' with an error)
    assert len(fake.sent) == 1
    fake.retcode = 10016
    with pytest.raises(BrokerError, match="rejected"):
        b.modify_position("77", stop_px=1999.0)
    with pytest.raises(BrokerError, match="not found"):
        b.modify_position("999", stop_px=1.0)


def test_modify_through_the_bridge_and_unsupported_venues_say_so():
    from goldsys.live.broker import Broker
    cost = CostModel(slippage_mean=0.0, slippage_std=0.0)
    paper = PaperBroker(cost, slippage=False)
    paper.set_quote(T0, 2400.0, 0.2)
    srv = make_server(paper, "t" * 32, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rb = RemoteBroker(f"http://127.0.0.1:{srv.server_address[1]}", "t" * 32)
        f = rb.open_position("s", 1, 0.1, 5.0, 10.0)
        p = rb.modify_position(f.position_id, 2398.0)
        assert p.stop_px == pytest.approx(2398.0) and p.tgt_px == pytest.approx(2410.2) and rb.positions()[0].stop_px == pytest.approx(2398.0)
        with pytest.raises(BrokerError):
            rb.modify_position("nope", 1.0)
    finally:
        srv.shutdown()

    class Bare(Broker):
        name = "bare"
        connected = quote = account = positions = open_position = close_position = poll_fills = lambda self, *a, **k: None
    with pytest.raises(BrokerError, match="cannot modify"):
        Bare().modify_position("1", 1.0)


def test_dashboard_state_shows_the_stop_now_not_only_the_entry_stop():
    led = Ledger(":memory:")
    led.open_trade("P1", "s", "XAUUSD", "demo", 1, 0.1, T0, 2400.2, 2395.2, None)
    led.open_trade("P2", "s2", "XAUUSD", "demo", 1, 0.1, T0, 2400.2, 2395.2, None)
    led.set("trail:P1", {"stop": 2403.5, "ext": 2404.5})
    st = control.state(led)
    by = {p["position_id"]: p for p in st["positions"]}
    assert by["P1"]["trailing"] and by["P1"]["stop_now"] == 2403.5 and by["P1"]["stop_px"] == 2395.2
    assert not by["P2"]["trailing"] and by["P2"]["stop_now"] == 2395.2
