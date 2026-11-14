"""Hybrid replay: no look-ahead, decisions, hidden outcomes, blind mode, persistence, report, and the HTTP write guard."""
import datetime as dt
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest
import yaml

from goldsys.chart import data as chartdata
from goldsys.dashboard.server import make_dashboard
from goldsys.data import store
from goldsys.data.synthetic import make_m1
from goldsys.replay import report, session, sim

SPEC_REV = {"name": "x_rev", "timeframe": "M15", "entry_long": {"op": "cross_above", "args": ["close", "ema_20"]}, "entry_short": "mirror",
            "exit": {"stop": {"atr_mult": 3.0, "atr_n": 14}, "target": None, "max_bars": 400}, "sizing": {"type": "fixed", "lots": 0.05, "max_lots": 1.0}}
SPEC_TRAIL = {"name": "x_trail", "timeframe": "M15", "entry_long": {"op": "cross_above", "args": ["close", "ema_20"]}, "entry_short": "mirror",
              "exit": {"stop": {"atr_mult": 2.0, "atr_n": 14}, "target": None, "max_bars": 300, "trail": {"atr_mult": 1.0, "atr_n": 14}}, "sizing": {"type": "fixed", "lots": 0.05}}
SPEC = {"name": "x_cross", "timeframe": "M15", "entry_long": {"op": "cross_above", "args": ["close", "ema_20"]}, "entry_short": "mirror",
        "exit": {"stop": {"atr_mult": 1.5, "atr_n": 14}, "target": {"r_mult": 2.0}, "max_bars": 40}, "sizing": {"type": "fixed", "lots": 0.05}}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("replay")
    (root / "bars").mkdir(); (root / "strat").mkdir()
    (root / "strat" / "x_cross.yaml").write_text(yaml.safe_dump(SPEC))
    (root / "strat" / "x_rev.yaml").write_text(yaml.safe_dump(SPEC_REV))
    (root / "strat" / "x_trail.yaml").write_text(yaml.safe_dump(SPEC_TRAIL))
    mp = pytest.MonkeyPatch()
    mp.setattr(store, "BAR_DIR", root / "bars"); mp.setattr(chartdata, "BAR_DIR", root / "bars")
    mp.setattr(chartdata, "STRATEGY_DIRS", (root / "strat",)); mp.setattr(chartdata, "ROOT", root)
    mp.setattr(session, "REPLAY_DB", root / "replay.sqlite")
    mp.setattr(session, "CALENDAR_FILE", root / "events.csv")
    store.write_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 4, 1), seed=21), "RP", "M1")
    for d in (chartdata._M1, chartdata._TF, chartdata._IND, chartdata._BT, sim._MARKETS, sim._CTX, session._SESSIONS):
        d.clear()
    yield {"sym": "RP", "rel": "strat/x_cross.yaml", "rev": "strat/x_rev.yaml", "trail": "strat/x_trail.yaml", "root": root}
    mp.undo()
    for d in (chartdata._M1, chartdata._TF, chartdata._IND, chartdata._BT, sim._MARKETS, sim._CTX, session._SESSIONS):
        d.clear()


def go(s, n):
    """Advance n M1 bars, skipping any proposal the clock stops at (the session will not move past an undecided one)."""
    st = s.advance(n)
    for p in st["pending"]:
        st = s.decide(p["id"], "skip", "other")
    return st


def fresh(env, blind=False, start="2024-01-15", rel=None):
    return session.new_session(env["sym"], [rel or env["rel"]], start, blind)


def test_clock_stops_at_each_proposal_and_refuses_to_move_until_decided(env):
    s = fresh(env)
    st = s.advance(to_signal=True)
    assert st["status"] == "awaiting" and len(st["pending"]) == 1
    with pytest.raises(session.ReplayError, match="pending"):
        s.advance(1)
    p = st["pending"][0]
    assert p["stop_est"] < p["entry_est"] if p["dir"] > 0 else p["stop_est"] > p["entry_est"]
    st = s.decide(p["id"], "skip", "chop_range")
    assert st["status"] == "ready" and st["decisions"] == {"take": 0, "skip": 1}


@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4", "D1"])
def test_page_never_receives_a_bar_after_the_clock(env, tf):
    s = fresh(env)
    for _ in range(6):
        go(s, 37)
        w = s.window(tf, 300, None, ["ema:20", "rsi:14"])
        now = s.state()["t"]
        assert max(w["t"]) <= now, "a bar opens after the clock"
        # the last bar is the PARTIAL bar the clock is inside: its range must be built only from M1 bars up to the cursor
        tf_s = chartdata.TF_SECONDS[tf]
        real_now = s._t(s.cursor)
        b0 = real_now - real_now % tf_s
        lo = np.searchsorted(s.M.ns, b0 * 10**9)
        seg = s.M.frame.iloc[lo:s.cursor + 1]
        assert w["h"][-1] == pytest.approx(seg["high"].max(), abs=1e-3) and w["l"][-1] == pytest.approx(seg["low"].min(), abs=1e-3)
        assert w["c"][-1] == pytest.approx(seg["close"].iloc[-1], abs=1e-3)
        assert all(len(i["series"][0]["v"]) == w["n"] for i in w["indicators"])


def test_older_paging_returns_only_earlier_complete_bars(env):
    s = fresh(env)
    go(s, 500)
    w = s.window("M15", 200, None, [])
    older = s.window("M15", 200, w["t"][0] - 1, [])
    assert older["t"][-1] < w["t"][0] and older["has_more"]


def test_outcome_is_hidden_until_the_clock_passes_it_then_matches_the_simulator(env):
    s = fresh(env)
    p = s.advance(to_signal=True)["pending"][0]
    st = s.decide(p["id"], "take", "trend_aligned")
    assert st["closed"] == [] and len(st["queued"]) == 1                     # not filled yet: fills at the NEXT bar's open
    row = s.pending() or s._rows("AND decision='take'")
    r = s._rows("AND decision='take'")[0]
    assert r["fill_ts"] > s._t(s.cursor)
    go(s, 1)
    assert len(s.state()["open"]) + len(s.state()["closed"]) == 1
    while not s.state()["closed"] and not s.state()["at_end"]:                # run the clock to the trade's exit, declining nothing that appears
        st = s.advance(200)
        for q in st["pending"]:
            s.decide(q["id"], "skip", "gut_feel")
        if s.cursor >= s._idx(r["exit_ts"]):
            break
    st = s.state()
    assert len(st["closed"]) == 1
    c = st["closed"][0]
    j = s._idx(p["t"] + s.shift)
    ref = sim.simulate_signal(s.M, s.ctx[env["rel"]], j, p["dir"], p["stop_dist"], p["target_dist"], p["lots"])
    assert c["pnl"] == pytest.approx(ref["pnl"], abs=0.01) and c["reason"] == ref["reason"]
    assert st["equity"] == pytest.approx(10_000 + ref["pnl"], abs=0.02)


def test_busy_strategy_does_not_propose_until_its_trade_has_exited(env):
    s = fresh(env)
    p = s.advance(to_signal=True)["pending"][0]
    s.decide(p["id"], "take", "good_rr")
    exit_ts = s._rows("AND decision='take'")[0]["exit_ts"]
    nxt = s.advance(to_signal=True)
    if nxt["pending"]:
        assert s._real(nxt["pending"][0]["t"]) >= exit_ts


def test_decision_validation(env):
    s = fresh(env)
    p = s.advance(to_signal=True)["pending"][0]
    for bad in [("take", "news_risk"), ("skip", "good_rr"), ("maybe", "other"), ("take", "")]:
        with pytest.raises(session.ReplayError):
            s.decide(p["id"], *bad)
    with pytest.raises(session.ReplayError):
        s.decide(99999, "skip", "other")


def test_report_counts_only_outcomes_the_clock_has_passed_until_finished(env):
    s = fresh(env, start="2024-02-01")
    n = 0
    while n < 8 and not s.state()["at_end"]:
        st = s.advance(to_signal=True)
        for p in st["pending"]:
            s.decide(p["id"], "take" if n % 2 else "skip", "good_rr" if n % 2 else "chop_range")
            n += 1
        if s.cursor > len(s.M.ns) - 500:
            break
    rep = report.analyse([s.id])
    assert rep["decisions"] >= 4
    assert rep["unresolved"] >= 1 and rep["resolved"] < rep["decisions"]    # the latest decisions' outcomes lie in the future of the clock
    s.finish()
    rep2 = report.analyse([s.id])
    assert rep2["resolved"] == rep2["decisions"] and rep2["unresolved"] == 0
    assert rep2["taken"]["n"] + rep2["skipped"]["n"] == rep2["resolved"]
    assert "edge" in rep2 and any("Fewer than" in w for w in rep2["warnings"])      # tiny sample is flagged, not interpreted
    assert "take:good_rr" in rep2["by_tag"] and "skip:chop_range" in rep2["by_tag"]
    assert "Hybrid replay report" in report.to_markdown(rep2)


def test_blind_mode_hides_the_date_but_not_weekday_or_time_of_day(env):
    s = fresh(env, blind=True)
    assert s.shift % session.WEEK == 0 and s.shift > 0
    st = go(s, 100)
    real = s._t(s.cursor)
    shown = st["t"]
    assert shown != real and (real - shown) == s.shift
    assert pd.Timestamp(shown, unit="s").dayofweek == pd.Timestamp(real, unit="s").dayofweek
    assert shown % 86400 == real % 86400
    assert pd.Timestamp(shown, unit="s").year < 2010
    w = s.window("H1", 100, None, [])
    assert max(w["t"]) <= shown and all(pd.Timestamp(t, unit="s").year < 2010 for t in w["t"])
    assert pd.Timestamp(w["first"], unit="s").year < 2010


def test_session_survives_a_restart(env):
    s = fresh(env)
    p = s.advance(to_signal=True)["pending"][0]
    cur = s.cursor
    session._SESSIONS.clear()
    s2 = session.get_session(s.id)
    assert s2.cursor == cur and [x["id"] for x in s2.state()["pending"]] == [p["id"]]
    s2.decide(p["id"], "skip", "other")
    session._SESSIONS.clear()
    assert session.get_session(s.id).state()["decisions"]["skip"] == 1


def test_start_validation(env):
    for start in ["2024-01-01", "not a date", "2030-01-01"]:
        with pytest.raises(session.ReplayError):
            fresh(env, start=start)
    with pytest.raises(session.ReplayError):
        session.new_session(env["sym"], ["../README.md"], "2024-02-01")
    with pytest.raises(session.ReplayError):
        session.new_session(env["sym"], [], "2024-02-01")


def test_http_writes_need_the_token_and_a_local_host(env):
    srv = make_dashboard(None, None, port=0, read_only=True)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        html = urllib.request.urlopen(base + "/chart").read().decode()
        assert "__TOKEN__" not in html and srv.token in html                  # the page carries this process's token

        def post(path, body, token=None, host=None):
            h = {"Content-Type": "application/json"}
            if token:
                h["X-Goldsys-Token"] = token
            if host:
                h["Host"] = host
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=h, method="POST")
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        body = {"symbol": env["sym"], "strategies": [env["rel"]], "start": "2024-02-01"}
        assert post("/api/replay/new", body)[0] == 401                          # no token
        assert post("/api/replay/new", body, token="wrong")[0] == 401
        assert post("/api/replay/new", body, token=srv.token, host="evil.example.com")[0] == 403
        code, st = post("/api/replay/new", body, token=srv.token)
        assert code == 200 and st["status"] == "ready"
        code, st = post("/api/replay/step", {"session": st["id"], "to_signal": True}, token=srv.token)
        assert code == 200 and st["status"] == "awaiting"
        got = json.loads(urllib.request.urlopen(f"{base}/api/replay/state?session={st['id']}").read())
        assert got["pending"][0]["id"] == st["pending"][0]["id"]
        assert json.loads(urllib.request.urlopen(f"{base}/api/replay/bars?session={st['id']}&tf=M15&limit=50").read())["n"] == 50
    finally:
        srv.shutdown()


# ================================================================ own levels, manual management, reversals, trailing, news

def take_first(s, tag="good_rr", **kw):
    p = s.advance(to_signal=True)["pending"][0]
    return p, s.decide(p["id"], "take", tag, **kw)


def run_until(s, pred, limit=400):
    """Advance in 25-bar steps, skipping new proposals, until pred(state) or the step limit."""
    st = s.state()
    for _ in range(limit):
        if pred(st):
            return st
        st = s.advance(25)
        for p in st["pending"]:
            st = s.decide(p["id"], "skip", "other")
    raise AssertionError("condition never reached")


def test_take_with_own_stop_target_and_size(env):
    s = fresh(env)
    p = s.advance(to_signal=True)["pending"][0]
    d, e = p["dir"], p["entry_est"]
    for bad in [dict(stop_px=e - d * 0.01), dict(stop_px=e + d * 5), dict(tgt_px=e - d * 5), dict(tgt_px=e + d * 0.01), dict(lots=0), dict(lots=99)]:
        with pytest.raises(session.ReplayError):
            s.decide(p["id"], "take", "good_rr", **bad)
    with pytest.raises(session.ReplayError, match="only apply when you take"):
        s.decide(p["id"], "skip", "other", stop_px=e - d * 5)
    stop, tgt = round(e - d * p["stop_dist"] * 0.6, 2), round(e + d * p["stop_dist"] * 3.0, 2)
    s.decide(p["id"], "take", "good_rr", stop_px=stop, tgt_px=tgt, lots=0.02)
    row = s._rows("AND decision='take'")[0]
    assert row["a_modified"] == 1 and row["a_lots"] == 0.02 and row["lots"] == 0.05
    assert row["a_stop_px"] == stop and row["a_tgt_px"] == tgt
    assert abs(row["entry_px"] - row["stop_px"]) == pytest.approx(p["stop_dist"], abs=0.01)        # the shadow keeps the STRATEGY'S stop and size
    run_until(s, lambda st: st["closed_n"] >= 1)
    c = s.state()["closed"][0]
    f = s._idx(row["fill_ts"])
    ref = sim.simulate(s.M, s.ctx[env["rel"]], f, p["dir"], 0.02, stop_px=stop, tgt_px=tgt, trail_dist=None)
    assert c["pnl"] == pytest.approx(ref["pnl"], abs=0.01) and c["lots"] == 0.02 and c["modified"] is True
    assert row["pnl"] != pytest.approx(ref["pnl"])                                       # the shadow outcome is a different (full-size) trade


def test_manual_stop_target_and_close_on_an_open_trade(env):
    s = fresh(env, rel=env["rev"])
    p, st = take_first(s, "trend_aligned")
    st = run_until(s, lambda st: len(st["open"]) == 1)
    t = st["open"][0]
    d, px, gap = t["dir"], st["price"], t["min_gap"]
    assert t["trailing"] is False and t["stop_px"] == pytest.approx(s._rows("AND decision='take'")[0]["a_stop_px"], abs=0.01)
    for bad in [dict(stop_px=px + d * 3), dict(tgt_px=px - d * 3), dict()]:
        with pytest.raises(session.ReplayError):
            s.manage(t["id"], **bad)
    with pytest.raises(session.ReplayError):
        s.manage(99999, close=True)
    new_stop = round(px - d * (gap + 1.0), 2)
    st = s.manage(t["id"], stop_px=new_stop, clear_target=True)
    t2 = st["open"][0]
    assert t2["stop_px"] == pytest.approx(new_stop) and t2["tgt_px"] is None and t2["modified"] is True
    row = s._rows("AND decision='take'")[0]
    ref = sim.simulate(s.M, s.ctx[env["rev"]], s._idx(row["fill_ts"]), d, row["a_lots"], stop_px=row["a_stop_px"], tgt_px=None,
                       adjust=[{"i": s.cursor + 1, "stop_px": new_stop, "clear_target": True}])
    assert row["a_exit_ts"] == s._t(ref["exit_i"]) and row["a_pnl"] == pytest.approx(ref["pnl"])
    assert st["closed_n"] == 0                                                            # the changed outcome is still in the clock's future
    st = s.manage(t["id"], close=True)                                                     # close at the next open
    assert st["open"][0]["closing"] is True
    row = s._rows("AND decision='take'")[0]
    assert row["a_reason"] == "manual" and row["a_exit_ts"] == s._t(s.cursor + 1)
    st = s.advance(1) if not s.pending() else st
    assert any(c["reason"] == "manual" for c in st["closed"])
    with pytest.raises(session.ReplayError, match="not open"):
        s.manage(t["id"], stop_px=px)


def test_trailing_strategy_ratchets_in_the_replay_and_a_manual_stop_switches_it_off(env):
    s = fresh(env, rel=env["trail"])
    take_first(s, "good_rr")
    st = run_until(s, lambda st: len(st["open"]) == 1 or st["closed_n"] > 0)
    if not st["open"]:
        pytest.skip("trade closed before it could be inspected")
    t = st["open"][0]
    assert t["trailing"] is True
    stops = [t["stop_px"]]
    for _ in range(40):
        st = s.advance(5)
        for p in st["pending"]:
            st = s.decide(p["id"], "skip", "other")
        if not st["open"]:
            break
        stops.append(st["open"][0]["stop_px"])
    d = t["dir"]
    assert all((b - a) * d >= -1e-9 for a, b in zip(stops, stops[1:])), "the trail must only ever tighten"
    if len(stops) > 3 and st["open"]:
        px = st["price"]
        st = s.manage(t["id"], stop_px=round(px - d * (t["min_gap"] + 2), 2))
        assert st["open"][0]["trailing"] is False


def test_reversal_is_offered_only_for_the_opposite_signal_and_closes_the_old_trade_at_the_new_fill(env):
    s = fresh(env, rel=env["rev"])
    p0, _ = take_first(s)
    old_id = s._rows("AND decision='take'")[0]["id"]
    seen_same_dir = 0
    rev = None
    for _ in range(80):
        st = s.advance(to_signal=True)
        if not st["pending"]:
            break
        p = st["pending"][0]
        if p["kind"] == "reverse":
            rev = p
            break
        s.decide(p["id"], "skip", "other")
    assert rev is not None and rev["dir"] == -p0["dir"] and rev["reverses"]["id"] == old_id
    old_before = s._row(old_id)
    s.decide(rev["id"], "take", "momentum_confirms")
    old, new = s._row(old_id), s._rows("AND decision='take'")[-1]
    assert old["a_reason"] == "reverse" and old["a_exit_ts"] == new["fill_ts"] and old["a_exit_ts"] > s._t(s.cursor) - 1
    assert old["a_exit_ts"] < old_before["a_exit_ts"]                                      # it really was cut short
    assert new["kind"] == "reverse" and new["dir"] == -old["dir"]
    # engine semantics for the exit: at the next open, adverse slippage
    f = s._idx(new["fill_ts"])
    exp = s.M.open[f] - s.M.slip[f] if old["dir"] > 0 else s.M.open[f] + s.M.spread[f] + s.M.slip[f]
    assert old["a_exit_px"] == pytest.approx(exp)
    assert new["inc_pnl"] is not None


def test_skipping_a_reversal_leaves_the_old_trade_alone_and_still_scores_it(env):
    s = fresh(env, rel=env["rev"])
    take_first(s)
    old = s._rows("AND decision='take'")[0]
    for _ in range(80):
        st = s.advance(to_signal=True)
        if not st["pending"]:
            pytest.skip("no reversal signal in this data")
        p = st["pending"][0]
        if p["kind"] == "reverse":
            s.decide(p["id"], "skip", "chop_range")
            break
        s.decide(p["id"], "skip", "other")
    row = [r for r in s._rows() if r["kind"] == "reverse"][0]
    assert s._row(old["id"])["a_exit_ts"] == old["a_exit_ts"] and row["inc_pnl"] is not None and row["decision"] == "skip"
    assert row["res_ts"] >= max(row["exit_ts"], old["a_exit_ts"])                           # not scored until BOTH paths have played out


def test_no_proposal_for_the_same_direction_while_in_a_trade(env):
    s = fresh(env, rel=env["rev"])
    p0, _ = take_first(s)
    for _ in range(60):
        st = s.advance(to_signal=True)
        if not st["pending"]:
            break
        p = st["pending"][0]
        open_now = [o for o in st["open"]]
        if open_now and p["kind"] == "entry":
            raise AssertionError("an entry was proposed while a trade of this strategy was open")
        assert not (open_now and p["dir"] == open_now[0]["dir"])
        s.decide(p["id"], "skip", "other")


def write_calendar(env, rows):
    (env["root"] / "events.csv").write_text("ts_utc,name,tier\n" + "".join(f"{t},{n},{k}\n" for t, n, k in rows))
    session._CAL.clear()


def test_news_proximity_is_recorded_and_the_schedule_is_not_look_ahead(env):
    write_calendar(env, [("2023-05-01 12:30", "US CPI", 1)])                                # a calendar that does not cover January 2024
    s = fresh(env)
    p = s.advance(to_signal=True)["pending"][0]
    assert p["news"] is None and s.state()["upcoming"] == []                                # unknown is not "no news"
    s.decide(p["id"], "skip", "other")
    sig = s._real(p["t"])
    t = lambda m: (pd.Timestamp(sig, unit="s") + pd.Timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M")        # noqa: E731
    write_calendar(env, [("2023-05-01 12:30", "old", 1), (t(-180), "US NFP", 1), (t(20), "FOMC Statement", 1), (t(300), "US PPI", 2), ("2024-12-31 12:30", "far", 1)])
    s2 = fresh(env)
    p2 = s2.advance(to_signal=True)["pending"][0]
    now = s2.state()["t"]
    if abs(p2["t"] - p["t"]) > 3600:
        pytest.skip("the two sessions stopped at different signals")
    assert p2["news"]["name"] == "FOMC Statement" or p2["news"]["name"] == "US NFP"
    assert any(e["name"] == "FOMC Statement" and 0 < e["in_min"] <= 24 * 60 for e in s2.state()["upcoming"])
    ev = s2.events_window(now - 86400, now + 86400)
    assert {e["name"] for e in ev} == {"US NFP"} and all(e["t"] <= now for e in ev)         # events in the clock's future are not drawn
    write_calendar(env, [])
    (env["root"] / "events.csv").unlink()
    session._CAL.clear()


def test_old_database_gains_the_new_columns(tmp_path, monkeypatch):
    import sqlite3
    old = tmp_path / "old.sqlite"
    con = sqlite3.connect(old)
    con.executescript("""CREATE TABLE sessions(id TEXT PRIMARY KEY, created TEXT, symbol TEXT, strategies TEXT, start_ts INTEGER, cursor_ts INTEGER, blind INTEGER, shift INTEGER, equity0 REAL, finished INTEGER DEFAULT 0, source TEXT);
    CREATE TABLE decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, seq INTEGER, strategy TEXT, dir INTEGER, signal_ts INTEGER, stop_dist REAL, target_dist REAL, lots REAL, ref_px REAL, decision TEXT, tag TEXT, note TEXT, decided_wall TEXT, fill_ts INTEGER, exit_ts INTEGER, entry_px REAL, exit_px REAL, stop_px REAL, tgt_px REAL, reason TEXT, pnl REAL, r REAL);
    INSERT INTO decisions(session,seq,strategy,dir,signal_ts) VALUES('x',1,'s',1,5);""")
    con.commit(); con.close()
    monkeypatch.setattr(session, "REPLAY_DB", old)
    with session._db() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(decisions)")}
        assert {"kind", "ref_id", "a_pnl", "a_lots", "news_min", "res_ts", "inc_pnl"} <= cols
        assert c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


def test_report_sections_for_management_reversals_and_news(env):
    write_calendar(env, [("2024-01-10 12:30", "x", 1), ("2024-03-30 12:30", "y", 1)])
    s = fresh(env, rel=env["rev"], start="2024-01-20")
    n_edit = 0
    for _ in range(60):
        st = s.advance(to_signal=True)
        if not st["pending"]:
            break
        p = st["pending"][0]
        if p["kind"] == "entry" and n_edit < 4:
            s.decide(p["id"], "take", "good_rr", stop_px=round(p["entry_est"] - p["dir"] * p["stop_dist"] * 0.7, 2), lots=0.03); n_edit += 1
        else:
            s.decide(p["id"], "skip", "chop_range")                     # reversals are skipped here so the edited trades run to their own exits
        if s.state()["at_end"]:
            break
    s.finish()
    rep = report.analyse([s.id])
    assert rep["resolved"] == rep["decisions"]
    assert "management" in rep and rep["management"]["n"] >= 1 and rep["management"]["interpretable"] is False
    assert "entry" in rep["by_kind"] and "reverse" in rep["by_kind"]
    assert "by_news" in rep
    md = report.to_markdown(rep)
    assert "Your levels, size and management" in md and "Near high-impact news" in md
    write_calendar(env, [])
    (env["root"] / "events.csv").unlink()
    session._CAL.clear()
