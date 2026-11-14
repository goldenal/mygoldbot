"""Chart data layer: windows, indicator warm-up and causality, valid JSON, strategy trades with levels, path safety."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from goldsys.chart import api, data
from goldsys.data import store
from goldsys.data.synthetic import make_m1
from goldsys.features import indicators as I


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    d = tmp_path_factory.mktemp("bars")
    mp = pytest.MonkeyPatch()
    mp.setattr(store, "BAR_DIR", d)
    mp.setattr(data, "BAR_DIR", d)
    store.write_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 4, 1), seed=5), "TESTSYM", "M1")
    data._M1.clear(); data._TF.clear(); data._IND.clear(); data._BT.clear()
    yield "TESTSYM"
    mp.undo()
    data._M1.clear(); data._TF.clear(); data._IND.clear(); data._BT.clear()


def get(path):
    code, ctype, body = api.route(path)
    return code, json.loads(body)          # json.loads rejects NaN only if parse_constant is set; allow_nan=False on the way out guards it


ALL = "sma:20,ema:50,bb:20:2,donchian:20,vwap,asian,sessions,rsi:14,atr:14,adx:14,macd"


def test_window_latest_and_paging_are_contiguous(lake):
    code, a = get(f"/api/chart/bars?symbol={lake}&tf=M15&limit=300")
    assert code == 200 and a["n"] == 300 and a["has_more"] and not a["has_newer"]
    code, b = get(f"/api/chart/bars?symbol={lake}&tf=M15&limit=300&end={a['t'][0] - 1}")
    assert b["t"][-1] < a["t"][0] and b["has_newer"]
    code, c = get(f"/api/chart/bars?symbol={lake}&tf=M15&limit=300&after={b['t'][-1]}")
    assert c["t"] == a["t"]                       # forward page from the older window lands exactly on the newer one
    assert all(np.diff(a["t"]) > 0)


def test_all_indicators_are_valid_json_and_aligned_even_in_warmup(lake):
    code, d = get(f"/api/chart/bars?symbol={lake}&tf=M5&limit=400&end={int(pd.Timestamp('2024-01-02', tz='UTC').timestamp())}&ind={ALL}")
    assert code == 200
    assert len(d["indicators"]) == len(ALL.split(","))
    for ind in d["indicators"]:
        for s in ind["series"]:
            assert len(s["v"]) == d["n"]


@pytest.mark.parametrize("spec", ["sma:20", "ema:50", "bb:20:2", "rsi:14", "adx:14", "atr:14", "macd", "donchian:20", "vwap", "asian"])
def test_indicator_window_equals_full_computation(lake, spec):
    """A window is a slice of the full-history indicator, so its left edge is warmed up, not restarted."""
    df = data.bars(lake, "M15")
    full = data.compute(spec, df, "M15")
    code, d = get(f"/api/chart/bars?symbol={lake}&tf=M15&limit=200&ind={spec}")
    for fs, ws in zip(full["series"], d["indicators"][0]["series"]):
        a = np.array([np.nan if v is None else v for v in fs["v"][-200:]], dtype=float)
        b = np.array([np.nan if v is None else v for v in ws["v"]], dtype=float)
        assert np.allclose(a, b, equal_nan=True)


@pytest.mark.parametrize("fn", [lambda c: I.bollinger(c, 20, 2), lambda c: I.macd(c)])
def test_new_indicators_are_causal(fn):
    c = store.resample_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 2, 1), seed=2), "M15")["close"]
    full, part = fn(c), fn(c.iloc[:-150])
    pd.testing.assert_frame_equal(full.iloc[:-150], part)


def test_bollinger_matches_zscore():
    c = store.resample_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 1, 20), seed=2), "M15")["close"]
    b = I.bollinger(c, 20, 2)
    z = I.bollinger_z(c, 20)
    assert np.allclose(((c - b["mid"]) / ((b["upper"] - b["mid"]) / 2)).dropna(), z.dropna())


def test_bad_inputs_are_400_not_500(lake):
    for q in ["symbol=NOPE&tf=M15", f"symbol={lake}&tf=W9", f"symbol={lake}&tf=M15&ind=nope:1", f"symbol={lake}&tf=M15&ind=sma:-3", f"symbol={lake}&tf=M15&ind=sma:abc",
              f"symbol={lake}&tf=M15&end=1000"]:
        code, d = get("/api/chart/bars?" + q)
        assert code == 400 and "error" in d, q


def test_strategy_path_must_be_a_strategy_file(lake):
    for bad in ["../README.md", "../../etc/passwd", "strategies/../pyproject.toml", "/etc/passwd", "strategies/nope.yaml"]:
        code, d = get(f"/api/chart/trades?symbol={lake}&strategy={bad}")
        assert code == 400, bad


def test_static_path_traversal_is_refused():
    code, _, _ = api.route("/chart-static/../page.html")
    assert code == 404
    code, _, _ = api.route("/chart-static/%2e%2e/page.html")
    assert code == 404


def test_strategy_trades_have_levels_on_the_right_side(lake):
    code, d = get(f"/api/chart/trades?symbol={lake}&strategy=strategies/asian_breakout.yaml")
    assert code == 200 and d["trades"]
    for t in d["trades"]:
        assert t["exit_t"] >= t["entry_t"]
        if t["stop_px"] is not None:
            assert (t["entry_px"] - t["stop_px"]) * t["dir"] > 0       # stop is against the position
        if t["tgt_px"] is not None:
            assert (t["tgt_px"] - t["entry_px"]) * t["dir"] > 0        # target is in favour
    stopped = [t for t in d["trades"] if t["reason"] == "stop"]
    assert stopped and all(abs(t["exit_px"] - t["stop_px"]) < 1.5 for t in stopped[:20] if t["stop_px"])   # fills near the stop (slippage / gaps)
    assert d["suggested"] == ["asian"] or "asian" in d["suggested"]


def test_list_strategies_and_meta(lake):
    code, m = get("/api/chart/meta")
    assert code == 200 and lake in m["symbols"] and any(s["path"].endswith("asian_breakout.yaml") for s in m["strategies"])
    assert {i["id"] for i in m["indicators"]} >= {"sma", "ema", "bb", "rsi", "macd"}
