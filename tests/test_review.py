"""Phase 5 monthly review: retire on drift / failed refresh, propose gated replacements, re-allocate."""
import datetime as dt

import pandas as pd
import pytest

from goldsys.data.synthetic import make_m1
from goldsys.live.ledger import Ledger
from goldsys.portfolio.build import build_portfolio
from goldsys.research.review import ReviewConfig, monthly_review, render_markdown
from goldsys.search.registry import Registry
from goldsys.validation.gate import GateConfig


@pytest.fixture(scope="module")
def bars():
    return make_m1(dt.date(2021, 1, 1), dt.date(2024, 1, 1), seed=5)


def test_portfolio_build_requires_the_gate_unless_told_otherwise(bars):
    pf, rep = build_portfolio(["strategies/vol_momentum.yaml"], bars)
    assert pf.strategies == [] and rep["rejected"][0]["name"] == "vol_momentum"
    pf, rep = build_portfolio(["strategies/vol_momentum.yaml", "strategies/asian_breakout.yaml"], bars, allow_unvalidated=True)
    assert {s.name for s in pf.strategies} == {"vol_momentum", "asian_breakout"}
    assert all(not s.validated and s.drift["window"] == 30 and s.drift["p05_r"] < 0 and s.spec_hash for s in pf.strategies)


def test_monthly_review_retires_on_live_drift_and_proposes_a_gated_replacement(bars, tmp_path):
    pf, _ = build_portfolio(["strategies/vol_momentum.yaml", "strategies/asian_breakout.yaml"], bars, allow_unvalidated=True)
    led = Ledger(tmp_path / "l.sqlite")
    t = pd.Timestamp("2023-11-01 09:00", tz="UTC")
    for i in range(30):                                   # thirty straight -1R trades for vol_momentum in the live ledger
        led.open_trade(f"P{i}", "vol_momentum", "XAUUSD", "demo", 1, 0.1, t + pd.Timedelta(hours=i), 2400.0, 2390.0, None)
        led.close_trade(f"P{i}", t + pd.Timedelta(hours=i, minutes=30), 2390.0, "stop", -100.0)
    cfg = ReviewConfig(wfa_max_points=4, tier1_trials=3, retire_if_oos_sharpe_below=-99, retire_if_last_windows_negative=0, candidates_dir=tmp_path / "cand",
                       gate=GateConfig(wfa_max_points=4, cpcv_max_points=3, pbo_trials=4, run_plateau=False, run_cpcv=False))
    reg = Registry(tmp_path / "reg.sqlite")
    before = pf.get("asian_breakout").weight
    new, rep = monthly_review(pf, bars, led, reg, cfg=cfg, verbose=False)
    by = {r["name"]: r for r in rep["reviews"]}
    assert by["vol_momentum"]["decision"] == "retire" and "drift" in by["vol_momentum"]["reasons"][0]
    assert by["asian_breakout"]["decision"] == "keep"
    e = new.get("vol_momentum")
    assert e.status == "retired" and e.weight == 0.0 and e.mode == "off"
    assert pf.get("vol_momentum").status == "active"       # the input portfolio is untouched: the review only proposes
    assert new.get("asian_breakout").weight != before      # survivors are re-allocated
    assert reg.n_trials() >= 3                             # every replacement trial is counted for the deflated Sharpe
    for c in rep["replacements"]:                          # synthetic data has no edge: proposals must fail the gate and stay out
        assert not c["passed"] and new.get(c["name"]) is None
    assert new.history[-1]["action"] == "monthly_review" and "retire" in render_markdown(rep)


def test_monthly_review_retires_when_the_refreshed_walk_forward_fails(bars):
    pf, _ = build_portfolio(["strategies/asian_breakout.yaml"], bars, allow_unvalidated=True)
    cfg = ReviewConfig(wfa_max_points=4, retire_if_oos_sharpe_below=99.0, propose_replacements=False)
    new, rep = monthly_review(pf, bars, None, Registry(":memory:"), cfg=cfg, verbose=False)
    assert rep["reviews"][0]["decision"] == "retire" and "walk-forward" in rep["reviews"][0]["reasons"][0]
    assert new.active() == []
