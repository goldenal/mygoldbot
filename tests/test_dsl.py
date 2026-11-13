import datetime as dt
from goldsys.data.synthetic import make_m1
from goldsys.dsl.spec import StrategySpec, mirror, substitute_params
from goldsys.dsl.compile import backtest_spec
from goldsys.engine.costs import CostModel


def test_substitute_and_mirror():
    e = {"op": "cross_above", "args": ["close", {"op": "+", "args": ["asian_high", {"op": "*", "args": ["atr_$n", "$k"]}]}]}
    s = substitute_params(e, {"n": 14, "k": 0.5})
    assert s["args"][1]["args"][1]["args"] == ["atr_14", 0.5]
    m = mirror(s)
    assert m["op"] == "cross_below" and m["args"][1]["op"] == "-" and m["args"][1]["args"][0] == "asian_low"


def test_yaml_roundtrip_and_hash():
    spec = StrategySpec.from_yaml("strategies/asian_breakout.yaml")
    again = StrategySpec.from_dict(spec.to_dict())
    assert again.hash() == spec.hash()
    assert spec.with_params(k=0.75).hash() != spec.hash()
    assert spec.complexity() > 5


def test_end_to_end_backtest_runs():
    bars = make_m1(dt.date(2023, 1, 1), dt.date(2023, 7, 1), seed=5)
    for f in ("asian_breakout", "vol_momentum", "vwap_reversion"):
        res = backtest_spec(StrategySpec.from_yaml(f"strategies/{f}.yaml"), bars, CostModel(), cache_key=None)
        assert len(res.equity) > 0
        assert res.trades["pnl"].notna().all()
