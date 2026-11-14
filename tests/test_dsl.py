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


def test_gp_operators_preserve_types():
    import random
    from goldsys.search.genetic import random_bool, crossover, mutate, _is_bool
    rng = random.Random(3)

    def well_typed(e):
        if isinstance(e, dict):
            op, args = e["op"], e["args"]
            if op in ("and", "or", "not"):
                return all(_is_bool(a) and well_typed(a) for a in args)
            if op in ("gt", "lt", "ge", "le", "cross_above", "cross_below", "between"):
                return all((not _is_bool(a)) and well_typed(a) for a in args)
            return all(well_typed(a) for a in args)
        return True

    pop = [random_bool(rng) for _ in range(50)]
    for _ in range(500):
        c = mutate(crossover(rng.choice(pop), rng.choice(pop), rng), rng)
        assert well_typed(c)
        pop.append(c)


# ---------------------------------------------------------------- trailing stop in the strategy format

def test_adding_the_trail_field_did_not_change_the_hash_of_existing_specs():
    import hashlib, json
    from goldsys.dsl.spec import StrategySpec
    s = StrategySpec.from_yaml("strategies/vol_momentum.yaml")
    d = s.to_dict(); d.pop("rationale", None); d["exit"].pop("trail", None)
    assert s.hash() == hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
    assert s.with_params(stop_atr=2.5).hash() != s.hash()
    t = StrategySpec.from_yaml("strategies/vol_momentum_trail.yaml")
    assert t.exit.trail and t.hash() != s.hash()


def test_trail_compiles_to_distances_and_validates():
    import numpy as np
    from goldsys.data.synthetic import make_m1
    from goldsys.data.store import resample_bars
    from goldsys.dsl.compile import compile_signals
    from goldsys.dsl.spec import StrategySpec
    from goldsys.features.registry import base_features
    import datetime as dt
    df = base_features(resample_bars(make_m1(dt.date(2024, 1, 1), dt.date(2024, 1, 20), seed=2), "M15"))
    base = {"name": "t", "timeframe": "M15", "entry_long": {"op": "gt", "args": ["close", "ema_20"]},
            "exit": {"stop": {"atr_mult": 2.0, "atr_n": 14}, "target": None}, "sizing": {"type": "fixed", "lots": 0.1}}
    none = compile_signals(StrategySpec.from_dict(base), df)
    assert np.isnan(none["trail_dist"]).all() and (none["trail_act"] == 0).all()
    a = compile_signals(StrategySpec.from_dict({**base, "exit": {**base["exit"], "trail": {"atr_mult": "$k", "atr_n": 14, "activate_r": 0.5}}, "params": {"k": 1.5}}), df)
    atr = df["close"].diff().abs().rolling(14).mean()
    ok = ~np.isnan(a["trail_dist"])
    assert ok.sum() > 100 and np.allclose(a["trail_act"][ok], 0.5 * a["stop_dist"][ok])
    f = compile_signals(StrategySpec.from_dict({**base, "exit": {**base["exit"], "trail": {"fixed": 3.0}}}), df)
    assert (f["trail_dist"] == 3.0).all()
    import pytest
    with pytest.raises(ValueError, match="trail must have"):
        compile_signals(StrategySpec.from_dict({**base, "exit": {**base["exit"], "trail": {"bogus": 1}}}), df)
