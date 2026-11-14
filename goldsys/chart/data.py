"""Chart data: bars at any timeframe, indicators, and strategy trades with their stop/target levels.

Everything here is read-only. Indicators are the same causal functions the strategies use
(goldsys.features.indicators), computed on the FULL resampled history and then sliced, so the
values at the left edge of a window are warmed up and identical to what a backtest sees.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BAR_DIR, ROOT
from ..data.sessions import asian_range, session_flags
from ..data.store import PANDAS_FREQ, read_bars, resample_bars
from ..features import indicators as I

TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
STRATEGY_DIRS = (ROOT / "strategies", ROOT / "strategies" / "candidates")
MAX_LIMIT = 5000

# id -> (label, pane?, default params). Parameters are positional in the spec string: "bb:20:2".
CATALOG = {
    "sma": ("SMA", False, [20]),
    "ema": ("EMA", False, [50]),
    "bb": ("Bollinger", False, [20, 2]),
    "donchian": ("Donchian", False, [20]),
    "vwap": ("VWAP (daily)", False, []),
    "asian": ("Asian range", False, []),
    "sessions": ("Sessions", False, []),
    "rsi": ("RSI", True, [14]),
    "atr": ("ATR", True, [14]),
    "adx": ("ADX", True, [14]),
    "macd": ("MACD", True, [12, 26, 9]),
}

_M1: dict[str, tuple] = {}
_TF: dict[tuple, pd.DataFrame] = {}
_IND: dict[tuple, dict] = {}
_BT: dict[tuple, dict] = {}


class ChartError(ValueError):
    pass


def symbols() -> list[str]:
    if not BAR_DIR.exists():
        return []
    return sorted(p.name for p in BAR_DIR.iterdir() if (p / "M1").is_dir() and any((p / "M1").glob("*.parquet")))


def _stamp(symbol: str) -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted((BAR_DIR / symbol / "M1").glob("*.parquet")))


def m1(symbol: str) -> pd.DataFrame:
    if symbol not in symbols():
        raise ChartError(f"unknown symbol {symbol!r}; have {symbols()}")
    st = _stamp(symbol)
    hit = _M1.get(symbol)
    if hit is None or hit[0] != st:
        for d in (_TF, _IND, _BT):          # data changed: drop everything derived from the old file set
            for k in [k for k in d if k[0] == symbol]:
                del d[k]
        _M1[symbol] = (st, read_bars(symbol, "M1"))
    return _M1[symbol][1]


def bars(symbol: str, tf: str) -> pd.DataFrame:
    if tf not in PANDAS_FREQ:
        raise ChartError(f"unknown timeframe {tf!r}; have {list(PANDAS_FREQ)}")
    base = m1(symbol)
    if tf == "M1":
        return base
    key = (symbol, tf)
    if key not in _TF:
        _TF[key] = resample_bars(base, tf)
    return _TF[key]


# ---------------------------------------------------------------- indicators

def parse_spec(spec: str) -> tuple[str, list[float]]:
    parts = spec.split(":")
    kind = parts[0]
    if kind not in CATALOG:
        raise ChartError(f"unknown indicator {kind!r}; have {sorted(CATALOG)}")
    try:
        args = [float(x) if "." in x else int(x) for x in parts[1:] if x != ""]
    except ValueError as e:
        raise ChartError(f"bad indicator parameters in {spec!r}") from e
    defaults = CATALOG[kind][2]
    args = args + defaults[len(args):]
    if len(args) > len(defaults) or any(a <= 0 for a in args) or any(a > 5000 for a in args):
        raise ChartError(f"bad indicator parameters in {spec!r}")
    return kind, args


def _series(name: str, values, typ: str = "line") -> dict:
    v = pd.Series(values).astype(float)
    return {"name": name, "type": typ, "v": v.round(4).astype(object).where(v.notna(), None).tolist()}


def compute(spec: str, df: pd.DataFrame, tf: str) -> dict:
    """One indicator over a whole frame -> {id, pane, label, series:[{name,type,v}], levels?}."""
    kind, a = parse_spec(spec)
    label, pane, _ = CATALOG[kind]
    c, h, l = df["close"], df["high"], df["low"]
    out: dict = {"id": spec, "pane": pane, "label": label, "series": []}
    if kind == "sma":
        out["label"] = f"SMA {a[0]}"
        out["series"] = [_series(out["label"], I.sma(c, a[0]))]
    elif kind == "ema":
        out["label"] = f"EMA {a[0]}"
        out["series"] = [_series(out["label"], I.ema(c, a[0]))]
    elif kind == "bb":
        b = I.bollinger(c, a[0], float(a[1]))
        out["label"] = f"Bollinger {a[0]}, {a[1]}"
        out["series"] = [_series("upper", b["upper"]), _series("middle", b["mid"]), _series("lower", b["lower"])]
    elif kind == "donchian":
        out["label"] = f"Donchian {a[0]}"
        out["series"] = [_series("high", I.donchian_high(h, a[0])), _series("low", I.donchian_low(l, a[0]))]
    elif kind == "vwap":
        out["series"] = [_series("VWAP", I.session_vwap(c, df["volume"], df.index.normalize()))]
    elif kind == "asian":
        ar = asian_range(df)
        out["series"] = [_series("Asian high", ar["asian_high"]), _series("Asian low", ar["asian_low"])]
    elif kind == "sessions":
        if TF_SECONDS[tf] > 14400:
            out["series"] = []
        else:
            f = session_flags(df.index)
            code = np.zeros(len(df), dtype=float)
            code[f["in_asia"].to_numpy()] = 1
            code[f["in_london"].to_numpy()] = 2
            code[f["in_newyork"].to_numpy()] = 3
            code[f["in_overlap"].to_numpy()] = 4
            s = _series("Sessions", np.where(code > 0, code, np.nan), "bands")
            out["series"] = [s]
    elif kind == "rsi":
        out["label"] = f"RSI {a[0]}"
        out["series"] = [_series(out["label"], I.rsi(c, a[0]))]
        out["levels"] = [30, 70]
    elif kind == "atr":
        out["label"] = f"ATR {a[0]}"
        out["series"] = [_series(out["label"], I.atr(h, l, c, a[0]))]
    elif kind == "adx":
        out["label"] = f"ADX {a[0]}"
        out["series"] = [_series(out["label"], I.adx(h, l, c, a[0]))]
        out["levels"] = [20, 25]
    elif kind == "macd":
        m = I.macd(c, a[0], a[1], a[2])
        out["label"] = f"MACD {a[0]}, {a[1]}, {a[2]}"
        out["series"] = [_series("hist", m["hist"], "hist"), _series("macd", m["macd"]), _series("signal", m["signal"])]
    return out


def indicators(symbol: str, tf: str, specs: list[str]) -> list[dict]:
    df = bars(symbol, tf)
    res = []
    for s in specs:
        key = (symbol, tf, s)
        if key not in _IND:
            if len(_IND) > 64:
                _IND.clear()
            _IND[key] = compute(s, df, tf)
        res.append(_IND[key])
    return res


# ---------------------------------------------------------------- bars window

def window(symbol: str, tf: str, end: int | None, limit: int, specs: list[str], after: int | None = None) -> dict:
    """The `limit` bars of timeframe `tf` that open at or before unix time `end` (default: the latest),
    or, with `after`, the first `limit` bars that open after that time. Plus every requested indicator over the same bars."""
    df = bars(symbol, tf)
    if df.empty:
        raise ChartError(f"no bars for {symbol} {tf}")
    limit = max(10, min(int(limit), MAX_LIMIT))
    ts = df.index.as_unit("s").asi8
    if after is not None:
        lo = int(np.searchsorted(ts, int(after), side="right"))
        hi = min(len(ts), lo + limit)
    else:
        hi = len(ts) if end is None else int(np.searchsorted(ts, int(end), side="right"))
        lo = max(0, hi - limit)
    if hi <= lo:
        raise ChartError("no bars in that direction")
    sl = df.iloc[lo:hi]
    out = {
        "symbol": symbol, "tf": tf, "first": int(ts[0]), "last": int(ts[-1]), "has_more": lo > 0, "has_newer": hi < len(ts), "n": int(hi - lo),
        "t": ts[lo:hi].tolist(),
        "o": sl["open"].round(3).tolist(), "h": sl["high"].round(3).tolist(), "l": sl["low"].round(3).tolist(), "c": sl["close"].round(3).tolist(),
        "vol": sl["volume"].round(0).tolist(), "spread": sl["spread"].round(3).astype(object).where(sl["spread"].notna(), None).tolist(),
        "indicators": [],
    }
    for ind in indicators(symbol, tf, specs):
        out["indicators"].append({**ind, "series": [{**s, "v": s["v"][lo:hi]} for s in ind["series"]]})
    return out


# ---------------------------------------------------------------- strategies and their trades

def list_strategies() -> list[dict]:
    from ..dsl.spec import StrategySpec
    res = []
    for d in STRATEGY_DIRS:
        for p in sorted(d.glob("*.yaml")):
            try:
                s = StrategySpec.from_yaml(str(p))
                res.append({"path": str(p.relative_to(ROOT)), "name": s.name, "timeframe": s.timeframe, "params": s.params})
            except Exception:  # noqa: BLE001  a malformed file must not hide the others
                res.append({"path": str(p.relative_to(ROOT)), "name": p.stem + " (unreadable)", "timeframe": "", "params": {}})
    return res


def _strategy_path(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if p.suffix != ".yaml" or not any(p.parent == d.resolve() for d in STRATEGY_DIRS) or not p.is_file():
        raise ChartError(f"not a strategy file: {rel!r}")
    return p


_FEAT = re.compile(r"^(sma|ema)_(\d+)$|^donchian_(?:high|low)_(\d+)$")


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def suggested_indicators(spec) -> list[str]:
    """Overlays that show what the strategy's rules look at (moving averages, Donchian channels, Asian range)."""
    from ..dsl.spec import substitute_params
    p = spec.params
    parts = [spec.filters, spec.entry_long, spec.entry_short if spec.entry_short != "mirror" else None, spec.exit.exit_long, spec.exit.stop, spec.exit.target]
    found: list[str] = []
    for part in parts:
        for s in _strings(substitute_params(part, p)):
            m = _FEAT.match(s)
            if m:
                ind = f"{m.group(1)}:{m.group(2)}" if m.group(1) else f"donchian:{m.group(3)}"
            elif s in ("asian_high", "asian_low", "asian_width"):
                ind = "asian"
            elif s in ("vwap", "vwap_z") or s.startswith("vwap_z_"):
                ind = "vwap"
            else:
                continue
            if ind not in found:
                found.append(ind)
    return found


def strategy_trades(symbol: str, rel: str) -> dict:
    """Run a strategy over the symbol's whole history and return its trades with stop/target levels."""
    from ..dsl.compile import compile_signals, feature_frame
    from ..dsl.spec import StrategySpec
    from ..engine.backtest import run_backtest
    from ..engine.costs import CostModel
    from ..engine.metrics import summarize

    path = _strategy_path(rel)
    base = m1(symbol)
    if base.empty:
        raise ChartError(f"no bars for {symbol}")
    key = (symbol, str(path), path.stat().st_mtime_ns, _M1[symbol][0])
    if key in _BT:
        return _BT[key]
    spec = StrategySpec.from_yaml(str(path))
    df = feature_frame(base, spec.timeframe, None)
    sig = compile_signals(spec, df)
    res = run_backtest(df[["open", "high", "low", "close", "spread"]], cost=CostModel(), init_cash=spec.sizing.equity,
                       max_bars_in_trade=spec.exit.max_bars or 10**9, params=dict(spec.params), seed=0, **sig)
    tr = res.trades
    ts_idx = df.index
    pos = ts_idx.get_indexer(pd.DatetimeIndex(tr["entry_ts"])) if len(tr) else np.array([], dtype=int)
    sd, td = sig["stop_dist"], sig["target_dist"]
    rows = []
    for k, r in enumerate(tr.itertuples()):
        i = int(pos[k]) - 1                      # the signal bar; the engine sets stop/target from its distances
        stop = float(sd[i]) if i >= 0 and np.isfinite(sd[i]) else None
        tgt = float(td[i]) if i >= 0 and np.isfinite(td[i]) else None
        d = int(r.dir)
        rows.append({
            "id": k, "dir": d, "entry_t": int(pd.Timestamp(r.entry_ts).timestamp()), "exit_t": int(pd.Timestamp(r.exit_ts).timestamp()),
            "entry_px": round(float(r.entry_px), 3), "exit_px": round(float(r.exit_px), 3),
            "stop_px": round(float(r.entry_px) - d * stop, 3) if stop is not None else None,
            "tgt_px": round(float(r.entry_px) + d * tgt, 3) if tgt is not None else None,
            "lots": float(r.lots), "pnl": round(float(r.pnl), 2),
            "r": None if pd.isna(getattr(r, "r_multiple", np.nan)) else round(float(r.r_multiple), 2), "reason": r.reason,
        })
    m = summarize(res.equity, res.trades) if len(tr) else {}
    out = {"strategy": rel, "name": spec.name, "timeframe": spec.timeframe, "params": spec.params,
           "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
           "suggested": suggested_indicators(spec), "trades": rows}
    _BT[key] = out
    return out
