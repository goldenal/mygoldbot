"""Signal sources for the live runner. Both produce, for one closed bar of the strategy timeframe,
the same scalars the screening engine consumes (sig_long, sig_short, exit_long, exit_short,
stop_dist, target_dist, lots, time_exit_flag) and both go through dsl.compile.compile_signals,
so there is exactly one implementation of a strategy's logic.

PrecomputedSignals  whole history evaluated once (features are causal, so row t only depends on
                    data <= t). Used for replays and for the shadow simulation.
BrokerBarSignals    live: each time a bar closes, re-reads the venue's own closed bars of the strategy
                    timeframe and evaluates the spec on them. Survives restarts and feed gaps because
                    it never depends on bars this process happened to assemble itself.
RollingSignals      re-evaluates the spec on a rolling window each time a bar closes. Used live.
                    Recursive indicators (EMA, ATR, RSI, ADX) need the window to be several times
                    their longest period before they match a full-history calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.store import BAR_COLS
from ..dsl.compile import compile_signals
from ..dsl.spec import StrategySpec
from ..features.registry import base_features

KEYS = ("sig_long", "sig_short", "exit_long", "exit_short", "stop_dist", "target_dist", "lots", "time_exit_flag")


def _row(sig: dict, i: int) -> dict:
    return {"sig_long": bool(sig["sig_long"][i]), "sig_short": bool(sig["sig_short"][i]), "exit_long": bool(sig["exit_long"][i]), "exit_short": bool(sig["exit_short"][i]),
            "stop_dist": float(np.nan_to_num(sig["stop_dist"][i], nan=np.inf)), "target_dist": float(np.nan_to_num(sig["target_dist"][i], nan=np.inf)),
            "lots": float(np.nan_to_num(sig["lots"][i], nan=0.0)), "time_exit_flag": bool(sig["time_exit_flag"][i])}


class PrecomputedSignals:
    def __init__(self, spec: StrategySpec, tf_frame: pd.DataFrame):
        """tf_frame: bars of the strategy timeframe with base features (dsl.compile.feature_frame)."""
        self.spec = spec
        self.sig = compile_signals(spec, tf_frame)
        self.ns = tf_frame.index.as_unit("ns").asi8

    def on_bar(self, ts_ns: int, bar: dict | None = None) -> dict | None:
        i = int(np.searchsorted(self.ns, ts_ns))
        if i >= len(self.ns) or self.ns[i] != ts_ns:
            return None
        return _row(self.sig, i)


class RollingSignals:
    def __init__(self, spec: StrategySpec, history: pd.DataFrame | None = None, window: int = 1500, min_bars: int = 300):
        """history: closed bars of the strategy timeframe (columns of data.store.BAR_COLS), used as warm-up."""
        self.spec, self.window, self.min_bars = spec, window, min_bars
        self.rows: list[tuple] = []
        if history is not None and len(history):
            h = history[BAR_COLS].tail(window)
            ns = h.index.as_unit("ns").asi8
            self.rows = [(int(t), *map(float, r)) for t, r in zip(ns, h.to_numpy())]

    def on_bar(self, ts_ns: int, bar: dict) -> dict | None:
        row = (int(ts_ns), *(float(bar[c]) for c in BAR_COLS))
        if self.rows and self.rows[-1][0] >= ts_ns:
            if self.rows[-1][0] > ts_ns:
                return None                     # out-of-order bar: ignore
            self.rows[-1] = row
        else:
            self.rows.append(row)
        self.rows = self.rows[-self.window:]
        if len(self.rows) < self.min_bars:
            return None
        arr = np.array(self.rows)
        df = pd.DataFrame(arr[:, 1:], columns=BAR_COLS, index=pd.DatetimeIndex(arr[:, 0].astype("int64"), tz="UTC", name="ts"))
        sig = compile_signals(self.spec, base_features(df))
        return _row(sig, len(df) - 1)


class BrokerBarSignals:
    def __init__(self, spec: StrategySpec, fetch, window: int = 1500, min_bars: int = 300):
        """fetch(timeframe, n) -> closed bars (BAR_COLS, UTC index by open time), e.g. broker.bars."""
        self.spec, self.fetch, self.window, self.min_bars = spec, fetch, window, min_bars

    def on_bar(self, ts_ns: int, bar: dict) -> dict | None:
        df = self.fetch(self.spec.timeframe, self.window)
        df = df[BAR_COLS] if len(df) else pd.DataFrame(columns=BAR_COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        ns = df.index.as_unit("ns").asi8
        df = df[ns <= ts_ns]
        if len(df) == 0 or df.index.as_unit("ns").asi8[-1] < ts_ns:
            # the venue has not published the bar that just closed: use the one assembled from minute bars
            last = pd.DataFrame([[float(bar[c]) for c in BAR_COLS]], columns=BAR_COLS, index=pd.DatetimeIndex([ts_ns], tz="UTC", name="ts"))
            df = pd.concat([df.astype(float), last])
        if len(df) < self.min_bars:
            return None
        sig = compile_signals(self.spec, base_features(df.astype(float)))
        return _row(sig, len(df) - 1)
