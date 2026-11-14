"""A simulated venue with its own clock: historical M1 bars played forward as if they were happening
now, behind the same Broker interface a real venue has (quotes, closed bars, orders, stops that
trigger on their own between polls).

It exists so the REAL-broker code path (LiveRunner.live_step, the bridge, the dashboard, the
approval queue) can be rehearsed end to end on any machine before a demo account is involved:
    python scripts/mt5_bridge_server.py --paper --replay-symbol XAUUSD --replay-start 2026-08-03 --speed 120
    python scripts/run_live.py broker --bridge http://127.0.0.1:8765
`speed` = simulated seconds per wall-clock second. Closed-market gaps are skipped.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..data.store import BAR_COLS, resample_bars
from ..dsl.compile import TF_MINUTES
from ..engine.costs import CostModel
from .broker import Account, PaperBroker, Quote

MIN_NS = 60_000_000_000


class SimVenue(PaperBroker):
    name = "sim"

    def __init__(self, bars_m1: pd.DataFrame, cost: CostModel | None = None, start: pd.Timestamp | None = None, speed: float = 60.0,
                 init_cash: float = 10_000.0, slippage: bool = True, seed: int = 0, manual_clock: bool = False, symbol: str = "XAUUSD"):
        super().__init__(cost, init_cash, symbol, slippage, seed)
        self.m1 = bars_m1[BAR_COLS]
        self.ns = self.m1.index.as_unit("ns").asi8
        self.cols = [self.m1[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close")]
        self.sp = self.cost.spread_at(self.m1.index, self.m1["spread"])
        start_ns = pd.Timestamp(start).as_unit("ns").value if start is not None else int(self.ns[min(len(self.ns) - 1, 3000)])
        self.i = int(np.searchsorted(self.ns, start_ns - MIN_NS, side="right"))      # bars already closed at the start
        self.speed, self.manual = speed, manual_clock
        self._sim_ns, self._wall = start_ns, time.time()

    # ------------------------------------------------------------------ clock
    def now(self) -> pd.Timestamp:
        if not self.manual:
            w = time.time()
            self._sim_ns += int((w - self._wall) * self.speed * 1e9)
            self._wall = w
            if self.i < len(self.ns) and self.ns[self.i] - self._sim_ns > 10 * MIN_NS:
                self._sim_ns = int(self.ns[self.i])                                   # market closed: jump to the next bar
        return pd.Timestamp(self._sim_ns, tz="UTC")

    def advance(self, seconds: float) -> None:
        self._sim_ns += int(seconds * 1e9)

    def _sync(self) -> int:
        now_ns = self.now().value
        o, h, l, c = self.cols
        while self.i < len(self.ns) and self.ns[self.i] + MIN_NS <= now_ns:
            i = self.i
            self.on_bar(pd.Timestamp(int(self.ns[i]) + MIN_NS, tz="UTC"), o[i], h[i], l[i], c[i], self.sp[i])    # venue-side fills are stamped at the bar's close
            self.i += 1
        if self.i > 0:
            j = self.i - 1
            self._q = Quote(pd.Timestamp(now_ns, tz="UTC"), float(c[j]), float(c[j] + self.sp[j]))
        return now_ns

    # ------------------------------------------------------------------ Broker
    def quote(self) -> Quote:
        self._sync()
        return super().quote()

    def account(self) -> Account:
        self._sync()
        a = super().account()
        return Account(a.balance, a.equity, a.currency, True)

    def positions(self):
        self._sync()
        return super().positions()

    def poll_fills(self):
        self._sync()
        return super().poll_fills()

    def bars(self, timeframe: str = "M1", n: int = 500) -> pd.DataFrame:
        self._sync()
        tfm = TF_MINUTES[timeframe]
        closed = self.m1.iloc[max(0, self.i - (n + 2) * tfm * 2): self.i]            # x2: sessions have gaps
        if timeframe == "M1" or closed.empty:
            return closed.tail(n)
        out = resample_bars(closed, timeframe)
        if len(out) and out.index[-1].as_unit("ns").value + tfm * MIN_NS > int(self.ns[self.i - 1]) + MIN_NS:
            out = out.iloc[:-1]                                                       # the last period is still forming
        return out.iloc[1:].tail(n) if len(out) > 1 else out                          # first period may be cut by the slice
