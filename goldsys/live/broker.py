"""Broker interface and the paper broker.

Every execution venue (paper, MT5 on a Windows VPS reached over the bridge, a future IBKR adapter)
implements `Broker`. Positions are per strategy and independent, so a real account must be a
HEDGING account (MT5 default for CFD brokers); strategies are told apart by the position comment.

PaperBroker follows the screening engine's conventions exactly: bars and quotes are BID, buys pay
bid + spread, a stop is assumed to be hit before a target inside one bar, a gap through the stop
fills at the open. tests/test_live_parity.py holds it trade-for-trade equal to the backtester.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import OZ_PER_LOT, SYMBOL
from ..engine.costs import CostModel


@dataclass
class Quote:
    ts: pd.Timestamp
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Position:
    id: str
    strategy: str
    symbol: str
    dir: int                   # +1 long, -1 short
    lots: float
    entry_px: float
    entry_ts: pd.Timestamp
    stop_px: float | None
    tgt_px: float | None


@dataclass
class Fill:
    position_id: str
    strategy: str
    symbol: str
    ts: pd.Timestamp
    dir: int                   # direction of the POSITION
    lots: float
    kind: str                  # "entry" | "exit"
    price: float
    reason: str = ""           # exits: stop | target | signal | time | reverse | manual | kill | end | external
    pnl: float = 0.0           # exits: net of commission and swap
    commission: float = 0.0
    swap: float = 0.0
    stop_px: float | None = None
    tgt_px: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Account:
    balance: float
    equity: float
    currency: str = "USD"
    demo: bool | None = None   # True = demo account, False = real money, None = simulated / unknown


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    name = "broker"
    simulated = False

    @abstractmethod
    def connected(self) -> bool: ...

    @abstractmethod
    def quote(self) -> Quote: ...

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def open_position(self, strategy: str, direction: int, lots: float, stop_dist: float, target_dist: float | None) -> Fill:
        """Market order with a protective stop (and target) at the given DISTANCES from the fill."""

    @abstractmethod
    def close_position(self, position_id: str, reason: str = "signal") -> Fill: ...

    @abstractmethod
    def poll_fills(self) -> list[Fill]:
        """Exits the venue did on its own (stop / target / manual close in the terminal) since the last call."""

    def bars(self, timeframe: str = "M1", n: int = 500) -> pd.DataFrame:
        """Most recent CLOSED bars (bid OHLC + spread), UTC index labelled by open time. Optional."""
        return pd.DataFrame()

    def now(self) -> pd.Timestamp:
        """The venue's clock (UTC). Real venues use wall time; a simulated venue has its own."""
        return pd.Timestamp.now(tz="UTC")

    def exit_fill(self, position_id: str) -> Fill | None:
        """The closing fill of a position that no longer exists, looked up from the venue's history
        (None if the venue has no record). Used to recover an exit that was never delivered by poll_fills."""
        return None


class PaperBroker(Broker):
    name = "paper"
    simulated = True

    def __init__(self, cost: CostModel | None = None, init_cash: float = 10_000.0, symbol: str = SYMBOL, slippage: bool = True, seed: int = 0):
        self.cost = cost or CostModel()
        self.cash = float(init_cash)
        self.symbol = symbol
        self.slippage = slippage
        self.rng = np.random.default_rng(seed)
        self._q: Quote | None = None
        self._pos: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._closed: dict[str, Fill] = {}
        self._n = 0

    # ------------------------------------------------------------------ market data in
    def set_quote(self, ts, bid: float, spread: float) -> None:
        self._q = Quote(ts, float(bid), float(bid) + float(spread))

    def _slip(self) -> float:
        if not self.slippage or (self.cost.slippage_mean == 0 and self.cost.slippage_std == 0):
            return 0.0
        return max(0.0, float(self.rng.normal(self.cost.slippage_mean, self.cost.slippage_std)))

    def on_bar(self, ts, o: float, h: float, l: float, c: float, spread: float) -> None:
        """Resolve stops and targets inside one bar (stop first), then leave the quote at the close."""
        for p in list(self._pos.values()):
            px, reason = None, ""
            stop = p.stop_px if p.stop_px is not None else (-math.inf if p.dir == 1 else math.inf)
            tgt = p.tgt_px if p.tgt_px is not None else (math.inf if p.dir == 1 else -math.inf)
            if p.dir == 1:
                if l <= stop:
                    px, reason = min(stop, o) - self._slip(), "stop"
                elif h >= tgt:
                    px, reason = (tgt if o < tgt else o), "target"
            else:
                if h + spread >= stop:
                    px, reason = max(stop, o + spread) + self._slip(), "stop"
                elif l + spread <= tgt:
                    px, reason = (tgt if o + spread > tgt else o + spread), "target"
            if px is not None:
                self._fills.append(self._close(p, ts, px, reason))
        self.set_quote(ts, c, spread)

    # ------------------------------------------------------------------ Broker
    def connected(self) -> bool:
        return True

    def quote(self) -> Quote:
        if self._q is None:
            raise BrokerError("no quote yet")
        return self._q

    def account(self) -> Account:
        eq = self.cash
        if self._q is not None:
            for p in self._pos.values():
                eq += ((self._q.bid - p.entry_px) if p.dir == 1 else (p.entry_px - self._q.ask)) * p.lots * OZ_PER_LOT
        return Account(self.cash, eq)

    def positions(self) -> list[Position]:
        return list(self._pos.values())

    def open_position(self, strategy: str, direction: int, lots: float, stop_dist: float, target_dist: float | None) -> Fill:
        q = self.quote()
        if lots <= 0:
            raise BrokerError("lots must be positive")
        px = q.ask + self._slip() if direction == 1 else q.bid - self._slip()
        stop = px - direction * stop_dist if stop_dist is not None and math.isfinite(stop_dist) else None
        tgt = px + direction * target_dist if target_dist is not None and math.isfinite(target_dist) else None
        self._n += 1
        p = Position(f"P{self._n}", strategy, self.symbol, direction, float(lots), px, q.ts, stop, tgt)
        self._pos[p.id] = p
        return Fill(p.id, strategy, self.symbol, q.ts, direction, p.lots, "entry", px, stop_px=stop, tgt_px=tgt)

    def _close(self, p: Position, ts, px: float, reason: str) -> Fill:
        commission = self.cost.commission_per_lot_rt * p.lots
        swap = self.cost.swap_cost(p.dir, p.lots, p.entry_ts, ts)
        pnl = (px - p.entry_px) * p.dir * p.lots * OZ_PER_LOT - commission + swap
        self.cash += pnl
        del self._pos[p.id]
        f = Fill(p.id, p.strategy, p.symbol, ts, p.dir, p.lots, "exit", px, reason, pnl, commission, swap)
        self._closed[p.id] = f
        return f

    def exit_fill(self, position_id: str) -> Fill | None:
        return self._closed.get(position_id)

    def close_position(self, position_id: str, reason: str = "signal") -> Fill:
        p = self._pos.get(position_id)
        if p is None:
            raise BrokerError(f"unknown position {position_id}")
        q = self.quote()
        px = q.bid - self._slip() if p.dir == 1 else q.ask + self._slip()
        return self._close(p, q.ts, px, reason)

    def poll_fills(self) -> list[Fill]:
        out, self._fills = self._fills, []
        return out
