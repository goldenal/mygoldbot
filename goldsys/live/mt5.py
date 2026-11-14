"""MetaTrader 5 broker adapter. WINDOWS ONLY: the official `MetaTrader5` package talks to a running,
logged-in MT5 terminal on the same machine. Run it on the VPS behind scripts/mt5_bridge_server.py
and connect from the research box with live.bridge.RemoteBroker.

NOT EXERCISED IN THIS REPOSITORY'S TESTS (no MT5 on macOS/Linux). Before trusting it: run
scripts/mt5_bridge_server.py --selftest on a DEMO account, which opens and closes one minimum-size
position and prints every field this adapter relies on.

Account requirements: a HEDGING account (each strategy holds its own position), symbol tradable
with market execution. Credentials come from the environment (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
or from an already logged-in terminal; they are never passed on the command line.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from ..config import SYMBOL
from .broker import Account, Broker, BrokerError, Fill, Position, Quote

MAGIC = 26111300
TF_ATTR = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}


class MT5Broker(Broker):
    name = "mt5"

    def __init__(self, symbol: str = SYMBOL, deviation_points: int = 30, server_utc_offset_hours: float | None = None, magic: int = MAGIC):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as e:
            raise BrokerError("the MetaTrader5 package is not installed (Windows only: pip install MetaTrader5)") from e
        self.mt5, self.symbol, self.deviation, self.magic = mt5, symbol, deviation_points, magic
        kw = {}
        if os.environ.get("MT5_LOGIN"):
            kw = {"login": int(os.environ["MT5_LOGIN"]), "password": os.environ.get("MT5_PASSWORD", ""), "server": os.environ.get("MT5_SERVER", "")}
        if not mt5.initialize(**kw):
            raise BrokerError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.symbol_select(symbol, True):
            raise BrokerError(f"symbol {symbol} is not available at this broker: {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        self.point, self.digits = info.point, info.digits
        self.vol_min, self.vol_step = info.volume_min, info.volume_step
        acct = mt5.account_info()
        self.is_demo = acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        if acct.margin_mode != mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING:
            raise BrokerError("the MT5 account is not a hedging account; goldsys needs one position per strategy")
        # MT5 timestamps are in broker SERVER time. Detect the offset from the latest tick unless given.
        self.offset_s = int(server_utc_offset_hours * 3600) if server_utc_offset_hours is not None else self._detect_offset()
        self._known: dict[str, Position] = {p.id: p for p in self.positions()}

    def _detect_offset(self) -> int:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.time == 0:
            raise BrokerError("cannot detect the server time offset (no tick); pass server_utc_offset_hours")
        off = round((tick.time - time.time()) / 1800) * 1800       # half-hour granularity
        if abs(tick.time - time.time() - off) > 300:
            raise BrokerError("market looks closed, so the server time offset cannot be detected; pass server_utc_offset_hours")
        return int(off)

    def _utc(self, server_epoch: float) -> pd.Timestamp:
        return pd.Timestamp(server_epoch - self.offset_s, unit="s", tz="UTC")

    # ------------------------------------------------------------------ Broker
    def connected(self) -> bool:
        ti = self.mt5.terminal_info()
        return bool(ti is not None and ti.connected)

    def quote(self) -> Quote:
        t = self.mt5.symbol_info_tick(self.symbol)
        if t is None:
            raise BrokerError(f"no tick: {self.mt5.last_error()}")
        return Quote(self._utc(t.time_msc / 1000.0), float(t.bid), float(t.ask))

    def account(self) -> Account:
        a = self.mt5.account_info()
        if a is None:
            raise BrokerError(f"no account info: {self.mt5.last_error()}")
        return Account(float(a.balance), float(a.equity), a.currency, self.is_demo)

    def positions(self) -> list[Position]:
        res = self.mt5.positions_get(symbol=self.symbol)
        if res is None:
            raise BrokerError(f"positions_get failed: {self.mt5.last_error()}")
        out = []
        for p in res:
            if p.magic != self.magic:
                continue                                   # not ours: leave manual / other EAs' positions alone
            out.append(Position(str(p.ticket), p.comment, p.symbol, 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1, float(p.volume), float(p.price_open),
                                self._utc(p.time), float(p.sl) or None, float(p.tp) or None))
        return out

    def _send(self, request: dict):
        r = self.mt5.order_send(request)
        if r is None:
            raise BrokerError(f"order_send returned nothing: {self.mt5.last_error()}")
        if r.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise BrokerError(f"order rejected: retcode {r.retcode} {r.comment}")
        return r

    def _filling(self) -> int:
        mode = self.mt5.symbol_info(self.symbol).filling_mode
        if mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        if mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def open_position(self, strategy: str, direction: int, lots: float, stop_dist: float, target_dist: float | None) -> Fill:
        mt5 = self.mt5
        lots = round(round(lots / self.vol_step) * self.vol_step, 2)
        if lots < self.vol_min:
            raise BrokerError(f"volume {lots} below the broker minimum {self.vol_min}")
        t = mt5.symbol_info_tick(self.symbol)
        px = t.ask if direction == 1 else t.bid
        sl = round(px - direction * stop_dist, self.digits) if stop_dist and stop_dist != float("inf") else 0.0
        tp = round(px + direction * target_dist, self.digits) if target_dist and target_dist != float("inf") else 0.0
        r = self._send({"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lots, "type": mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
                        "price": px, "sl": sl, "tp": tp, "deviation": self.deviation, "magic": self.magic, "comment": strategy[:31],
                        "type_time": mt5.ORDER_TIME_GTC, "type_filling": self._filling()})
        pid = str(r.order)                                 # on hedging accounts the position ticket is the opening order's ticket
        pos = Position(pid, strategy, self.symbol, direction, float(r.volume), float(r.price), pd.Timestamp.now(tz="UTC"), sl or None, tp or None)
        self._known[pid] = pos
        return Fill(pid, strategy, self.symbol, pos.entry_ts, direction, pos.lots, "entry", pos.entry_px, stop_px=pos.stop_px, tgt_px=pos.tgt_px)

    def close_position(self, position_id: str, reason: str = "signal") -> Fill:
        mt5 = self.mt5
        res = mt5.positions_get(ticket=int(position_id))
        if not res:
            fills = [f for f in self.poll_fills() if f.position_id == position_id]
            if fills:
                return fills[0]                            # the stop/target got there first
            raise BrokerError(f"position {position_id} not found")
        p = res[0]
        d = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
        t = mt5.symbol_info_tick(self.symbol)
        self._send({"action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": p.volume, "type": mt5.ORDER_TYPE_SELL if d == 1 else mt5.ORDER_TYPE_BUY,
                    "position": p.ticket, "price": t.bid if d == 1 else t.ask, "deviation": self.deviation, "magic": self.magic, "comment": reason[:31],
                    "type_time": mt5.ORDER_TIME_GTC, "type_filling": self._filling()})
        f = self._exit_fill(position_id, reason)
        self._known.pop(position_id, None)
        return f

    def _exit_fill(self, position_id: str, reason: str | None = None) -> Fill:
        """Build the exit fill from the deal history of one position (price, profit, commission, swap)."""
        mt5 = self.mt5
        known = self._known.get(position_id)
        deals = None
        for _ in range(10):                                # history can lag the order result by a moment
            deals = mt5.history_deals_get(position=int(position_id))
            if deals and any(d.entry == mt5.DEAL_ENTRY_OUT for d in deals):
                break
            time.sleep(0.2)
        outs = [d for d in (deals or []) if d.entry == mt5.DEAL_ENTRY_OUT]
        if not outs:
            raise BrokerError(f"no closing deal found for position {position_id}")
        vol = sum(d.volume for d in outs)
        px = sum(d.price * d.volume for d in outs) / vol
        commission = sum(d.commission for d in deals)
        swap = sum(d.swap for d in deals)
        pnl = sum(d.profit for d in deals) + commission + swap + sum(getattr(d, "fee", 0.0) for d in deals)
        last = outs[-1]
        if reason is None:
            reason = {mt5.DEAL_REASON_SL: "stop", mt5.DEAL_REASON_TP: "target"}.get(last.reason, "external")
        ins = [d for d in deals if d.entry == mt5.DEAL_ENTRY_IN]
        direction = known.dir if known else (1 if ins and ins[0].type == mt5.DEAL_TYPE_BUY else -1)
        strategy = known.strategy if known else (ins[0].comment if ins else "")
        return Fill(position_id, strategy, self.symbol, self._utc(last.time_msc / 1000.0), direction, float(vol), "exit", float(px), reason, float(pnl), float(-commission), float(swap))

    def exit_fill(self, position_id: str) -> Fill | None:
        try:
            return self._exit_fill(position_id)
        except BrokerError:
            return None

    def poll_fills(self) -> list[Fill]:
        live = {p.id for p in self.positions()}
        out = []
        for pid in [k for k in self._known if k not in live]:
            try:
                out.append(self._exit_fill(pid))
            except BrokerError:
                continue                                   # history not there yet: try again on the next poll
            self._known.pop(pid, None)
        return out

    def bars(self, timeframe: str = "M1", n: int = 500) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(self.symbol, getattr(self.mt5, TF_ATTR[timeframe]), 1, n)      # start at 1: skip the forming bar
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df["time"] - self.offset_s, unit="s", utc=True)
        df.index.name = "ts"
        return pd.DataFrame({"open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"], "spread": df["spread"] * self.point,
                             "ticks": df["tick_volume"].astype(float), "volume": df["real_volume"].astype(float)}, index=df.index)

    def shutdown(self) -> None:
        self.mt5.shutdown()
