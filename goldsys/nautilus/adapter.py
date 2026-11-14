"""Faithful-engine adapter (Phase 2b): compile a StrategySpec into a NautilusTrader Strategy.

NautilusTrader is optional (`pip install nautilus_trader`). The generated strategy recomputes the
same DSL signals bar by bar from the Nautilus bar stream and submits bracket orders, so the exact
same spec drives screening, tick-level validation and live trading. This module provides:
  - `spec_to_nautilus_config(spec)`: the config dict a Nautilus node needs
  - `DSLStrategy`: a Nautilus Strategy subclass that evaluates the spec on each closed bar
If nautilus_trader is not installed, importing DSLStrategy raises ImportError with instructions.
"""
from __future__ import annotations

import pandas as pd

from ..dsl.compile import compile_signals
from ..dsl.spec import StrategySpec
from ..features.registry import base_features

try:
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig  # type: ignore
    from nautilus_trader.model.data import Bar, BarType  # type: ignore
    from nautilus_trader.model.enums import OrderSide  # type: ignore
    from nautilus_trader.model.identifiers import InstrumentId  # type: ignore
    from nautilus_trader.model.objects import Quantity, Price  # type: ignore
    HAVE_NAUTILUS = True
except Exception:  # pragma: no cover
    HAVE_NAUTILUS = False


def spec_to_nautilus_config(spec: StrategySpec, instrument_id: str = "XAUUSD.SIM") -> dict:
    tf = {"M1": "1-MINUTE", "M5": "5-MINUTE", "M15": "15-MINUTE", "M30": "30-MINUTE", "H1": "1-HOUR", "H4": "4-HOUR", "D1": "1-DAY"}[spec.timeframe]
    return {"instrument_id": instrument_id, "bar_type": f"{instrument_id}-{tf}-BID-EXTERNAL", "spec": spec.to_dict(), "warmup_bars": 400}


if HAVE_NAUTILUS:

    class DSLStrategyConfig(StrategyConfig, frozen=True):  # type: ignore[misc]
        instrument_id: str
        bar_type: str
        spec: dict
        warmup_bars: int = 400

    class DSLStrategy(Strategy):  # type: ignore[misc]
        """Evaluates the DSL on a rolling window of closed bars (recomputed each bar: simple and
        guaranteed identical to the screening engine's signal logic; optimise later if needed)."""

        def __init__(self, config: DSLStrategyConfig):
            super().__init__(config)
            self.spec = StrategySpec.from_dict(config.spec)
            if self.spec.exit.trail:
                raise NotImplementedError("trailing stops are not implemented in the Nautilus adapter (brackets are fixed at entry)")
            self.instrument_id = InstrumentId.from_str(config.instrument_id)
            self.bar_type = BarType.from_str(config.bar_type)
            self.rows: list[dict] = []
            self.warmup = config.warmup_bars

        def on_start(self):
            self.instrument = self.cache.instrument(self.instrument_id)
            self.subscribe_bars(self.bar_type)

        def on_bar(self, bar: Bar):
            ts = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC")
            self.rows.append({"ts": ts, "open": float(bar.open), "high": float(bar.high), "low": float(bar.low), "close": float(bar.close), "spread": 0.2, "ticks": 1, "volume": float(bar.volume)})
            if len(self.rows) < self.warmup:
                return
            self.rows = self.rows[-max(self.warmup, 600):]
            df = pd.DataFrame(self.rows).set_index("ts")
            df = base_features(df)
            sig = compile_signals(self.spec, df)
            i = len(df) - 1
            if self.portfolio.is_flat(self.instrument_id):
                if sig["sig_long"][i] and sig["lots"][i] > 0:
                    self._bracket(OrderSide.BUY, sig["lots"][i], sig["stop_dist"][i], sig["target_dist"][i], float(bar.close))
                elif sig["sig_short"][i] and sig["lots"][i] > 0:
                    self._bracket(OrderSide.SELL, sig["lots"][i], sig["stop_dist"][i], sig["target_dist"][i], float(bar.close))
            else:
                net_long = self.portfolio.is_net_long(self.instrument_id)
                if (net_long and (sig["exit_long"][i] or sig["time_exit_flag"][i])) or ((not net_long) and (sig["exit_short"][i] or sig["time_exit_flag"][i])):
                    self.close_all_positions(self.instrument_id)

        def _bracket(self, side, lots: float, stop_dist: float, target_dist: float, px: float):
            qty = self.instrument.make_qty(round(lots * 100, 0))   # oz
            sl = px - stop_dist if side == OrderSide.BUY else px + stop_dist
            tp = None
            if target_dist != float("inf"):
                tp = px + target_dist if side == OrderSide.BUY else px - target_dist
            order_list = self.order_factory.bracket(
                instrument_id=self.instrument_id, order_side=side, quantity=qty,
                sl_trigger_price=self.instrument.make_price(sl),
                tp_price=self.instrument.make_price(tp) if tp is not None else None,
            )
            self.submit_order_list(order_list)
else:

    class DSLStrategy:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            raise ImportError("nautilus_trader is not installed. pip install 'goldsys[nautilus]' (Linux/macOS x86_64/arm64; see docs.nautilustrader.io)")
