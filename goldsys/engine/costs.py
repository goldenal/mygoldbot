"""Cost model: time-of-day spread, slippage, commission, swap. Versioned, serialisable."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from ..config import OZ_PER_LOT

# Default $ spread per UTC hour for a raw-spread CFD broker. Replace with your broker's sampled table.
DEFAULT_SPREAD_BY_HOUR = [0.45, 0.35, 0.30, 0.30, 0.30, 0.30, 0.28, 0.20, 0.18, 0.18, 0.18, 0.18,
                          0.17, 0.17, 0.17, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.60, 1.50, 0.80]


@dataclass
class CostModel:
    version: str = "v1-default"
    spread_by_hour: list[float] = field(default_factory=lambda: list(DEFAULT_SPREAD_BY_HOUR))
    spread_multiplier: float = 1.0            # for cost-sensitivity tests
    use_bar_spread: bool = True               # if bars carry a measured 'spread' column, prefer it
    slippage_mean: float = 0.05               # $ adverse slippage per fill (market/stop orders)
    slippage_std: float = 0.05
    slippage_vol_scale: float = 0.5           # extra slippage = scale * (bar_range / median_range - 1)+
    commission_per_lot_rt: float = 6.0        # $ per round trip per standard lot
    swap_long_per_lot_day: float = -4.0       # $ per lot per day held past rollover
    swap_short_per_lot_day: float = -2.0
    triple_swap_weekday: int = 2              # Wednesday
    rollover_hour_utc: int = 22
    news_spread_multiplier: float = 3.0
    news_window_minutes: int = 2

    def spread_at(self, idx: pd.DatetimeIndex, bar_spread: pd.Series | None = None) -> np.ndarray:
        table = np.asarray(self.spread_by_hour, dtype=float)[idx.hour.to_numpy()]
        if self.use_bar_spread and bar_spread is not None:
            s = bar_spread.to_numpy(dtype=float)
            s = np.where(np.isnan(s), table, s)
            # never below the table (table is the broker floor), protects against unrealistic tick data
            table = np.maximum(s, table * 0.5)
        return table * self.spread_multiplier

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CostModel":
        return cls(**d)

    def round_trip_commission(self, lots: float) -> float:
        return self.commission_per_lot_rt * lots

    def swap_cost(self, direction: int, lots: float, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> float:
        """Total swap in $ for a position held from entry_ts to exit_ts (negative = cost)."""
        per_day = self.swap_long_per_lot_day if direction > 0 else self.swap_short_per_lot_day
        # count rollovers crossed
        roll = pd.Timestamp(entry_ts).floor("D") + pd.Timedelta(hours=self.rollover_hour_utc)
        if roll <= entry_ts:
            roll += pd.Timedelta(days=1)
        total = 0.0
        while roll < exit_ts:
            mult = 3.0 if roll.dayofweek == self.triple_swap_weekday else 1.0
            total += per_day * lots * mult
            roll += pd.Timedelta(days=1)
        return total
