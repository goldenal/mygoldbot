"""Economic calendar for the news blackout. Point-in-time by construction: only the SCHEDULE is used
(event time, name, tier), never the released value.

CSV at data/calendar/events.csv with a header and columns:
    ts_utc,name,tier[,currency]
    2026-11-06 13:30,US Non-Farm Payrolls,1,USD
Supply the schedule from your calendar provider; there is deliberately no built-in list, because a
blackout that silently misses a rescheduled release is worse than one that says it has no data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CALENDAR_FILE


class EventCalendar:
    def __init__(self, events: pd.DataFrame | None = None):
        if events is None or events.empty:
            events = pd.DataFrame({"ts": pd.DatetimeIndex([], tz="UTC"), "name": [], "tier": []})
        ev = events.copy()
        ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
        ev["tier"] = ev["tier"].astype(int)
        self.events = ev.sort_values("ts").reset_index(drop=True)
        ns = pd.DatetimeIndex(self.events["ts"]).as_unit("ns").asi8
        tier, names = self.events["tier"].to_numpy(), self.events["name"].astype(str).to_numpy()
        self._tiers = {t: (ns[tier <= t], names[tier <= t]) for t in (1, 2, 3)}

    @classmethod
    def from_csv(cls, path: str | Path = CALENDAR_FILE) -> "EventCalendar":
        p = Path(path)
        if not p.exists():
            return cls()
        df = pd.read_csv(p)
        df = df.rename(columns={"ts_utc": "ts"})
        return cls(df[["ts", "name", "tier"]])

    def __len__(self) -> int:
        return len(self.events)

    def covers(self, ts: pd.Timestamp) -> bool:
        """True if the schedule extends past ts (an expired calendar must not look like 'no news')."""
        return len(self.events) > 0 and self.events["ts"].iloc[-1] >= ts

    def nearest(self, ts: pd.Timestamp, tier: int = 1) -> tuple[float, str] | None:
        """(signed minutes to the nearest event of this tier or more important, name). Negative = already released."""
        arr, names = self._tiers[tier]
        if len(arr) == 0:
            return None
        t = pd.Timestamp(ts).as_unit("ns").value
        i = int(np.searchsorted(arr, t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(arr)]
        j = min(cands, key=lambda k: abs(int(arr[k]) - t))
        return float((int(arr[j]) - t) / 60e9), str(names[j])

    def in_blackout(self, ts: pd.Timestamp, minutes: float, tier: int = 1) -> str | None:
        """Name of the event if ts is within +-minutes of one, else None."""
        if minutes <= 0:
            return None
        n = self.nearest(ts, tier)
        if n is not None and abs(n[0]) <= minutes:
            return n[1]
        return None
