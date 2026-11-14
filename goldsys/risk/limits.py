"""Risk layer (PLAN 5.8): hard limits checked before every entry, and a kill switch.

The risk manager never originates a trade. It can only shrink an order, block it, or tell the runner
to flatten. Exits are never blocked. State (day/week start equity, high-water mark, loss streak,
kill switch) is persisted through a small key-value store so a restart cannot reset a tripped limit.

Trading day = rollover to rollover (22:00 UTC by default). Trading week starts at the Sunday open.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import OZ_PER_LOT
from ..engine.costs import CostModel
from .calendar import EventCalendar


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 1.5          # of the day's starting equity; blocks entries until the next rollover
    max_weekly_loss_pct: float = 3.0
    max_open_risk_pct: float = 2.0           # sum of (entry - stop) * size over open positions, % of equity
    max_lots_per_trade: float = 1.0
    max_total_lots: float = 2.0
    max_positions: int = 4
    consecutive_losses_pause: int = 5        # after this many losing trades in a row ...
    pause_minutes: int = 240                 # ... no new entries for this long
    spread_guard_mult: float = 1.5           # no entry if live spread > cost-model table for the hour x this
    news_blackout_minutes: int = 30          # +- minutes around tier-1 events
    equity_floor_pct: float = 10.0           # kill switch if equity is this far below its high-water mark
    max_data_staleness_s: int = 180          # live only: kill if no new bar for this long while the market is open
    max_disconnect_s: int = 120              # live only
    flatten_on_daily_limit: bool = True
    flatten_on_kill: bool = True
    rollover_hour_utc: int = 22
    min_lots: float = 0.01

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    allowed: bool
    lots: float
    reasons: list[str] = field(default_factory=list)


def market_open(ts: pd.Timestamp) -> bool:
    """Spot gold hours: Sunday 22:00 UTC to Friday 21:00 UTC with a daily 21:00-22:00 break."""
    wd, h = ts.day_of_week, ts.hour
    if wd == 5 or h == 21:
        return False
    if wd == 4 and h >= 21:
        return False
    if wd == 6 and h < 22:
        return False
    return True


class MemoryState:
    """Default state store (tests, replays). The live ledger provides the persistent one."""

    def __init__(self):
        self.d: dict = {}

    def get(self, key: str, default=None):
        return self.d.get(key, default)

    def set(self, key: str, value) -> None:
        self.d[key] = value


class RiskManager:
    def __init__(self, cfg: RiskConfig | None = None, cost: CostModel | None = None, calendar: EventCalendar | None = None, store=None, on_event=None):
        self.cfg = cfg or RiskConfig()
        self.cost = cost or CostModel()
        self.calendar = calendar or EventCalendar()
        self.store = store or MemoryState()
        self.on_event = on_event or (lambda level, kind, msg, data=None: None)
        s = self.store.get("risk_state") or {}
        self.s = {"day": None, "day_start_equity": None, "week": None, "week_start_equity": None, "hwm": None,
                  "consec_losses": 0, "pause_until": None, "day_blocked": False, "week_blocked": False}
        self.s.update(s)
        self._dirty = False

    # ------------------------------------------------------------------ persisted controls
    def _save(self) -> None:
        self.store.set("risk_state", self.s)
        self._dirty = False

    def flush(self) -> None:
        """Persist slow-moving state (high-water mark). Limit trips and pauses are saved as they happen."""
        if self._dirty:
            self._save()

    def kill_state(self) -> dict:
        return self.store.get("kill") or {"active": False}

    def kill(self, ts, reason: str) -> bool:
        """Trip the kill switch. Returns True if it was newly tripped."""
        if self.kill_state().get("active"):
            return False
        self.store.set("kill", {"active": True, "reason": reason, "ts": str(ts)})
        self.on_event("critical", "kill_switch", f"KILL SWITCH: {reason}", {"ts": str(ts)})
        return True

    def reset_kill(self, by: str = "operator") -> None:
        self.store.set("kill", {"active": False, "reset_by": by})
        self.on_event("warning", "kill_reset", f"kill switch reset by {by}")

    def risk_off(self) -> bool:
        return bool((self.store.get("risk_off") or {}).get("active"))

    # ------------------------------------------------------------------ clock / equity
    def _day_key(self, ts: pd.Timestamp) -> int:
        """Trading-day number: days since the epoch, with the day starting at the rollover hour.
        (Integer arithmetic: this runs once per bar in replays.)"""
        return (ts.value + (24 - self.cfg.rollover_hour_utc) * 3_600_000_000_000) // 86_400_000_000_000

    def _week_key(self, ts: pd.Timestamp) -> int:
        return (self._day_key(ts) + 3) // 7          # weeks start at the Sunday-evening open (= Monday's trading day)

    def update(self, ts: pd.Timestamp, equity: float) -> list[str]:
        """Feed the current equity. Returns actions the runner must take: 'flatten' and/or 'kill'."""
        s, c, actions = self.s, self.cfg, []
        dk, wk = self._day_key(ts), self._week_key(ts)
        changed = False
        if s["day"] != dk:
            s.update(day=dk, day_start_equity=equity, day_blocked=False)
            changed = True
        if s["week"] != wk:
            s.update(week=wk, week_start_equity=equity, week_blocked=False)
            changed = True
        if s["hwm"] is None or equity > s["hwm"]:
            s["hwm"] = equity
            self._dirty = True
        if not s["day_blocked"] and s["day_start_equity"] and equity <= s["day_start_equity"] * (1 - c.max_daily_loss_pct / 100):
            s["day_blocked"] = changed = True
            self.on_event("warning", "daily_loss_limit", f"daily loss limit hit: equity {equity:,.0f} vs day start {s['day_start_equity']:,.0f}", {"ts": str(ts)})
            if c.flatten_on_daily_limit:
                actions.append("flatten")
        if not s["week_blocked"] and s["week_start_equity"] and equity <= s["week_start_equity"] * (1 - c.max_weekly_loss_pct / 100):
            s["week_blocked"] = changed = True
            self.on_event("warning", "weekly_loss_limit", f"weekly loss limit hit: equity {equity:,.0f} vs week start {s['week_start_equity']:,.0f}", {"ts": str(ts)})
            if c.flatten_on_daily_limit and "flatten" not in actions:
                actions.append("flatten")
        if s["hwm"] and equity <= s["hwm"] * (1 - c.equity_floor_pct / 100):
            if self.kill(ts, f"equity {equity:,.0f} is {c.equity_floor_pct:g}% below high-water mark {s['hwm']:,.0f}"):
                actions.append("kill")
        if changed:
            self._save()
        return actions

    def on_trade_closed(self, ts: pd.Timestamp, pnl: float) -> None:
        s, c = self.s, self.cfg
        if pnl < 0:
            s["consec_losses"] += 1
            if c.consecutive_losses_pause and s["consec_losses"] >= c.consecutive_losses_pause:
                s["pause_until"] = str(ts + pd.Timedelta(minutes=c.pause_minutes))
                self.on_event("warning", "loss_streak_pause", f"{s['consec_losses']} consecutive losses: entries paused until {s['pause_until']}", {"ts": str(ts)})
                s["consec_losses"] = 0
        else:
            s["consec_losses"] = 0
        self._save()

    def check_health(self, now: pd.Timestamp, last_bar_ts: pd.Timestamp | None, bar_seconds: int, connected: bool, disconnected_since: pd.Timestamp | None) -> list[str]:
        """Live only. Kill on stale data or a broker that stays disconnected."""
        c, actions = self.cfg, []
        if not connected and disconnected_since is not None and (now - disconnected_since).total_seconds() > c.max_disconnect_s:
            if self.kill(now, f"broker disconnected for more than {c.max_disconnect_s}s"):
                actions.append("kill")
        if connected and last_bar_ts is not None and market_open(now) and market_open(now - pd.Timedelta(seconds=c.max_data_staleness_s + bar_seconds)):
            age = (now - last_bar_ts).total_seconds() - bar_seconds      # last_bar_ts is the bar's OPEN time
            if age > c.max_data_staleness_s:
                if self.kill(now, f"market data stale for {age:.0f}s"):
                    actions.append("kill")
        return actions

    # ------------------------------------------------------------------ pre-trade
    def check_entry(self, ts: pd.Timestamp, strategy: str, lots: float, stop_dist: float, spread: float, equity: float, open_positions: list) -> Decision:
        """open_positions: objects with .lots, .entry_px, .stop_px."""
        c, s, why = self.cfg, self.s, []
        ks = self.kill_state()
        if ks.get("active"):
            why.append(f"kill switch active ({ks.get('reason', '')})")
        if self.risk_off():
            why.append("manual risk-off")
        if s["day_blocked"]:
            why.append("daily loss limit reached")
        if s["week_blocked"]:
            why.append("weekly loss limit reached")
        if s["pause_until"] and ts < pd.Timestamp(s["pause_until"]):
            why.append(f"loss-streak pause until {s['pause_until']}")
        ev = self.calendar.in_blackout(ts, c.news_blackout_minutes)
        if ev:
            why.append(f"news blackout: {ev}")
        table = float(np.asarray(self.cost.spread_by_hour)[ts.hour]) * self.cost.spread_multiplier
        if c.spread_guard_mult > 0 and spread > table * c.spread_guard_mult + 1e-9:
            why.append(f"spread {spread:.2f} > {c.spread_guard_mult:g} x table {table:.2f}")
        if len(open_positions) >= c.max_positions:
            why.append(f"max positions ({c.max_positions}) open")
        if why:
            return Decision(False, 0.0, why)
        notes = []
        allowed_lots = min(lots, c.max_lots_per_trade)
        room = c.max_total_lots - sum(p.lots for p in open_positions)
        allowed_lots = min(allowed_lots, room)
        if np.isfinite(stop_dist) and stop_dist > 0 and equity > 0:
            open_risk = sum(abs(p.entry_px - p.stop_px) * p.lots * OZ_PER_LOT for p in open_positions if p.stop_px is not None)
            budget = equity * c.max_open_risk_pct / 100 - open_risk
            allowed_lots = min(allowed_lots, budget / (stop_dist * OZ_PER_LOT))
        allowed_lots = float(np.floor(allowed_lots * 100 + 1e-9) / 100)
        if allowed_lots < c.min_lots:
            return Decision(False, 0.0, ["no room under lot / open-risk limits"])
        if allowed_lots < lots - 1e-9:
            notes.append(f"size cut {lots:.2f} -> {allowed_lots:.2f} by risk limits")
        return Decision(True, allowed_lots, notes)
