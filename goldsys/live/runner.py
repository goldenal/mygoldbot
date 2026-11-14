"""Live runner: closed bars in, risk-checked and (optionally) human-approved orders out.

For each strategy the order of events inside one bar of its timeframe is the screening engine's:
    open   entries / reversals decided on the PREVIOUS bar's signal
    intra  stops and targets (the broker's job; PaperBroker resolves them on the bar's range)
    close  signal / time / max-bars exits, and the entry decision for the next open
so a replay through PaperBroker reproduces the backtester (tests/test_live_parity.py).

Two ways to drive it:
    feed_bar(...)    simulated execution: replays and paper trading on a live bar feed
    live_step(now)   a real broker: poll closed M1 bars, act at the current quote

Between the signal and the order sit, in this order: strategy mode (auto / approve / off), the
risk manager (limits, kill switch, news blackout, spread guard) and the approval queue. Exits are
never blocked by any of them. Everything is written to the ledger.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.store import resample_bars
from ..dsl.compile import TF_MINUTES, feature_frame
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..risk.limits import RiskManager
from .broker import Broker, Fill
from .ledger import Ledger
from .signals import PrecomputedSignals, RollingSignals

MIN_NS = 60_000_000_000
MODES = ("auto", "approve", "off")


@dataclass
class Slot:
    name: str
    spec: StrategySpec
    source: object
    weight: float = 1.0
    mode: str = "auto"
    drift: dict = field(default_factory=dict)       # {"window": 30, "p05_r": -0.4}
    tf_ns: int = 0
    max_bars: int = 10**9
    time_exit_min: int | None = None
    agg: list | None = None                         # [start_ns, o, h, l, c, spread_sum, n, ticks, volume]
    n_closed: int = 0
    pending: dict | None = None
    position_id: str | None = None
    pos_dir: int = 0
    entry_idx: int = -1
    live_r: list = field(default_factory=list)
    drift_tripped: bool = False

    def __post_init__(self):
        self.tf_ns = TF_MINUTES[self.spec.timeframe] * MIN_NS
        self.max_bars = self.spec.exit.max_bars or 10**9
        t = self.spec.exit.time_exit_utc
        if t:
            h, m = t.split(":")
            self.time_exit_min = int(h) * 60 + int(m)


def make_slots(specs: list[StrategySpec], bars_m1: pd.DataFrame | None = None, precomputed: bool = True, weights: dict | None = None,
               modes: dict | None = None, drift: dict | None = None, window: int = 1500, cache_key: str | None = None) -> list[Slot]:
    """precomputed=True: signals for the whole of bars_m1 are evaluated once (replay / shadow).
    precomputed=False: rolling re-evaluation, with bars_m1 (if given) as indicator warm-up (live)."""
    slots = []
    for spec in specs:
        if precomputed:
            src = PrecomputedSignals(spec, feature_frame(bars_m1, spec.timeframe, cache_key))
        else:
            hist = resample_bars(bars_m1, spec.timeframe) if bars_m1 is not None and len(bars_m1) else None
            src = RollingSignals(spec, hist, window=window)
        slots.append(Slot(spec.name, spec, src, (weights or {}).get(spec.name, 1.0), (modes or {}).get(spec.name, "auto"), (drift or {}).get(spec.name, {})))
    return slots


class LiveRunner:
    def __init__(self, broker: Broker, ledger: Ledger, slots: list[Slot], risk: RiskManager | None = None, cost: CostModel | None = None,
                 stage: str = "paper", notify=None, size_factor: float = 1.0, exec_minutes: int = 1, approval_ttl_s: int = 900,
                 approval_max_drift_r: float = 0.25, snapshot_minutes: int = 60, store_bars: bool = True, stale_s: int = 120, realtime: bool = False):
        self.broker, self.ledger, self.slots = broker, ledger, slots
        self.cost = cost or CostModel()
        sink = notify or (lambda level, kind, msg, data=None, ts=None: ledger.event(level, kind, msg, data, ts))
        self._said: dict[tuple, float] = {}

        def _notify(level, kind, msg, data=None, ts=None):
            # a fault that persists is reported when it starts and then at most every 15 minutes, not on every poll
            if level in ("error", "critical"):
                t = time.time()
                if t - self._said.get((kind, msg), -1e18) < 900:
                    return
                self._said[(kind, msg)] = t
            sink(level, kind, msg, data, ts)

        self.notify = _notify
        self.risk = risk or RiskManager(cost=self.cost, store=ledger)
        self.risk.on_event = lambda level, kind, msg, data=None: self.notify(level, kind, msg, data, (data or {}).get("ts"))
        self.stage, self.size_factor = stage, size_factor
        self.exec_ns = exec_minutes * MIN_NS
        self.approval_ttl = pd.Timedelta(seconds=approval_ttl_s)
        self.approval_max_drift_r = approval_max_drift_r
        self.snapshot_ns = snapshot_minutes * MIN_NS
        self.store_bars, self.stale_s = store_bars, stale_s
        self.realtime = realtime            # True: poll operator controls (kill switch) on every step, not just at snapshots
        self._by_pos: dict[str, Slot] = {}
        self._pending: dict[int, tuple[Slot, dict]] = {}
        self._last_snapshot_ns = -1
        self._bars_since_commit = 0
        self.last_bar_ns: int | None = None
        self._disconnected_since: pd.Timestamp | None = None
        self._orphans_seen: set[str] = set()
        self._lock_wall = 0.0
        if realtime:
            self._take_lock()
        for iid in [r["id"] for r in ledger.intents("pending")] + [r["id"] for r in ledger.intents("approved")]:
            ledger.set_intent(iid, "expired", reasons="runner restarted")       # approvals do not survive a restart
        for s in slots:
            t = ledger.trades(closed_only=True, strategy=s.name)
            s.live_r = [float(x) for x in t["r_multiple"].dropna()]

    # ------------------------------------------------------------------ one runner per ledger
    def _take_lock(self) -> None:
        """Two runners on one ledger (and so on one account) would each act on the other's positions."""
        lock = self.ledger.get("runner_lock") or {}
        if lock.get("pid") not in (None, os.getpid()) and time.time() - lock.get("wall", 0) < 60:
            raise RuntimeError(f"another runner (pid {lock['pid']}) is active on this ledger; stop it first, or wait 60s if it crashed")
        self._beat()

    def _beat(self) -> None:
        self._lock_wall = time.time()
        self.ledger.set("runner_lock", {"pid": os.getpid(), "wall": self._lock_wall})

    def release_lock(self) -> None:
        if (self.ledger.get("runner_lock") or {}).get("pid") == os.getpid():
            self.ledger.set("runner_lock", {})

    # ------------------------------------------------------------------ driving: simulated execution
    def feed_bar(self, ts_ns: int, o: float, h: float, l: float, c: float, spread: float, ticks: float = 0.0, volume: float = 0.0) -> None:
        """One closed execution bar, with a simulated broker (PaperBroker)."""
        ts = pd.Timestamp(ts_ns, tz="UTC")
        self.broker.set_quote(ts, o, spread)
        self.on_open(ts_ns, ts)
        self.broker.on_bar(ts, o, h, l, c, spread)
        self._sync_fills()
        self.on_close(ts_ns, ts, o, h, l, c, spread, ticks, volume)
        self._housekeeping(ts_ns, ts)

    def replay(self, bars: pd.DataFrame, finish: bool = True) -> None:
        """Feed a frame of execution bars (index = open time; 'spread' already the cost model's)."""
        ns = bars.index.as_unit("ns").asi8
        cols = [bars[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "spread")]
        tk = bars["ticks"].to_numpy(dtype=float) if "ticks" in bars else np.zeros(len(bars))
        vol = bars["volume"].to_numpy(dtype=float) if "volume" in bars else np.zeros(len(bars))
        o, h, l, c, sp = cols
        for i in range(len(ns)):
            self.feed_bar(int(ns[i]), o[i], h[i], l[i], c[i], sp[i], tk[i], vol[i])
        if finish:
            self.finish()

    def finish(self, reason: str = "end") -> None:
        self.flatten_all(reason)
        if self.last_bar_ns is not None:
            acct = self.broker.account()
            self.ledger.snapshot_equity(pd.Timestamp(self.last_bar_ns + self.exec_ns, tz="UTC"), acct.balance, acct.equity, sum(p.lots for p in self.broker.positions()))
        self.ledger.conn.commit()
        self.risk.flush()

    # ------------------------------------------------------------------ driving: real broker
    def live_step(self, now: pd.Timestamp) -> int:
        """One iteration of the live loop. Returns the number of new bars processed."""
        connected = False
        try:
            connected = self.broker.connected()
        except Exception as e:  # noqa: BLE001
            self.notify("error", "broker", f"connection check failed: {e}", None, now)
        if connected:
            self._disconnected_since = None
        elif self._disconnected_since is None:
            self._disconnected_since = now
            self.notify("warning", "broker", "broker disconnected", None, now)
        last_ts = pd.Timestamp(self.last_bar_ns, tz="UTC") if self.last_bar_ns is not None else None
        if "kill" in self.risk.check_health(now, last_ts, self.exec_ns // 1_000_000_000, connected, self._disconnected_since) and connected and self.risk.cfg.flatten_on_kill:
            self.flatten_all("kill")
        if not connected:
            return 0
        self._sync_fills()
        bars = self.broker.bars("M1", 300)
        n_new = 0
        if len(bars):
            ns = bars.index.as_unit("ns").asi8
            for i in range(len(bars)):
                if self.last_bar_ns is not None and ns[i] <= self.last_bar_ns:
                    continue
                r = bars.iloc[i]
                self.on_close(int(ns[i]), bars.index[i], float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["spread"]),
                              float(r.get("ticks", 0.0)), float(r.get("volume", 0.0)))
                n_new += 1
        if n_new:
            open_ns = self.last_bar_ns + self.exec_ns
            fresh = (now - pd.Timestamp(open_ns, tz="UTC")).total_seconds() < self.stale_s
            self.on_open(open_ns, now, fresh=fresh)
        self._process_approvals(now)
        self._reconcile_positions(now)
        self._housekeeping(int(now.as_unit("ns").value), now)
        return n_new

    def adopt_positions(self, now: pd.Timestamp) -> None:
        """After a restart with a real broker: pick up positions that are still open at the venue."""
        open_rows = {r["position_id"]: r for r in self.ledger.open_trades()}
        by_name = {s.name: s for s in self.slots}
        for p in self.broker.positions():
            s = by_name.get(p.strategy)
            if s is None or p.id not in open_rows:
                continue
            s.position_id, s.pos_dir = p.id, p.dir
            held = int((now - pd.Timestamp(open_rows[p.id]["entry_ts"])).total_seconds() * 1e9 // s.tf_ns)
            s.entry_idx = s.n_closed - held
            self._by_pos[p.id] = s
            self.notify("info", "adopt", f"adopted open position {p.id} for {s.name}", None, now)

    # ------------------------------------------------------------------ bar events
    def on_open(self, ts_ns: int, ts: pd.Timestamp, fresh: bool = True) -> None:
        for s in self.slots:
            if s.agg is not None and ts_ns >= s.agg[0] + s.tf_ns:
                self._close_tf(s, ts)                 # bar closed late (missing final minute / session gap)
            if s.pending is not None:
                intent, s.pending = s.pending, None
                self._act(s, intent, ts_ns, ts, fresh)
        if self._pending:
            self._process_approvals(ts)

    def on_close(self, ts_ns: int, ts: pd.Timestamp, o: float, h: float, l: float, c: float, spread: float, ticks: float = 0.0, volume: float = 0.0) -> None:
        self.last_bar_ns = ts_ns
        if self.store_bars:
            self.ledger.store_bar(ts, o, h, l, c, spread, ticks, volume, commit=False)
            self._bars_since_commit += 1
            if not self.broker.simulated or self._bars_since_commit >= 500:
                self.ledger.conn.commit()
                self._bars_since_commit = 0
        for s in self.slots:
            start = ts_ns - ts_ns % s.tf_ns
            a = s.agg
            if a is not None and a[0] != start:
                self._close_tf(s, ts)
                a = None
            if a is None:
                s.agg = a = [start, o, h, l, c, spread, 1, ticks, volume]
            else:
                if h > a[2]:
                    a[2] = h
                if l < a[3]:
                    a[3] = l
                a[4] = c
                a[5] += spread
                a[6] += 1
                a[7] += ticks
                a[8] += volume
            if ts_ns + self.exec_ns >= start + s.tf_ns:
                self._close_tf(s, ts)

    def _close_tf(self, s: Slot, ts: pd.Timestamp) -> None:
        a, s.agg = s.agg, None
        sig = s.source.on_bar(a[0], {"open": a[1], "high": a[2], "low": a[3], "close": a[4], "spread": a[5] / a[6], "ticks": a[7], "volume": a[8]})
        i = s.n_closed
        s.n_closed += 1
        if sig is None:
            s.pending = None
            return
        if s.position_id is not None:
            self._sync_fills()
        if s.position_id is not None:
            if (sig["exit_long"] if s.pos_dir == 1 else sig["exit_short"]) or sig["time_exit_flag"] or (i - s.entry_idx) >= s.max_bars:
                self._close(s, "time" if sig["time_exit_flag"] else "signal", ts)
        want = 1 if (sig["sig_long"] and not sig["sig_short"]) else -1 if (sig["sig_short"] and not sig["sig_long"]) else 0
        s.pending = {"dir": want, "lots": sig["lots"], "stop_dist": sig["stop_dist"], "target_dist": sig["target_dist"], "bar_ns": a[0]} if want != 0 and sig["lots"] > 0 else None

    # ------------------------------------------------------------------ entries
    def _mode(self, s: Slot) -> str:
        m = self.ledger.get(f"mode:{s.name}")
        return m["mode"] if m and m.get("mode") in MODES else s.mode

    def _act(self, s: Slot, intent: dict, ts_ns: int, ts: pd.Timestamp, fresh: bool) -> None:
        d = intent["dir"]
        if s.position_id is not None:
            self._sync_fills()
        if s.pos_dir == d and s.position_id is not None:
            return                                    # already positioned that way
        if s.time_exit_min is not None and (ts_ns // MIN_NS) % 1440 >= s.time_exit_min:
            return                                    # no entries at/after the time exit
        if s.position_id is not None:
            self._close(s, "reverse", ts)             # opposite signal: the exit half is never blocked
        q = self.broker.quote()
        ref_px = q.ask if d == 1 else q.bid
        lots = math.floor(intent["lots"] * s.weight * self.size_factor * 100 + 1e-9) / 100
        base = dict(ts=ts, strategy=s.name, direction=d, lots_signal=intent["lots"], ref_px=ref_px, stop_dist=intent["stop_dist"], target_dist=intent["target_dist"])
        if not fresh:
            self.ledger.add_intent(lots=lots, status="stale", reasons="signal bar arrived too late to act on", **base)
            return
        mode = self._mode(s)
        if mode == "off":
            self.ledger.add_intent(lots=lots, status="off", reasons="strategy mode is off", **base)
            return
        if lots < self.risk.cfg.min_lots:
            self.ledger.add_intent(lots=lots, status="too_small", reasons="size after weight/scale is below the minimum lot", **base)
            return
        dec = self.risk.check_entry(ts, s.name, lots, intent["stop_dist"], q.spread, self.broker.account().equity, self.broker.positions())
        if not dec.allowed:
            self.ledger.add_intent(lots=lots, status="blocked", reasons=dec.reasons, **base)
            return
        if mode == "approve":
            for iid, (ps, _) in list(self._pending.items()):
                if ps is s:
                    self.ledger.set_intent(iid, "expired", ts=ts, reasons="superseded by a newer signal", only_if_status="pending")
                    del self._pending[iid]
            iid = self.ledger.add_intent(lots=dec.lots, status="pending", reasons=dec.reasons, expires_ts=ts + self.approval_ttl, **base)
            self._pending[iid] = (s, dict(intent, lots_live=dec.lots, ref_px=ref_px))
            self.notify("warning", "approval_needed", f"#{iid} {s.name} {'BUY' if d == 1 else 'SELL'} {dec.lots:.2f} lots @ ~{ref_px:.2f} awaits approval (expires {ts + self.approval_ttl})", None, ts)
            return
        iid = self.ledger.add_intent(lots=dec.lots, status="accepted", reasons=dec.reasons, **base)
        self._execute(s, iid, d, dec.lots, intent["stop_dist"], intent["target_dist"], ts)

    def _execute(self, s: Slot, intent_id: int, d: int, lots: float, stop_dist: float, target_dist: float, ts: pd.Timestamp, by: str = "system") -> None:
        try:
            f = self.broker.open_position(s.name, d, lots, stop_dist, target_dist)
        except Exception as e:  # noqa: BLE001
            self.ledger.set_intent(intent_id, "failed", ts=ts, reasons=f"order failed: {e}")
            self.notify("error", "order_failed", f"{s.name}: entry order failed: {e}", None, ts)
            return
        s.position_id, s.pos_dir, s.entry_idx = f.position_id, d, s.n_closed
        self._by_pos[f.position_id] = s
        self.ledger.open_trade(f.position_id, s.name, f.symbol, self.stage, d, f.lots, f.ts, f.price, f.stop_px, f.tgt_px, intent_id)
        self.ledger.set_intent(intent_id, "executed", by=by, ts=ts, position_id=f.position_id)

    def _process_approvals(self, ts: pd.Timestamp) -> None:
        for iid, (s, intent) in list(self._pending.items()):
            row = self.ledger.conn.execute("SELECT status, decided_by, expires_ts FROM intents WHERE id=?", (iid,)).fetchone()
            status = row["status"] if row else "vetoed"
            if status == "pending":
                if ts >= pd.Timestamp(row["expires_ts"]):
                    self.ledger.set_intent(iid, "expired", ts=ts, reasons="not approved in time")
                    del self._pending[iid]
                continue
            del self._pending[iid]
            if status != "approved":
                continue
            by = row["decided_by"]
            if ts >= pd.Timestamp(row["expires_ts"]):
                self.ledger.set_intent(iid, "expired", by=by, ts=ts, reasons="approved after expiry")
                continue
            if s.position_id is not None:
                self.ledger.set_intent(iid, "expired", by=by, ts=ts, reasons="strategy already has a position")
                continue
            q = self.broker.quote()
            px = q.ask if intent["dir"] == 1 else q.bid
            if abs(px - intent["ref_px"]) > self.approval_max_drift_r * intent["stop_dist"]:
                self.ledger.set_intent(iid, "expired", by=by, ts=ts, reasons=f"price moved {px - intent['ref_px']:+.2f} since the signal")
                continue
            dec = self.risk.check_entry(ts, s.name, intent["lots_live"], intent["stop_dist"], q.spread, self.broker.account().equity, self.broker.positions())
            if not dec.allowed:
                self.ledger.set_intent(iid, "blocked", by=by, ts=ts, reasons="; ".join(dec.reasons))
                continue
            self._execute(s, iid, intent["dir"], dec.lots, intent["stop_dist"], intent["target_dist"], ts, by=by)

    # ------------------------------------------------------------------ exits
    def _close(self, s: Slot, reason: str, ts: pd.Timestamp) -> None:
        try:
            f = self.broker.close_position(s.position_id, reason)
        except Exception as e:  # noqa: BLE001
            self.notify("error", "order_failed", f"{s.name}: close ({reason}) failed: {e}", None, ts)
            return
        self._on_exit(f)

    def _sync_fills(self) -> None:
        for f in self.broker.poll_fills():
            if f.kind == "exit":
                self._on_exit(f)

    def _on_exit(self, f: Fill) -> None:
        s = self._by_pos.pop(f.position_id, None)
        row = self.ledger.close_trade(f.position_id, f.ts, f.price, f.reason, f.pnl, f.commission, f.swap)
        if s is None:
            return
        s.position_id, s.pos_dir = None, 0
        self.risk.on_trade_closed(f.ts, f.pnl)
        if row and row.get("r_multiple") is not None:
            s.live_r.append(row["r_multiple"])
            self._check_drift(s, f.ts)

    def _check_drift(self, s: Slot, ts) -> None:
        w, thr = s.drift.get("window"), s.drift.get("p05_r")
        if not w or thr is None or s.drift_tripped or len(s.live_r) < w:
            return
        m = float(np.mean(s.live_r[-w:]))
        if m < thr:
            s.drift_tripped = True
            self.ledger.set(f"mode:{s.name}", {"mode": "off", "by": "drift_monitor", "ts": str(ts), "reason": f"rolling {w}-trade mean R {m:.3f} < {thr:.3f}"})
            self.notify("critical", "drift_breach", f"{s.name}: rolling {w}-trade expectancy {m:.3f}R is below the backtest 5th percentile {thr:.3f}R. Strategy switched off.", None, ts)

    def flatten_all(self, reason: str, ts: pd.Timestamp | None = None) -> None:
        self._sync_fills()
        for s in self.slots:
            if s.position_id is not None:
                self._close(s, reason, ts)
            s.pending = None

    # ------------------------------------------------------------------ housekeeping
    def _housekeeping(self, ts_ns: int, ts: pd.Timestamp) -> None:
        acct = self.broker.account()
        actions = self.risk.update(ts, acct.equity)
        if ("kill" in actions and self.risk.cfg.flatten_on_kill) or "flatten" in actions:
            self.flatten_all("kill" if "kill" in actions else "risk", ts)
        due = ts_ns - self._last_snapshot_ns >= self.snapshot_ns or self._last_snapshot_ns < 0
        if self.realtime and time.time() - self._lock_wall > 10:
            self._beat()
        if (due or self.realtime) and any(s.position_id for s in self.slots) and self.risk.cfg.flatten_on_kill and self.risk.kill_state().get("active"):
            self.flatten_all("kill", ts)              # the operator pressed the kill switch from another process
            acct = self.broker.account()
        if due:
            self._last_snapshot_ns = ts_ns
            self.ledger.snapshot_equity(ts, acct.balance, acct.equity, sum(p.lots for p in self.broker.positions()))
            self.risk.flush()

    def _reconcile_positions(self, ts: pd.Timestamp) -> None:
        """Real brokers: what the venue holds must equal what we think we hold."""
        try:
            held = {p.id: p for p in self.broker.positions()}
        except Exception as e:  # noqa: BLE001
            self.notify("error", "reconcile", f"could not read positions: {e}", None, ts)
            return
        ours = {s.position_id: s for s in self.slots if s.position_id}
        for pid, s in ours.items():
            if pid not in held:
                self._sync_fills()
                if s.position_id != pid:
                    continue
                f = None
                try:
                    f = self.broker.exit_fill(pid)            # the exit happened but was never delivered: ask the venue's history
                except Exception as e:  # noqa: BLE001
                    self.notify("error", "reconcile", f"{s.name}: could not look up the exit of {pid}: {e}", None, ts)
                    continue
                if f is not None:
                    self.notify("warning", "reconcile", f"{s.name}: exit of {pid} recovered from the venue's history ({f.reason}, {f.pnl:+.2f})", None, ts)
                    self._on_exit(f)
                else:
                    # nothing at the venue and nothing in its history: close our record at the current quote, flagged, and stop
                    q = self.broker.quote()
                    row = next((r for r in self.ledger.open_trades() if r["position_id"] == pid), None)
                    px = q.bid if s.pos_dir == 1 else q.ask
                    pnl = (px - row["entry_px"]) * s.pos_dir * row["lots"] * 100.0 if row else 0.0
                    self._on_exit(Fill(pid, s.name, row["symbol"] if row else "", ts, s.pos_dir, row["lots"] if row else 0.0, "exit", px, "lost", pnl))
                    self.risk.kill(ts, f"position {pid} ({s.name}) vanished at the broker with no record of its exit; its P&L in the ledger is an ESTIMATE")
        names = {s.name for s in self.slots}
        for pid, p in held.items():
            if pid not in ours and p.strategy in names and pid not in self._orphans_seen:
                self._orphans_seen.add(pid)
                self.notify("critical", "reconcile", f"orphan position {pid} ({p.strategy} {p.lots} lots) at the broker is not tracked by the runner", None, ts)
