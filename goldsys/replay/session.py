"""Hybrid replay session: the clock walks M1 bars; strategies propose trades; you take or skip each one and say why.

Rules that keep it honest:
  * The page only ever receives bars up to the clock. Future bars, and the outcome of any proposal (taken or skipped), stay on the server
    until the clock has reached them. The report only counts decisions whose outcome the clock has passed, unless the session is finished.
  * The clock stops at every proposal; it will not move until you decide.
  * Blind mode relabels dates by a whole number of weeks (weekday and time of day stay real), so you cannot tell which year you are in.

What you can change: on Take you may set your own stop, target and size; while a trade is open you may move its stop (which switches off any
trailing rule), move or clear its target, or close it at the next open; and when a strategy that is already in a trade signals the OTHER way you are
offered a reversal (close at the next open and enter the other way), as the backtester would do. The strategy's own levels are always simulated too
(the "shadow"), so the report can separate what your filter did from what your levels and management did.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import secrets
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..chart import data as chartdata
from ..config import CALENDAR_FILE, OZ_PER_LOT, REPLAY_DB
from ..risk.calendar import EventCalendar
from . import sim

WEEK = 7 * 86400
WARM = 1500          # bars of history before the window, so indicators are warmed up

TAGS = {
    "take": [("trend_aligned", "With the higher-timeframe trend"), ("clean_level", "Clean level / structure"), ("momentum_confirms", "Momentum confirms"),
             ("session_timing", "Good session / timing"), ("good_rr", "Good risk:reward"), ("gut_feel", "Gut feel"), ("other", "Other")],
    "skip": [("against_htf_trend", "Against the higher-timeframe trend"), ("news_risk", "News / event risk"), ("weak_level", "Weak level / poor structure"),
             ("overextended", "Price overextended"), ("chop_range", "Choppy / ranging"), ("poor_rr", "Poor risk:reward"), ("late_in_session", "Late in the session"),
             ("spread_wide", "Spread / liquidity"), ("gut_feel", "Gut feel"), ("other", "Other")],
}


class ReplayError(ValueError):
    pass


_LOCK = threading.RLock()
_SESSIONS: dict[str, "Session"] = {}


# ---------------------------------------------------------------- storage

_DECISION_COLS = [
    ("kind", "TEXT DEFAULT 'entry'"), ("ref_id", "INTEGER"), ("trail_dist", "REAL"), ("trail_act", "REAL"), ("news_name", "TEXT"), ("news_min", "REAL"),
    ("inc_pnl", "REAL"), ("a_lots", "REAL"), ("a_stop_px", "REAL"), ("a_tgt_px", "REAL"), ("a_modified", "INTEGER DEFAULT 0"), ("a_exit_ts", "INTEGER"),
    ("a_exit_px", "REAL"), ("a_reason", "TEXT"), ("a_pnl", "REAL"), ("a_r", "REAL"), ("res_ts", "INTEGER"),
]


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(REPLAY_DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript("""
    CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, created TEXT, symbol TEXT, strategies TEXT, start_ts INTEGER, cursor_ts INTEGER,
        blind INTEGER, shift INTEGER, equity0 REAL, finished INTEGER DEFAULT 0, source TEXT);
    CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, seq INTEGER, strategy TEXT, dir INTEGER, signal_ts INTEGER,
        stop_dist REAL, target_dist REAL, lots REAL, ref_px REAL, decision TEXT, tag TEXT, note TEXT, decided_wall TEXT,
        fill_ts INTEGER, exit_ts INTEGER, entry_px REAL, exit_px REAL, stop_px REAL, tgt_px REAL, reason TEXT, pnl REAL, r REAL);
    CREATE INDEX IF NOT EXISTS ix_dec_session ON decisions(session, seq);
    CREATE TABLE IF NOT EXISTS adjustments(id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, decision_id INTEGER, eff_ts INTEGER, stop_px REAL, tgt_px REAL,
        clear_target INTEGER DEFAULT 0, close INTEGER DEFAULT 0, reason TEXT, wall TEXT);
    CREATE INDEX IF NOT EXISTS ix_adj ON adjustments(decision_id);
    """)
    have = {r["name"] for r in con.execute("PRAGMA table_info(decisions)")}
    for name, typ in _DECISION_COLS:                       # databases created before these features gain the columns in place
        if name not in have:
            con.execute(f"ALTER TABLE decisions ADD COLUMN {name} {typ}")
    return con


_CAL: dict = {}


def calendar() -> EventCalendar:
    """The economic calendar file, re-read when it changes. Empty when the file does not exist."""
    p = Path(CALENDAR_FILE)
    stamp = p.stat().st_mtime_ns if p.exists() else None
    if _CAL.get("stamp") != stamp or _CAL.get("path") != str(p):
        _CAL.update(stamp=stamp, path=str(p), cal=EventCalendar.from_csv(p))
    return _CAL["cal"]


def list_sessions() -> list[dict]:
    with _db() as c:
        rows = c.execute("SELECT s.*, (SELECT COUNT(*) FROM decisions d WHERE d.session=s.id AND d.decision IS NOT NULL) AS n_decisions FROM sessions s ORDER BY created DESC").fetchall()
    return [{"id": r["id"], "created": r["created"], "symbol": r["symbol"], "strategies": json.loads(r["strategies"]), "blind": bool(r["blind"]),
             "finished": bool(r["finished"]), "decisions": r["n_decisions"], "source": r["source"]} for r in rows]


# ---------------------------------------------------------------- session

def _nan(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


class Session:
    def __init__(self, row: sqlite3.Row):
        self.id = row["id"]
        self.symbol = row["symbol"]
        self.rels = json.loads(row["strategies"])
        self.blind = bool(row["blind"])
        self.shift = int(row["shift"])
        self.equity0 = float(row["equity0"])
        self.finished = bool(row["finished"])
        self.M = sim.market(self.symbol)
        self.ctx = {rel: sim.build_ctx(self.M, rel) for rel in self.rels}
        self.cursor = int(np.searchsorted(self.M.ns, int(row["cursor_ts"]) * 10**9))
        self.cursor = min(self.cursor, len(self.M.ns) - 1)

    # ---- helpers
    def _t(self, i: int) -> int:                      # real epoch seconds of M1 index i
        return int(self.M.ns[i] // 10**9)

    def _disp(self, ts: int | None) -> int | None:    # what the page sees
        return None if ts is None else int(ts) - self.shift

    def _real(self, ts: int) -> int:
        return int(ts) + self.shift

    def _idx(self, ts: int) -> int:
        return int(np.searchsorted(self.M.ns, int(ts) * 10**9))

    def _rows(self, where: str = "", args: tuple = ()) -> list[sqlite3.Row]:
        with _db() as c:
            return c.execute(f"SELECT * FROM decisions WHERE session=? {where} ORDER BY seq, id", (self.id, *args)).fetchall()

    def _row(self, did: int) -> sqlite3.Row | None:
        with _db() as c:
            return c.execute("SELECT * FROM decisions WHERE session=? AND id=?", (self.id, int(did))).fetchone()

    def _save_cursor(self) -> None:
        with _db() as c:
            c.execute("UPDATE sessions SET cursor_ts=? WHERE id=?", (self._t(self.cursor), self.id))

    def pending(self) -> list[sqlite3.Row]:
        return self._rows("AND decision IS NULL")

    def _adjustments(self, did: int) -> list[dict]:
        with _db() as c:
            rows = c.execute("SELECT * FROM adjustments WHERE decision_id=? ORDER BY id", (int(did),)).fetchall()
        out = []
        for r in rows:
            a: dict = {"i": self._idx(r["eff_ts"])}
            if r["stop_px"] is not None:
                a["stop_px"] = float(r["stop_px"])
            if r["tgt_px"] is not None:
                a["tgt_px"] = float(r["tgt_px"])
            if r["clear_target"]:
                a["clear_target"] = True
            if r["close"]:
                a["close"] = True
                a["reason"] = r["reason"] or "manual"
            out.append(a)
        return out

    def _add_adjustment(self, did: int, eff_i: int, *, stop_px=None, tgt_px=None, clear_target=False, close=False, reason="manual") -> None:
        with _db() as c:
            c.execute("INSERT INTO adjustments(session,decision_id,eff_ts,stop_px,tgt_px,clear_target,close,reason,wall) VALUES(?,?,?,?,?,?,?,?,?)",
                      (self.id, int(did), self._t(eff_i), stop_px, tgt_px, int(clear_target), int(close), reason, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))

    def _taken_sim(self, r: sqlite3.Row, extra: list[dict] = (), until_i: int | None = None) -> dict | None:
        """Re-run a taken trade with its own levels and every manual change made so far (plus `extra`)."""
        C = self.ctx[r["strategy"]]
        return sim.simulate(self.M, C, self._idx(r["fill_ts"]), int(r["dir"]), float(r["a_lots"]), stop_px=float(r["a_stop_px"]),
                            tgt_px=None if r["a_tgt_px"] is None else float(r["a_tgt_px"]), trail_dist=r["trail_dist"], trail_act=float(r["trail_act"] or 0.0),
                            adjust=self._adjustments(r["id"]) + list(extra), until_i=until_i)

    def _store_actual(self, did: int, res: dict, risk: float) -> None:
        with _db() as c:
            c.execute("UPDATE decisions SET a_exit_ts=?,a_exit_px=?,a_reason=?,a_pnl=?,a_r=? WHERE id=?",
                      (self._t(res["exit_i"]), res["exit_px"], res["reason"], res["pnl"], (res["pnl"] / risk) if risk > 0 else None, int(did)))

    def _open_trades(self) -> dict[str, list[tuple[int, int, int, int]]]:
        """Per strategy, its TAKEN trades as (fill index, exit index, direction, decision id)."""
        out: dict[str, list] = {}
        for r in self._rows("AND decision='take'"):
            out.setdefault(r["strategy"], []).append((self._idx(r["fill_ts"]), self._idx(r["a_exit_ts"]), int(r["dir"]), int(r["id"])))
        return out

    def _seen(self) -> set[tuple[str, int]]:
        return {(r["strategy"], int(r["signal_ts"])) for r in self._rows()}

    @staticmethod
    def _kind_at(trades: list, j: int, d: int) -> tuple[str, int | None] | None:
        """What a signal at strategy-bar close j means for a strategy with these taken trades: a new entry, a reversal of the open trade, or nothing."""
        for fi, xi, dd, did in trades:
            if fi <= j < xi:                               # a trade is open at that close
                return None if dd == d else ("reverse", did)
        return ("entry", None)

    # ---- clock
    def _next_signal(self, limit: int) -> int | None:
        """Smallest M1 index in (cursor, limit] at which some strategy has an eligible, not-yet-seen proposal."""
        trades, seen = self._open_trades(), self._seen()
        best = None
        for rel, C in self.ctx.items():
            p = int(np.searchsorted(C.cand, self.cursor, "right"))
            while p < len(C.cand) and C.cand[p] <= limit and (best is None or C.cand[p] < best):
                j = int(C.cand[p])
                if (rel, self._t(j)) not in seen and self._kind_at(trades.get(rel, []), j, int(C.cand_dir[p])) is not None:
                    best = j
                    break
                p += 1
        return best

    def _news(self, ts: int) -> tuple[str | None, float | None]:
        cal = calendar()
        if len(cal) == 0:
            return None, None
        t = pd.Timestamp(ts, unit="s", tz="UTC")
        ev = cal.events["ts"]
        if t < ev.iloc[0] - pd.Timedelta(days=1) or t > ev.iloc[-1] + pd.Timedelta(days=1):
            return None, None                       # the calendar does not cover this date: unknown, not "no news"
        n = cal.nearest(t, 1)
        return (n[1], round(n[0], 1)) if n else (None, None)

    def _propose(self, j: int) -> None:
        trades, seen = self._open_trades(), self._seen()
        nm, nmin = self._news(self._t(j))
        with _db() as c:
            seq = (c.execute("SELECT COALESCE(MAX(seq),0) FROM decisions WHERE session=?", (self.id,)).fetchone()[0]) + 1
            for rel, C in self.ctx.items():
                p = int(np.searchsorted(C.cand, j, "left"))
                if p >= len(C.cand) or C.cand[p] != j or (rel, self._t(j)) in seen:
                    continue
                k = self._kind_at(trades.get(rel, []), j, int(C.cand_dir[p]))
                if k is None:
                    continue
                c.execute("INSERT INTO decisions(session,seq,kind,ref_id,strategy,dir,signal_ts,stop_dist,target_dist,lots,ref_px,trail_dist,trail_act,news_name,news_min) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (self.id, seq, k[0], k[1], rel, int(C.cand_dir[p]), self._t(j), float(C.cand_stop[p]), _nan(C.cand_target[p]), float(C.cand_lots[p]), float(self.M.close[j]),
                           _nan(C.cand_trail[p]), float(C.cand_trail_act[p]), nm, nmin))
                seq += 1

    def advance(self, n: int = 1, to_signal: bool = False, to_tf_close: str | None = None) -> dict:
        with _LOCK:
            if self.finished:
                raise ReplayError("this session is finished")
            if self.pending():
                raise ReplayError("decide the pending proposal first")
            last = len(self.M.ns) - 1
            if self.cursor >= last:
                raise ReplayError("end of data")
            if to_signal:
                target = last
            elif to_tf_close:
                if to_tf_close not in chartdata.TF_SECONDS:
                    raise ReplayError(f"unknown timeframe {to_tf_close!r}")
                tf_ns = chartdata.TF_SECONDS[to_tf_close] * 10**9
                bucket = (self.M.ns // tf_ns) * tf_ns
                end = int(np.searchsorted(bucket, bucket[self.cursor], "right")) - 1
                if end <= self.cursor:                                      # already at a close: go to the next one
                    end = int(np.searchsorted(bucket, bucket[min(self.cursor + 1, last)], "right")) - 1
                target = min(end, last)
            else:
                target = min(self.cursor + max(1, min(int(n), 20000)), last)
            j = self._next_signal(target)
            if j is not None:
                self.cursor = j
                self._propose(j)
            else:
                self.cursor = target
            self._save_cursor()
            return self.state()

    # ---- decisions
    def _min_gap(self, spread: float) -> float:
        return max(0.3, 3.0 * float(spread))

    def decide(self, proposal_id: int, decision: str, tag: str, note: str = "", *, stop_px: float | None = None, tgt_px: float | None = None,
               clear_target: bool = False, lots: float | None = None) -> dict:
        with _LOCK:
            if decision not in ("take", "skip"):
                raise ReplayError("decision must be 'take' or 'skip'")
            if tag not in {t for t, _ in TAGS[decision]}:
                raise ReplayError(f"pick a reason tag for '{decision}': {[t for t, _ in TAGS[decision]]}")
            rows = [r for r in self.pending() if r["id"] == int(proposal_id)]
            if not rows:
                raise ReplayError("no such pending proposal")
            r = rows[0]
            if decision == "skip" and (stop_px is not None or tgt_px is not None or lots is not None or clear_target):
                raise ReplayError("levels and size only apply when you take a trade")
            C = self.ctx[r["strategy"]]
            j = self._idx(r["signal_ts"])
            d = int(r["dir"])
            f = j + 1
            tdist = r["target_dist"]
            # the strategy's own trade (the shadow): its levels, its size
            sh = sim.simulate(self.M, C, f, d, float(r["lots"]), stop_dist=float(r["stop_dist"]), target_dist=tdist, trail_dist=r["trail_dist"], trail_act=float(r["trail_act"] or 0.0))
            if sh is None:
                raise ReplayError("there is no later bar to fill on (end of data); skip it")
            risk0 = float(r["stop_dist"]) * float(r["lots"]) * OZ_PER_LOT
            inc = None
            old = old_trunc = None
            res_ts = self._t(sh["exit_i"])
            if r["kind"] == "reverse":
                old = self._row(r["ref_id"])
                if old is None or old["decision"] != "take":
                    raise ReplayError("the trade this would reverse is gone")
                if not (self._idx(old["fill_ts"]) <= j < self._idx(old["a_exit_ts"])):
                    raise ReplayError("the trade this would reverse is no longer open")
                old_trunc = self._taken_sim(old, [{"i": f, "close": True, "reason": "reverse"}])
                inc = old_trunc["pnl"] + sh["pnl"] - float(old["a_pnl"])
                res_ts = max(self._t(sh["exit_i"]), int(old["a_exit_ts"]))        # the answer needs the old trade's full path too
            # what you actually do
            spread = float(self.M.spread[j])
            entry_est = float(r["ref_px"]) + (spread if d > 0 else 0.0)
            a_stop_px, a_tgt_px, a_lots, modified = sh["stop_px"], sh["tgt_px"], float(r["lots"]), False
            act = sh
            if decision == "take":
                gap = self._min_gap(spread)
                if stop_px is not None:
                    if (d > 0 and stop_px >= entry_est - gap) or (d < 0 and stop_px <= entry_est + gap):
                        raise ReplayError(f"stop must be at least {gap:.2f} on the losing side of the entry (~{entry_est:.2f})")
                    a_stop_px, modified = float(stop_px), True
                if clear_target:
                    a_tgt_px, modified = None, True
                elif tgt_px is not None:
                    if (d > 0 and tgt_px <= entry_est + gap) or (d < 0 and tgt_px >= entry_est - gap):
                        raise ReplayError(f"target must be at least {gap:.2f} on the winning side of the entry (~{entry_est:.2f})")
                    a_tgt_px, modified = float(tgt_px), True
                if lots is not None:
                    ml = float(sim.StrategySpec.from_yaml(str(sim.strategy_path(r["strategy"]))).sizing.max_lots)
                    if not (0.01 <= float(lots) <= ml):
                        raise ReplayError(f"size must be between 0.01 and {ml} lots")
                    a_lots, modified = round(float(lots), 2), (modified or round(float(lots), 2) != float(r["lots"]))
                if modified:
                    act = sim.simulate(self.M, C, f, d, a_lots, stop_px=a_stop_px, tgt_px=a_tgt_px, trail_dist=r["trail_dist"], trail_act=float(r["trail_act"] or 0.0))
            with _db() as c:
                c.execute("UPDATE decisions SET decision=?,tag=?,note=?,decided_wall=?,fill_ts=?,exit_ts=?,entry_px=?,exit_px=?,stop_px=?,tgt_px=?,reason=?,pnl=?,r=?,inc_pnl=?,res_ts=? WHERE id=?",
                          (decision, tag, (note or "")[:300], dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), self._t(sh["fill_i"]), self._t(sh["exit_i"]),
                           sh["entry_px"], sh["exit_px"], sh["stop_px"], sh["tgt_px"], sh["reason"], sh["pnl"],
                           ((inc if inc is not None else sh["pnl"]) / risk0) if risk0 > 0 else None, inc, res_ts, r["id"]))
                if decision == "take":
                    c.execute("UPDATE decisions SET a_lots=?,a_stop_px=?,a_tgt_px=?,a_modified=?,a_exit_ts=?,a_exit_px=?,a_reason=?,a_pnl=?,a_r=? WHERE id=?",
                              (a_lots, a_stop_px, a_tgt_px, int(modified), self._t(act["exit_i"]), act["exit_px"], act["reason"], act["pnl"],
                               (act["pnl"] / (abs(act["entry_px"] - a_stop_px) * a_lots * OZ_PER_LOT)) if abs(act["entry_px"] - a_stop_px) > 0 else None, r["id"]))
            if decision == "take" and old is not None:                        # close the old trade at the new trade's fill, as the backtester does on a reversal
                self._add_adjustment(old["id"], f, close=True, reason="reverse")
                self._store_actual(old["id"], old_trunc, abs(old_trunc["entry_px"] - float(old["a_stop_px"])) * float(old["a_lots"]) * OZ_PER_LOT)
            return self.state()

    def manage(self, decision_id: int, *, stop_px: float | None = None, tgt_px: float | None = None, clear_target: bool = False, close: bool = False) -> dict:
        """Change a trade that is open right now. Takes effect from the next bar; the new outcome stays hidden like any other."""
        with _LOCK:
            if self.finished:
                raise ReplayError("this session is finished")
            r = self._row(decision_id)
            if r is None or r["decision"] != "take":
                raise ReplayError("no such open trade")
            c, last = self.cursor, len(self.M.ns) - 1
            fi, xi = self._idx(r["fill_ts"]), self._idx(r["a_exit_ts"])
            if not (fi <= c < xi):
                raise ReplayError("that trade is not open at the clock")
            if c >= last:
                raise ReplayError("end of data")
            d = int(r["dir"])
            px = float(self.M.close[c]) + (float(self.M.spread[c]) if d < 0 else 0.0)
            gap = self._min_gap(float(self.M.spread[c]))
            if not (close or stop_px is not None or tgt_px is not None or clear_target):
                raise ReplayError("nothing to change")
            if stop_px is not None and ((d > 0 and stop_px >= px - gap) or (d < 0 and stop_px <= px + gap)):
                raise ReplayError(f"stop must be at least {gap:.2f} beyond the current price ({px:.2f}) on the losing side")
            if tgt_px is not None and ((d > 0 and tgt_px <= px + gap) or (d < 0 and tgt_px >= px - gap)):
                raise ReplayError(f"target must be at least {gap:.2f} beyond the current price ({px:.2f}) on the winning side")
            self._add_adjustment(r["id"], c + 1, stop_px=stop_px, tgt_px=tgt_px, clear_target=clear_target, close=close, reason="manual")
            res = self._taken_sim(r)
            self._store_actual(r["id"], res, abs(res["entry_px"] - float(r["a_stop_px"])) * float(r["a_lots"]) * OZ_PER_LOT)
            with _db() as cx:
                cx.execute("UPDATE decisions SET a_modified=1 WHERE id=?", (r["id"],))
            return self.state()

    def finish(self) -> dict:
        with _LOCK:
            if self.pending():
                raise ReplayError("decide the pending proposal first")
            with _db() as c:
                c.execute("UPDATE sessions SET finished=1 WHERE id=?", (self.id,))
            self.finished = True
            return self.state()

    # ---- what the page may see
    def _news_json(self, r: sqlite3.Row) -> dict | None:
        return None if r["news_name"] is None else {"name": r["news_name"], "minutes": r["news_min"]}

    def _proposal_json(self, r: sqlite3.Row) -> dict:
        d = int(r["dir"])
        j = self._idx(r["signal_ts"])
        spread = float(self.M.spread[j])
        est = float(r["ref_px"]) + (spread if d > 0 else 0.0)
        sd, td = float(r["stop_dist"]), r["target_dist"]
        out = {"id": r["id"], "kind": r["kind"], "strategy": r["strategy"], "name": self.ctx[r["strategy"]].name, "tf": self.ctx[r["strategy"]].tf, "dir": d,
               "t": self._disp(r["signal_ts"]), "ref_px": round(float(r["ref_px"]), 3), "entry_est": round(est, 3), "stop_dist": round(sd, 3),
               "stop_est": round(est - d * sd, 3), "target_dist": None if td is None else round(float(td), 3),
               "target_est": None if td is None else round(est + d * float(td), 3), "rr": None if td is None else round(float(td) / sd, 2), "lots": float(r["lots"]),
               "risk_usd": round(sd * float(r["lots"]) * OZ_PER_LOT, 2), "spread": round(spread, 3), "news": self._news_json(r),
               "trail_dist": None if r["trail_dist"] is None else round(float(r["trail_dist"]), 3), "min_gap": round(self._min_gap(spread), 2),
               "max_lots": float(sim.StrategySpec.from_yaml(str(sim.strategy_path(r["strategy"]))).sizing.max_lots)}
        if r["kind"] == "reverse":
            old = self._row(r["ref_id"])
            c = self.cursor
            px = float(self.M.close[c]) if int(old["dir"]) > 0 else float(self.M.close[c] + self.M.spread[c])
            out["reverses"] = {"id": old["id"], "dir": int(old["dir"]), "entry_px": round(float(old["entry_px"]), 3), "lots": float(old["a_lots"]),
                               "unrealised": round((px - float(old["entry_px"])) * int(old["dir"]) * float(old["a_lots"]) * OZ_PER_LOT, 2)}
        return out

    def upcoming_events(self, hours: float = 24.0, max_tier: int = 2) -> list[dict]:
        """Scheduled events in the next `hours` of clock time. The schedule is public in advance, so this is not look-ahead (no values, only times)."""
        cal = calendar()
        if len(cal) == 0:
            return []
        now = pd.Timestamp(self._t(self.cursor), unit="s", tz="UTC")
        ev = cal.events
        m = (ev["ts"] > now) & (ev["ts"] <= now + pd.Timedelta(hours=hours)) & (ev["tier"] <= max_tier)
        return [{"t": self._disp(int(e.ts.timestamp())), "name": e.name, "tier": int(e.tier), "in_min": round((e.ts - now).total_seconds() / 60.0)} for e in ev[m].itertuples()]

    def events_window(self, start: int, end: int, max_tier: int = 2) -> list[dict]:
        """Past events (up to the clock) between two display times, for markers on the chart."""
        cal = calendar()
        if len(cal) == 0:
            return []
        lo = pd.Timestamp(self._real(start), unit="s", tz="UTC")
        hi = min(pd.Timestamp(self._real(end), unit="s", tz="UTC"), pd.Timestamp(self._t(self.cursor), unit="s", tz="UTC"))
        ev = cal.events
        m = (ev["ts"] >= lo) & (ev["ts"] <= hi) & (ev["tier"] <= max_tier)
        return [{"t": self._disp(int(e.ts.timestamp())), "name": e.name, "tier": int(e.tier)} for e in ev[m].itertuples()]

    def state(self) -> dict:
        c = self.cursor
        now_ts = self._t(c)
        taken = self._rows("AND decision='take'")
        closed, opened, queued, realised = [], [], [], 0.0
        for r in taken:
            d = int(r["dir"])
            base = {"id": r["id"], "strategy": r["strategy"], "name": self.ctx[r["strategy"]].name, "dir": d, "lots": float(r["a_lots"]), "tag": r["tag"],
                    "entry_t": self._disp(r["fill_ts"]), "entry_px": round(float(r["entry_px"]), 3), "modified": bool(r["a_modified"]), "kind": r["kind"],
                    "stop_px": round(float(r["a_stop_px"]), 3), "tgt_px": None if r["a_tgt_px"] is None else round(float(r["a_tgt_px"]), 3)}
            if r["fill_ts"] > now_ts:
                queued.append({k: base[k] for k in ("id", "strategy", "name", "dir", "lots")})
            elif r["a_exit_ts"] <= now_ts:
                realised += float(r["a_pnl"])
                closed.append({**base, "exit_t": self._disp(r["a_exit_ts"]), "exit_px": round(float(r["a_exit_px"]), 3), "reason": r["a_reason"], "pnl": round(float(r["a_pnl"]), 2),
                               "r": None if r["a_r"] is None else round(float(r["a_r"]), 2)})
            else:
                px = float(self.M.close[c]) if d > 0 else float(self.M.close[c] + self.M.spread[c])
                lv = self._taken_sim(r, until_i=c)
                trailing = bool(r["trail_dist"]) and not any(a.get("stop_px") is not None for a in self._adjustments(r["id"]))
                cur_stop = lv["stop_px"] if lv and lv.get("open") else float(r["a_stop_px"])
                cur_tgt = lv["tgt_px"] if lv and lv.get("open") else r["a_tgt_px"]
                opened.append({**base, "stop_px": round(cur_stop, 3), "tgt_px": None if cur_tgt is None else round(float(cur_tgt), 3), "trailing": trailing and not (lv and lv.get("closing")), "closing": bool(lv and lv.get("closing")),
                               "unrealised": round((px - float(r["entry_px"])) * d * float(r["a_lots"]) * OZ_PER_LOT, 2), "min_gap": round(self._min_gap(float(self.M.spread[c])), 2)})
        decided = self._rows("AND decision IS NOT NULL")
        recent = [{"id": r["id"], "t": self._disp(r["signal_ts"]), "name": self.ctx[r["strategy"]].name, "dir": int(r["dir"]), "decision": r["decision"], "tag": r["tag"],
                   "note": r["note"], "kind": r["kind"], "edited": bool(r["a_modified"])} for r in decided[-12:]][::-1]
        pend = [self._proposal_json(r) for r in self.pending()]
        return {"id": self.id, "symbol": self.symbol, "blind": self.blind, "finished": self.finished,
                "strategies": [{"path": rel, "name": C.name, "tf": C.tf} for rel, C in self.ctx.items()],
                "t": self._disp(now_ts), "price": round(float(self.M.close[c]), 3), "spread": round(float(self.M.spread[c]), 3), "at_end": c >= len(self.M.ns) - 1,
                "status": "awaiting" if pend else "ready", "pending": pend, "open": opened, "queued": queued, "closed": closed[-40:][::-1], "closed_n": len(closed),
                "equity0": self.equity0, "equity": round(self.equity0 + realised + sum(o["unrealised"] for o in opened), 2), "realised": round(realised, 2),
                "decisions": {"take": sum(1 for r in decided if r["decision"] == "take"), "skip": sum(1 for r in decided if r["decision"] == "skip")},
                "upcoming": self.upcoming_events(), "calendar": len(calendar()) > 0,
                "recent": recent, "tags": {k: [{"id": a, "label": b} for a, b in v] for k, v in TAGS.items()}}

    # ---- chart window (bars <= clock; the last bar is the partial one the clock is inside)
    def window(self, tf: str, limit: int, end: int | None, specs: list[str]) -> dict:
        if tf not in chartdata.TF_SECONDS:
            raise ReplayError(f"unknown timeframe {tf!r}")
        limit = max(10, min(int(limit), chartdata.MAX_LIMIT))
        M, c = self.M, self.cursor
        tf_ns = chartdata.TF_SECONDS[tf] * 10**9
        cur_bucket = (int(M.ns[c]) // tf_ns) * tf_ns
        full = chartdata.bars(self.symbol, tf)
        fns = full.index.as_unit("ns").asi8
        stop_at = int(np.searchsorted(fns, cur_bucket, "left"))                      # complete bars: strictly before the bucket the clock is in
        partial = None
        if end is not None:
            stop_at = min(stop_at, int(np.searchsorted(fns, self._real(end) * 10**9, "right")))
        else:
            lo = int(np.searchsorted(M.ns, cur_bucket, "left"))
            seg = M.frame.iloc[lo:c + 1]
            partial = pd.DataFrame({"open": [seg["open"].iloc[0]], "high": [seg["high"].max()], "low": [seg["low"].min()], "close": [seg["close"].iloc[-1]],
                                    "spread": [seg["spread"].mean()], "ticks": [seg["ticks"].sum()], "volume": [seg["volume"].sum()]},
                                   index=pd.DatetimeIndex([pd.Timestamp(cur_bucket, unit="ns", tz="UTC")], name="ts"))
        take = max(0, stop_at - (limit + WARM))
        frame = full.iloc[take:stop_at]
        if partial is not None:
            frame = pd.concat([frame, partial])
        if frame.empty:
            raise ReplayError("no bars before that time")
        n_out = min(limit, len(frame))
        sl = frame.iloc[-n_out:]
        times = (sl.index.as_unit("s").asi8 - self.shift).tolist()
        nn = lambda s: s.round(3).astype(object).where(s.notna(), None).tolist()   # noqa: E731
        out = {"symbol": self.symbol, "tf": tf, "n": n_out, "has_more": (len(frame) - n_out) > 0 or take > 0, "t": times,
               "o": nn(sl["open"]), "h": nn(sl["high"]), "l": nn(sl["low"]), "c": nn(sl["close"]), "vol": sl["volume"].round(0).tolist(), "spread": nn(sl["spread"]),
               "first": self._disp(self._t(0)), "last": self._disp(self._t(c)), "has_newer": False, "indicators": []}
        for spec in specs:
            ind = chartdata.compute(spec, frame, tf)
            out["indicators"].append({**ind, "series": [{**s, "v": s["v"][-n_out:]} for s in ind["series"]]})
        return out


# ---------------------------------------------------------------- lifecycle

def new_session(symbol: str, rels: list[str], start: str, blind: bool = False, equity: float = 10_000.0) -> Session:
    if not rels:
        raise ReplayError("pick at least one strategy")
    if len(rels) > 6:
        raise ReplayError("at most 6 strategies per session")
    M = sim.market(symbol)
    for rel in rels:
        try:
            sim.strategy_path(rel)
        except chartdata.ChartError as e:
            raise ReplayError(str(e)) from e
    try:
        t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    except Exception as e:  # noqa: BLE001
        raise ReplayError(f"bad start date {start!r}; use YYYY-MM-DD") from e
    i = int(np.searchsorted(M.ns, t0 * 10**9))
    if i < 2000:
        raise ReplayError(f"start at least a few days after the first bar ({pd.Timestamp(int(M.ns[0]), tz='UTC').date()}) so indicators have history")
    if i >= len(M.ns) - 100:
        raise ReplayError("start is too close to the end of the data")
    shift = 0
    if blind:
        target = random.Random(secrets.randbits(64)).randrange(int(dt.datetime(1996, 1, 1).timestamp()), int(dt.datetime(2008, 12, 31).timestamp()))
        shift = ((t0 - target) // WEEK) * WEEK
    sid = secrets.token_hex(4)
    meta = {}
    try:
        from ..data import mt5csv
        meta = mt5csv.read_source(symbol) or {}
    except Exception:  # noqa: BLE001
        pass
    with _db() as c:
        c.execute("INSERT INTO sessions(id,created,symbol,strategies,start_ts,cursor_ts,blind,shift,equity0,source) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (sid, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), symbol, json.dumps(rels), int(M.ns[i] // 10**9), int(M.ns[i] // 10**9),
                   int(blind), int(shift), float(equity), meta.get("source_name", "")))
        row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    s = Session(row)
    _SESSIONS[sid] = s
    return s


def get_session(sid: str) -> Session:
    with _LOCK:
        s = _SESSIONS.get(sid)
        if s is not None and s.M.stamp == chartdata._M1.get(s.symbol, (None,))[0]:
            return s
        with _db() as c:
            row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise ReplayError("no such session")
        s = Session(row)
        _SESSIONS[sid] = s
        return s
