"""Live/paper ledger (SQLite): the system's own verified track record plus everything needed to
replay a decision.

Tables
  bars      every closed execution bar the runner saw (so a bad trade can be re-run in the backtester)
  intents   every entry signal and what happened to it: executed / pending / vetoed / expired / blocked / off
  trades    one row per position, filled in at entry and completed at exit
  chain     append-only, hash-chained log of entries and exits. Each row stores sha256(prev_hash + payload),
            so any later edit or deletion of a trade breaks verification (`verify_chain`).
  equity    equity snapshots
  events    risk events, alerts, overrides, reconciliation findings
  kv        persisted state and operator controls (kill switch, risk-off, per-strategy mode)
The runner, the CLI and the dashboard are separate processes sharing this file (WAL mode).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from ..config import LIVE_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (ts TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, spread REAL, ticks REAL, volume REAL);
CREATE TABLE IF NOT EXISTS intents (
  id INTEGER PRIMARY KEY, ts TEXT, strategy TEXT, dir INTEGER, lots_signal REAL, lots REAL, ref_px REAL,
  stop_dist REAL, target_dist REAL, status TEXT, reasons TEXT, decided_by TEXT, decided_ts TEXT, expires_ts TEXT, position_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_intents_status ON intents(status);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY, position_id TEXT UNIQUE, strategy TEXT, symbol TEXT, stage TEXT, dir INTEGER, lots REAL,
  entry_ts TEXT, entry_px REAL, stop_px REAL, tgt_px REAL, risk_usd REAL, intent_id INTEGER,
  exit_ts TEXT, exit_px REAL, reason TEXT, pnl REAL, commission REAL, swap REAL, r_multiple REAL
);
CREATE TABLE IF NOT EXISTS chain (seq INTEGER PRIMARY KEY, ts TEXT, kind TEXT, payload TEXT, prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, balance REAL, equity REAL, open_lots REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, wall REAL, ts TEXT, level TEXT, kind TEXT, message TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

GENESIS = "0" * 64


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


class Ledger:
    def __init__(self, path: str | Path = LIVE_DB):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ kv (state + controls)
    def get(self, key: str, default=None):
        r = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, key: str, value) -> None:
        self.conn.execute("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, default=str)))
        self.conn.commit()

    # ------------------------------------------------------------------ events
    def event(self, level: str, kind: str, message: str, data: dict | None = None, ts=None) -> None:
        self.conn.execute("INSERT INTO events (wall, ts, level, kind, message, data) VALUES (?,?,?,?,?,?)",
                          (time.time(), str(ts) if ts is not None else None, level, kind, message, json.dumps(data or {}, default=str)))
        self.conn.commit()

    def events(self, n: int = 100, min_level: str | None = None) -> list[dict]:
        rows = [dict(r) for r in self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,))]
        if min_level:
            order = ["info", "warning", "error", "critical"]
            rows = [r for r in rows if order.index(r["level"]) >= order.index(min_level)]
        return rows

    # ------------------------------------------------------------------ bars / equity
    def store_bar(self, ts, o, h, l, c, spread, ticks=0.0, volume=0.0, commit: bool = True) -> None:
        self.conn.execute("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", (str(ts), o, h, l, c, spread, ticks, volume))
        if commit:
            self.conn.commit()

    def bars(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM bars ORDER BY ts", self.conn)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df.set_index("ts")

    def snapshot_equity(self, ts, balance: float, equity: float, open_lots: float) -> None:
        self.conn.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)", (str(ts), balance, equity, open_lots))
        self.conn.commit()

    def equity_curve(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM equity ORDER BY ts", self.conn)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df.set_index("ts")

    # ------------------------------------------------------------------ intents (approval queue)
    def add_intent(self, ts, strategy: str, direction: int, lots_signal: float, lots: float, ref_px: float, stop_dist: float, target_dist: float,
                   status: str, reasons: list[str] | str = "", expires_ts=None, decided_by: str = "system") -> int:
        if isinstance(reasons, list):
            reasons = "; ".join(reasons)
        cur = self.conn.execute(
            "INSERT INTO intents (ts, strategy, dir, lots_signal, lots, ref_px, stop_dist, target_dist, status, reasons, decided_by, decided_ts, expires_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(ts), strategy, direction, lots_signal, lots, ref_px, float(stop_dist), None if target_dist is None or target_dist == float("inf") else float(target_dist),
             status, reasons, decided_by, str(ts), None if expires_ts is None else str(expires_ts)))
        self.conn.commit()
        return int(cur.lastrowid)

    def set_intent(self, intent_id: int, status: str, by: str = "system", ts=None, reasons: str | None = None, position_id: str | None = None, only_if_status: str | None = None) -> bool:
        q = "UPDATE intents SET status=?, decided_by=?, decided_ts=?, reasons=COALESCE(?, reasons), position_id=COALESCE(?, position_id) WHERE id=?"
        args = [status, by, str(ts) if ts is not None else str(pd.Timestamp.now(tz="UTC")), reasons, position_id, intent_id]
        if only_if_status:
            q += " AND status=?"
            args.append(only_if_status)
        cur = self.conn.execute(q, args)
        self.conn.commit()
        return cur.rowcount > 0

    def intents(self, status: str | None = None, n: int = 200) -> list[dict]:
        if status:
            return [dict(r) for r in self.conn.execute("SELECT * FROM intents WHERE status=? ORDER BY id DESC LIMIT ?", (status, n))]
        return [dict(r) for r in self.conn.execute("SELECT * FROM intents ORDER BY id DESC LIMIT ?", (n,))]

    def intents_frame(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM intents ORDER BY id", self.conn)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    # ------------------------------------------------------------------ trades + hash chain
    def _append_chain(self, ts, kind: str, payload: dict) -> str:
        r = self.conn.execute("SELECT hash FROM chain ORDER BY seq DESC LIMIT 1").fetchone()
        prev = r[0] if r else GENESIS
        body = _canon(payload)
        h = hashlib.sha256((prev + kind + body).encode()).hexdigest()
        self.conn.execute("INSERT INTO chain (ts, kind, payload, prev_hash, hash) VALUES (?,?,?,?,?)", (str(ts), kind, body, prev, h))
        return h

    def open_trade(self, position_id: str, strategy: str, symbol: str, stage: str, direction: int, lots: float, entry_ts, entry_px: float,
                   stop_px: float | None, tgt_px: float | None, intent_id: int | None = None) -> None:
        risk = abs(entry_px - stop_px) * lots * 100.0 if stop_px is not None else None
        self.conn.execute(
            "INSERT INTO trades (position_id, strategy, symbol, stage, dir, lots, entry_ts, entry_px, stop_px, tgt_px, risk_usd, intent_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (position_id, strategy, symbol, stage, direction, lots, str(entry_ts), entry_px, stop_px, tgt_px, risk, intent_id))
        self._append_chain(entry_ts, "entry", {"position_id": position_id, "strategy": strategy, "symbol": symbol, "stage": stage, "dir": direction,
                                               "lots": round(lots, 2), "entry_ts": str(entry_ts), "entry_px": round(entry_px, 5)})
        self.conn.commit()

    def close_trade(self, position_id: str, exit_ts, exit_px: float, reason: str, pnl: float, commission: float = 0.0, swap: float = 0.0) -> dict | None:
        row = self.conn.execute("SELECT * FROM trades WHERE position_id=?", (position_id,)).fetchone()
        if row is None or row["exit_ts"] is not None:
            return None
        r_mult = pnl / row["risk_usd"] if row["risk_usd"] else None
        self.conn.execute("UPDATE trades SET exit_ts=?, exit_px=?, reason=?, pnl=?, commission=?, swap=?, r_multiple=? WHERE position_id=?",
                          (str(exit_ts), exit_px, reason, pnl, commission, swap, r_mult, position_id))
        self._append_chain(exit_ts, "exit", {"position_id": position_id, "exit_ts": str(exit_ts), "exit_px": round(exit_px, 5), "reason": reason,
                                             "pnl": round(pnl, 2), "commission": round(commission, 2), "swap": round(swap, 2)})
        self.conn.commit()
        out = dict(row)
        out.update(exit_ts=str(exit_ts), exit_px=exit_px, reason=reason, pnl=pnl, r_multiple=r_mult)
        return out

    def trades(self, closed_only: bool = False, strategy: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM trades", []
        cond = []
        if closed_only:
            cond.append("exit_ts IS NOT NULL")
        if strategy:
            cond.append("strategy=?")
            args.append(strategy)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        df = pd.read_sql_query(q + " ORDER BY id", self.conn, params=args)
        for c in ("entry_ts", "exit_ts"):
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")      # real-time fills carry fractional seconds, bar-time ones do not
        return df

    def open_trades(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM trades WHERE exit_ts IS NULL ORDER BY id")]

    def verify_chain(self) -> dict:
        """Recompute the hash chain and cross-check it against the trades table."""
        prev, n = GENESIS, 0
        chained: dict[str, dict] = {}
        for r in self.conn.execute("SELECT * FROM chain ORDER BY seq"):
            if r["prev_hash"] != prev or hashlib.sha256((prev + r["kind"] + r["payload"]).encode()).hexdigest() != r["hash"]:
                return {"ok": False, "rows": n, "error": f"chain broken at seq {r['seq']}"}
            prev = r["hash"]
            n += 1
            p = json.loads(r["payload"])
            chained.setdefault(p["position_id"], {}).update({r["kind"]: p})
        for t in self.conn.execute("SELECT * FROM trades ORDER BY id"):
            c = chained.pop(t["position_id"], None)
            if c is None or "entry" not in c:
                return {"ok": False, "rows": n, "error": f"trade {t['position_id']} is not in the chain"}
            e = c["entry"]
            if (e["dir"], e["lots"], e["entry_px"], e["entry_ts"]) != (t["dir"], round(t["lots"], 2), round(t["entry_px"], 5), t["entry_ts"]):
                return {"ok": False, "rows": n, "error": f"trade {t['position_id']} entry differs from the chain"}
            if (t["exit_ts"] is None) != ("exit" not in c):
                return {"ok": False, "rows": n, "error": f"trade {t['position_id']} exit state differs from the chain"}
            if "exit" in c and (c["exit"]["exit_px"], c["exit"]["pnl"]) != (round(t["exit_px"], 5), round(t["pnl"], 2)):
                return {"ok": False, "rows": n, "error": f"trade {t['position_id']} exit differs from the chain"}
        if chained:
            return {"ok": False, "rows": n, "error": f"{len(chained)} chained trade(s) missing from the trades table: {sorted(chained)[:3]}"}
        return {"ok": True, "rows": n, "head": prev}
