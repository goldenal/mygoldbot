"""SQLite experiment registry. Every backtest trial is a row; the trial COUNT feeds DSR."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from ..config import REGISTRY_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
  id INTEGER PRIMARY KEY,
  ts REAL, source TEXT, family TEXT, strategy TEXT, spec_hash TEXT,
  params TEXT, spec TEXT, data_start TEXT, data_end TEXT, cost_version TEXT,
  trades INTEGER, net_profit REAL, sharpe REAL, max_drawdown REAL, profit_factor REAL,
  oos_sharpe REAL, passed INTEGER, notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_trials_hash ON trials(spec_hash);
CREATE TABLE IF NOT EXISTS hypotheses (
  id INTEGER PRIMARY KEY, ts REAL, source TEXT, name TEXT, rationale TEXT, spec TEXT, spec_hash TEXT, outcome TEXT
);
"""


class Registry:
    def __init__(self, path: str | Path = REGISTRY_DB):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def log_trial(self, source: str, spec, metrics: dict, data_start, data_end, cost_version: str, oos_sharpe=None, passed=None, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO trials (ts, source, family, strategy, spec_hash, params, spec, data_start, data_end, cost_version, trades, net_profit, sharpe, max_drawdown, profit_factor, oos_sharpe, passed, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), source, spec.family, spec.name, spec.hash(), json.dumps(spec.params, default=str), spec.to_yaml(), str(data_start), str(data_end), cost_version,
             int(metrics.get("trades", 0)), float(metrics.get("net_profit", 0)), float(metrics.get("sharpe", 0)), float(metrics.get("max_drawdown", 0)),
             float(metrics.get("profit_factor", 0)) if metrics.get("profit_factor") not in (None, float("inf")) else 99.0, oos_sharpe, None if passed is None else int(passed), notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_hypothesis(self, source: str, name: str, rationale: str, spec_yaml: str, spec_hash: str, outcome: str) -> None:
        self.conn.execute("INSERT INTO hypotheses (ts, source, name, rationale, spec, spec_hash, outcome) VALUES (?,?,?,?,?,?,?)", (time.time(), source, name, rationale, spec_yaml, spec_hash, outcome))
        self.conn.commit()

    def n_trials(self, family: str | None = None) -> int:
        if family:
            return int(self.conn.execute("SELECT COUNT(*) FROM trials WHERE family=?", (family,)).fetchone()[0])
        return int(self.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def best(self, n: int = 20, order: str = "oos_sharpe") -> list[dict]:
        rows = self.conn.execute(f"SELECT strategy, family, spec_hash, params, trades, net_profit, sharpe, max_drawdown, oos_sharpe, passed FROM trials ORDER BY {order} DESC LIMIT ?", (n,)).fetchall()
        keys = ["strategy", "family", "spec_hash", "params", "trades", "net_profit", "sharpe", "max_drawdown", "oos_sharpe", "passed"]
        return [dict(zip(keys, r)) for r in rows]

    def tested_hashes(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT DISTINCT spec_hash FROM trials")}

    def hypotheses(self, n: int = 50) -> list[dict]:
        rows = self.conn.execute("SELECT name, rationale, outcome, spec_hash FROM hypotheses ORDER BY ts DESC LIMIT ?", (n,)).fetchall()
        return [dict(zip(["name", "rationale", "outcome", "spec_hash"], r)) for r in rows]
