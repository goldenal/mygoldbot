"""Alerts: every alert is written to the ledger's event log and the console; warnings and above are
also pushed to Telegram when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in the environment
(or .env). Sending never raises: a failed alert must not take the trading loop down."""
from __future__ import annotations

import os
import sys
import time

LEVELS = ["info", "warning", "error", "critical"]


def load_dotenv(path: str = ".env") -> None:
    """Minimal KEY=VALUE loader so secrets stay out of the shell history and the repo."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Alerter:
    def __init__(self, ledger=None, push_level: str = "warning", console: bool = True, min_gap_s: float = 30.0):
        self.ledger, self.console = ledger, console
        self.push_level = LEVELS.index(push_level)
        self.token, self.chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        self.min_gap_s = min_gap_s
        self._last: dict[str, float] = {}

    def __call__(self, level: str, kind: str, message: str, data: dict | None = None, ts=None) -> None:
        if self.ledger is not None:
            self.ledger.event(level, kind, message, data, ts)
        if self.console:
            print(f"[{level.upper():8s}] {ts if ts is not None else ''} {kind}: {message}", file=sys.stderr, flush=True)
        if LEVELS.index(level) >= self.push_level and self.token and self.chat:
            now = time.time()
            if now - self._last.get(kind, 0.0) < self.min_gap_s:      # do not flood on a repeating fault
                return
            self._last[kind] = now
            try:
                import requests
                requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage", json={"chat_id": self.chat, "text": f"goldsys {level.upper()} [{kind}]\n{message}"}, timeout=10)
            except Exception as e:  # noqa: BLE001
                print(f"[alert] telegram push failed: {e}", file=sys.stderr, flush=True)
