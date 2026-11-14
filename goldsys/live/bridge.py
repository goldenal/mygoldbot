"""Bridge: run a Broker in one process (the Windows VPS with MT5) and use it from another (the
research box) over HTTP + JSON.

    VPS:        scripts/mt5_bridge_server.py            -> serve(MT5Broker(...), token)
    research:   RemoteBroker("http://127.0.0.1:8765", token)

Security model: the server can place real orders, so (1) every request must carry the shared token
(constant-time compare), (2) only the Broker methods listed in METHODS can be called, and (3) it
binds to 127.0.0.1 by default. The transport is plain HTTP: reach it through an SSH tunnel or a
private network (WireGuard/Tailscale), never by opening the port to the internet.
"""
from __future__ import annotations

import hmac
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import requests

from .broker import Account, Broker, BrokerError, Fill, Position, Quote

METHODS = ("connected", "quote", "account", "positions", "open_position", "close_position", "poll_fills", "bars", "now", "exit_fill")


def _enc(o):
    if isinstance(o, pd.DataFrame):
        return {"__df__": True, "index": [str(t) for t in o.index], "columns": list(o.columns), "data": o.to_numpy().tolist()}
    if isinstance(o, (Quote, Account, Position, Fill)):
        return {"__t__": type(o).__name__, **{k: _enc(v) for k, v in o.__dict__.items()}}
    if isinstance(o, pd.Timestamp):
        return {"__ts__": str(o)}
    if isinstance(o, float) and not math.isfinite(o):
        return {"__f__": "inf" if o > 0 else "-inf" if o < 0 else "nan"}
    if isinstance(o, list):
        return [_enc(x) for x in o]
    return o


_TYPES = {"Quote": Quote, "Account": Account, "Position": Position, "Fill": Fill}


def _dec(o):
    if isinstance(o, list):
        return [_dec(x) for x in o]
    if isinstance(o, dict):
        if "__ts__" in o:
            return pd.Timestamp(o["__ts__"])
        if "__f__" in o:
            return float(o["__f__"])
        if "__df__" in o:
            return pd.DataFrame(o["data"], columns=o["columns"], index=pd.DatetimeIndex(pd.to_datetime(o["index"], utc=True), name="ts"))
        if "__t__" in o:
            return _TYPES[o["__t__"]](**{k: _dec(v) for k, v in o.items() if k != "__t__"})
        return {k: _dec(v) for k, v in o.items()}
    return o


def make_server(broker: Broker, token: str, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if not token or len(token) < 16:
        raise ValueError("bridge token must be at least 16 characters")
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _reply(self, code: int, body: dict):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            auth = self.headers.get("Authorization", "")
            if not hmac.compare_digest(auth.encode(), f"Bearer {token}".encode()):
                return self._reply(401, {"error": "unauthorized"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                method = req.get("method")
                if self.path != "/rpc" or method not in METHODS:
                    return self._reply(404, {"error": f"unknown method {method!r}"})
                with lock:
                    result = getattr(broker, method)(*_dec(req.get("args", [])))
                self._reply(200, {"result": _enc(result)})
            except BrokerError as e:
                self._reply(200, {"broker_error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._reply(500, {"error": f"{type(e).__name__}: {e}"})

    return ThreadingHTTPServer((host, port), H)


class RemoteBroker(Broker):
    name = "remote"

    def __init__(self, url: str, token: str, timeout: float = 15.0):
        self.url, self.timeout = url.rstrip("/") + "/rpc", timeout
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def _call(self, method: str, *args):
        try:
            r = self.s.post(self.url, json={"method": method, "args": _enc(list(args))}, timeout=self.timeout)
        except requests.RequestException as e:
            raise BrokerError(f"bridge unreachable: {e}") from e
        if r.status_code != 200:
            raise BrokerError(f"bridge error {r.status_code}: {r.text[:200]}")
        body = r.json()
        if "broker_error" in body:
            raise BrokerError(body["broker_error"])
        return _dec(body["result"])

    def connected(self) -> bool:
        try:
            return bool(self._call("connected"))
        except BrokerError:
            return False

    def quote(self) -> Quote:
        return self._call("quote")

    def account(self) -> Account:
        return self._call("account")

    def positions(self) -> list[Position]:
        return self._call("positions")

    def open_position(self, strategy: str, direction: int, lots: float, stop_dist: float, target_dist: float | None) -> Fill:
        return self._call("open_position", strategy, direction, lots, stop_dist, target_dist)

    def close_position(self, position_id: str, reason: str = "signal") -> Fill:
        return self._call("close_position", position_id, reason)

    def poll_fills(self) -> list[Fill]:
        return self._call("poll_fills")

    def bars(self, timeframe: str = "M1", n: int = 500) -> pd.DataFrame:
        return self._call("bars", timeframe, n)

    def now(self) -> pd.Timestamp:
        return self._call("now")

    def exit_fill(self, position_id: str) -> Fill | None:
        return self._call("exit_fill", position_id)
