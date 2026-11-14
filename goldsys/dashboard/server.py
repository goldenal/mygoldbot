"""Local operations dashboard: one page over the ledger, with the operator controls.

Standard library only. Binds to 127.0.0.1. Reads are open to anything on this machine; every
state-changing request must carry the per-process token that is embedded in the served page, so
another website open in the same browser cannot press the kill switch (it can neither read the
page nor guess the token)."""
from __future__ import annotations

import getpass
import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..chart import api as chart_api
from ..replay import api as replay_api
from ..live import control
from ..live.ledger import Ledger
from ..portfolio.config import Portfolio

PAGE = Path(__file__).with_name("page.html")


def make_dashboard(db: str | None, portfolio_path: str | None = None, host: str = "127.0.0.1", port: int = 8787, read_only: bool = False) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(24)
    who = f"{getpass.getuser()}@dashboard"

    def portfolio():
        return Portfolio.load(portfolio_path) if portfolio_path and Path(portfolio_path).exists() else None

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):
            hit = replay_api.get(self.path) or chart_api.route(self.path)
            if hit is not None:
                body = hit[2].replace(b"__TOKEN__", token.encode()) if self.path.split("?")[0] == "/chart" else hit[2]
                return self._send(hit[0], body, hit[1])
            if db is None and self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/chart")
                self.end_headers()
                return
            if db is None and self.path == "/api/state":
                return self._json(404, {"error": "no ledger: chart-only mode"})
            if self.path == "/":
                html = PAGE.read_text().replace("__TOKEN__", "" if read_only else token).replace("__DB__", Path(db).name)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                try:
                    st = control.state(Ledger(db), portfolio())
                    st["read_only"], st["db"] = read_only, Path(db).name
                    return self._json(200, st)
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/api/replay/"):           # replay writes only replay.sqlite, so it is allowed even when the ops dashboard is read-only
                host = self.headers.get("Host", "").split(":")[0]
                if host not in ("127.0.0.1", "localhost", "[::1]"):
                    return self._json(403, {"error": "bad host"})
                if not hmac.compare_digest(self.headers.get("X-Goldsys-Token", "").encode(), token.encode()):
                    return self._json(401, {"error": "bad token"})
                try:
                    req = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", "0")), 65536)) or b"{}")
                except ValueError:
                    return self._json(400, {"error": "bad json"})
                hit = replay_api.post(self.path, req if isinstance(req, dict) else {})
                return self._send(hit[0], hit[2], hit[1])
            if read_only or db is None or self.path != "/api/action":
                return self._json(403, {"error": "read-only"})
            if not hmac.compare_digest(self.headers.get("X-Goldsys-Token", "").encode(), token.encode()):
                return self._json(401, {"error": "bad token"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                led, act = Ledger(db), req.get("action")
                if act == "approve":
                    ok = control.approve(led, int(req["id"]), who)
                elif act == "veto":
                    ok = control.veto(led, int(req["id"]), who, str(req.get("reason", ""))[:200])
                elif act == "mode":
                    control.set_mode(led, str(req["strategy"]), str(req["mode"]), who)
                    ok = True
                elif act in ("risk_off", "risk_on"):
                    control.risk_off(led, act == "risk_off", who, str(req.get("reason", ""))[:200])
                    ok = True
                elif act == "kill":
                    control.kill(led, who, str(req.get("reason", "manual"))[:200])
                    ok = True
                elif act == "reset_kill":
                    control.reset_kill(led, who)
                    ok = True
                else:
                    return self._json(400, {"error": f"unknown action {act!r}"})
                self._json(200, {"ok": bool(ok)})
            except Exception as e:  # noqa: BLE001
                self._json(400, {"error": f"{type(e).__name__}: {e}"})

    srv = ThreadingHTTPServer((host, port), H)
    srv.token = token  # type: ignore[attr-defined]
    return srv
