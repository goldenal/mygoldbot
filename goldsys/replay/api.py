"""HTTP routes for hybrid replay. GET is read-only; POST changes only replay.sqlite and requires the per-process token."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

from ..chart import data as chartdata
from . import report, session


def _json(code: int, obj) -> tuple[int, str, bytes]:
    return code, "application/json", json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()


def _f(x):
    """Optional number from JSON: None / "" mean 'not given'; NaN or infinity are refused."""
    if x is None or x == "":
        return None
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("not a finite number")
    return v


def _err(e: Exception):
    code = 400 if isinstance(e, (session.ReplayError, chartdata.ChartError, ValueError, KeyError)) else 500
    return _json(code, {"error": f"{type(e).__name__}: {e}"})


def get(raw_path: str):
    u = urlsplit(raw_path)
    p, q = u.path, {k: v[-1] for k, v in parse_qs(u.query).items()}
    if not p.startswith("/api/replay/"):
        return None
    try:
        if p == "/api/replay/sessions":
            return _json(200, {"sessions": session.list_sessions(), "strategies": chartdata.list_strategies(), "symbols": chartdata.symbols(),
                               "tags": {k: [{"id": a, "label": b} for a, b in v] for k, v in session.TAGS.items()}})
        if p == "/api/replay/state":
            return _json(200, session.get_session(q.get("session", "")).state())
        if p == "/api/replay/bars":
            s = session.get_session(q.get("session", ""))
            specs = [x for x in q.get("ind", "").split(",") if x]
            return _json(200, s.window(q.get("tf", "M15"), int(q.get("limit", 600)), int(q["end"]) if q.get("end") else None, specs))
        if p == "/api/replay/events":
            s = session.get_session(q.get("session", ""))
            return _json(200, {"events": s.events_window(int(q["start"]), int(q["end"]), int(q.get("tier", 2))), "upcoming": s.upcoming_events()})
        if p == "/api/replay/report":
            ids = [q["session"]] if q.get("session") else None
            rep = report.analyse(ids)
            if q.get("format") == "md":
                return 200, "text/markdown; charset=utf-8", report.to_markdown(rep).encode()
            return _json(200, rep)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return _json(404, {"error": "not found"})


def post(path: str, body: dict):
    if not path.startswith("/api/replay/"):
        return None
    try:
        if path == "/api/replay/new":
            s = session.new_session(str(body.get("symbol", "")), [str(x) for x in body.get("strategies", [])], str(body.get("start", "")), bool(body.get("blind")),
                                    float(body.get("equity", 10_000)))
            return _json(200, s.state())
        s = session.get_session(str(body.get("session", "")))
        if path == "/api/replay/step":
            return _json(200, s.advance(int(body.get("n", 1)), bool(body.get("to_signal")), body.get("to_tf_close")))
        if path == "/api/replay/decide":
            return _json(200, s.decide(int(body.get("proposal", -1)), str(body.get("decision", "")), str(body.get("tag", "")), str(body.get("note", "")),
                                       stop_px=_f(body.get("stop_px")), tgt_px=_f(body.get("tgt_px")), clear_target=bool(body.get("clear_target")), lots=_f(body.get("lots"))))
        if path == "/api/replay/manage":
            return _json(200, s.manage(int(body.get("trade", -1)), stop_px=_f(body.get("stop_px")), tgt_px=_f(body.get("tgt_px")),
                                       clear_target=bool(body.get("clear_target")), close=bool(body.get("close"))))
        if path == "/api/replay/finish":
            return _json(200, s.finish())
    except Exception as e:  # noqa: BLE001
        return _err(e)
    return _json(404, {"error": "not found"})
