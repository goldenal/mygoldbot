"""HTTP routes for the chart page. Pure functions of (path, query) so the dashboard server can delegate to them.
Read-only: nothing here changes the ledger or any file."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import data

HERE = Path(__file__).parent
PAGE = HERE / "page.html"
STATIC = HERE / "static"
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".txt": "text/plain; charset=utf-8"}


def _json(code: int, obj) -> tuple[int, str, bytes]:
    return code, "application/json", json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()


def route(raw_path: str) -> tuple[int, str, bytes] | None:
    """Returns (status, content-type, body), or None if the path is not a chart path."""
    u = urlsplit(raw_path)
    p, q = u.path, {k: v[-1] for k, v in parse_qs(u.query).items()}
    if p == "/chart":
        return 200, "text/html; charset=utf-8", PAGE.read_bytes()
    if p.startswith("/chart-static/"):
        f = (STATIC / p[len("/chart-static/"):]).resolve()
        if f.parent != STATIC.resolve() or not f.is_file() or f.suffix not in STATIC_TYPES:
            return _json(404, {"error": "not found"})
        return 200, STATIC_TYPES[f.suffix], f.read_bytes()
    if not p.startswith("/api/chart/"):
        return None
    try:
        if p == "/api/chart/meta":
            return _json(200, {
                "symbols": data.symbols(), "timeframes": list(data.TF_SECONDS), "tf_seconds": data.TF_SECONDS,
                "indicators": [{"id": k, "label": v[0], "pane": v[1], "defaults": v[2]} for k, v in data.CATALOG.items()],
                "strategies": data.list_strategies(),
            })
        if p == "/api/chart/bars":
            specs = [s for s in q.get("ind", "").split(",") if s]
            end = int(q["end"]) if q.get("end") else None
            after = int(q["after"]) if q.get("after") else None
            return _json(200, data.window(q.get("symbol", ""), q.get("tf", "M15"), end, int(q.get("limit", 1500)), specs, after))
        if p == "/api/chart/events":
            from ..replay import session as _rs                       # the calendar loader lives next to the replay's use of it
            import pandas as pd
            cal = _rs.calendar()
            ev = cal.events
            lo, hi = pd.Timestamp(int(q["start"]), unit="s", tz="UTC"), pd.Timestamp(int(q["end"]), unit="s", tz="UTC")
            m = (ev["ts"] >= lo) & (ev["ts"] <= hi) & (ev["tier"] <= int(q.get("tier", 2)))
            return _json(200, {"loaded": len(cal) > 0, "events": [{"t": int(e.ts.timestamp()), "name": e.name, "tier": int(e.tier)} for e in ev[m].itertuples()]})
        if p == "/api/chart/trades":
            return _json(200, data.strategy_trades(q.get("symbol", ""), q.get("strategy", "")))
    except (data.ChartError, ValueError, KeyError) as e:
        return _json(400, {"error": f"{type(e).__name__}: {e}"})
    except Exception as e:  # noqa: BLE001
        return _json(500, {"error": f"{type(e).__name__}: {e}"})
    return _json(404, {"error": "not found"})
