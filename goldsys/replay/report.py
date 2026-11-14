"""Score the decisions: did your takes beat the strategies' raw signals, and which of your reasons actually predicted anything?

Every proposal has a shadow outcome (what taking it would have produced, same simulator, same costs). A decision counts only once the
replay clock has passed its exit, so the report of a running session cannot tell you how a pending idea will end. Finishing a session
reveals everything. Outcomes are in R (profit divided by the money risked at the stop)."""
from __future__ import annotations

import numpy as np

from .session import _db

MIN_N = 30          # below this a mean is shown but not interpreted
BOOT = 4000


def _stats(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"n": 0, "mean_r": None, "total_r": 0.0, "win_rate": None}
    return {"n": int(len(r)), "mean_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 2), "win_rate": round(float((r > 0).mean()), 3)}


def _load(session_ids: list[str] | None) -> tuple[list[dict], dict]:
    with _db() as c:
        sess = c.execute("SELECT id,cursor_ts,finished,symbol,blind FROM sessions" + (" WHERE id IN (%s)" % ",".join("?" * len(session_ids)) if session_ids else ""),
                         tuple(session_ids or ())).fetchall()
        info = {s["id"]: s for s in sess}
        rows = c.execute("SELECT * FROM decisions WHERE decision IS NOT NULL AND r IS NOT NULL" + (" AND session IN (%s)" % ",".join("?" * len(session_ids)) if session_ids else ""),
                         tuple(session_ids or ())).fetchall() if sess else []
    return [dict(r) for r in rows], info


def analyse(session_ids: list[str] | None = None) -> dict:
    rows, info = _load(session_ids)
    resolved, hidden = [], 0
    for r in rows:
        s = info.get(r["session"])
        # a decision counts once the clock has passed the end of everything it depends on (its shadow trade, and for a reversal the old trade's path)
        if s and (s["finished"] or (r.get("res_ts") or r["exit_ts"]) <= s["cursor_ts"]):
            resolved.append(r)
        else:
            hidden += 1
    out: dict = {"sessions": len(info), "decisions": len(rows), "resolved": len(resolved), "unresolved": hidden, "min_n": MIN_N, "warnings": []}
    if not resolved:
        out["warnings"].append("No resolved decisions yet.")
        return out
    r = np.array([x["r"] for x in resolved], dtype=float)
    take = np.array([x["decision"] == "take" for x in resolved])
    out["all_signals"] = _stats(r)
    out["taken"] = _stats(r[take])
    out["skipped"] = _stats(r[~take])
    out["take_rate"] = round(float(take.mean()), 3)
    # does the filter add anything? edge = mean R of what you took minus mean R of everything proposed
    if take.sum() and (~take).sum():
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(r), (BOOT, len(r)))
        rr, tt = r[idx], take[idx]
        ok = (tt.sum(1) > 0) & ((~tt).sum(1) > 0)
        m_take = (rr * tt).sum(1)[ok] / tt.sum(1)[ok]
        m_skip = (rr * ~tt).sum(1)[ok] / (~tt).sum(1)[ok]
        diff = m_take - m_skip
        out["edge"] = {"taken_minus_skipped_r": round(float(r[take].mean() - r[~take].mean()), 3), "ci95": [round(float(np.percentile(diff, 2.5)), 3), round(float(np.percentile(diff, 97.5)), 3)],
                       "taken_minus_all_r": round(float(r[take].mean() - r.mean()), 3)}
    if take.sum() < MIN_N or (~take).sum() < MIN_N:
        out["warnings"].append(f"Fewer than {MIN_N} taken or skipped decisions ({int(take.sum())} taken, {int((~take).sum())} skipped): differences this size are what luck alone produces. Keep going before reading anything into them.")
    elif out.get("edge") and out["edge"]["ci95"][0] <= 0 <= out["edge"]["ci95"][1]:
        out["warnings"].append("The 95% interval for 'taken minus skipped' includes zero: no evidence yet that your filter beats the raw signals.")
    by_tag = {}
    for key in sorted({(x["decision"], x["tag"]) for x in resolved}):
        m = np.array([(x["decision"], x["tag"]) == key for x in resolved])
        st = _stats(r[m])
        st["interpretable"] = st["n"] >= MIN_N
        by_tag[f"{key[0]}:{key[1]}"] = st
    out["by_tag"] = by_tag
    kind = np.array([x.get("kind") or "entry" for x in resolved])
    out["by_kind"] = {}
    for k in ("entry", "reverse"):
        m = kind == k
        if m.any():
            out["by_kind"][k] = {"all": _stats(r[m]), "taken": _stats(r[m & take]), "skipped": _stats(r[m & ~take])}
    # your levels, size and management: what the taken trades made versus what the strategy's own levels would have made (same fills, $)
    s_by = {sid: info[sid] for sid in info}
    mg = [x for x in resolved if x["decision"] == "take" and x.get("a_modified") and x.get("a_reason") != "reverse" and x.get("a_pnl") is not None
          and (s_by[x["session"]]["finished"] or x["a_exit_ts"] <= s_by[x["session"]]["cursor_ts"])]
    if mg:
        delta = np.array([x["a_pnl"] - x["pnl"] for x in mg], dtype=float)
        out["management"] = {"n": len(mg), "mean_usd": round(float(delta.mean()), 2), "total_usd": round(float(delta.sum()), 2), "better": int((delta > 0).sum()), "worse": int((delta < 0).sum()),
                             "interpretable": len(mg) >= MIN_N}
        if len(mg) < MIN_N:
            out["warnings"].append(f"Only {len(mg)} trades with your own levels, size or management: too few to say whether your edits help.")
    nm = np.array([x.get("news_min") if x.get("news_min") is not None else np.nan for x in resolved], dtype=float)
    known = ~np.isnan(nm)
    if known.any():
        near = known & (np.abs(nm) <= 60)
        out["by_news"] = {"within_60min_of_tier1": {"all": _stats(r[near]), "taken": _stats(r[near & take]), "skipped": _stats(r[near & ~take])},
                          "otherwise": {"all": _stats(r[known & ~near]), "taken": _stats(r[known & ~near & take]), "skipped": _stats(r[known & ~near & ~take])},
                          "calendar_unknown": int((~known).sum())}
    by_strat = {}
    for s_ in sorted({x["strategy"] for x in resolved}):
        m = np.array([x["strategy"] == s_ for x in resolved])
        by_strat[s_] = {"all": _stats(r[m]), "taken": _stats(r[m & take]), "skipped": _stats(r[m & ~take])}
    out["by_strategy"] = by_strat
    if any(info[x["session"]]["symbol"].endswith("_SYN") for x in resolved):
        out["warnings"].append("Some of these decisions are on synthetic data, which has no edge by construction.")
    return out


def to_markdown(rep: dict) -> str:
    L = ["# Hybrid replay report", "", f"Sessions {rep['sessions']} · decisions {rep['decisions']} · counted {rep['resolved']} · not yet resolved by the clock {rep['unresolved']}", ""]
    for w in rep.get("warnings", []):
        L += [f"> {w}", ""]
    if "all_signals" not in rep:
        return "\n".join(L)
    f = lambda s: f"{s['n']} trades · mean {s['mean_r']}R · total {s['total_r']}R · win {None if s['win_rate'] is None else round(100 * s['win_rate'])}%"  # noqa: E731
    L += ["| group | result |", "|---|---|", f"| every signal proposed | {f(rep['all_signals'])} |", f"| you took | {f(rep['taken'])} |", f"| you skipped | {f(rep['skipped'])} |", ""]
    if "edge" in rep:
        e = rep["edge"]
        L += [f"Taken minus skipped: **{e['taken_minus_skipped_r']}R** per trade (95% interval {e['ci95'][0]} to {e['ci95'][1]}). Taken minus all signals: {e['taken_minus_all_r']}R.", ""]
    L += ["## By reason tag", "", "| decision:tag | n | mean R | total R | win | |", "|---|---|---|---|---|---|"]
    for k, s in rep["by_tag"].items():
        L.append(f"| {k} | {s['n']} | {s['mean_r']} | {s['total_r']} | {None if s['win_rate'] is None else round(100 * s['win_rate'])}% | {'' if s['interpretable'] else 'too few'} |")
    L += ["", "A skip tag with a NEGATIVE mean R is a reason that saved you money; a skip tag with a positive mean R is a reason that cost you.", ""]
    if rep.get("by_kind"):
        L += ["## New entries vs reversals", "", "| kind | group | n | mean R | total R |", "|---|---|---|---|---|"]
        for k, g in rep["by_kind"].items():
            for name, st in g.items():
                L.append(f"| {k} | {name} | {st['n']} | {st['mean_r']} | {st['total_r']} |")
        L += ["", "For a reversal, R is the extra profit from reversing instead of holding the old trade, in units of the new trade's risk.", ""]
    if rep.get("management"):
        m = rep["management"]
        L += ["## Your levels, size and management", "", f"{m['n']} trades with your own changes: {m['total_usd']:+} $ in total ({m['mean_usd']:+} per trade) against the strategy's own levels; "
              f"{m['better']} better, {m['worse']} worse." + ("" if m["interpretable"] else " Too few to interpret."), ""]
    if rep.get("by_news"):
        L += ["## Near high-impact news", "", "| | group | n | mean R | total R |", "|---|---|---|---|---|"]
        for k in ("within_60min_of_tier1", "otherwise"):
            for name, st in rep["by_news"][k].items():
                L.append(f"| {k} | {name} | {st['n']} | {st['mean_r']} | {st['total_r']} |")
        L += ["", f"Proposals without calendar coverage: {rep['by_news']['calendar_unknown']}.", ""]
    return "\n".join(L)
