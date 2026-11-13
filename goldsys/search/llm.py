"""Tier 3: LLM hypothesis loop.

The model never touches data or engine code. It receives: gold market facts, the DSL grammar,
available features, what has already been tested (registry), and the validation reports of
earlier attempts. It returns a hypothesis (rationale) + a DSL spec as JSON. The system compiles
it, runs the gate, logs the trial (counted in DSR like any other trial), and feeds the report
back for refinement. Look-ahead protection lives in the engine, not in the prompt.

Runs offline in `dry_run` mode (writes the prompt to disk and uses built-in seed hypotheses)
so the loop can be exercised without an API key.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..config import REPORT_DIR
from ..dsl.spec import StrategySpec
from ..engine.costs import CostModel
from ..features.registry import available_feature_names
from ..validation.gate import GateConfig, run_gate
from .registry import Registry

MODEL = "claude-opus-5-5"

SYSTEM = """You are a quantitative researcher formulating rule-based XAUUSD (spot gold) trading strategies for an automated research platform.
You cannot run code. You output ONE strategy per reply as JSON matching the DSL below. The platform backtests it with realistic
spread/slippage/swap costs and a strict validation gate (walk-forward, CPCV, deflated Sharpe, PBO, Monte Carlo). Most ideas fail; that is expected.
Prefer simple, causal, economically motivated rules over complex ones: complexity is penalised and look-ahead is impossible (the engine only exposes past data).

DSL (JSON):
{
 "name": "snake_case_name", "family": "session_breakout|trend_following|mean_reversion|news|cross_asset",
 "rationale": "2-4 sentences: the market mechanism this exploits and when it should fail",
 "timeframe": "M5|M15|M30|H1|H4",
 "filters": [ <bool expr>, ... ],                       # all must be true to allow entries
 "entry_long": <bool expr>, "entry_short": <bool expr> | "mirror",
 "exit": {"stop": {"atr_mult": x, "atr_n": 14} | {"fixed": $}, "target": {"r_mult": x} | {"atr_mult": x} | null,
          "time_exit_utc": "HH:MM" | null, "max_bars": n | null, "exit_long": <bool expr>|null, "exit_short": <bool expr>|"mirror"|null},
 "sizing": {"type": "risk_pct", "risk_pct": 0.5, "equity": 10000, "max_lots": 2.0},
 "params": {"name": value, ...},  "params_space": {"name": [lo, hi, step], ...}   # 2-5 params max
}
Expressions: numeric = feature name string | number | {"op": "+|-|*|/", "args": [a, b]} | {"op": "shift", "args": [a, n]}
             boolean = {"op": "gt|lt|ge|le|cross_above|cross_below", "args": [a, b]} | {"op": "between", "args": [a, lo, hi]}
                       | {"op": "and|or", "args": [...]} | {"op": "not", "args": [x]} | boolean feature name
Parameters are referenced as "$name" inside strings, e.g. "ema_$fast" or {"op":"*","args":["atr_14","$k"]}.
Features: parametric (suffix _N): %(parametric)s ; simple: %(simple)s ; base columns: %(base)s
Session booleans: in_asia (Tokyo 09-15 local), in_london (08:00-16:30 London), in_newyork (08-17 New York), in_overlap.
asian_high/asian_low/asian_width = 00:00-07:00 UTC range, known only after 07:00 UTC.

Gold facts: 1 lot = 100 oz; spread ~$0.17-0.30 in London/NY, $0.3-0.5 Asia, $1-2 at 21:00-23:00 UTC rollover; swap negative both sides;
largest moves 12:30-18:00 UTC around US data; strong multi-month trends at daily horizon; weak mean reversion at minute horizon outside news.
Reply with a short reasoning paragraph, then exactly one ```json fenced block containing the strategy."""


def build_prompt(registry: Registry | None, previous_reports: list[dict], n_history: int = 12) -> str:
    parts = []
    if registry is not None:
        hyps = registry.hypotheses(n_history)
        if hyps:
            parts.append("Already tested (do not repeat; learn from them):")
            for h in hyps:
                parts.append(f"- {h['name']}: {h['outcome']} | {h['rationale'][:200]}")
            top = registry.best(5)
            parts.append("Best trials so far (robust score): " + json.dumps(top, default=str)[:1500])
    for r in previous_reports[-3:]:
        parts.append(f"Validation report for {r['strategy']}: passed={r.get('passed')} reasons={r.get('reasons')} in_sample={ {k: round(v, 3) if isinstance(v, float) else v for k, v in r.get('in_sample', {}).items() if k in ('trades', 'net_profit', 'sharpe', 'max_drawdown', 'profit_factor', 'win_rate')} } wfe={r.get('walk_forward', {}).get('wfe')} oos_sharpe={r.get('walk_forward', {}).get('oos_sharpe')} pbo={r.get('pbo', {}).get('pbo')}")
    parts.append("Propose the next strategy. Either refine the most promising previous one (change one thing, say what and why) or propose a new mechanism not yet tested.")
    return "\n".join(parts)


def parse_spec(text: str) -> StrategySpec:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    d = json.loads(raw)
    return StrategySpec.from_dict(d)


def ask_llm(prompt: str, system: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {msg.stop_details}")
    return "".join(b.text for b in msg.content if b.type == "text")


SEED_HYPOTHESES = [
    {"name": "ny_open_momentum", "family": "session_breakout", "timeframe": "M15",
     "rationale": "US data and the NY cash open (13:30 UTC) set the direction for the next hours; follow a Donchian break in the first NY hours and exit before the close.",
     "filters": [{"op": "and", "args": ["in_newyork", {"op": "between", "args": ["minute_of_day", 810, 960]}]}],
     "entry_long": {"op": "gt", "args": ["close", "donchian_high_$dc"]}, "entry_short": "mirror",
     "exit": {"stop": {"atr_mult": "$stop_atr", "atr_n": 14}, "target": {"r_mult": "$r_mult"}, "time_exit_utc": "20:00"},
     "sizing": {"type": "risk_pct", "risk_pct": 0.5, "equity": 10000, "max_lots": 2.0},
     "params": {"dc": 20, "stop_atr": 1.0, "r_mult": 2.0}, "params_space": {"dc": [10, 40, 10], "stop_atr": [0.5, 2.0, 0.5], "r_mult": [1.0, 3.0, 0.5]}},
    {"name": "trend_pullback_h1", "family": "trend_following", "timeframe": "H1",
     "rationale": "In an established hourly uptrend, short-term oversold pullbacks (RSI2) get bought; stop below recent swing, exit on RSI recovery.",
     "filters": [{"op": "gt", "args": ["adx_14", "$adx"]}],
     "entry_long": {"op": "and", "args": [{"op": "gt", "args": ["ema_20", "ema_100"]}, {"op": "lt", "args": ["rsi_2", "$rsi_lo"]}]},
     "entry_short": {"op": "and", "args": [{"op": "lt", "args": ["ema_20", "ema_100"]}, {"op": "gt", "args": ["rsi_2", {"op": "-", "args": [100, "$rsi_lo"]}]}]},
     "exit": {"stop": {"atr_mult": "$stop_atr", "atr_n": 14}, "target": None, "exit_long": {"op": "gt", "args": ["rsi_2", 70]}, "exit_short": {"op": "lt", "args": ["rsi_2", 30]}, "max_bars": 48},
     "sizing": {"type": "risk_pct", "risk_pct": 0.5, "equity": 10000, "max_lots": 2.0},
     "params": {"adx": 20, "rsi_lo": 10, "stop_atr": 2.0}, "params_space": {"adx": [15, 30, 5], "rsi_lo": [5, 25, 5], "stop_atr": [1.5, 3.0, 0.5]}},
]


def hypothesis_loop(bars_m1: pd.DataFrame, cost: CostModel | None = None, iterations: int = 3, registry: Registry | None = None, dry_run: bool = False, gate_cfg: GateConfig | None = None, verbose: bool = True) -> list[dict]:
    cost = cost or CostModel()
    registry = registry or Registry()
    reports: list[dict] = []
    system = SYSTEM % available_feature_names()
    for i in range(iterations):
        prompt = build_prompt(registry, reports)
        if dry_run:
            (REPORT_DIR / f"llm_prompt_{i}.txt").write_text(system + "\n\n---\n\n" + prompt)
            if i >= len(SEED_HYPOTHESES):
                break
            spec = StrategySpec.from_dict(SEED_HYPOTHESES[i])
            if verbose:
                print(f"[tier3] dry run: seed hypothesis '{spec.name}' (prompt saved to reports/llm_prompt_{i}.txt)")
        else:
            text = ask_llm(prompt, system)
            try:
                spec = parse_spec(text)
            except Exception as e:
                if verbose:
                    print(f"[tier3] could not parse reply: {e}")
                registry.log_hypothesis("tier3", f"unparseable_{i}", text[:500], "", "", "unparseable")
                continue
        try:
            rep = run_gate(spec, bars_m1, cost, gate_cfg, n_trials_so_far=registry.n_trials() + 1, verbose=verbose)
        except Exception as e:
            rep = {"strategy": spec.name, "passed": False, "reasons": [f"error: {e}"], "in_sample": {}}
            if verbose:
                print(f"[tier3] gate error for {spec.name}: {e}")
        outcome = "PASS" if rep.get("passed") else "FAIL: " + "; ".join(rep.get("reasons", []))[:300]
        registry.log_hypothesis("tier3", spec.name, spec.rationale, spec.to_yaml(), spec.hash(), outcome)
        registry.log_trial("tier3", spec, rep.get("in_sample", {}), bars_m1.index[0], bars_m1.index[-1], cost.version, oos_sharpe=rep.get("walk_forward", {}).get("oos_sharpe"), passed=rep.get("passed"))
        reports.append(rep)
    return reports
