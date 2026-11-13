"""Strategy specification (the DSL). A strategy is DATA, not code.

Expressions are nested dicts / lists / strings / numbers:
  numeric:  "close" | "atr_14" | 1.5 | {"op": "+", "args": [a, b]}   ops: + - * / shift abs max min
  boolean:  {"op": "gt", "args": [a, b]}   ops: gt lt ge le cross_above cross_below between and or not
            "in_london" (boolean feature column)
Parameter references: any string containing "$name" is substituted from `params`
  e.g. "atr_$atr_n"  or  {"op": "*", "args": ["atr_14", "$k"]}
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

BOOL_OPS = {"gt", "lt", "ge", "le", "cross_above", "cross_below", "between", "and", "or", "not"}
NUM_OPS = {"+", "-", "*", "/", "shift", "abs", "max", "min"}


@dataclass
class ExitSpec:
    stop: dict = field(default_factory=lambda: {"atr_mult": 1.0, "atr_n": 14})
    target: dict | None = field(default_factory=lambda: {"r_mult": 2.0})
    time_exit_utc: str | None = None          # "16:00" -> flat at/after this time, no new entries
    max_bars: int | None = None
    exit_long: Any = None                      # boolean expression
    exit_short: Any = None


@dataclass
class SizingSpec:
    type: str = "fixed"                        # fixed | risk_pct | vol_target
    lots: float = 0.1
    risk_pct: float = 0.5                      # % of equity risked per trade (risk_pct)
    annual_vol: float = 0.10                   # target (vol_target)
    max_lots: float = 2.0
    min_lots: float = 0.01
    equity: float = 10_000.0


@dataclass
class StrategySpec:
    name: str
    timeframe: str = "M15"
    filters: list = field(default_factory=list)
    entry_long: Any = None
    entry_short: Any = None                    # "mirror" => auto-mirror entry_long (gt<->lt etc.)
    exit: ExitSpec = field(default_factory=ExitSpec)
    sizing: SizingSpec = field(default_factory=SizingSpec)
    params: dict = field(default_factory=dict)
    params_space: dict = field(default_factory=dict)   # name -> [lo, hi, step] or list of choices
    rationale: str = ""
    family: str = ""

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec":
        d = copy.deepcopy(d)
        ex = d.pop("exit", {}) or {}
        sz = d.pop("sizing", {}) or {}
        return cls(exit=ExitSpec(**ex), sizing=SizingSpec(**sz), **d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategySpec":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def with_params(self, **params) -> "StrategySpec":
        s = copy.deepcopy(self)
        s.params.update(params)
        return s

    def hash(self) -> str:
        d = self.to_dict()
        d.pop("rationale", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ helpers
    def complexity(self) -> int:
        """Node count of all expressions: used as an overfitting penalty."""
        return sum(_count(x) for x in (self.filters, self.entry_long, self.entry_short, self.exit.exit_long, self.exit.exit_short))


def _count(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, dict):
        return 1 + sum(_count(a) for a in x.get("args", []))
    if isinstance(x, list):
        return sum(_count(a) for a in x)
    return 1


def substitute_params(expr: Any, params: dict) -> Any:
    """Replace "$name" tokens inside strings with param values."""
    if isinstance(expr, str):
        if "$" in expr:
            out = expr
            for k in sorted(params, key=len, reverse=True):
                out = out.replace(f"${k}", _fmt(params[k]))
            if out.strip().lstrip("-").replace(".", "", 1).isdigit():
                return float(out)
            return out
        return expr
    if isinstance(expr, dict):
        return {k: substitute_params(v, params) for k, v in expr.items()}
    if isinstance(expr, list):
        return [substitute_params(v, params) for v in expr]
    return expr


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


MIRROR = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge", "cross_above": "cross_below", "cross_below": "cross_above"}
MIRROR_FEATURES = {"donchian_high": "donchian_low", "donchian_low": "donchian_high", "asian_high": "asian_low", "asian_low": "asian_high"}


def mirror(expr: Any) -> Any:
    """Build the short-side expression from the long-side one.
    Comparisons flip, +/- offsets flip sign where they are added to a price, and high/low features swap."""
    if isinstance(expr, dict):
        op = expr.get("op")
        args = [mirror(a) for a in expr.get("args", [])]
        if op in MIRROR:
            op = MIRROR[op]
        if op in ("+", "-") and len(args) == 2:
            op = "-" if op == "+" else "+"
        return {"op": op, "args": args}
    if isinstance(expr, list):
        return [mirror(a) for a in expr]
    if isinstance(expr, str):
        for k, v in MIRROR_FEATURES.items():
            if expr.startswith(k):
                return v + expr[len(k):]
    return expr
