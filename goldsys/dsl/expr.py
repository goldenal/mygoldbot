"""Evaluate DSL expressions against a feature DataFrame (lazy feature construction)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..features.registry import ensure_feature
from ..features import indicators as I


def eval_expr(expr: Any, df: pd.DataFrame) -> pd.Series:
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return pd.Series(float(expr), index=df.index)
    if isinstance(expr, bool):
        return pd.Series(expr, index=df.index)
    if isinstance(expr, str):
        s = ensure_feature(df, expr)
        return s
    if isinstance(expr, list):  # implicit AND of a list of conditions
        if not expr:
            return pd.Series(True, index=df.index)
        out = eval_expr(expr[0], df).astype(bool)
        for e in expr[1:]:
            out = out & eval_expr(e, df).astype(bool)
        return out
    if isinstance(expr, dict):
        op = expr["op"]
        args = expr.get("args", [])
        if op == "and":
            return eval_expr(args, df)
        if op == "or":
            out = eval_expr(args[0], df).astype(bool)
            for e in args[1:]:
                out = out | eval_expr(e, df).astype(bool)
            return out
        if op == "not":
            return ~eval_expr(args[0], df).astype(bool)
        a = eval_expr(args[0], df)
        if op == "abs":
            return a.abs()
        if op == "shift":
            return a.shift(int(args[1]))
        b = eval_expr(args[1], df)
        if op == "gt":
            return (a > b).fillna(False)
        if op == "lt":
            return (a < b).fillna(False)
        if op == "ge":
            return (a >= b).fillna(False)
        if op == "le":
            return (a <= b).fillna(False)
        if op == "cross_above":
            return I.cross_above(a, b).fillna(False)
        if op == "cross_below":
            return I.cross_below(a, b).fillna(False)
        if op == "between":
            c = eval_expr(args[2], df)
            return ((a >= b) & (a <= c)).fillna(False)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b.replace(0, np.nan)
        if op == "max":
            return np.maximum(a, b)
        if op == "min":
            return np.minimum(a, b)
        raise ValueError(f"unknown op {op}")
    raise TypeError(f"bad expression {expr!r}")
