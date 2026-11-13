"""Performance metrics from an equity curve and trade list."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sharpe(daily: pd.Series) -> float:
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return 0.0
    return float(daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(daily: pd.Series) -> float:
    if len(daily) < 2:
        return 0.0
    dn = daily[daily < 0]
    d = np.sqrt((dn ** 2).mean()) if len(dn) else 0.0
    if d == 0:
        return 0.0 if daily.mean() <= 0 else 10.0
    return float(daily.mean() / d * np.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    if years <= 0:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    return float(ratio ** (1 / years) - 1) if ratio > 0 else -1.0


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    daily = equity.resample("1D").last().dropna().pct_change().dropna()
    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gp = float(wins["pnl"].sum()) if n else 0.0
    gl = float(-losses["pnl"].sum()) if n else 0.0
    m = {
        "trades": int(n),
        "net_profit": float(equity.iloc[-1] - equity.iloc[0]) if len(equity) else 0.0,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) else 0.0,
        "cagr": cagr(equity),
        "sharpe": sharpe(daily),
        "sortino": sortino(daily),
        "max_drawdown": max_drawdown(equity),
        "calmar": 0.0,
        "win_rate": float(len(wins) / n) if n else 0.0,
        "profit_factor": float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "expectancy": float(trades["pnl"].mean()) if n else 0.0,
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else 0.0,
        "avg_bars_held": float(trades["bars_held"].mean()) if n and "bars_held" in trades else 0.0,
        "largest_trade_share": float(trades["pnl"].max() / gp) if gp > 0 and n else 0.0,
        "days": int(len(daily)),
    }
    m["calmar"] = float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else 0.0
    if n:
        monthly = trades.set_index("exit_ts")["pnl"].resample("1ME").sum()
        m["best_month_share"] = float(monthly.max() / monthly[monthly > 0].sum()) if (monthly > 0).any() else 0.0
        m["months_positive"] = float((monthly > 0).mean())
    return m
