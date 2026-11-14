"""Assemble a LiveRunner from portfolio.yaml for each way of running it."""
from __future__ import annotations

import os

import pandas as pd

from ..dsl.compile import feature_frame
from ..engine.costs import CostModel
from ..portfolio.config import Portfolio
from ..risk.calendar import EventCalendar
from ..risk.limits import RiskManager
from .alerts import Alerter
from .broker import Broker, PaperBroker
from .ledger import Ledger
from .runner import LiveRunner, Slot
from .scaling import REAL_MONEY
from .signals import BrokerBarSignals, PrecomputedSignals


class StageError(RuntimeError):
    pass


def check_stage(pf: Portfolio, broker: Broker, confirm_live: bool = False) -> None:
    """Refuse to connect a portfolio to an account its stage does not allow."""
    acct = broker.account()
    if pf.stage == "research":
        raise StageError("portfolio stage is 'research': replay / paper only. Promote it to 'demo' first (scripts/portfolio.py promote).")
    if pf.stage == "demo" and acct.demo is False:
        raise StageError("portfolio stage is 'demo' but the broker account is a REAL-MONEY account. Refusing to trade.")
    if pf.stage in REAL_MONEY:
        unval = [s.name for s in pf.active() if not s.validated]
        if unval:
            raise StageError(f"real-money stage with strategies that never passed the validation gate: {', '.join(unval)}")
        if not confirm_live or os.environ.get("GOLDSYS_ALLOW_LIVE") != "1":
            raise StageError("real-money stage: pass --confirm-live AND set GOLDSYS_ALLOW_LIVE=1 to trade it.")


def build_runner(pf: Portfolio, broker: Broker, ledger: Ledger, cost: CostModel | None = None, stage: str | None = None, bars_m1: pd.DataFrame | None = None,
                 feed: Broker | None = None, calendar: EventCalendar | None = None, console: bool = True, **runner_kw) -> LiveRunner:
    """bars_m1 given  -> signals precomputed over that history (replay).
    otherwise       -> signals from the venue's own bars (`feed` if given, else `broker`)."""
    cost = cost or CostModel()
    slots = []
    for e in pf.active():
        spec = e.load_spec()
        if spec.exit.trail:
            raise StageError(f"{e.name}: this strategy uses a trailing stop, which the live runner cannot manage yet (stops are set once, at entry). "
                             "Trade it in the replay, or remove the trail, until live trailing is built.")
        if spec.hash() != e.spec_hash:
            raise StageError(f"{e.name}: the spec file changed since the portfolio was built (hash {spec.hash()} != {e.spec_hash}). Rebuild the portfolio.")
        if bars_m1 is not None:
            src = PrecomputedSignals(spec, feature_frame(bars_m1, spec.timeframe, None))
        else:
            src = BrokerBarSignals(spec, (feed or broker).bars)
        slots.append(Slot(e.name, spec, src, weight=e.weight * pf.size_factor(spec), mode=e.mode, drift=e.drift))
    notify = Alerter(ledger, console=console)
    calendar = calendar if calendar is not None else EventCalendar.from_csv()
    risk = RiskManager(pf.risk_config(), cost, calendar, store=ledger)
    runner = LiveRunner(broker, ledger, slots, risk, cost, stage=stage or pf.stage, notify=notify, **runner_kw)
    if len(calendar) == 0 and pf.risk_config().news_blackout_minutes > 0:
        notify("warning", "calendar", "no economic calendar loaded (data/calendar/events.csv): the news blackout is INACTIVE")
    if ledger.get(f"stage_started:{runner.stage}") is None:
        ledger.set(f"stage_started:{runner.stage}", {"wall": str(pd.Timestamp.now(tz='UTC'))})
    return runner


def paper_broker(pf: Portfolio, cost: CostModel, ledger: Ledger | None = None, slippage: bool = True, seed: int = 0) -> PaperBroker:
    """Paper account at the portfolio's reference equity; on a restart, realised P&L and open positions come back from the ledger."""
    b = PaperBroker(cost, init_cash=pf.account_equity, symbol=pf.symbol, slippage=slippage, seed=seed)
    if ledger is not None:
        closed = ledger.trades(closed_only=True)
        b.cash += float(closed["pnl"].sum()) if len(closed) else 0.0
        from .broker import Position
        for r in ledger.open_trades():
            b._n = max(b._n, int(str(r["position_id"]).lstrip("P") or 0))
            b._pos[r["position_id"]] = Position(r["position_id"], r["strategy"], r["symbol"], r["dir"], r["lots"], r["entry_px"], pd.Timestamp(r["entry_ts"]), r["stop_px"], r["tgt_px"])
        b._n = max([b._n] + [int(str(x).lstrip("P")) for x in ledger.trades()["position_id"] if str(x).lstrip("P").isdigit()])
    return b
