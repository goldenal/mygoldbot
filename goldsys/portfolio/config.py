"""portfolio.yaml: the strategies that are (or may go) live, their weights, modes and risk limits.
This file is the hand-over between research and execution, and every change to it is appended to
its own `history` list, so it doubles as the audit trail of what was live when."""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import PORTFOLIO_FILE, ROOT, SYMBOL
from ..dsl.spec import StrategySpec
from ..risk.limits import RiskConfig

STAGES = ("research", "demo", "tiny_live", "live")       # research: replays only. demo: broker demo account.


@dataclass
class StrategyEntry:
    name: str
    spec_path: str
    spec_hash: str = ""
    weight: float = 1.0                 # multiplies the spec's own lot sizing
    mode: str = "approve"               # auto | approve | off
    status: str = "active"              # active | candidate | retired
    validated: bool = False             # a passing validation-gate report exists for exactly this spec hash
    drift: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    notes: str = ""

    def load_spec(self) -> StrategySpec:
        p = Path(self.spec_path)
        return StrategySpec.from_yaml(p if p.is_absolute() else ROOT / p)


@dataclass
class Portfolio:
    version: int = 1
    created: str = ""
    updated: str = ""
    stage: str = "demo"
    symbol: str = SYMBOL
    account_equity: float = 10_000.0    # reference equity that lots are scaled to (reviewed monthly, not compounded tick by tick)
    target_annual_vol: float = 0.10
    scale: float = 1.0                  # live size multiplier; only moves through live/scaling.py
    risk: dict = field(default_factory=lambda: RiskConfig().to_dict())
    strategies: list[StrategyEntry] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path = PORTFOLIO_FILE) -> "Portfolio":
        d = yaml.safe_load(Path(path).read_text())
        d["strategies"] = [StrategyEntry(**s) for s in d.get("strategies", [])]
        return cls(**d)

    def save(self, path: str | Path = PORTFOLIO_FILE) -> Path:
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.created = self.created or now
        self.updated = now
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))
        return Path(path)

    def log(self, action: str, detail: str = "") -> None:
        self.history.append({"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "action": action, "detail": detail})

    def active(self) -> list[StrategyEntry]:
        return [s for s in self.strategies if s.status == "active"]

    def get(self, name: str) -> StrategyEntry | None:
        return next((s for s in self.strategies if s.name == name), None)

    def risk_config(self) -> RiskConfig:
        return RiskConfig.from_dict(self.risk)

    def size_factor(self, spec: StrategySpec) -> float:
        """Lots multiplier common to a strategy's orders: account size relative to the spec's sizing equity, times the live scale."""
        return self.account_equity / spec.sizing.equity * self.scale
