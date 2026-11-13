# goldsys: a gold (XAUUSD) trading research platform

Phases 1-3 of the plan in `PLAN.md`: data lake, feature library, strategy DSL, screening backtester,
validation gate, and a three-tier strategy formulation engine. Everything is Python; the hot loop is numba.

```
goldsys/
  data/        dukascopy.py (free tick feed downloader), store.py (parquet lake + resampling),
               sessions.py (DST-aware London/NY/Asia flags, Asian range), quality.py, synthetic.py
  features/    indicators.py (causal, unit-tested for look-ahead), registry.py (lazy "atr_14"-style features)
  dsl/         spec.py (strategy = YAML/JSON data), expr.py (expression evaluator), compile.py (spec -> backtest)
  engine/      costs.py (time-of-day spread, slippage, commission, swap), backtest.py (numba), metrics.py
  validation/  walkforward.py (WFA + CPCV), stats.py (PSR/DSR/PBO), montecarlo.py, sensitivity.py, gate.py
  search/      registry.py (sqlite trial log -> DSR), parametric.py (Tier 1 Optuna), genetic.py (Tier 2 GP + MAP-Elites),
               llm.py (Tier 3 Claude hypothesis loop)
  nautilus/    adapter.py (DSL -> NautilusTrader strategy, optional dependency)
strategies/    three templates: asian_breakout, vol_momentum, vwap_reversion
scripts/       download_data.py, make_synthetic.py, run_backtest.py, run_gate.py, run_search.py
tests/         look-ahead, engine hand-checks, DSL, statistics
```

## Quick start

```bash
pip install -e ".[dev,llm]"
python -m pytest -q

# 1. data: real ticks (rate-limited; ~1-2 s per hour file, resumable) ...
python scripts/download_data.py 2019-01-01 2026-11-01
# ... or synthetic bars for development
python scripts/make_synthetic.py 2017-01-01 2026-11-01

# 2. one backtest
python scripts/run_backtest.py strategies/asian_breakout.yaml --symbol XAUUSD_SYN

# 3. full validation gate (walk-forward, CPCV, DSR, PBO, Monte Carlo, cost sensitivity, plateau)
python scripts/run_gate.py strategies/vol_momentum.yaml --symbol XAUUSD_SYN            # add --quick for a fast pass

# 4. strategy formulation
python scripts/run_search.py tier1 strategies/asian_breakout.yaml --symbol XAUUSD_SYN --trials 100
python scripts/run_search.py tier2 --symbol XAUUSD_SYN --generations 10 --pop 30
python scripts/run_search.py tier3 --symbol XAUUSD_SYN --iterations 5 [--dry-run]       # needs ANTHROPIC_API_KEY unless --dry-run
```

Survivors land in `strategies/candidates/`; run `scripts/run_gate.py` on them. Reports are written to `reports/` as JSON + Markdown.
Every backtest trial is logged in `registry.sqlite`; the trial count is fed into the Deflated Sharpe Ratio.

## Conventions that matter
- Bars are BID. Buys pay bid + spread (+ slippage); sells receive bid.
- Signals are computed at bar close and filled at the next bar's open. No feature may use future data (`tests/test_lookahead.py`).
- Inside a bar, if stop and target are both touched, the stop is assumed first.
- Cost model is a versioned object (`CostModel.version`); replace `DEFAULT_SPREAD_BY_HOUR` with your broker's sampled table.
- Synthetic data has no edge by construction. Results on `XAUUSD_SYN` only prove the plumbing works.

## Writing a strategy
See `strategies/*.yaml`. A strategy is data: filters, entry expressions, exit rules, sizing, params and a params_space.
`"$name"` inside strings refers to a parameter, so `"ema_$fast"` with `fast: 20` becomes `ema_20`.
`entry_short: mirror` auto-generates the short side (gt<->lt, +<->-, donchian_high<->donchian_low, asian_high<->asian_low).

## Status
- Done: Phase 1 (data, features, screening engine), Phase 2 (validation gate; NautilusTrader adapter written but untested here
  because the package is not installed), Phase 3 (all three formulation tiers, registry).
- Next: download real ticks on your machine (the Dukascopy feed blocks cloud egress IPs), sample your broker's spread table,
  install nautilus_trader and validate the adapter, then Phase 4 (risk layer, execution bridge, forward test).
