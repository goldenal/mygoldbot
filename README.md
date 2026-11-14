# goldsys: a gold (XAUUSD) trading research platform

All five phases of the plan in `PLAN.md`: data lake, feature library, strategy DSL, screening backtester,
validation gate, three-tier strategy formulation, portfolio step, risk and hybrid (human) layer, execution
through a broker bridge, forward-test reconciliation, and the monthly live review. Everything is Python;
the hot loops are numba.

```
goldsys/
  data/        dukascopy.py (free tick feed downloader), mt5csv.py (MetaTrader 5 bar-export importer), store.py (parquet lake + resampling),
               sessions.py (DST-aware London/NY/Asia flags, Asian range), quality.py, synthetic.py
  features/    indicators.py (causal, unit-tested for look-ahead), registry.py (lazy "atr_14"-style features)
  dsl/         spec.py (strategy = YAML/JSON data), expr.py (expression evaluator), compile.py (spec -> backtest)
  engine/      costs.py (time-of-day spread, slippage, commission, swap), backtest.py (numba), metrics.py
  validation/  walkforward.py (WFA + CPCV), stats.py (PSR/DSR/PBO), montecarlo.py, sensitivity.py, gate.py
  search/      registry.py (sqlite trial log -> DSR), parametric.py (Tier 1 Optuna), genetic.py (Tier 2 GP + MAP-Elites),
               llm.py (Tier 3 Claude hypothesis loop)
  portfolio/   allocate.py (correlation cap, inverse-vol / risk-parity weights, vol target), build.py, config.py (portfolio.yaml)
  risk/        limits.py (loss limits, open-risk and lot caps, spread guard, loss-streak pause, kill switch), calendar.py (news blackout)
  live/        runner.py (signal -> risk -> approval queue -> broker -> ledger), broker.py (interface + paper broker),
               signals.py, ledger.py (SQLite, hash-chained trades), control.py (operator actions), drift.py, reconcile.py
               (forward test vs shadow backtest), scaling.py (stage and size rules), statement.py, alerts.py (Telegram),
               mt5.py (MetaTrader 5, Windows), bridge.py (HTTP bridge to the VPS), simvenue.py (simulated live venue)
  dashboard/   server.py + page.html (local operations dashboard, stdlib only)
  research/    review.py (monthly: refresh walk-forward, retire, propose replacements, re-allocate)
  chart/       data.py (bars at any timeframe, indicators, strategy trades with stop/target levels), api.py, page.html
               (chart viewer + replay UI; TradingView Lightweight Charts vendored in static/, Apache-2.0)
  replay/      sim.py (M1-level trade simulator, tested trade-for-trade against the backtester), session.py (clock, proposals,
               decisions, blind mode, sqlite), report.py (taken vs skipped, by reason tag), api.py
  nautilus/    adapter.py (DSL -> NautilusTrader strategy, optional dependency)
strategies/    three templates: asian_breakout, vol_momentum, vwap_reversion
scripts/       research: download_data, make_synthetic, run_backtest, run_gate, run_search
               live:     portfolio, run_live, ops, dashboard, forward_report, ledger, monthly_review, mt5_bridge_server
tests/         look-ahead, engine hand-checks, DSL, statistics, risk limits, live-vs-backtester parity, hybrid layer,
               ledger tampering, bridge, simulated-venue live loop, monthly review
```

## Chart viewer

```bash
python scripts/chart.py            # http://127.0.0.1:8787/chart   (read-only; also serves the ops dashboard at / if live.sqlite exists)
```
Scroll back through history (older bars load as you go), switch M1-D1, jump to a date (UTC), and add SMA / EMA / Bollinger / Donchian /
VWAP / Asian range / session shading / RSI / ATR / ADX / MACD. Indicators are the strategies' own causal functions, computed on the
full history so the left edge is warmed up. Pick a strategy and "Show trades" to see every entry, exit, R-multiple and the stop/target
levels on the chart; click a trade in the list to jump to it. Bars and trades are the real backtester's output, not a re-implementation.

## Hybrid replay: the strategies propose, you decide

In the chart page, open **Replay**: pick a start date and one or more strategies (optionally *blind*, which hides the date), then
**Next signal**. The clock walks M1 bars and stops at every proposal, showing entry, stop, target, size and risk as ghost lines on the
chart. You **Take** or **Skip** and must give a reason from a fixed list. A taken trade fills at the next bar's open and is managed
bar by bar with the backtester's costs; the replay also plays forward, steps a minute, or runs to the close of the current chart bar.

- Nothing from the future reaches the page: bars stop at the clock, and the outcome of every proposal (taken or skipped) stays on the
  server until the clock passes it. `Report` counts only outcomes the clock has passed; `Finish & reveal` ends the session and counts all.
- Every proposal also gets a *shadow* outcome from the same simulator, so "what your skips would have made" is measured, not guessed.
  The report compares what you took with what you skipped (with a bootstrap interval and an explicit "too few decisions" warning) and
  breaks results down by reason tag. A skip tag with a negative mean saved you money; a positive one cost you.
- **Your own levels.** Open "Use your own stop, target or size" on a proposal: the ghost lines follow what you type. The strategy's own trade is
  always simulated too (the shadow), so the report separates what your *filter* did from what your *levels, size and management* did.
- **Manage open trades.** Move the stop (this switches off any trailing rule: you are managing it now), move or clear the target, or close at the next
  open. A change takes effect from the next bar and its outcome stays hidden like any other.
- **Reversals.** When a strategy already in your trade signals the other way you are offered "Reverse": close at the next open and enter the other
  way, exactly as the backtester does. Same-direction signals while you are in a trade are not proposed. A reversal is scored as the extra profit
  from reversing instead of holding, in units of the new trade's risk (and is only counted once both paths have played out).
- **Trailing stops.** A strategy can declare `exit.trail` (see below); the replay shows the stop ratcheting. Manual moves override it.
- **News.** Each proposal records the nearest tier-1 event (minutes to/since), the panel lists the next 24h of scheduled events, and the chart marks
  released events. The schedule is public in advance so showing upcoming times is not look-ahead; no values are ever stored. The report splits results
  within 60 minutes of a tier-1 event from the rest.
- Sessions persist in `replay.sqlite`; resume them from the same panel. `python scripts/replay_report.py` prints the report.
- Blind mode shifts dates by whole weeks, so weekday and time of day stay real, but price levels still hint at the era.

## Trailing stops in strategies

```yaml
exit:
  stop:  {atr_mult: 3.0, atr_n: 14}
  trail: {atr_mult: 2.5, atr_n: 14, activate_r: 1.0}     # or {fixed: 5.0}; activate_r = profit (in initial-stop multiples) before it starts
```
From the bar after the fill the stop follows the best price reached, `trail` behind it, once the trade is `activate_r` x the initial stop in profit. It only
tightens, and a bar's own extreme moves it for the NEXT bar (inside a bar the order of high and low is unknowable). A trailed stop exits as `trail`.
`strategies/vol_momentum_trail.yaml` is a working template. The backtester works on strategy-timeframe bars; the replay trails on every M1 bar, so an H1
strategy's replay can differ slightly from its backtest (an M1 strategy matches trade for trade; `tests/test_replay_sim.py`). **The live runner and the
Nautilus adapter refuse trailing strategies** (stops are set once, at entry) rather than ignore the trail.

## News calendar

```bash
python scripts/import_calendar.py --fomc                       # Fed calendar page: FOMC statements + minutes 2021-2027 (no key)
FRED_API_KEY=... python scripts/import_calendar.py --fred      # jobs report, CPI, PPI, PCE, GDP, retail sales; full history + schedule (free key)
python scripts/import_calendar.py --ics ~/Downloads/bls.ics    # or a CSV: ts_utc,name,tier   (the BLS blocks automated downloads: this one is manual)
```
Writes `data/calendar/events.csv` (schedule only, never released values). Nothing is guessed from rules like "first Friday": the FRED source has the real
dates including delayed or moved releases, and the importer refuses to label a FRED release whose name it does not recognise. It prints a coverage table and
lists any event types that are missing. The same file drives the live news blackout, the chart's event markers and the replay's news proximity.

## Importing MetaTrader 5 history

```bash
python scripts/import_mt5.py ~/Documents/XAUUSD_M1_....csv --check              # profile, timezone detection, hourly spread table; writes nothing
python scripts/import_mt5.py ~/Documents/XAUUSD_M1_....csv --source-name "Deriv MT5"
```
MT5 > View > Symbols > Bars > M1 > Request > Export Bars. The importer parses the tab/comma export, works out the broker's clock from where the
last bar of each week falls in New York time (Friday close = 17:00 there, whatever the server says; override with `--tz utc|+2|ny+7|<IANA>`),
converts to UTC, turns spread points into $/oz (0 points = unknown, so the cost model uses its table), and writes `data/bars/<SYMBOL>/M1/`
plus `SOURCE.json`. One source per symbol: importing over a different source is refused unless `--replace`. Re-importing the same source is safe.

## Quick start

```bash
uv venv .venv --python 3.13 && source .venv/bin/activate      # or: python3 -m venv .venv
uv pip install -e ".[dev,llm]"                                # or: pip install -e ".[dev,llm]"
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

## From research to live (Phases 4 and 5)

```bash
# 5. portfolio: only strategies with a passing gate report are eligible
python scripts/portfolio.py build strategies/candidates/*.yaml --symbol XAUUSD
#    (plumbing test on synthetic data, where nothing can pass: add --symbol XAUUSD_SYN --allow-unvalidated)

# 6. replay the portfolio through the live stack (risk limits, drift monitor, ledger) on history
python scripts/run_live.py replay --symbol XAUUSD --start 2026-08-01 --fresh
python scripts/forward_report.py --db live_replay.sqlite --warmup-symbol XAUUSD
python scripts/dashboard.py --db live_replay.sqlite              # http://127.0.0.1:8787

# 7. rehearse the REAL-broker loop with no broker: a simulated venue plays lake history forward as if live
export GOLDSYS_BRIDGE_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
python scripts/mt5_bridge_server.py --paper --replay-symbol XAUUSD --replay-start 2026-08-03 --speed 60 &
python scripts/portfolio.py promote --confirm                    # research -> demo
python scripts/run_live.py broker --bridge http://127.0.0.1:8765 --db live_rehearsal.sqlite
python scripts/ops.py --db live_rehearsal.sqlite status          # pending / approve N / veto N / mode / risk-off / kill

# 8. demo account (>= 8 weeks). On the Windows VPS, next to a logged-in MT5 DEMO terminal:
#      set GOLDSYS_BRIDGE_TOKEN=...   &&   python scripts\mt5_bridge_server.py --selftest   &&   python scripts\mt5_bridge_server.py
#    on this machine:  ssh -L 8765:127.0.0.1:8765 user@vps
python scripts/run_live.py paper  --bridge http://127.0.0.1:8765     # live bars, simulated fills, no orders sent
python scripts/run_live.py broker --bridge http://127.0.0.1:8765     # orders on the demo account
python scripts/forward_report.py --stage demo                        # must PASS before ...
python scripts/portfolio.py promote --confirm                        # ... demo -> tiny_live (size x0.1), later -> live

# 9. Phase 5, monthly
python scripts/download_data.py 2026-11-01 2026-12-01
python scripts/monthly_review.py --symbol XAUUSD [--gp-generations 5] [--apply]
python scripts/portfolio.py scale [--apply]                          # size moves only on realised live results
python scripts/ledger.py verify && python scripts/ledger.py statement --stage live
```

How the pieces fit:
- **Stages.** `portfolio.yaml` carries a stage: research -> demo -> tiny_live -> live. `run_live.py broker` refuses a research
  portfolio, refuses a demo-stage portfolio on a real-money account, and for real money needs validated strategies plus both
  `--confirm-live` and `GOLDSYS_ALLOW_LIVE=1`. Promotion needs a passing forward-test reconciliation for the current stage
  (>= 8 weeks demo, >= 4 weeks tiny live).
- **Risk layer.** Checked before every entry: kill switch, manual risk-off, daily (1.5%) and weekly (3%) loss limits, loss-streak
  pause, news blackout, spread guard (live spread > table x 1.5), max positions / lots / open risk. It can shrink or block an
  order and flatten the book; it never originates one, and it never blocks an exit. Limits live in `portfolio.yaml` under `risk:`.
- **Kill switch** trips on stale data, a broker that stays disconnected, equity 10% below its high-water mark, a position that
  vanishes at the broker, or the operator. It flattens and blocks entries until `ops.py reset-kill`. It survives restarts.
- **Hybrid layer.** Each strategy runs in mode `auto`, `approve` (signals wait in a queue for `ops.py approve N` or the dashboard;
  they expire, and are refused if price has moved more than 0.25 x the stop distance since the signal) or `off` (logged, not traded).
  Every override is logged with who did it, and `forward_report.py` prices the signals that were not taken.
- **Drift.** Each strategy's rolling 30-trade expectancy (in R) is compared with the 5th percentile of the same statistic
  bootstrapped from its backtest; below it, the strategy is switched off and you are alerted.
- **Ledger.** `live.sqlite` holds every bar seen, signal, decision, fill and alert. Entries and exits are hash-chained, so an
  edited or deleted trade fails `ledger.py verify`. Replay, paper and broker runs use separate ledger files.
- **Alerts** go to the console and the ledger; warnings and above also to Telegram if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
  are in `.env`.
- **News blackout** needs `data/calendar/events.csv` (`ts_utc,name,tier`). Without it the runner says so at start-up and the
  blackout is inactive.

## Conventions that matter
- Bars are BID. Buys pay bid + spread (+ slippage); sells receive bid.
- Signals are computed at bar close and filled at the next bar's open. No feature may use future data (`tests/test_lookahead.py`).
- Per bar: entry at the open, then stop/target on that bar's range (the entry bar included), then signal/time exits at the close.
  Inside a bar, if stop and target are both touched, the stop is assumed first.
- The live runner replays exactly that order event by event; `tests/test_live_parity.py` holds it trade-for-trade equal to the
  backtester, so the screening engine and the engine that trades are two implementations that are tested against each other.
- If a backtest's equity reaches zero the account is liquidated and trading stops (exit reason `ruin`).
- Cost model is a versioned object (`CostModel.version`); replace `DEFAULT_SPREAD_BY_HOUR` with your broker's sampled table.
- Synthetic data has no edge by construction. Results on `XAUUSD_SYN` only prove the plumbing works.

## Writing a strategy
See `strategies/*.yaml`. A strategy is data: filters, entry expressions, exit rules, sizing, params and a params_space.
`"$name"` inside strings refers to a parameter, so `"ema_$fast"` with `fast: 20` becomes `ema_20`.
`entry_short: mirror` auto-generates the short side (gt<->lt, +<->-, donchian_high<->donchian_low, asian_high<->asian_low).

## Status
- Phases 1-3 run on this machine (Python 3.13 venv). Phase 4 (risk layer, approval queue, execution, reconciliation, dashboard,
  alerts) and Phase 5 (stage and size rules, monthly review, verified ledger) are implemented and tested on synthetic data.
- Not yet exercised, because it cannot be from here:
  - **MT5** (`goldsys/live/mt5.py`): Windows-only package, written but never run. Start with `mt5_bridge_server.py --selftest`
    on a DEMO account. Everything on this side of the bridge is tested against the paper broker and the simulated venue.
  - **Real data**: a Deriv MT5 M1 export (2023-03 to 2026-11, bid, spread in the file) is imported as `XAUUSD`. The Dukascopy tick
    download has not completed (their feed was timing out). The validation gate has NOT been run on the real data yet, and the
    cost model still uses its default commission and slippage; match them to the account you will trade.
  - **Forward test**: the 8 weeks on demo and 4 weeks at tiny size are calendar time; the tooling that judges them is ready.
  - **NautilusTrader adapter**: the package installs (1.231) and the adapter imports, but no tick-level Nautilus backtest has
    been run. The bar-level second-engine check in `tests/test_live_parity.py` does not replace a tick replay.
  - **Tier 3 with the real Claude API** (only `--dry-run` was run) and **Telegram** delivery.
- The economic calendar is yours to supply; there is no built-in event list.
