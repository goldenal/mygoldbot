# Building Your Own Gold (XAUUSD) Trading System
## Research, algorithms, and a build plan for a system that can both *formulate* and *test* strategies

Date: 2026-11-13

---

## 0. Executive summary

- **Sonic AI** is a gold-only copy-trading service, not a transparent algorithm. Vendor pages describe it as three human traders assisted by AI forecasting tools ("hybrid"), with trades mirrored into client MetaTrader 5 (MT5) accounts via a copier. Reported stats (~+30% total, ~0.86% max drawdown, 600+ trades, 23 green months) are self-reported and inconsistent across their own pages. There is nothing to reverse-engineer; what you can take from it is the *product shape*: one instrument, session-aware, tight drawdown, human override on top of model signals.
- **Your system should be a research platform first, a bot second.** The hard part of gold trading is not generating signals, it is proving a signal survives costs and out-of-sample testing. Build the test harness before the strategies.
- **Recommended architecture:** Python, with a two-engine design. A fast vectorised engine (vectorbt-style) for screening thousands of candidate strategies, and an event-driven engine (NautilusTrader) for faithful tick-level validation and live deployment with the same code.
- **Strategy formulation** is done with a three-tier generator: (1) parametric grid/Bayesian search over hand-designed strategy templates, (2) genetic programming over a small domain-specific language (DSL) of indicators and rules, (3) an LLM hypothesis loop that proposes rule-based strategies in that DSL and reads back the validation report. All three feed the same validation gate.
- **Validation gate (non-negotiable):** realistic spread/slippage/swap model, walk-forward analysis, combinatorial purged cross-validation (CPCV), Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), Monte Carlo trade reshuffling, and a mandatory forward (demo) period before any live capital.
- **Timeline:** roughly 5 phases over 4–6 months part-time. Phase 1 (data + backtester) is the foundation and should not be skipped.

---

## 1. What "Sonic AI bot trading" actually is

Findings from vendor and affiliate pages (no independent audit found):

| Aspect | What the sources say |
|---|---|
| Instrument | XAU/USD only |
| Model | Copy trading. Signals come from a "professional gold trading desk" (three traders) using AI tools for forecasting; some pages call it "fully automated". The word *hybrid* = discretionary humans + AI analytics |
| Delivery | Trades mirrored into client MT5 accounts through a copier (CopyX / Tag Markets) |
| Reported results | +29–31% total gain, 0.86% max drawdown, 604–634 live trades, "23 consecutive green months", claimed MyFXBook verification |
| Transparency | None. No rules, no code, no audited statement. Different names across pages (Sonic AI, SONIC, Sonic AI Gold, Sonic AI CopyX) |

**Interpretation.** A 0.86% max drawdown over 600+ trades and ~30% return is an extraordinarily smooth equity curve. In retail gold services that profile is most often produced by (a) very small position sizing with frequent small wins, (b) grid/martingale or averaging-down that hides open losses from closed-trade statistics, or (c) a short, lucky sample in a strongly trending gold market (2024–2026 was one). You cannot tell which from the outside. Do not try to clone it; build something you can audit.

**What to borrow from the product shape:**
1. Single instrument specialisation (gold has unique session/news behaviour; focus pays).
2. Session awareness (London/New York concentration).
3. Human-in-the-loop override (a "hybrid" layer that can veto model trades around news or regime breaks).
4. A verified public track record as the deliverable (your own MyFXBook/FXBlue-style ledger, even if private).

---

## 2. Gold market research: what you are actually trading

### 2.1 Instrument facts that shape the system
- **Spot XAUUSD via CFD/forex broker** (what Sonic uses): quoted per troy ounce, 1 lot = 100 oz, so a $1 move = $100 per lot. Typical raw spreads 10–30 cents in liquid hours, widening sharply (often $1–3+) at the daily rollover, Asian open, and around US data releases. Overnight swap on gold is usually negative on both sides and material for multi-day holds.
- **Alternatives**: COMEX GC futures (centralised, transparent, no swap, 100 oz contracts; micro MGC is 10 oz), GLD/IAU ETFs (equity hours only). Futures give cleaner data and execution; CFDs give 24/5 access and small sizing. Decide early because the cost model differs.
- **Trading hours**: 23 hours a day Sunday 22:00 UTC to Friday 21:00 UTC with a 1-hour daily break (broker dependent).

### 2.2 Statistical character of gold
- **Volatility clustering and intraday seasonality.** Sources agree range concentrates in London and New York hours; one 2024–2026 sample measured ~$23 average hourly range in the 08:00–12:00 New York window versus ~$13–14 elsewhere. The London–New York overlap (roughly 13:00–17:00 UTC) is the highest-activity window. Asian session is typically range-bound with wider spreads.
- **Scheduled event spikes.** US CPI, NFP, FOMC, PCE, and Fed speakers produce the largest short-horizon moves and the worst fills. Gold is more sensitive than most FX pairs to rate expectations.
- **Trend persistence at daily/weekly horizon, mean reversion at minutes horizon.** Gold shows long momentum regimes (2019–2020, 2023–2026) separated by multi-month ranges. At 1–15 minute horizons, after accounting for spread, autocorrelation is weakly negative (mean-reverting) outside news.
- **Fat tails.** Daily returns show kurtosis well above 3; weekend gaps are common. Any system must survive 3–5% single-day moves.

### 2.3 Macro drivers (for regime filters and features)
Mixed evidence; the relationships have shifted since 2022:
- **Real yields (10y TIPS)**: historically the dominant driver (negative relation). Deutsche Bank's model finds rate changes, not the dollar, explain most fluctuations. Saxo and others argue the link weakened in 2024–2026 as central bank and fiscal-concern buying added a structural bid.
- **USD index (DXY)**: still a tailwind/headwind but weaker quantitative support.
- **Central bank buying**: described as the "floor" of the market; World Gold Council quarterly data. Low frequency, useful only for regime context.
- **ETF flows**: Granger-cause tests suggest price leads flows, not the reverse, so flows are a confirmation feature, not a predictor.
- **Risk sentiment / geopolitics**: episodic, hard to model; treat as a volatility regime feature (VIX, MOVE) rather than direction.

**Implication for the system:** use macro series as *regime features* (daily/weekly) that gate or scale intraday strategies, not as the intraday signal itself.

### 2.4 Cost reality (the thing most retail backtests ignore)
A gold scalping backtest on M5 bars with zero spread will look spectacular and be worthless. Model, per fill:
- Spread: time-of-day dependent table (e.g., 0.15 in London/NY overlap, 0.35 Asian, 1.5+ at rollover, 2–5 in the 2 minutes around tier-1 news).
- Slippage: random, skewed positive (against you), scaled by volatility and order size; larger for stops than limits.
- Commission: broker dependent (~$3–7 per lot round trip on raw-spread accounts).
- Swap: tri-weekly triple swap on Wednesday (broker dependent), negative both sides.
- MT5 "every tick" modelling on bar data synthesises intra-candle paths and creates phantom fills; use real tick data for anything with stops inside the bar.

---

## 3. Strategy families and algorithms suited to gold

Each family below is a *template* with a parameter space. The formulation engine (Section 6) searches within and across these templates. Rules are written as pseudocode so they can be implemented in the DSL.

### 3.1 Session breakout (trend, intraday)
Rationale: Asian range compression followed by London/NY expansion.
```
range = high/low of 00:00–07:00 UTC window
entry: buy stop at range_high + k*ATR(14, M15), sell stop at range_low - k*ATR
filter: range_width < q-th percentile of last N days (compression)
        and no tier-1 news in next 30 min
stop: opposite side of range or 1.0*ATR; target: 1.5–2.5 R or time exit at 16:00 UTC
params: window, k, q, R-multiple, time exit
```

### 3.2 Volatility-adjusted momentum (trend, swing)
```
signal = sign( EMA(close, fast) - EMA(close, slow) ) on H1/H4
confirm: ADX(14) > 20 and close > Donchian(20) high (or <) 
size = target_vol / realised_vol(20)   (vol targeting)
stop: chandelier 3*ATR trailing; re-entry allowed
regime gate: daily 200-SMA slope and/or DXY 20d change
```

### 3.3 Intraday mean reversion (counter-trend, M5–M15, non-news hours)
```
z = (close - VWAP_session) / stdev(close - VWAP, 60 bars)
entry: fade when |z| > 2.0 and RSI(2) extreme and no news within ±15 min
exit: z crosses 0, or time stop 60–120 min, or hard stop 1.2*ATR
only in Asian session or post-NY-close; skip the first 30 min of London/NY
```

### 3.4 Opening range / NY open momentum
```
OR = high/low of first 15–30 min after 13:30 UTC (US data / NY cash open)
entry on break of OR with volume/tick-count confirmation
stop at OR midpoint, target 2R or 16:00 UTC flat
```

### 3.5 News-reaction strategies (two variants)
- **Avoidance** (recommended default): flatten/skip ±N minutes around tier-1 events; used as a universal filter.
- **Post-spike fade/continuation**: after a tier-1 release, wait 3–5 minutes, trade in direction of the first candle only if the move exceeds 1.5× average release move. Needs tick data and a proper economic calendar with historical surprise values.

### 3.6 Statistical / cross-asset
- Gold vs DXY and vs real-yield proxy (TLT/TIP or 10y futures) rolling-beta residual mean reversion at H1/H4.
- Gold/silver ratio extremes as a regime feature.
- Gold vs GLD/GC futures basis (if you have both feeds) for data sanity rather than alpha.

### 3.7 Machine learning layer (after the rule-based layer works)
- **Meta-labelling** (López de Prado): keep the primary rule-based entry, train a gradient-boosted classifier to predict whether a given signal will hit target before stop, use its probability to size or veto. This is the most robust way to add ML; it does not try to predict price.
- Features: session flags, realised vol ratios, ATR percentiles, distance to session VWAP, DXY/yield changes, time to next news, recent win/loss streak of the base rule.
- Label: triple-barrier (profit target, stop, time) outcome.
- Validation: purged, embargoed CPCV only; never random k-fold.

### 3.8 What to avoid
- Grid/martingale/averaging-down. They manufacture smooth closed-trade curves and blow up in a trend. Suspect this is behind many "0.8% drawdown" gold products.
- Deep neural price prediction on raw OHLC. Low signal-to-noise; overfits instantly.
- Strategies whose edge depends on sub-$0.10 precision; retail spread kills them.

---

## 4. System architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       1. DATA LAYER                                   │
│  tick/M1 history (Dukascopy .bi5, broker MT5)  │ economic calendar  │
│  macro daily (FRED: real yields, DXY; WGC)     │ sessions/holidays  │
│  -> Parquet lake, point-in-time, UTC, validated, resampled on demand │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│                  2. FEATURE & INDICATOR LIBRARY                       │
│  vectorised (NumPy/Polars/Numba) indicators, session/news features,  │
│  regime features; every function has a look-ahead unit test          │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│                  3. STRATEGY DSL                                      │
│  declarative JSON/YAML spec: entries, exits, filters, sizing, params │
│  compiles to BOTH engines below -> one source of truth               │
└──────┬─────────────────────────────────────────────┬─────────────────┘
       │                                             │
┌──────▼─────────────────────┐          ┌────────────▼────────────────┐
│ 4a. SCREENING ENGINE       │          │ 4b. FAITHFUL ENGINE         │
│ vectorised, bar-based,     │          │ event-driven, tick-level,   │
│ thousands of runs/minute   │ promote  │ order book-aware fills,     │
│ coarse cost model          │ ───────► │ full cost model, same code  │
│ (vectorbt / custom Numba)  │          │ goes live (NautilusTrader)  │
└──────┬─────────────────────┘          └────────────┬────────────────┘
       │                                             │
┌──────▼─────────────────────────────────────────────▼─────────────────┐
│                  5. VALIDATION GATE                                   │
│  walk-forward, CPCV, DSR, PBO, Monte Carlo, stress (2020-03, 2024-25 │
│  spikes), parameter-plateau test, cost-sensitivity, trial registry   │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│                  6. STRATEGY FORMULATION ENGINE                       │
│  Tier 1 parametric search  │ Tier 2 genetic programming over DSL     │
│  Tier 3 LLM hypothesis loop (proposes DSL specs, reads reports)      │
│  all write to an experiment registry; all pass through gate 5        │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│          7. PORTFOLIO, RISK & HYBRID (HUMAN) LAYER                    │
│  allocation across surviving strategies, vol targeting, daily/weekly  │
│  loss limits, kill switch, news blackout, manual veto/approve queue  │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│          8. EXECUTION & MONITORING                                    │
│  MT5 bridge (Windows VPS) or broker API / IBKR futures; reconcile    │
│  fills vs simulated; live-vs-backtest drift dashboard; alerts        │
└──────────────────────────────────────────────────────────────────────┘
```

Design principles:
1. **One strategy definition, two engines.** Screening speed and live fidelity are different problems; the DSL keeps them consistent.
2. **Everything is point-in-time.** Data, calendar, and features must only use information available at that timestamp. Enforce with tests, not prompts.
3. **Every backtest is logged.** The number of trials you ran is an input to DSR/PBO. If you do not count trials, you cannot correct for selection bias.
4. **Costs are first-class.** The cost model is a versioned object, not a number.

---

## 5. Component detail

### 5.1 Data layer
- **Primary history:** Dukascopy free tick feed (.bi5 LZMA files, no auth, rate-limited ~5 req/s). Use a mirror/downloader such as `tickvault` or `tickterial`, or write your own with backoff. Coverage for XAUUSD goes back to the 2000s. Store raw ticks as Parquet partitioned by day.
- **Broker-specific history:** pull from your actual MT5 broker via the official `MetaTrader5` Python package (Windows only) to calibrate spreads/swaps. Broker M1 history is usually only a few months deep, so use it for cost calibration, Dukascopy for depth.
- **Futures alternative:** CME GC/MGC via a data vendor (Databento, Portara) if you go the futures route.
- **Economic calendar:** historical events with actual/forecast/previous (free sources are patchy; a paid calendar API is worth it for news strategies). Store as point-in-time: the forecast known *before* release.
- **Macro:** FRED (DFII10 real yield, DTWEXBGS dollar index, VIXCLS), WGC central bank data. Daily, lagged by publication date.
- **Sessions/holidays:** explicit UTC session table including DST shifts for London and New York; broker rollover time.
- **Quality checks:** gap detection, duplicate ticks, negative spreads, bars outside trading hours, weekend data, Z-score outlier flags. Run on every ingest.

### 5.2 Feature library
- Resampling: tick -> M1 -> M5/M15/H1/H4/D1 with proper OHLCV and tick count; dollar/volume bars as an option.
- Indicators: ATR, EMA/SMA, RSI, ADX, Donchian, Bollinger, Keltner, VWAP (session-anchored), realised vol, Hurst exponent, z-scores.
- Session/news features: minutes since session open, minutes to next tier-1 event, Asian range width percentile.
- Regime features: daily trend state, vol regime (ATR percentile), macro deltas.
- **Look-ahead tests:** for each feature, assert that `f(data[:t]) == f(data)[t]`.

### 5.3 Strategy DSL (the key to "formulate" + "test")
A strategy is a data structure, not Python code the generator has to parse. Example:

```yaml
name: asian_breakout_v3
timeframe: M15
filters:
  - session_in: [london, newyork]
  - news_blackout_minutes: 30
  - regime: { daily_sma200_slope: ">", value: 0 }   # longs only in uptrend
entry_long:
  all:
    - { cross_above: [close, asian_range_high + 0.3 * atr14] }
    - { lt: [asian_range_width_pct, 40] }
entry_short: mirror
exit:
  stop: { atr_mult: 1.0 }
  target: { r_mult: 2.0 }
  time_exit_utc: "16:00"
sizing: { type: vol_target, annual_vol: 0.10, max_lots: 1.0 }
params_space:            # what the search engines are allowed to vary
  atr_mult: [0.5, 2.0, 0.25]
  r_mult: [1.0, 3.0, 0.5]
  asian_range_width_pct: [20, 60, 10]
```
- Grammar: typed expression tree over features, comparison and logical operators, a fixed set of exit and sizing types. Keep it small; a small grammar is both faster to search and harder to overfit with.
- Compiles to: (a) vectorised signal arrays for the screening engine, (b) a NautilusTrader `Strategy` class for the faithful engine and live.
- Complexity score (node count) is stored and penalised in the validation gate.

### 5.4 Screening engine (fast)
- Bar-based (M1 or M5), vectorised with NumPy/Polars + Numba; vectorbt (or vectorbt PRO) is a good base.
- Simplified fill model: next-bar open with time-of-day spread table plus fixed slippage; stops/targets resolved using M1 high/low with conservative tie-breaking (assume stop hit first when both touched in one bar).
- Output: trade list, equity curve, standard metrics (CAGR, Sharpe, Sortino, max DD, Calmar, profit factor, expectancy, trade count, exposure, average holding time).
- Throughput target: a 10-year M5 history for one parameter set in well under a second; parameter grids run in parallel across cores.

### 5.5 Faithful engine (NautilusTrader)
- Rust core, deterministic event-driven, nanosecond clock, same strategy code in backtest and live. Official adapters include Interactive Brokers (futures route); for MT5 there is only a community adapter (`mt5connect`, Windows + logged-in terminal), so plan an MT5 bridge of your own if that is your broker.
- Replay real ticks with the time-of-day spread, latency model (e.g., 50–200 ms), partial fills optional, swaps applied at rollover.
- Used for: final validation of every strategy that passes screening, and as the live runtime.

### 5.6 Validation gate (strategy testing protocol)
Run in this order; a strategy must pass each step to advance.

1. **Sanity:** ≥ 200 trades in-sample (≥ 300 for intraday), no single trade > 10% of profit, no single month > 25% of profit, exposure and holding time make sense.
2. **Cost sensitivity:** re-run with spread ×1.5 and ×2. Edge should degrade gracefully, not vanish.
3. **Parameter plateau:** heat-map of neighbouring parameters; reject sharp isolated peaks.
4. **Walk-forward analysis (WFA):** anchored or rolling, e.g., optimise 18 months → trade 6 months, step 6 months, over the full history. Walk-forward efficiency (OOS/IS performance ratio) > 0.5 and OOS profitable in the majority of windows.
5. **CPCV** (combinatorial purged cross-validation with embargo) to get a distribution of OOS Sharpe rather than a single path; library options: `purgedcv`, `ml4t-diagnostic`.
6. **DSR / PBO:** Deflated Sharpe Ratio using the number of trials from the experiment registry; Probability of Backtest Overfitting from CPCV paths. Require DSR > 0 at 95% confidence and PBO < 0.2.
7. **Monte Carlo:** reshuffle trade order and bootstrap returns to get drawdown distribution; require 95th-percentile drawdown within risk budget. Also randomise entry timing ±1 bar and skip 10% of trades at random.
8. **Stress replay:** March 2020, Aug 2024 unwind, 2025–2026 record-high spikes, FOMC days; check worst day and gap behaviour.
9. **Faithful-engine confirmation:** tick-level replay of the OOS period in NautilusTrader; results within tolerance of the screening engine (document the tolerance; large gaps mean the screening fill model is wrong).
10. **Forward test:** ≥ 8 weeks on a demo account, then ≥ 4 weeks tiny live size, comparing live fills vs simulated fills per trade.

Produce a one-page **validation report** per strategy (metrics table, equity curves IS/OOS, WFA windows, CPCV distribution, DSR/PBO, cost-sensitivity, stress table). This report is also what the LLM tier reads.

### 5.7 Strategy formulation engine

**Tier 1 — Parametric search over templates (build first).**
- Templates from Section 3 with declared parameter spaces.
- Search: coarse grid → Bayesian optimisation (Optuna) on the screening engine; objective = OOS-robust metric (e.g., mean WFA-OOS Sharpe minus complexity penalty), never in-sample return.
- Every trial logged to the registry (strategy hash, params, data range, metrics, cost model version).

**Tier 2 — Genetic programming over the DSL.**
- Individuals = DSL expression trees for entry/exit conditions plus a parameter vector.
- Fitness = WFA-OOS Sortino × sqrt(trade count factor) − λ × tree size, with a hard minimum trade count.
- Operators: subtree crossover, point mutation, parameter jitter; typed grammar prevents nonsense (comparing price to RSI).
- Diversity: MAP-Elites niches by (holding time, session, trade frequency) so you get a portfolio of different behaviours instead of 100 clones. The open-source `pwb-alphaevolve` project is a reference implementation of this loop.
- Budget control: cap trials per run and feed the count into DSR.

**Tier 3 — LLM hypothesis loop.**
- Prompt an LLM with: gold market facts (Section 2), the DSL grammar, the list of available features, the registry of already-tested ideas (so it does not repeat them), and validation reports of recent attempts.
- The LLM returns a *hypothesis* (one paragraph of market rationale) plus a DSL spec. Rationale is stored; specs without a causal rationale are deprioritised.
- The system compiles, runs the gate, and returns the report. The LLM proposes refinements or abandons the idea.
- Guardrails learned from recent research (AlgoEvolve, "What survives honest evaluation?"): the LLM must never touch the data or engine; it only emits DSL. Look-ahead protection is enforced by the engine, not by instructions. Count LLM trials in DSR exactly like GP trials. Expect most LLM ideas to fail the gate; that is the point.

**Formulation output:** a ranked list of surviving strategies with their reports, plus the correlation matrix of their OOS returns, feeding the portfolio layer.

### 5.8 Portfolio, risk and the hybrid (human) layer
- Allocate across survivors by inverse volatility or risk parity with a correlation cap; target portfolio vol (e.g., 8–12% annual) rather than fixed lots.
- Hard limits: max daily loss (e.g., 1.5%), max weekly loss (3%), max open risk, max lots, consecutive-loss pause, spread guard (do not trade if live spread > table × 1.5), news blackout.
- Kill switch on: data staleness, broker disconnect, equity below threshold, strategy live-vs-backtest drift beyond threshold (e.g., rolling 30-trade expectancy below the 5th percentile of the Monte Carlo distribution).
- **Hybrid layer:** a queue where model signals can be set to auto, require approval, or be vetoed; a manual "risk-off" toggle for macro events; a daily human review of the drift dashboard. This is the one place discretion belongs.

### 5.9 Execution and monitoring
- **MT5 route:** Windows VPS near the broker's server; Python `MetaTrader5` package for orders; your NautilusTrader strategy talks to a small bridge service (ZeroMQ/gRPC) that wraps MT5. Reconcile positions every minute.
- **Futures route:** Interactive Brokers via the official NautilusTrader adapter; cleaner, no copier/bridge needed, needs more capital per contract (MGC helps).
- Monitoring: Grafana/Streamlit dashboard of equity, open risk, spread, latency, fills vs simulated, per-strategy drift; alerts via Telegram/email.
- Logging: every order event and every decision input, so a bad trade can be replayed in the backtester.

---

## 6. Tech stack (recommended)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12+, Numba for hot loops | Ecosystem; Rust core comes via NautilusTrader |
| Storage | Parquet + DuckDB/Polars | Fast columnar queries over years of ticks |
| Screening engine | vectorbt (or custom NumPy/Numba) | Thousands of runs/min |
| Faithful + live engine | NautilusTrader | Same code backtest → live, tick-level, deterministic |
| Search | Optuna (Tier 1), DEAP or custom GP (Tier 2), Claude API (Tier 3) | Mature, well documented |
| Validation | purgedcv / ml4t-diagnostic for CPCV, DSR, PBO; own Monte Carlo | Avoids writing statistics from scratch |
| Registry | SQLite/Postgres + MLflow | Count trials, reproduce runs |
| Broker | MT5 broker with raw spreads (CFD) or IBKR (futures) | Decide in Phase 0 |
| Infra | Linux box for research; Windows VPS only if MT5 | MT5 Python API is Windows-only |

---

## 7. Phased roadmap

### Phase 0 — Decisions and scaffolding (Week 1)
- Choose instrument route (CFD at MT5 broker vs COMEX futures). Open demo accounts.
- Set up repo, environments, Parquet lake layout, experiment registry schema.
- Write the cost model object with a first spread table from your broker's live quotes (sample for a week).
- **Exit criteria:** cost model v1 documented; data layout decided.

### Phase 1 — Data and screening backtester (Weeks 2–5)
- Dukascopy tick downloader with caching and validation; resample pipeline; session/holiday table; FRED macro ingest; calendar ingest.
- Feature library with look-ahead tests.
- Screening engine: signals → fills → trades → metrics; unit tests against hand-computed trades.
- Implement two templates (session breakout, vol-adjusted momentum) end to end.
- **Exit criteria:** a 10-year M5 backtest with realistic costs runs in seconds and matches a hand-checked sample of trades.

### Phase 2 — Validation gate and faithful engine (Weeks 6–9)
- WFA, CPCV, DSR, PBO, Monte Carlo, stress replays, cost sensitivity, parameter plateau; validation report generator.
- NautilusTrader integration: DSL compiler to Nautilus strategy; tick replay with cost model; tolerance comparison against screening engine.
- **Exit criteria:** both engines agree within tolerance on the two templates; first validation reports produced.

### Phase 3 — Strategy formulation (Weeks 10–15)
- Finish the DSL grammar and compilers.
- Tier 1 Optuna search over all Section 3 templates.
- Tier 2 GP with MAP-Elites niches and trial counting.
- Tier 3 LLM loop with the registry and report feedback.
- Build the portfolio/correlation step.
- **Exit criteria:** a ranked set of strategies that pass the full gate (expect only a handful out of thousands), with a combined portfolio report.

### Phase 4 — Hybrid risk layer, execution, forward test (Weeks 16–20)
- Risk limits, kill switch, approval queue, dashboard, alerts.
- MT5 bridge or IBKR adapter; paper trading with live data; fill reconciliation.
- Run ≥ 8 weeks demo. Compare live vs backtest trade by trade.
- **Exit criteria:** live-vs-simulated slippage and expectancy within tolerance; no unhandled failure modes during the forward period.

### Phase 5 — Live and continuous research (ongoing)
- Go live at small size; scale only on realised, not backtested, performance.
- Monthly: re-run WFA with new data, retire strategies whose drift exceeds threshold, let the formulation engine propose replacements.
- Keep your own verified ledger; that is the real product.

---

## 8. Pitfalls checklist (read before every milestone)
- [ ] Spread/slippage/swap modelled per time of day, not as one number.
- [ ] Stops inside a bar resolved conservatively or with tick data.
- [ ] No feature uses future data (tests exist).
- [ ] Calendar and macro data are point-in-time.
- [ ] Trial count recorded and used in DSR; optimisation target is OOS-robust, not in-sample.
- [ ] Minimum trade counts enforced; results not dominated by a few trades.
- [ ] No grid/martingale; no unlimited averaging down.
- [ ] Survivorship of "gold only goes up" 2023–2026 regime checked with earlier ranging periods (2013–2018).
- [ ] Forward test completed before real money; scaling tied to live results.
- [ ] Human override exists but is logged and reviewed; discretion is measured, too.

---

## 9. Sources consulted

Sonic AI (vendor/affiliate, unverified):
- https://sonicaitrading.com/ , https://sonicaitrading.com/how-it-works , https://sonicaitrading.com/performance , https://sonicaitrading.com/faqs
- https://sonicaigold.com/ , https://blog.sonicaigold.com/blog/aitech-sonic-ai-what-the-live-track-record-actually-shows
- https://sonicaicopyx.com/ , https://www.sonicai.tech/

Gold sessions, costs, strategies (mostly vendor content, directional only):
- https://www.mql5.com/en/blogs/post/770149
- https://www.pro-scalper.com/xauusd-trading-sessions
- https://www.tradingviewcopier.com/blog/xauusd-automated-trading.html
- https://quant-signals.com/xauusd-trading-strategies/
- https://fxnx.com/en/blog/mt5-strategy-tester-xauusd-modelling-modes-measured
- https://algotradingspace.com/xauusd-scalping-strategy

Gold macro drivers:
- https://www.tickmill.com/blog/institutional-insights-deutsche-bank-who-drives-gold-prices
- https://www.kitco.com/news/article/2026-09-22/etf-flows-show-gold-price-decoupled-real-yields-fiscal-concerns-now-drive
- https://www.khaleejtimes.com/business-technology-review/golds-strongest-support-may-come-from-central-banks-not-crisis-headlines
- https://www.gold.org/goldhub/research/gold-demand-trends/gold-demand-trends-full-year-2025/outlook
- https://www.ssga.com/sg/en/individual/insights/gold-2026-outlook-can-the-structural-bull-cycle-continue-to-5000

Data:
- https://glama.ai/mcp/servers/dt4w4y97py (Dukascopy feed description)
- https://keyhankamyar.github.io/posts/tickvault-introduction/
- https://github.com/drui9/tickterial
- https://www.mql5.com/en/forum/489956
- https://strategyquant.com/blog/get-100-high-quality-tick-data-for-metatrader-5-free-tutorial/

Frameworks:
- https://docs.nautilustrader.io/concepts/overview.html , https://nautilustrader.io/blog/why-nautilustrader-exists/
- https://pypi.org/project/mt5connect/
- https://dev.to/pickuma/quantconnect-vs-backtrader-vs-vectorbt-which-to-start-with-in-2026-4954
- https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks
- https://pypi.org/project/purgedcv/ , https://pypi.org/project/ml4t-diagnostic/0.1.0a9/

Automated strategy discovery and overfitting:
- https://arxiv.org/abs/2606.26173 (AlgoEvolve)
- https://arxiv.org/pdf/2608.27734 (What survives honest evaluation?)
- https://arxiv.org/pdf/2512.12924 (Walk-forward validation framework)
- https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-41061-5 (GP + LLM alpha discovery)
- https://github.com/paperswithbacktest/pwb-alphaevolve
