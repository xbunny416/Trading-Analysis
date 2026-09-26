# Trading-Analysis

This repository just for my own coding trading workspace.

## EURUSD H1 long/short trend-following — walk-forward research

**Status: FAIL.** The full 8-trial budget was used and no configuration passed the
walk-forward gates. `python harness.py` exits **1** on the final code. The mandate says to open
a pull request only once the harness exits 0, so none was opened.

| Gate | Target | Best trial (T5) | Final trial (T8) |
|---|---|---|---|
| (a) OOS trades / year | > 100 | 167.1 ✅ | 136.7 ✅ |
| (b) OOS Sharpe | ≥ 1.50 | 0.32 ❌ | 0.20 ❌ |
| (c) OOS max drawdown | < 12 % | 7.8 % ✅ | 5.9 % ✅ |
| (d) WFE = OOS Sharpe / IS Sharpe | ≥ 0.60 | 0.54 ❌ | 0.21 ❌ |
| (e) Perturbation look-ahead test | pass | pass ✅ | pass ✅ |

Every number here comes from `harness.py` running on real data. None of it is estimated.
The complete history is in [`trials.log`](trials.log), and each trial is its own git commit.

---

### Files

| File | Purpose |
|---|---|
| `strategy.py` | Strategy (final trial, T8). It has two independent implementations of the same rules: a **vectorised pandas research path** (`compute_features` → `signal_frame` with `.shift(1)` → `backtest_vectorized`) and an **event-driven OOP path** (`MarketData` → `SignalEngine` → `TrendStrategy` → `RiskManager` → `ExecutionHandler`, orchestrated by `EventDrivenBacktester`). |
| `harness.py` | Loads the data, runs the 5-split rolling WFA, applies gates (a)–(e) and the integrity tests, prints the JSON report, and sets the exit code. It also appends to `trials.log` and enforces the 8-trial budget. |
| `scripts/eval_wfa.py` | Entry point named in `.claude/skills/quant-wfa`. It forwards to `harness.main()` with the same flags and exit code. |
| `trials.log` | One block per completed real-data evaluation: hypothesis, grid, data hash, metrics, per-split results, failure reasons. |
| `results/wfa_latest.json` | Full report of the last run, including the in-sample grid of every split. |

### Reproduce

```bash
pip install -r requirements.txt
python harness.py --selftest   # synthetic-data unit tests only; no real-data metrics, not a trial
python harness.py              # full evaluation on real EURUSD H1 data (exit 0 = PASS, 1 = FAIL)
python scripts/eval_wfa.py     # same thing, via the skill's entry point
```

On the first run the harness does a sparse, blob-less git checkout of the pinned source commit
(about 100 MB of monthly CSVs) and caches the hourly bars in `data/`, which is git-ignored. A full
run takes about 4 s after that. Each run is deterministic: re-running any trial commit reproduces
its logged metrics exactly (checked for T1 and T5).

**Trial budget.** The harness fingerprints `strategy.py` and `harness.py` together. If the
fingerprint is already in `trials.log`, the run is a re-run and nothing is logged. Any other code
change counts as a new trial and is refused once 8 trials have been logged. To start a new research
cycle, start a fresh `trials.log`, and ideally use data these 8 trials have never seen.

---

### Data

| | |
|---|---|
| Source | OANDA **mid** candles, 1-minute, UTC, from the public repo [FutureSharks/financial-data](https://github.com/FutureSharks/financial-data) at pinned commit `7ba1d404aa8b0e1c0f71321acebadcbfb9bcca8d` |
| Resampling | 1-hour bars: open = first, high = max, low = min, close = last |
| Span | 2015-01-01 22:00 → 2020-05-14 07:00 UTC: **5.36 years, 33,340 bars** (~6,215 bars/yr) |
| Quality | Clean after validation; 4 intra-week gaps longer than 1 h; the longest gap (76 h) is a holiday weekend |
| Hash | `sha256(hash_pandas_object(OHLC))` = `4a5f8acc…eeda5`, printed in every report |

**Why not yfinance:** the sandbox's network policy blocks Yahoo Finance, as well as Dukascopy,
HistData, Stooq, FRED and the ECB. Only GitHub and package registries are reachable. yfinance also
caps 1-hour history at 730 days, below the 3-year minimum. This is the most recent multi-year
intraday EURUSD history that was reachable, so the sample ends in May 2020. To run on a newer
export, use `python harness.py --data your.csv` with columns `time,open,high,low,close`.

---

### Strategy — final trial (T8): parsimonious shock momentum

It uses one indicator (ATR) and three tunable parameters: `atr_n` (n), `shock_k` (k) and
`risk_pct` (r). The WFA searches `shock_k ∈ {2, 2.5, 3, 3.5}` and `atr_n ∈ {6, 12, 24, 48}`;
`risk_pct` is fixed at 0.25 %.

```math
TR_t = \max\left(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|\right), \qquad
ATR_t = \frac{1}{n}\sum_{i=0}^{n-1} TR_{t-i}
```

```math
\text{Shock}_t = \operatorname{sign}(C_t - C_{t-1}) \cdot \mathbf{1}\left[\,TR_t > k \cdot ATR_{t-1}\right]
```

Every decision that takes effect in bar *t* uses bars ≤ *t−1* only, and fills at **Open_t**:

* **Entry (flat):** take direction `Shock_{t-1}` when it is non-zero.
* **Trailing stop (chandelier):** the initial stop is `C_{e-1} − d·k·ATR_{e-1}`. Each bar, exit if
  `d·(C_{t-1} − S) < 0`; otherwise ratchet `S ← max_d(S, C_{t-1} − d·k·ATR_{t-1})`.
* **Reverse:** when `Shock_{t-1} = −d`.
* **Size** (volatility targeting, as the mandate specifies):

```math
\text{Units}_t = \min\left(\frac{NAV_{t-1}\cdot r}{k \cdot ATR_{t-1}},\; \frac{L_{\max}\cdot NAV_{t-1}}{C_{t-1}}\right), \qquad L_{\max}=10
```

  Any degenerate input (ATR ≤ 0, NaN or ∞, NAV ≤ 0, price ≤ 0) sizes to 0.

### Execution and risk model (fixed, not tuned)

* **Friction:** 0.5 pip half-spread plus 0.5 pip slippage on **every fill**, so 2.0 pips per round
  trip. This is the conservative reading of "1.0 pip round-trip spread + 0.5 pip slippage per trade".
* **Daily-loss halt:** if NAV at a bar close is ≥ 2.5 % below NAV at the start of the trading day
  (the day rolls at 17:00 New York, DST-aware), go flat at the next open and open nothing for 24 h.
* **Max-drawdown kill:** if NAV is ≥ 10 % below its running peak, go flat and halt permanently.
* No calendar or time-of-day filters anywhere. The harness greps `strategy.py` for calendar
  attributes as part of a complexity audit.

### Walk-forward design

There are 5 rolling windows of equal length W, split **70 % IS / 30 % OOS**. The window steps by
0.3·W, so the five OOS segments tile the last 68 % of the data back to back (3.66 OOS years,
2016-09 → 2020-05).

* **Selection:** within each IS window, every grid point is backtested. Among those trading more
  than 100 times a year, the harness picks the best **plateau score**: mean Sharpe over the
  point's 3×3 grid neighbourhood. This penalises isolated spikes.
* **Stitched OOS:** one continuous OOS simulation. Parameters switch at segment boundaries (open
  positions are liquidated at the boundary close), while NAV, peak and circuit-breaker state carry
  over, so a kill in one segment silences every later segment.
* **Sharpe:** daily (trading-day) NAV returns, annualised with √252, risk-free rate 0.
  WFE = stitched OOS Sharpe / mean of the IS Sharpes of the chosen parameters.

### WFA results — final trial (T8), per split

| Split | IS window | OOS window | Chosen (k, n) | IS Sharpe | OOS Sharpe | OOS return | OOS max DD | OOS trades/yr | OOS win % | OOS PF | Split WFE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2015-01-01 → 2016-09-16 | 2016-09-16 → 2017-06-09 | 2.5, 48 | 1.22 | **1.11** | +4.49 % | 3.59 % | 124.7 | 40.7 | 1.40 | 0.91 |
| 2 | 2015-09-27 → 2017-06-09 | 2017-06-11 → 2018-03-02 | 2.5, 48 | 1.01 | **0.52** | +1.64 % | 3.14 % | 125.9 | 39.6 | 1.15 | 0.52 |
| 3 | 2016-06-19 → 2018-03-02 | 2018-03-04 → 2018-11-26 | 2.5, 48 | 0.88 | **0.54** | +1.64 % | 3.26 % | 132.7 | 38.1 | 1.09 | 0.61 |
| 4 | 2017-03-13 → 2018-11-26 | 2018-11-26 → 2019-08-21 | 3.0, 12 | 1.46 | **−0.62** | −2.06 % | 5.17 % | 135.3 | 38.4 | 0.67 | −0.43 |
| 5 | 2017-12-05 → 2019-08-21 | 2019-08-21 → 2020-05-14 | 2.5, 24 | 0.11 | **−0.83** | −2.56 % | 4.77 % | 166.8 | 33.6 | 0.68 | −7.68 |
| **Stitched** | | 2016-09-16 → 2020-05-14 | | **0.93** (mean) | **0.20** | **+3.02 %** | **5.95 %** | **136.7** | 37.8 | 0.99 | **0.21** |

Stitched OOS also shows CAGR 0.82 %, annualised volatility 4.46 %, an average net trade of −0.12
pips, no daily-loss halts, and no kill.

### Drawdown statistics (stitched OOS, final trial)

| | |
|---|---|
| Max drawdown | **5.95 %** (peak 2018-10-02 → trough 2020-05-05, not recovered by the end of the data) |
| Longest time under water | 589 days (2018-10-02 → end of sample) |
| Worst OOS segment drawdown | 5.17 % (split 4) |
| Daily-loss halts / kill switch | 0 / not triggered |
| Circuit-breaker behaviour | Verified by tests (see below), not by chance events in the sample |

Across all 8 trials the stitched OOS max drawdown ranged from 5.9 % to 10.1 %. T3 and T7 tripped
the 10 % kill switch, which then silenced their later splits.

### All trials

| # | Hypothesis (full text in `trials.log`) | OOS Sharpe | IS Sharpe (mean) | WFE | OOS max DD | OOS trades/yr | Failed gates |
|---|---|---|---|---|---|---|---|
| 1 | Donchian breakout + ATR chandelier stop, stop-and-reverse | −0.10 | −0.42 | n/a | 8.7 % | 117.6 | b, d |
| 2 | Slow Donchian regime, buy fast pullbacks inside it | −0.60 | −0.76 | n/a | 9.8 % | 181.0 | b, d |
| 3 | Slow Donchian regime, trade fast breakouts aligned with it | −0.46 | 0.00 | −285.7 | 10.0 % (kill) | 115.1 | b, d |
| 4 | Range-expansion "shock" momentum, fixed holding clock | −0.66 | 0.61 | −1.09 | 9.0 % | 132.1 | b, d |
| 5 | Shock entry + chandelier trailing exit | **0.32** | 0.60 | 0.54 | 7.8 % | 167.1 | b, d |
| 6 | T5 + close-location confirmation of the shock bar | 0.25 | 0.64 | 0.40 | 7.9 % | 147.1 | b, d |
| 7 | T5 shocks only in the direction of a slow Donchian regime | −0.38 | 0.07 | −5.16 | 10.1 % (kill) | 91.0 | a, b, d |
| 8 | T5 with the stop tied to the shock threshold; search `atr_n` | 0.20 | 0.93 | 0.21 | 5.9 % | 136.7 | b, d |

Each change was motivated by the previous trial's **in-sample** grid and trade diagnostics; the
reasoning is recorded in each trial's `HYPOTHESIS`. Every trial is also committed separately, so
`git checkout <trial commit> && python harness.py` reproduces its logged numbers exactly.

### Integrity tests (run on every evaluation)

* **Perturbation look-ahead test (gate e).** At 8 random bars *t*, the harness scrambles all data
  from bar *t+1* on (required) and, separately, from bar *t* itself (stricter). It then asserts that
  (1) the signal frame rows ≤ *t*, (2) the positions held in bars ≤ *t* and (3) the NAV before the
  perturbation are bit-for-bit identical in **both** engines. It runs on synthetic data and on a
  1-year slice of real data.
* **Canaries.** Two deliberately leaky signal frames must be caught: one reads bar *t+1*, one
  reads bar *t*'s close. A mutation test also showed that deleting the `.shift(1)` guard is caught,
  but only by the strict bar-*t* perturbation. That is why both perturbations are required.
* **Engine parity.** The vectorised and event-driven engines must agree on every bar: same
  direction, units within 1e-9, same trade count. On the real stitched OOS run (22,701 bars, 500
  trades) the NAV difference is exactly 0.0.
* **Edge cases** (required by `.claude/skills/tdd-enforcer`):
  * zero, NaN or ∞ ATR, NAV and price in position sizing
  * flat prices (ATR = 0 → no trades, NAV unchanged)
  * missing bars, NaN rows, duplicate or out-of-order timestamps, corrupt OHLC rows
  * an 8 % gap with a 10 % range bar
  * a ~5 % daily loss that must halt trading for exactly 24 bars and then resume
  * a ~15 % loss that must kill trading permanently
  * exact friction per round trip
  * parity with state carried across segments

  Mutation checks confirmed that the breaker test fails when the breakers are disabled or the
  halt length is set to 0.
* **Complexity audit.** At most 4 tunable parameters and at most 3 indicators, and no calendar
  attributes appear in `strategy.py`.

---

### Why it failed, and what would change the outcome

1. **The frequency gate and the edge pull in opposite directions.** In every in-sample grid, the
   highest Sharpe settings trade fewer than 100 times a year: longer Donchian lookbacks in T1,
   shock thresholds `k ≥ 3` in T5 and T6. Requiring more than 100 round trips a year on one pair
   pushes selection into the noisiest region.
2. **Friction is the same size as the edge.** In the final trial, average net P&L per trade in each
   OOS split ranged from −5.9 to +7.6 pips (stitched −0.1), against 2 pips of cost per round trip.
3. **Non-stationarity.** Shock momentum worked in-sample from 2015 to 2018 and faded from late
   2018 through 2020 (T4, T5 and T8 splits 4–5). A 21-month rolling IS window cannot anticipate
   that.
4. **Statistical power.** With 3.66 OOS years, the standard error of an annualised Sharpe is about
   0.52. The best OOS Sharpe (0.32) is about 0.6 standard errors from zero, and the expected
   maximum of 8 pure-noise trials is about 1.4 standard errors (≈ 0.7 Sharpe). The trial history
   is consistent with **no exploitable edge at this frequency**.

Changes that would genuinely alter the picture, none of which were tried here because they
change the mandate:

* a multi-pair portfolio, where diversification is how real trend-following programmes reach
  Sharpe > 1
* a lower trade-frequency floor, or daily-bar trend horizons
* modelling financing/swap costs
* tick-level stop execution
* **a fresh 2020–2026 holdout that none of these 8 trials has seen** before any further tuning

### Known limitations

* The sample ends 2020-05-14; see "Data" above.
* Prices are mid, and friction is a fixed 2 pips per round trip. Variable spreads (for example
  around news) and overnight swap/financing are not modelled.
* Stops are evaluated on hourly closes and filled at the next open, not intrabar.
* Sharpe uses daily NAV returns with a risk-free rate of 0.
