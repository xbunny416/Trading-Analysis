# Trading-Analysis

This repository just for my own coding trading workspace.

## FX trend-following research on NautilusTrader (cycle 2)

**Status: FAIL.** 7 of the 8 allowed trials were run on your MT5 data and none passed the
walk-forward gates. I stopped with one slot unused (reason below). `python harness.py` exits **1**,
so no pull request was opened (the mandate opens one only when the harness exits 0).

| Gate (5-pair portfolio, stitched out-of-sample) | Target | Best trial (T1) | Final trial (T7) |
|---|---|---|---|
| (a) OOS trades / year | > 100 | 590 ✅ | 469 ✅ |
| (b) OOS Sharpe | ≥ 1.50 | 0.66 ❌ | −0.64 ❌ |
| (c) OOS max drawdown | < 12 % | 2.7 % ✅ | 6.4 % ✅ |
| (d) WFE = OOS Sharpe / IS Sharpe | ≥ 0.60 | 0.93 ✅ | −0.86 ❌ |
| (e) Perturbation look-ahead test | pass | pass ✅ | pass ✅ |

Every number comes from NautilusTrader backtests on your data; nothing is estimated. Each trial is
its own git commit, and [`trials.log`](trials.log) holds the full history.

### What changed from cycle 1

| | Cycle 1 (archived in `research/cycle1/`) | Cycle 2 (this code) |
|---|---|---|
| Engine | custom vectorised + event-driven Python | **NautilusTrader 1.221** (`BacktestEngine`) |
| Data | OANDA EURUSD 1-minute bars, 2015-01 → 2020-05 | **your MT5 M15 exports**, 5 pairs, 2020-11-30 → 2022-12-30 |
| Universe | EURUSD only | **one USD account trading EURUSD, GBPUSD, USDJPY, USDCAD, EURJPY** |
| Costs | fixed 1.0 pip spread + 0.5 pip slippage per fill | **your broker's per-bar spread** (1.0 pip floor) + 0.5 pip slippage per fill |
| Minimum data span | 3 years | **2 years** (your decision: the exports cover 2.1 years) |

Cycle 1 ended 8/8 FAIL; its best idea (T5, "shock entry + trailing stop") became cycle 2's Trial 1.

### Files

| File | Purpose |
|---|---|
| `backtest.py` | Nautilus plumbing: MT5 CSV → UTC, hourly MID bars, executable quotes, instruments, a reusable `BacktestEngine`. |
| `strategy.py` | The final trial's strategy (T7). Engine-agnostic `UniverseSignalEngine`/`SignalEngine`, `PositionLogic` and `RiskManager` are wired to Nautilus orders by `PortfolioTrendStrategy` (a Nautilus `Strategy`). `signal_frame` (pandas, `.shift(1)`) and `reference_backtest` form an independent vectorised path used for the parity and leak tests. |
| `harness.py` | Walk-forward analysis (in-sample grid on 4 worker processes, each running Nautilus), gates (a)–(e), integrity tests, JSON report, `trials.log`, exit code. |
| `scripts/eval_wfa.py` | Entry point named in `.claude/skills/quant-wfa`; forwards to `harness.main()`. |
| `trials.log` | Cycle 2 trial history. Cycle 1's log is in `research/cycle1/trials.log`. |
| `results/wfa_latest.json` | Full report of the last run (T7), including each split's in-sample grid. |

### Reproduce

```bash
pip install -r requirements.txt           # nautilus_trader==1.221.0
# put the MT5 exports at data/mt5/EURUSD.csv, GBPUSD.csv, USDJPY.csv, USDCAD.csv, EURJPY.csv (git-ignored)
python harness.py --selftest              # synthetic tests only, ~20 s, never a trial
python harness.py                         # full walk-forward on data/mt5, ~2 min on 4 cores
python harness.py --data-dir other/       # same pipeline on other exports; reported, never logged as a trial
```

The harness fingerprints `strategy.py`, `harness.py` and `backtest.py`. Re-running logged code is
recognised as a re-run and not logged again. Any code change is a new trial, and 7 of the 8 slots are
used.

---

### How the Nautilus backtest works

* **Clock.** MT5 server time follows EU daylight saving (EET/EEST), so it is converted from
  `Europe/Athens` to UTC. That is why 6 weeks a year open at Sunday 23:00 server time.
* **Signals.** Hourly **MID** bars, where mid = bid + spread/2 and the M15 bars are resampled to
  1 hour. Each bar is published at its **close**.
* **Execution quotes.** Each hour gets two quotes: its first M15 bar at open + 1 ms, and its last
  M15 bar at close − 1 ms. They are built as

  ```
  bid/ask = mid ∓ ( max(spread, 1.0 pip)/2 + 0.5 pip )
  ```

  Median broker spreads are EURUSD 1.6, USDJPY 1.5, USDCAD 1.8, GBPUSD 1.9 and EURJPY 2.0 pips, so
  a round trip costs about **2.5–3.0 pips**. The spread spikes at the 17:00 NY rollover (6–9 pips)
  are kept as they are.
* **Timing.** Orders are submitted on a bar-close event and carry a 1 ns order latency, so every
  market order fills at the **next hour's open quote**: decide at the close of bar t−1, execute at
  the open of bar t. Positions are marked at the bid (longs) or ask (shorts), i.e. at liquidation
  value.
* **Venue.** SIM, NETTING, MARGIN account in USD, zero fees (all costs are in the quotes). JPY and
  CAD P&L is converted to USD by Nautilus's exchange-rate cache.
* **Nautilus pitfall found and fixed.** `BacktestEngine.reset()` clears the cache's FX conversion
  table (`Cache._xrate_symbols`). After a reset, JPY/CAD profit silently never reached the USD
  balance. `Market.run()` re-registers the instruments after each reset, and a friction test (a JPY
  round trip on the second run) guards the fix.

### Strategy — final trial (T7): parsimonious shock momentum

It uses one indicator (ATR) and three tunable parameters: `atr_n` (n), `shock_k` (k) and `risk_pct`
(r). The WFA searches `shock_k ∈ {2, 2.5, 3, 3.5}` × `atr_n ∈ {12, 24, 48, 96}`, with r = 0.1 % per
pair. The same rules and parameters run on all 5 pairs.

```math
TR_t = \max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|),\qquad ATR_t = \tfrac{1}{n}\textstyle\sum_{i=0}^{n-1} TR_{t-i}
```

```math
\text{Shock}_t = \operatorname{sign}(C_t - C_{t-1})\cdot \mathbf{1}\!\left[TR_t > k\cdot ATR_{t-1}\right]
```

Each decision is made at the close of bar t−1 and filled at the open quote of bar t:

* **Entry (flat):** take direction `Shock_{t−1}`.
* **Exit:** a chandelier stop `S ← max_d(S, C_{t−1} − d·k·ATR_{t−1})`; exit when `d·(C_{t−1} − S) < 0`.
  The stop distance equals the shock threshold k.
* **Reverse:** on an opposite shock.
* **Size:**

```math
\text{units} = \min\!\left(\frac{NAV\cdot r}{k\cdot ATR\cdot x_{quote\to USD}},\ \frac{10\cdot NAV}{C\cdot x_{quote\to USD}}\right)\ \text{floored to 1{,}000-unit lots; 0 for any degenerate input}
```

**Portfolio circuit breakers**, checked at every hourly close and acted on at the next open:

* a trading-day loss ≥ 2.5 % (the day rolls at 17:00 NY) → flatten everything and open nothing for
  24 h;
* a drawdown ≥ 10 % from peak → flatten and halt permanently.

### Walk-forward design

There are 5 rolling windows, each 70 % in-sample (≈ 8 months) and 30 % out-of-sample (≈ 3.4 months).
The out-of-sample segments tile 2021-07-30 → 2022-12-30 (1.42 years).

* **In-sample.** Each of the 16 grid points is a separate Nautilus run. The harness picks the best
  3×3-neighbourhood ("plateau") Sharpe among points trading more than 100 times a year.
* **Out-of-sample.** One continuous Nautilus run. Parameters switch at segment boundaries (positions
  from the old parameters are liquidated there), and NAV and circuit-breaker state carry over.
* **Sharpe.** Daily (trading-day) returns, annualised with √252.

### WFA results — final trial (T7), per split

| Split | IS window | OOS window | Chosen (k, n) | IS Sharpe | OOS Sharpe | OOS return | OOS max DD | OOS trades/yr | Win % | Net pips/trade |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2020-11-30 → 2021-07-30 | 2021-07-30 → 2021-11-11 | 3.5, 96 | 0.67 | 3.11 | +1.77 % | 0.72 % | 166 | 53.2 | +11.8 |
| 2 | 2021-03-14 → 2021-11-11 | 2021-11-11 → 2022-02-22 | 3.5, 96 | 1.65 | −1.19 | −0.79 % | 1.54 % | 204 | 41.4 | −9.6 |
| 3 | 2021-06-27 → 2022-02-22 | 2022-02-22 → 2022-06-06 | 3.0, 96 | 1.29 | −1.22 | −1.15 % | 1.66 % | 307 | 33.3 | −8.2 |
| 4 | 2021-10-07 → 2022-06-06 | 2022-06-06 → 2022-09-18 | 3.0, 96 | 0.13 | 0.64 | +0.63 % | 2.04 % | 375 | 36.5 | +0.8 |
| 5 | 2022-01-19 → 2022-09-18 | 2022-09-18 → 2022-12-30 | 2.0, 96 | −0.03 | −2.01 | −4.38 % | 5.11 % | 1,297 | 34.6 | −9.0 |
| **Stitched** | | 2021-07-30 → 2022-12-30 | | **0.74** (mean) | **−0.64** | **−3.95 %** | **6.36 %** | **469** | 36.7 | −5.9 |

### WFA results — best trial (T1), per split

T1 used the same shock rules, with the stop as a separate parameter, `shock_k` 2–3.5 and
`stop_mult` 1.5–5.

| Split | Chosen (k, stop) | IS Sharpe | OOS Sharpe | OOS return | OOS max DD | OOS trades/yr | Net pips/trade |
|---|---|---|---|---|---|---|---|
| 1 | 3.5, 1.5 | 0.79 | 1.58 | +1.09 % | 0.97 % | 236 | +3.2 |
| 2 | 3.5, 2.5 | 0.96 | 0.24 | +0.22 % | 1.87 % | 250 | −3.3 |
| 3 | 2.0, 5.0 | 1.32 | −0.32 | −0.28 % | 1.32 % | 930 | −0.4 |
| 4 | 2.0, 5.0 | −0.06 | 1.01 | +0.86 % | 1.43 % | 953 | +1.3 |
| 5 | 2.5, 5.0 | 0.56 | 0.94 | +1.04 % | 1.09 % | 581 | +13.4 |
| **Stitched** | | **0.71** | **0.66** | **+2.97 %** | **2.75 %** | **590** | +2.8 |

T1's OOS net pips per trade by pair:

| USDJPY | EURJPY | USDCAD | EURUSD | GBPUSD |
|---|---|---|---|---|
| +15.2 | +9.1 | +0.3 | −1.8 | −7.8 |

### Drawdown statistics

| | Final trial (T7) | Best trial (T1) |
|---|---|---|
| Stitched OOS max drawdown | **6.36 %** (peak 2022-02-04 → trough 2022-12-30, not recovered) | **2.75 %** |
| Longest time under water | 329 days (2022-02-04 → end of data) | — |
| Worst OOS split drawdown | 5.11 % (split 5) | 1.87 % (split 2) |
| Daily-loss halts / kill switch | 0 / not triggered | 0 / not triggered |
| OOS annualised volatility | 4.1 % | 3.1 % |

The circuit breakers never fired in these runs. Their behaviour is proven by the tests below.

### All trials (cycle 2)

| # | Hypothesis (full text in `trials.log`) | OOS Sharpe | IS Sharpe | WFE | OOS max DD | OOS trades/yr | Failed |
|---|---|---|---|---|---|---|---|
| 1 | Shock entry + chandelier stop, replicating cycle 1's best idea on unseen data | **0.66** | 0.71 | **0.93** | 2.7 % | 590 | b |
| 2 | T1, high-conviction shocks only (k 3–5) | 0.12 | 0.54 | 0.22 | 3.5 % | 205 | b, d |
| 3 | Classic multi-day Donchian breakout (5–30 days), stop-and-reverse | 0.02 | 0.69 | 0.03 | 2.6 % | 161 | b, d |
| 4 | T1 with wider stops (3.5–10 ATR) | 0.20 | 0.55 | 0.36 | 2.6 % | 669 | b, d |
| 5 | Cross-sectional currency-strength trend, zero-cross entries | −0.13 | −0.80 | n/a | 1.7 % | 602 | b, d |
| 6 | Currency strength with a hysteresis band | −0.67 | 0.36 | −1.86 | 4.4 % | 396 | b, d |
| 7 | T1 with the stop tied to the threshold, searching the ATR baseline (robustness test) | −0.64 | 0.74 | −0.86 | 6.4 % | 469 | b, d |
| 8 | *not used* | | | | | | |

### Integrity tests (run on every evaluation)

* **Look-ahead (gate e).**
  * *Vectorised signals:* scramble every pair from bar t+1 (and, stricter, from bar t). Signal rows
    ≤ t must be bit-identical. Two canaries must be caught: one frame reads bar t+1, the other reads
    bar t's close.
  * *NautilusTrader:* rebuild the engine with every pair's M15 data scrambled from a cut time c.
    Every order and every equity mark stamped ≤ c must be identical. The cut is placed both at an
    hour boundary and **15/30/45 minutes into an hour**, which catches a bar that is secretly
    published at its open.
* **Parity.** Nautilus and the vectorised reference must produce identical order lists and fill
  prices, with equity within one cent per fill (Nautilus books P&L in cents). This is checked on
  EURUSD with the real 5-segment walk-forward plan, and on synthetic data with corrupt bars, a
  volatility spike, and 3 parameter segments.
* **Edge cases.**
  * zero, NaN or ∞ inputs to position sizing
  * flat prices (no orders, NAV unchanged)
  * missing, NaN, negative, inconsistent and duplicated M15 rows
  * an 8 % gap with a 10 % range bar (leverage cap respected)
  * a ~5 % daily loss must flatten, block entries for 24 h, then resume
  * a ~15 % loss must kill trading permanently
  * exact friction: $26.00 on 100k EURUSD, and $27.27 on 100k EURJPY after JPY→USD conversion, on a
    fresh engine and after a reset
* **Mutation-tested.** Each of these deliberate bugs makes the tests fail:
  * bars stamped at their open (look-ahead) → caught by the mid-hour cuts;
  * zero order latency → caught by parity;
  * removing the xrate workaround → caught by the friction test;
  * disabled circuit breakers → caught by the breaker test.
* **Complexity audit.** At most 4 tunable parameters and at most 3 indicators, and no calendar
  attributes appear in `strategy.py`.

---

### Why it failed, and why I stopped at 7

1. **The only positive result was fragile.** T1 (0.66, WFE 0.93) is the one out-of-sample success,
   and it did not survive its two robustness tests:
   * shifting half the grid (T4) → 0.20;
   * removing the stop parameter (T7) → −0.64.

   A real edge should not depend on grid layout.
2. **No family generalised.** Every family tested failed out-of-sample:
   * range shocks (T1, T2, T4, T7)
   * multi-day breakouts (T3)
   * currency-level momentum (T5, T6)

   Most of their in-sample grids are noise, and the high in-sample cells never carried forward.
3. **Statistical power.** 1.42 OOS years puts the standard error of an annualised Sharpe at about
   0.84. T1's 0.66 is about 0.8 SE from zero, and the best of 7 pure-noise trials would be expected
   at about 1.35 SE (≈ 1.1 Sharpe). The results are consistent with **no exploitable edge**.
4. **Pair effects are real but can't be selected after the fact.** USDJPY was profitable
   out-of-sample in every trial and GBPUSD lost in every trial. Keeping only the winners would be
   look-ahead selection.
5. **Hindsight caveat.** I know from public history that 2022 had strong USD and JPY trends, so any
   trend hypothesis chosen now could be biased by that. T3 was picked as the textbook rule for that
   reason. The trend trials failed anyway, so the bias did not produce a false pass.

Why trial 8 was left unused: no untried hypothesis had in-sample support. Running it anyway would
mostly add selection bias, since each extra attempt makes a lucky pass more likely. The slot is still
available.

**What would genuinely change the picture:**

* **Longer data.** Exporting M15 history back to about 2015 from MT5 would meet the original 3-year
  rule and give roughly 5× more out-of-sample time. This matters most.
* **A true holdout.** Once the longer history is in, keep 2023 onward untouched until a candidate
  exists.
* **Diversification across more currencies** (AUD, NZD, CHF, crosses) rather than across signals.

### Known limitations

* The MT5 `spread` field is one value per M15 bar. Intra-bar spread changes and swap/financing are
  not modelled.
* Stops are evaluated on hourly closes and filled at the next open quote, not intrabar.
* Tick volume from the exports is not used.
* The sample is 2.1 years: below the original 3-year rule, and relaxed by your decision.

<details>
<summary>Cycle 1 summary (custom engine, OANDA EURUSD 2015-01 → 2020-05)</summary>

All 8 trials failed. The best out-of-sample Sharpe was 0.32 (shock entry + chandelier stop), with
2.0-pip round-trip costs. Full log: `research/cycle1/trials.log`. Code: git history up to commit
`fd14177`, one commit per trial.
</details>
