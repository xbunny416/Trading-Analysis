# Trading-Analysis

This repository just for my own coding trading workspace.

## FX strategy search: cycle 5 (adaptive, fully logged, until a strategy passes)

**Status: running.** No strategy has passed yet. The leaderboard below is updated after every batch of trials.

### Protocol

You asked for a search loop that runs until a strategy passes. It uses your 5 FX pairs and has no trial cap.

- **Gates (Stage 1):** the development walk-forward (2005 → 2022, five rolling 70/30 splits, stitched OOS 2011-10 →
  2022-12). Every one of these must hold:
  - (a) more than 100 fills a year;
  - (b) OOS Sharpe ≥ 1.5;
  - (c) max drawdown < 12 %;
  - (d) walk-forward efficiency OOS ÷ IS **> 0.5**;
  - (e) the look-ahead test and every integrity test.
- **Engine:** event-driven NautilusTrader for every run. Fills happen at each pair's own next open. Costs are your
  broker's spreads (1-pip floor) plus 0.5 pip slippage, and overnight financing carries a 0.5 % mark-up.
- **Every look at real data is a logged trial** with an immutable ID (`C5-001`, …). Its module is committed and pushed
  before it runs, and there is no unlogged screening. Changed code needs a new ID.
- **Multiple testing:** a search like this will eventually produce a Stage-1 pass by luck. So each trial reports a
  **deflated Sharpe ratio (DSR)**: the probability that its Sharpe beats the best one expected from all the trials run
  on this data, cycle-4 trials included.
- **The verdict:** a Stage-1 pass unlocks the untouched **2023 holdout**, once. Only a holdout pass counts as a PASS
  and opens a pull request. Every holdout look is reported.

### Leaderboard (Stage 1, development data)

| Trial | Family | OOS Sharpe | WFE | max DD | fills/yr | DSR (N) | Failed gates |
|---|---|---|---|---|---|---|---|
| C5-001 | time-series mean reversion (hourly z-score) | -0.38 | n/a | 10.0% | 132 | 0.0013 (9) | b, d |
| C5-002 | cross-sectional short-term reversal | -0.78 | n/a | 10.0% | 640 | 0.0 (10) | b, d |
| C5-003 | trend: A3 with a signal-strength threshold | 0.23 | 0.79 | 9.2% | 276 | 0.0999 (11) | b |
| C5-004 | trend: A2 that refuses to pay carry | -0.05 | n/a | 10.0% | 494 | 0.0141 (12) | b, d |
| C5-005 | trend: A2 only in efficient (trending) regimes | -0.09 | n/a | 10.0% | 482 | 0.0107 (13) | b, d |
| C5-006 | trend: dollar consensus of A3 over the USD pairs | 0.25 | 1.04 | 6.2% | 585 | 0.114 (14) | b |

### Batch notes

- **Batch 1 (C5-001…003):** three different return sources. Cycle 4's trend strategies were one bet (daily-return
  correlation 0.75–0.91), and carry offset trend (−0.6). So this batch probes short-horizon mean reversion, a
  market-neutral short-term reversal, and a cheaper version of the only positive trend (A3).
  - **Result:** all fail.
  - **Mean reversion (C5-001)** loses even before costs (−$33k): it wins 60–70 % of trades, but FX keeps trending
    over 1–7 days and the losers are large. The kill switch fired.
  - **Short-term reversal (C5-002)** has a tiny edge before costs (+$12k). About 5,000 entries and exits, costing
    $94k of spread, destroy it. The kill switch fired.
  - **A3 with a threshold (C5-003)** matches A3 (0.23 vs 0.25). The threshold barely changes it, and overnight
    financing (−$30k) is its main drag, not spread ($7k).
  - **Lesson:** trend is the only source with a real edge before costs (about +$124k, a Sharpe of roughly 0.33).
    Batch 2 therefore works on the trend book itself.
- **Batch 2 (C5-004…006), registered:**
  - **C5-004:** trend that refuses positions paying more than κ % a year of carry, since financing is trend's main
    drag.
  - **C5-005:** trend held only in trending regimes, measured by a normalised efficiency ratio.
  - **C5-006:** A3's trend averaged into one dollar view, traded through the four USD pairs.
  - **Harness:** a new multi-pair signal-parity test checks that, at every decision, the signal the engine trades
    equals the vectorised one for every pair, cross-pair signals included. A deliberate bug in the ranking was
    caught 8,394 times.
  - **Result:** all fail.
    - **The carry filter (C5-004)** and **the regime filter (C5-005)** both turn a +$75k–81k edge before costs
      negative. Switching positions on and off costs $82k–90k of spread, and both hit the kill switch.
    - **The dollar consensus (C5-006)** is the most efficient trend book so far: 0.25 Sharpe, WFE 1.04, 6.2 %
      drawdown, only $4k of spread. But its edge before costs (about $84k) is no bigger.
  - **Lesson:** FX trend on these pairs is worth about 0.35 before costs and 0.25 after, however it is packaged.
    Batch 3 leaves trend.

### Reproduce

```bash
pip install -r requirements.txt                    # nautilus_trader==1.221.0, pandas, pyarrow
# MT5 exports at data/mt5/<PAIR>.csv for EURUSD GBPUSD USDJPY USDCAD EURJPY (git-ignored)
python harness.py --check-data                     # splice + rate-table checks, no backtest
python harness.py --hypothesis C5-001 --selftest   # synthetic integrity tests, ~1-2 min, never a trial
python harness.py --hypothesis C5-001              # Stage 1; re-running logged code reproduces it, logs nothing
python harness.py --hypothesis C5-001 --holdout    # Stage 2; refused unless C5-001 passed Stage 1
python scripts/leaderboard.py                      # the leaderboard above, from results/cycle5/
```

The OANDA history is fetched on first use (a sparse git checkout of `FutureSharks/financial-data` at `7ba1d40`) and
cached in `data/spliced/`. A Stage-1 run takes 6–13 minutes on 4 cores.

### Data, costs and engine

- **Prices.** OANDA 1-minute mids 2005-01 → 2019-11 are spliced with your MT5 M15 exports from 2019-12-01 22:00 UTC.
  - Before the splice, USD/JPY = EUR/JPY ÷ EUR/USD.
  - OANDA's weekend and Sunday pre-open quotes are dropped.
  - On the six-month overlap, OANDA and MT5 hourly closes agree to a median 0.1–0.35 pips.
- **Costs.**
  - Your broker's spread: the MT5 bar spread; before the splice, the median MT5 spread for that pair and UTC hour.
    Floor 1 pip, +0.5 pip slippage per fill.
  - Overnight financing at each 17:00 New York roll: `qty·mid·(r_base − r_quote)/365 − |qty|·mid·0.5 %/365`.
  - Rates are OECD 3-month rates to 2020-06, then central-bank policy rates, committed in `data/rates/`.
  - The carry signal sees only published rates: a monthly average from the next month, a decision from the next
    day.
- **Sizing** (the mandate's formula): `units = NAV · 0.3 % · |s| / (ATR · 10 · quote→USD)`.
  - Wilder ATR over 1,440 hourly bars; whole 1,000-unit lots; 10× cap per position.
  - About 1.5 % annualised volatility per position at full signal.
- **Execution.**
  - The decision is taken at an hourly close; each pair fills at its own next open quote.
  - A no-trade band sets rebalancing, with no calendar rule.
  - Circuit breakers: 2.5 % daily loss → 24 h halt; 10 % drawdown → permanent halt.
- **Walk-forward.**
  - Five rolling splits (70 % IS / 30 % OOS) after an 18-month warm-up.
  - Plateau selection among combinations with more than 100 in-sample fills a year.
  - One stitched NautilusTrader OOS run.
- **Integrity tests, on every run:**
  - parity between Nautilus and an independent vectorised reference, including financing;
  - multi-pair fill timing;
  - look-ahead tests (bar-boundary and mid-bar cuts, canaries) and rate-publication-lag tests;
  - financing checked against hand calculations;
  - the execution rule, circuit breakers, friction (including JPY→USD) and corrupt data;
  - splice checks and the holdout lock.

### Files

| Path | Purpose |
|---|---|
| `PREREGISTRATION.md` | Cycle-4 protocol, data and rate hashes, code fingerprints; Addenda 1–2. |
| `hypotheses/academic_base.py` | Shared engine: sizing, no-trade-band execution, breakers, Nautilus strategy, vectorised reference, signal kernels. |
| `hypotheses/c5_*.py` | Cycle-5 trials, one module per trial ID, hypothesis and formulas in the docstring. |
| `hypotheses/a1..a6_*.py`, `a3f_*.py` | Cycle-4 signals, kept as a library. |
| `backtest.py` | Nautilus plumbing, OANDA + MT5 splice, rate table, overnight-financing module. |
| `harness.py` | Stage 1 / Stage 2 walk-forward, gates, integrity tests, trial and holdout guards. |
| `trials.log`, `holdout.log` | Cycle-5 Stage-1 and Stage-2 entries. |
| `results/cycle5/` | JSON report per cycle-5 trial. |
| `scripts/leaderboard.py` | Prints the leaderboard below from the reports. |
| `data/rates/` | OECD short-term rates and central-bank policy decisions. |
| `research/cycle1..4/` | Earlier cycles: logs, reports and write-ups. |

<details>
<summary>Earlier cycles (all failed the gates honestly)</summary>

* **Cycle 1:** custom engine, OANDA EURUSD 2015 → 2020, 8 adaptive trials; best OOS Sharpe 0.32.
* **Cycle 2:** NautilusTrader, your 5-pair MT5 data 2020-11 → 2022-12, 7 adaptive trials; the best (0.66) proved
  fragile.
* **Cycle 3:** the 7 cycle-2 intraday hypotheses pre-registered and re-tested on 2019-12 → 2022-12. All failed Stage 1
  (best 0.63), and the 2023 holdout was never used. See [`research/cycle3/README.md`](research/cycle3/README.md).
* **Cycle 4:** six published FX factor strategies (trend, carry, cross-sectional momentum), pre-registered, on
  2005 → 2022 data, plus a post-hoc fast re-test of the best. All failed Stage 1; best OOS Sharpe 0.25 (Baz et al.
  multi-speed EWMA crossover). See [`research/cycle4/README.md`](research/cycle4/README.md).
</details>
