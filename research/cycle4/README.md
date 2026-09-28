# Cycle 4 (archived): published FX factor strategies, pre-registered

This is the cycle-4 write-up as it stood in the main README at commit `c447b91`. The code that produced it is still
in the repository (`hypotheses/a1..a6_*.py`, `a3f_ewma_fast_wfa.py`, `academic_base.py`); the logs and JSON reports
now live in this folder. Reproduce a logged result exactly with `git checkout c447b91 && python harness.py
--hypothesis A3`.

## FX factor research: cycle 4 (published strategies, pre-registered, 2005 → 2022, NautilusTrader)

**Verdict: no strategy passes.**
- All six pre-registered strategies from the academic and institutional literature failed **Stage 1**, the
  mandate's walk-forward gates on the development period.
- Trial 8, a post-hoc re-test of the best one (A3) with a fast rolling walk-forward (Addendum 2), failed as well.
- None became eligible for **Stage 2**, so the **2023 holdout was never loaded**.
- The work was merged to `main` at your request (PR #3); the protocol itself would open a PR only for a strategy
  that passes both stages.

### How bias and data mining were ruled out

Everything was fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) and pushed (`0e7822b`) **before any backtest on
real data**:
- the strategies and their order;
- the signal parameters, taken from the papers;
- the grids, restricted to values the papers report plus a cost band;
- the data splice, costs, overnight financing and gates.

Each run is logged in `trials.log` under a code fingerprint, and the harness refuses any other order. One bug fix
after trial 1 is disclosed in **Addendum 1**, and the post-hoc trial 8 in **Addendum 2** (see below). All 8 trials
of the budget are now used.

### Stage 1 results (development data 2005-01 → 2022-12; stitched out-of-sample 2011-10-02 → 2022-12-30, 11.25 years)

| ID | Strategy (paper) | OOS Sharpe | mean IS Sharpe | WFE | OOS max DD | fills/yr | Kill switch | Failed gates |
|---|---|---|---|---|---|---|---|---|
| A1 | time-series momentum (Moskowitz, Ooi & Pedersen 2012) | −0.03 | −0.12 | n/a | 10.0 % | 175 | 2016-08 | b, d |
| A2 | 1/3/12-month trend blend (Hurst, Ooi & Pedersen 2017) | 0.05 | −0.07 | n/a | 10.0 % | 502 | 2017-05 | b, d |
| A3 | multi-speed EWMA crossover (Baz et al. 2015, Man AHL) | **0.25** | 0.32 | **0.79** | **8.9 %** | 483 | — | **b only** |
| A4 | carry (Koijen, Moskowitz, Pedersen & Vrugt 2018) | −0.72 | −0.37 | n/a | 10.0 % | 2.4 | 2015-03 | a, b, d |
| A5 | cross-sectional momentum (Menkhoff, Sarno, Schmeling & Schrimpf 2012) | −0.23 | −0.11 | n/a | 10.0 % | 395 | 2020-02 | b, d |
| A6 | equal blend of A2, A4 and A5 (Asness, Moskowitz & Pedersen 2013) | −0.46 | −0.38 | n/a | 10.0 % | 690 | 2020-11 | b, d |
| *A1, trial 1* | *before the execution fix (Addendum 1)* | *−0.03* | *−0.13* | *n/a* | *10.0 %* | *174* | *2016-08* | *b, d + integrity* |

Gates: (a) fills/yr > 100, (b) OOS Sharpe ≥ 1.5, (c) max DD < 12 %, (d) WFE ≥ 0.60, (e) look-ahead test.
- Every run passed (c) and (e) and every integrity test, except trial 1's position-desync test.
- "Kill switch" is the mandate's permanent halt at a 10 % drawdown. After it fires, the rest of the stitched run is
  flat and is still counted in the Sharpe.

#### Walk-forward breakdown: OOS Sharpe per split (parameters re-chosen on each 5.25-year in-sample window)

| Split (OOS window) | A1 | A2 | A3 | A4 | A5 | A6 |
|---|---|---|---|---|---|---|
| 1 (2011-10 → 2013-12) | 0.13 | −0.05 | 0.04 | −0.61 | 0.16 | −0.35 |
| 2 (2013-12 → 2016-04) | 0.06 | 0.67 | 0.96 | −1.51 (killed) | 0.32 | −0.22 |
| 3 (2016-04 → 2018-07) | −0.60 (killed) | −0.94 (killed) | −0.12 | flat | −0.57 | −0.80 |
| 4 (2018-07 → 2020-09) | flat | flat | −0.76 | flat | −1.17 (killed) | −0.58 |
| 5 (2020-09 → 2022-12) | flat | flat | 0.78 | flat | flat | −1.40 (killed) |

The chosen parameters, all in-sample grids, per-pair statistics and every integrity test are in
`research/cycle4/results/<ID>.json`.

#### Where the money went (stitched OOS, USD on a 1,000,000 account)

| ID | price P&L before costs | spread + slippage | net overnight financing | net result |
|---|---|---|---|---|
| A1 | +103,300 | −89,400 | −30,900 | −17,000 |
| A2 | +121,200 | −86,900 | −23,600 | +10,800 |
| A3 | +127,900 | −6,200 | −30,100 | +91,700 |
| A4 | −89,400 | −200 | +2,400 | −87,200 |
| A5 | +22,700 | −55,400 | −23,100 | −55,800 |
| A6 | −51,000 | −45,400 | +1,700 | −94,700 |

#### Drawdown statistics

- **Five of the six hit the permanent 10 % kill switch.** A3 did not: its maximum drawdown was 8.9 %, with the worst
  split drawdown 7.0 % (2018–2020).
- **No daily-loss halt** (2.5 % in a day) fired in any run.
- Out-of-sample volatility was 1–3.3 % a year, as sized.

### What the results mean

1. **Nothing is close to 1.5.**
   - The best, A3, reached 0.25. With 11.25 out-of-sample years the standard error of a Sharpe ratio is about
     0.30, so 0.25 is 0.8 SE from zero.
   - The best of six strategies with no edge would be expected at about +0.38.
   - The results are consistent with **no exploitable edge after costs** on these five pairs.
2. **Trend was there before costs.**
   - A1–A3 made +$103k to +$128k over the active period before costs. That is the modest, positive FX trend premium
     the literature reports for the 2010s.
   - For A1 and A2, spreads took almost all of it, because both flip positions often. For A1, plateau selection
     picked 1-month lookbacks in 2014–2018; A2's blend always includes a 1-month component.
   - A3 trades in small rebalances, so its spread cost was only about $6k. But it pays roughly $30k of financing
     mark-up, which the literature's futures-based returns do not bear.
3. **Carry lost money before costs.**
   - From 2011 to 2015 carry was long CAD, GBP and EUR against USD and JPY, straight into the dollar rally.
   - With near-zero rates in most of these currencies there was little carry to earn. This matches the literature:
     G10 carry has been weak since 2008.
   - Carry also cannot meet the 100 trades/yr gate, since rate differentials rarely change.
4. **Cross-sectional momentum is weak in G10**, as Menkhoff et al. found; most of it sits in emerging-market
   currencies. With five currencies, the middle-ranked currency keeps dropping to zero weight and re-entering,
   which costs spread.
5. **The blend did not diversify.** Its components were weak or losing after costs over this period. Averaging them
   kept the churn and the losses.
6. **The gates and the universe don't match these strategies.**
   - Institutional trend and multi-factor programs reach about 0.5–0.8 Sharpe by trading 50–150 markets across
     commodities, bonds, equity indices and currencies, with futures-level costs.
   - Four independent USD crosses plus EURJPY, retail spreads and a financing mark-up cannot deliver 1.5. With a
     permanent 10 % kill switch over an 11-year window, a strategy near zero Sharpe will almost surely be stopped.

### Addendum 1: an execution bug, found by trial 1 and fixed

Trial 1's integrity test for position desyncs failed (USDCAD, Christmas 2009). The root cause was two engine bugs,
present since cycle 2.

1. **Stale fills.** Nautilus processes due orders after every data point. With a fixed order latency, every pair
   except the first in the data stream filled at its *previous close* quote, not its next open. This is not
   look-ahead, since the price predates the decision, but it is optimistic.
2. **Late decisions.** When only some pairs traded in an hour (holidays), the decision waited for the next bar
   event, sometimes days later, and doubled orders.

The fix:
- A close is decided once every pair with a bar has reported.
- Each pair's order goes out on its own next quote and fills there.
- A new five-pair test with random gaps and a one-pair holiday checks every fill price. It flagged 155 of 175 fills
  on the old code.

A1 was re-run under the fixed code (trial 2) with an unchanged specification; both results are shown above. Cycles 2
and 3 used the same execution code; the bias favoured them, and their failing verdicts stand.

### Addendum 2 (trial 8): A3 with a fast rolling walk-forward

At your request, A3 was re-tested with its code unchanged, but with a faster walk-forward and a stricter efficiency
gate:
- 28 rolling (not anchored) out-of-sample windows of 146.7 days, under half a year, covering the same 2011-10 →
  2022-12 span as before.
- Each is preceded by its own 342-day in-sample window (70 : 30), and the band is re-chosen on every window.
- Gate (d) was raised to WFE ≥ 2/3.

The protocol was pre-registered and pushed (`8acfd39`) before the run. It was chosen after seeing A3's result, so
only the 2023 holdout could have confirmed a pass.

| | Standard WFA (trial 4) | Fast rolling WFA (trial 8) |
|---|---|---|
| OOS windows | 5 × 2.25 years | 28 × 147 days |
| OOS Sharpe (gate b ≥ 1.5) | 0.25 ✗ | **0.21 ✗** |
| mean IS Sharpe | 0.32 | 0.19 |
| WFE (gate d) | 0.79 (≥ 0.60 ✓) | 1.13 (≥ 2/3 ✓) |
| max drawdown | 8.9 % | 9.3 % |
| fills / yr | 483 | 617 |

**What the fast walk-forward shows:**
- **In-sample results don't predict the next window.** Across the 28 windows, the correlation between in-sample and
  out-of-sample Sharpe is **0.00**.
- **The overall efficiency figure is misleading.** WFE passes only because a one-year in-sample Sharpe averages a
  low 0.19. Window by window, WFE is defined in 15 of the 28 windows and reaches 2/3 in just 5 of them (median
  −0.28).
- **Window Sharpes are noise.** They range from −2.4 to +2.6; 15 of 28 are positive and 4 exceed 1.5. That spread is
  what a strategy with a Sharpe of about 0.2 produces at this window length, and good windows don't persist.
- **Re-optimising faster changes nothing.** The band choice barely matters, so the out-of-sample Sharpe stayed at
  0.21. As predicted in the addendum, the result still falls far short of 1.5.

### What could legitimately change the outcome

- **More markets.** The published Sharpe of trend and multi-factor strategies comes from diversification across
  dozens of markets, not from five FX pairs. This needs more data, such as index, bond and commodity futures, or at
  least G10 plus EM FX.
- **Institutional costs.** Futures or prime-broker financing has no 0.5 % mark-up, and raw spreads are cheaper. Trend
  was positive before costs here. These costs must come from real account data, not assumptions.
- **Gates that fit the strategy class.** A 1.5 Sharpe and 100 trades a year target fast, high-Sharpe strategies,
  while published factor premia are slower and smaller. Changing gates after seeing results would be data mining,
  so a new protocol must be pre-registered first.
- **The 2023 holdout is still untouched** and available for exactly that. This cycle's 8-trial budget is used up,
  so any further test needs a new pre-registered cycle.

### Reproduce

```bash
pip install -r requirements.txt                 # nautilus_trader==1.221.0, pandas, pyarrow
# MT5 exports at data/mt5/<PAIR>.csv for EURUSD GBPUSD USDJPY USDCAD EURJPY (git-ignored)
python harness.py --check-data                  # splice + rate-table checks, no backtest
python harness.py --hypothesis A1 --selftest    # synthetic integrity tests, ~1-2 min, never a trial
python harness.py --hypothesis A1               # Stage 1; re-running logged code reproduces it and logs nothing
python harness.py --hypothesis A1 --holdout     # Stage 2; refused unless A1 passed Stage 1 (none did)
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
| `PREREGISTRATION.md` | Cycle-4 protocol, data and rate hashes, code fingerprints; Addendum 1. |
| `hypotheses/academic_base.py` | Shared engine: sizing, no-trade-band execution, breakers, Nautilus strategy, vectorised reference, signal kernels. |
| `hypotheses/a1..a6_*.py` | The six pre-registered signals, each with its paper and formulas in the docstring. |
| `backtest.py` | Nautilus plumbing, OANDA + MT5 splice, rate table, overnight-financing module. |
| `harness.py` | Stage 1 / Stage 2 walk-forward, gates, integrity tests, trial and holdout guards. |
| `trials.log`, `holdout.log` | Cycle-4 Stage-1 entries (8) and Stage-2 entries (none). |
| `research/cycle4/results/` | JSON report per run (`A1_trial1.json` is the pre-fix run; `A3F.json` is trial 8). |
| `data/rates/` | OECD short-term rates and central-bank policy decisions. |
| `research/cycle1..3/` | Earlier cycles: logs, reports and write-ups. |

