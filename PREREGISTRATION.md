# Cycle 4 pre-registration: published FX factor strategies

This file is committed and pushed **before any backtest on real data**. Nothing below changes after results are seen.
The git history is the audit trail: this commit precedes every entry in `trials.log` and `holdout.log`.

## Why this cycle, and what to expect

Cycles 1–3 tested intraday technical rules and all of them failed honestly. The best out-of-sample Sharpe was 0.63
against the 1.5 gate. As you asked, cycle 4 switches to strategies that institutions run and the academic literature
documents: time-series momentum, EWMA-crossover trend, carry, cross-sectional momentum, and a blend of them.
Every signal and every signal parameter is taken from the papers; this repository chooses none of them.

**Expectation, stated now:**
- The published Sharpe ratios for these factors on FX alone are roughly 0.3–0.9, gross of retail costs.
- MOP 2012 report FX time-series momentum at about 0.5.
- G10 carry did well before 2008, crashed in 2008 and was weak through the 2010s.
- Menkhoff et al. find G10 currency momentum weak; most of it sits in emerging-market currencies.
- A 1.5 out-of-sample Sharpe is therefore **two to four times the literature**. The likely outcome is that
  nothing passes.

What this cycle does deliver is a trustworthy answer over about 11.25 out-of-sample years (2011-10 → 2022-12). The
standard error of an annualised Sharpe is about 0.30 there, against 0.69 in cycle 3. The 2023 holdout stays locked.

## Data

### Prices (5 pairs: EURUSD GBPUSD USDJPY USDCAD EURJPY)

| Period (UTC) | Source | Notes |
|---|---|---|
| 2005-01-02 → 2019-11-29 | OANDA 1-minute mid candles, `github.com/FutureSharks/financial-data` at commit `7ba1d404aa8b` | **USDJPY = EURJPY / EURUSD**: EURJPY minutes divided by the latest EURUSD close, at most 5 min old (OANDA has no USD_JPY) |
| 2019-12-01 22:00 → 2022-12-30 | your MT5 M15 exports (`data/mt5/<PAIR>.csv`) | unchanged from cycle 3 |

- **Splice point:** the first common MT5 bar, 2019-12-01 22:00 UTC. OANDA bars before it, MT5 bars from it.
- **Weekly closure:** OANDA minutes between Friday 17:00 and Sunday 17:00 New York are dropped. OANDA carries
  indicative Saturday quotes (2011) and Sunday pre-open quotes; your broker has no bars there. After this rule
  every year has about 6,200–6,260 hourly bars, as MT5 does. The rule was set after checking bar counts, which is
  a data-quality check that involves no returns or backtests.
- **OANDA-era costs:** for each pair and UTC hour, the median MT5 spread over the development period (MT5 bars
  before 2023 only). Then the unchanged rules apply: floor 1.0 pip, +0.5 pip slippage per fill.
  **Disclosure:** spreads in 2005–2008 were wider than today's, so early-year costs are somewhat optimistic.
- **Hashes:**
  - MT5 exports, first 16 hex of SHA-256: EURUSD `bc4b9cfd06f8c3dc`, GBPUSD `01e555bc13d01094`,
    USDJPY `1af85499df842f9b`, USDCAD `6f93d394bc220851`, EURJPY `99467048dc35ee9f`.
  - Development frames as loaded (`pandas.util.hash_pandas_object`, SHA-256 prefix): EURUSD `a44796505eb3e309`,
    GBPUSD `facbbd724baca0f8`, USDJPY `290d38bf151e6ed2`, USDCAD `79a851f9aec35f49`, EURJPY `b583721233caffb2`.
    These are recorded in every report's `provenance`.

**Splice checks** (`python harness.py --check-data`; they also run inside every Stage-1 run). On the overlap
2019-12-01 → 2020-05-29 (2,766 hours), OANDA-derived and MT5 hourly mid closes agree:

| Pair | median abs diff (pips) | p95 (pips) | same, shifted 1 h (pips) | median range ratio OANDA/MT5 | gap at splice |
|---|---|---|---|---|---|
| EURUSD | 0.10 | 1.10 | 4.5 | 1.004 | 48 h (weekend) |
| GBPUSD | 0.35 | 2.30 | 7.2 | 1.002 | 48 h |
| USDJPY (derived) | 0.20 | 2.80 | 4.5 | 1.063 | 48 h |
| USDCAD | 0.20 | 1.59 | 5.7 | 1.000 | 48 h |
| EURJPY | 0.25 | 2.15 | 6.0 | 0.991 | 48 h |

Pass rule: median below 1 pip, more than 1,000 overlap hours, gap ≤ 72 h, and a 1-hour shift at least 3× worse
(timestamps aligned).

### Interest rates (overnight financing for all hypotheses; the carry signal for A4 and A6)

| File | Content | SHA-256 prefix |
|---|---|---|
| `data/rates/oecd_short_term.csv` | OECD monthly 3-month interbank rates, USD EUR GBP JPY CAD, 2003-01 → 2020-06, copied from `nautilus_trader` v1.221.0 `tests/test_data/short-term-interest.csv` (USD 2020-04 and JPY 2020-06 missing, so the previous value carries forward) | `bfac90a21a3d8a14` |
| `data/rates/policy_rates.csv` | central-bank policy-rate decisions with effective dates, 2016 → 2023: Fed target-range midpoint, ECB deposit rate, BoE Bank Rate, BoJ policy-balance rate, BoC overnight target | `a6db1f84c1a4f5de` |

- **Hand-over:** OECD monthly values before 2020-07-01, policy rates from then on.
- **Two uses of the rates:**
  - *Accrual* (financing): the rate in force that day, i.e. month M's OECD average during month M.
  - *Signal* (carry): only what was public. Month M's OECD average becomes known on the first day of month M+1. A
    policy decision becomes known at 00:00 UTC the day **after** its effective date: the BoE and the BoC announce
    during the day their new rate takes effect.
- **Transcription disclosure:**
  - The official rate sites (FRED, BIS, ECB, BoE and others) are blocked from this environment, so the policy table
    was transcribed from the published decisions. You can verify it or replace it with your broker's swap history.
  - The cross-check against OECD over 2020-01 → 2020-06 passes. The per-currency median |OECD − policy| is 0.05 pp
    (USD), 0.12 (EUR), 0.14 (GBP), 0.11 (JPY) and 0.13 (CAD), against a 0.3 pp limit.
  - The planned rule was "every month within 0.5 pp". It failed for USD in March 2020 (0.69 pp) and GBP in April
    2020 (0.55 pp). Those are the COVID funding-stress months, when 3-month interbank rates sat above policy
    rates, not transcription errors. The rule was therefore changed to the per-currency median before any
    backtest.

### Split

- **Development:** every bar before 2023-01-01 00:00 UTC. The first 18 months (to 2006-07-02) are warm-up only;
  the split geometry starts there.
- **Holdout:** 2023-01-01 → 2023-12-29, **locked**. Stage-1 runs load no price bar and no rate row dated in 2023,
  and a test asserts it on every run.
- **Five rolling splits** (70 % IS, 30 % OOS, step 0.3 W):

| Split | IS | OOS |
|---|---|---|
| 1 | 2006-07-02 → 2011-10-02 | 2011-10-02 → 2013-12-31 |
| 2 | 2008-10-01 → 2013-12-31 | 2013-12-31 → 2016-04-01 |
| 3 | 2011-01-02 → 2016-04-01 | 2016-04-01 → 2018-07-01 |
| 4 | 2013-04-01 → 2018-07-01 | 2018-07-01 → 2020-09-30 |
| 5 | 2015-07-02 → 2020-09-30 | 2020-09-30 → 2022-12-30 |

## Engine, costs, sizing, execution (identical for every hypothesis; `hypotheses/academic_base.py`)

- **Engine.**
  - NautilusTrader 1.221: SIM venue, NETTING, USD margin account, zero fees (costs are in the quotes).
  - Orders are decided at an hourly bar's close and filled at the next hour's open quote,
    `bid/ask = mid ∓ (max(spread, 1 pip)/2 + 0.5 pip)`.
- **Overnight financing.** At every 17:00 New York roll (one per calendar day; a weekend is three rolls, and there
  is no weekday rule), each position is credited or debited
  `qty_signed · mid · (r_base − r_quote)/365 − |qty| · mid · 0.5 %/365`. The amount is converted to USD and posted
  to the account before that data point's fills. The 0.5 % p.a. is a broker mark-up charged on both sides.
  - Nautilus's `FXRolloverInterestModule` is not used: it multiplies an unsigned quantity (so it ignores the side)
    and triples on both Wednesday and Friday.
- **Sizing** (the mandate's formula, volatility targeting): `units = NAV · 0.3 % · |s| / (ATR · 10 · quote→USD)`.
  - s is the signal in [−1, 1]; ATR is Wilder's ATR of hourly bars with period 1,440 (≈ 60 trading days).
  - Positions are whole 1,000-unit lots, capped at 10× NAV per position.
  - This gives about 1.5 % annualised volatility per position at |s| = 1. The level was chosen for the permanent
    10 % drawdown kill switch over an 11-year run. It is fixed and not tuned.
- **Execution: a no-trade band, no rebalancing calendar.** At every hourly close, with P the position and T the
  target:
  - ENTRY if P = 0; EXIT if T = 0; FLIP (one order) on a sign change;
  - otherwise REBAL if `|T − P| ≥ band · max(|T|, |P|)`.
  - A band below 1 can never block a sign change or an exit, since `|T − P| ≥ max(|T|, |P|)` there.
  - Walk-forward parameter changes do not liquidate positions.
- **Risk limits (the mandate):** a trading-day loss ≥ 2.5 % flattens everything and halts for 24 h; a drawdown
  ≥ 10 % flattens everything permanently.
- **Gate (a) counts every fill** (entries, exits, flips and rebalances), as you decided. The in-sample eligibility
  filter (> 100 per year) counts fills too.
- Units: 1 month = 520 hourly bars, 1 day = 24 bars. Every lookback is counted in bars, never by calendar.

## Hypotheses: fixed list, fixed order A1 → A6, no additions, no edits

Code fingerprint = `sha256(harness.py + backtest.py + hypotheses/academic_base.py + hypothesis file)[:12]` at this
commit.

| ID | Paper | Signal s (decided at the close of bar t−1) | Fixed (published) | WFA grid | Indicators | Fingerprint |
|---|---|---|---|---|---|---|
| A1 | Moskowitz, Ooi & Pedersen 2012, JFE | `sign(C − C_{−L})` per pair | — | L ∈ {1, 3, 6, 12} months × band ∈ {0.1, 0.2, 0.3, 0.5} | past return, ATR | `21ac561d0549` |
| A2 | Hurst, Ooi & Pedersen 2017, JPM | `(sign r_1m + sign r_3m + sign r_12m) / 3` | horizons 1, 3, 12 months | band (4) | past returns, ATR | `92f3a5db88da` |
| A3 | Baz, Granger, Harvey, Le Roux & Rattray 2015 (Man AHL) | `mean_k z_k·e^{−z_k²/4}/0.89`, with `z_k = [(EWMA_{S_k} − EWMA_{L_k}) / σ_63d(C)] / σ_252d(·)` | (S, L) = (8, 24), (16, 48), (32, 96) days; EWMA(n): α = 1/(24n) | band (4) | EWMA crossover, price σ, ATR | `6fe275e77b68` |
| A4 | Koijen, Moskowitz, Pedersen & Vrugt 2018, JFE; Lustig, Roussanov & Verdelhan 2011, RFS | `sign(d)` if `abs(d) > θ` else 0, d = r_base − r_quote **known at the time** | — | θ ∈ {0, 0.25, 0.5, 1.0} % × band (4) | rate differential, ATR | `bc14e5151448` |
| A5 | Menkhoff, Sarno, Schmeling & Schrimpf 2012, JFE | Rank USD, EUR, GBP, JPY, CAD by L-month performance vs USD (average ranks for ties). `w = (rank − 3)/2`, dollar-neutral. EURUSD/GBPUSD get +w, USDJPY/USDCAD get −w, EURJPY gets 0. | — | L ∈ {1, 3, 6, 12} months × band (4) | past returns (rank), ATR | `382250221dde` |
| A6 | Asness, Moskowitz & Pedersen 2013, JF; KMPV 2018 | `(s_A2 + s_A4(θ=0) + s_A5(L=12m)) / 3`, netted per pair | all sleeve parameters | band (4) | past returns, rate differential, ATR | `2f4b544db738` |

- Each hypothesis has at most 3 indicators and at most 2 tuned parameters (the audit counts every
  `StrategyParams` field), and no calendar filter. The complexity audit scans the hypothesis file and
  `academic_base.py`.
- The grids use only values the papers themselves report, plus the cost band.
- **All six run in this order, whatever the outcomes.** The harness refuses any other order. This uses 6 of the
  8-trial budget; slots 7–8 are not used.
- A later fingerprint different from this table (e.g. a harness bug fix) will show in the logs and must be
  explained in the README.

## Stage 1: the mandate's gates (development period)

`python harness.py --hypothesis An`

- 5 rolling splits as above. The grid is evaluated in-sample, one Nautilus run per combination and split.
- Plateau selection: the best 3^k-neighbourhood mean Sharpe, among combinations with more than 100 in-sample fills
  per year (all combinations if none qualify).
- One stitched Nautilus OOS run from a fresh USD 1,000,000 account.

| Gate | Requirement |
|---|---|
| (a) | OOS fills per year > 100 |
| (b) | OOS Sharpe ≥ 1.5 (trading-day returns, √252) |
| (c) | OOS max drawdown < 12 % |
| (d) | WFE = OOS Sharpe / mean IS Sharpe ≥ 0.60 |
| (e) | perturbation look-ahead test: vectorised and Nautilus, bar-boundary and mid-bar cuts, two canaries |

**Every integrity test must also pass:**
- **Parity** of Nautilus vs an independent vectorised reference: identical orders and fill prices, equity within
  one cent per fill and per financing booking. It runs on synthetic data (segmented, active, late start, corrupt
  bars, volatility spike) and on real EURUSD over the stitched plan, and it must exercise ENTRY, REBAL and
  FLIP/EXIT fills.
- **Rate-lag test with canaries:** scrambling every rate published from month M must leave all known rates and
  signals before M unchanged. It runs on synthetic data, on real 2005–2006 data, and on real 2019–2021 data around
  the policy hand-over.
- **Financing** against hand calculations: long and short EUR/USD, EUR/JPY → USD, and the engine-reset path.
- **Execution-rule table**, circuit breakers (daily halt and kill switch), friction including JPY → USD, zero
  division, flat prices, corrupt bars and the volatility spike.
- Splice checks, the complexity audit, the holdout lock, and no order desyncs.

**Mutation checks before this commit.** Each of these was caught by the suite:
- same-month rates in the carry signal (rate-lag test);
- a policy decision known on its effective day rather than the next day (rate-lag test);
- a financing booking that ignores the position side (financing test);
- an unlagged signal (leak test);
- a band that ignores or mis-measures the position (execution-rule test);
- the reference seeded from an earlier bar than the Nautilus run (late-start parity).

## Stage 2: unbiased confirmation (holdout)

`python harness.py --hypothesis An --holdout`

- **Eligibility:** only a hypothesis with a logged Stage-1 PASS under the same code fingerprint.
- **Exactly once:** a second attempt is refused.
- **Method:** the development geometry continues into 2023. Parameters are re-selected on the preceding 5.25-year
  in-sample window by the same rule, then run from a fresh account.
- **Gates:** (a)–(d) on 2023. The integrity tests rerun; the leak test carries over from Stage 1 (same code).

## Disclosures

- **Prior exposure:**
  - EURUSD 2015–2020 (cycle 1) and 2019-12 → 2022-12 (cycles 2–3) were used before, only with unrelated intraday
    rules.
  - These factors' published performance is public knowledge, and choosing them is informed by that literature.
    None of their parameters is fitted here.
- **Multiple testing:** six hypotheses mean six chances. This is reported next to any pass.
- **Stage-1 overlap:** Stage-1 OOS 2019-12 → 2022-12 overlaps data mined in cycles 2–3. A Stage-1 pass is therefore
  necessary but not independent evidence; Stage 2 decides.

## Verdict

- A hypothesis **PASSES** only if Stage 1 **and** Stage 2 pass.
- Every Stage-1 and Stage-2 result is reported.
- A pull request to `main` is opened only if some hypothesis passes both stages.

---

## Addendum 1 (after trial 1): execution-timing bug fix

Added after trial 1. The protocol above is unchanged except for the code fingerprints and the re-run rule
described here.

**What happened.**
- Trial 1 (A1, code `21ac561d0549`) completed and failed the gates: OOS Sharpe −0.03, and the kill switch fired in
  split 3.
- Its integrity test `no_order_desyncs` also failed: 8 of the 80 in-sample runs, all USDCAD around 25 December 2009.

**Root cause.** Two engine bugs, present since cycle 2.
1. **Stale fills.** Nautilus processes due orders after every single data point. With a fixed 1-ns latency, all
   orders decided at a close were processed on the first next-open quote in the stream (EURUSD's). Every other
   pair therefore filled against its previous close quote, not its next open. This is not look-ahead (the price
   predates the decision), and the leak tests correctly passed. It does violate "execute at the open of bar t", and
   it is optimistic across gaps. The single-pair parity tests could not see it.
2. **Late decisions at incomplete closes.** A close was decided only once all five pairs had a bar. When only some
   pairs traded (Christmas 2009: USDCAD alone), the decision waited for the next bar event, days later. It then went
   out against stale bookkeeping, so orders were doubled (the desyncs).

**Fix.**
- A close is decided as soon as every pair with a bar at that time has reported; the strategy learns this from the
  bar index.
- Each pair's order is sent when that pair's next quote (its next open) arrives, and fills against exactly that
  quote with zero latency.
- Position bookkeeping counts orders still waiting for their quote.
- New test `fill_timing_multi_pair`: five synthetic pairs, random missing bars, and a two-day holiday in which only
  USDCAD trades. Every fill must be at its own pair's first open after the decision, with no desyncs. On the old
  code, 155 of 175 fills were wrong. The mutation "send the order at the decision" makes it fail again (175 of 175
  wrong). All six selftests pass, and the real Christmas-2009 case now has no desyncs.

**What does not change.** Signals, grids, constants, data, costs, gates, the split and the order A1 → A6.

**What changes.**
- Every code fingerprint, since `harness.py`, `backtest.py` and `academic_base.py` all changed. The new table is
  below.
- The harness now expects, in the pre-registered order, the first hypothesis with no logged trial under its current
  code. A1 therefore runs again under the fixed code (trial 2), then A2 … A6 (trials 3–7): 7 of the 8-trial budget.
- Trial 1 stays in `trials.log`, and its report is kept as `results/cycle4/A1_trial1.json`. Both A1 results are
  reported. A1's trial-1 outcome is known before its re-run, but the re-run involves no researcher choice.

| ID | Code fingerprint after the fix |
|---|---|
| A1 | `de5192d10c9c` |
| A2 | `787ac7b42f1b` |
| A3 | `9e9382b1678d` |
| A4 | `0f24b7bcc11b` |
| A5 | `6327f8573e23` |
| A6 | `af992ad79220` |

**Earlier cycles.** Cycles 2 and 3 used the same execution code, so their non-EURUSD fills were at the previous
close quote. That bias favoured them, and all of them failed anyway; their verdicts stand.
