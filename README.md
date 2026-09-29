# Trading-Analysis

This repository just for my own coding trading workspace.

## FX strategy search: cycle 5 (adaptive, fully logged, until a strategy passes)

**Status: no pass after 47 trials.** The best is C5-039 (value + dollar trend, trend normalised over 32/126 days), with OOS Sharpe 0.68, drawdown 2.0 % and WFE 1.78. It passes every gate except (b), Sharpe ≥ 1.5. The leaderboard below is updated after every batch of trials.

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
| C5-007 | relative value: EURUSD-GBPUSD spread mean reversion | -0.45 | n/a | 6.4% | 133 | 0.0001 (15) | b, d |
| C5-008 | large-shock follow-through, bar-count hold | -0.32 | -1.57 | 10.1% | 141 | 0.001 (16) | b, d |
| C5-009 | large-shock reversal, bar-count hold | -0.87 | n/a | 10.0% | 92 | 0.0 (17) | a, b, d |
| C5-010 | trend: volatility-managed dollar consensus | 0.36 | 1.36 | 5.2% | 344 | 0.1183 (18) | b |
| C5-011 | trend: dollar consensus, de-risk-only volatility scaling | 0.28 | 1.07 | 6.0% | 405 | 0.0668 (19) | b |
| C5-012 | trend: vol-managed blend of dollar consensus and per-pair A3 | 0.28 | 0.91 | 6.8% | 622 | 0.0607 (20) | b |
| C5-013 | trend: vol-managed per-pair A3 | 0.33 | 1.03 | 7.9% | 251 | 0.0802 (21) | b |
| C5-014 | trend: vol-managed dollar consensus of A2 TSMOM | -0.15 | n/a | 10.0% | 1164 | 0.0013 (22) | b, d |
| C5-015 | trend: fast vol-managed dollar consensus (A3 spans x 1/8..1/2) | 0.26 | 1.73 | 6.8% | 894 | 0.0517 (23) | b |
| C5-016 | trend: Donchian / Turtle breakout, per pair | -0.10 | n/a | 10.0% | 168 | 0.0023 (24) | b, d |
| C5-017 | carry: vol-managed dollar carry (LRV 2014) | -0.33 | n/a | 10.0% | 5 | 0.0001 (25) | a, b, d |
| C5-018 | rate momentum: change in the known rate differential, vol-managed | -0.28 | n/a | 10.0% | 9 | 0.0003 (26) | a, b, d |
| C5-019 | trend: slow vol-managed dollar consensus (A3 spans x 1..2) | 0.41 | 1.78 | 5.1% | 390 | 0.1288 (27) | b |
| C5-020 | value: 3-year cross-sectional reversal (nominal proxy), vol-managed | 0.03 | 3.05 | 9.3% | 414 | 0.0077 (28) | b |
| C5-021 | ensemble: 50/50 dollar trend + currency value, vol-managed | 0.25 | 1.33 | 4.0% | 568 | 0.0484 (29) | b |
| C5-022 | value: 3-year value with smooth z-score weights, vol-managed | 0.28 | 0.95 | 4.3% | 551 | 0.0569 (30) | b |
| C5-023 | ensemble: 50/50 dollar trend + smooth value, vol-managed | 0.49 | 1.55 | 3.1% | 1013 | 0.1779 (31) | b |
| C5-024 | XS momentum with smooth z-score weights, vol-managed | -0.17 | n/a | 6.9% | 2810 | 0.0008 (32) | b, d |
| C5-025 | ensemble: dollar trend + smooth value + smooth XS momentum | 0.29 | 1.24 | 2.6% | 2419 | 0.0531 (33) | b |
| C5-026 | carry: cross-sectional HML_FX, smooth z-score weights, vol-managed | -0.29 | n/a | 8.6% | 123 | 0.0002 (34) | b, d |
| C5-027 | ensemble: value + trend with trend speed and value horizon in the grid | 0.56 | 1.71 | 2.2% | 1101 | 0.2181 (35) | b |
| C5-028 | value: dollar value (USD long-horizon reversal), vol-managed | -0.14 | -0.29 | 10.0% | 44 | 0.0009 (36) | a, b, d |
| C5-029 | ensemble: C5-027 with the grid extended past its edges | 0.50 | 1.62 | 3.5% | 1304 | 0.1598 (37) | b |
| C5-030 | mining: value + trend with the blend weight in the grid | 0.45 | 1.51 | 3.1% | 1278 | 0.1142 (38) | b |
| C5-031 | mining: value + trend, value averaged over 3 and 4 years | 0.59 | 2.04 | 2.3% | 946 | 0.216 (39) | b |
| C5-032 | mining: value + trend, both legs horizon-averaged | 0.53 | 1.93 | 2.0% | 1177 | 0.1459 (40) | b |
| C5-033 | mining: value + trend, value averaged over 2, 3 and 4 years | 0.53 | 2.34 | 2.4% | 709 | 0.1391 (41) | b |
| C5-034 | mining: C5-031 with Moreira-Muir variance scaling | 0.60 | 1.79 | 2.3% | 789 | 0.1832 (42) | b |
| C5-035 | mining: C5-031 without volatility scaling (control) | 0.50 | 1.89 | 2.7% | 2049 | 0.1032 (43) | b |
| C5-036 | mining: value + trend with a Fisher-adjusted (real) value leg | 0.58 | 1.51 | 2.1% | 676 | 0.1558 (44) | b |
| C5-037 | mining (hindsight): C5-034 with JPY removed from the value leg | 0.47 | 1.67 | 2.8% | 916 | 0.0796 (45) | b |
| C5-038 | mining: C5-034 with a capped linear trend response | 0.53 | 3.23 | 2.8% | 890 | 0.1128 (46) | b |
| C5-039 | mining: C5-034 with A3 normalisation windows halved | 0.68 | 1.78 | 2.0% | 792 | 0.2248 (47) | b |

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
- **Batch 3 (C5-007…009), registered:**
  - **C5-007:** EUR/GBP relative value, fading z-score extremes of ln(EURUSD) − ln(GBPUSD).
  - **C5-008:** large-shock follow-through: after an hourly move beyond k × ATR, hold its direction for H bars.
    This is cycles 2–3's best intraday idea, now on the full history and with correct fills.
  - **C5-009:** the mirror image of C5-008, fading the shock, logged as its own trial.
  - **Engine:** gains an `on_close` hook for state built from several pairs.
  - **Tests:** synthetic test data now has rare fat-tail jumps, so shock strategies trade in the tests. The
    signal-parity recorder now reads signals exactly where the decision does. All selftests pass.
  - **Result:** all fail, and all three are negative before costs.
    - **EUR/GBP relative value** has no reversion edge (−$5k before costs, −$35k of spread).
    - **Hourly shocks** neither continue (C5-008, −$26k before costs) nor reverse (C5-009, −$76k). At this frequency
      the pairs look efficient.
  - **Lesson:** after 17 trials on this data, trend is the only source with a real edge before costs (short-term
    reversal had a small one that spreads erased). Batch 4 tries volatility management of the best trend book
    (Moreira & Muir 2017).
- **Batch 4 (C5-010…011), registered:** the dollar-trend consensus (C5-006), sized by recent volatility.
  - **C5-010:** scale by ATR₁₄₄₀ / ATR_F, clipped to 0.5–2.
  - **C5-011:** the same scale, but capped at 1, so it only de-risks.
  - The engine now exposes each bar's high and low; all eleven cycle-5 selftests pass.
  - **Result:** both fail gate (b), but volatility management helps, as Moreira & Muir found.
    - **C5-010** reaches **0.36**, the best of cycle 5 (WFE 1.36, drawdown 5.2 %, +$114k before costs vs C5-006's
      +$84k).
    - **C5-011** (de-risk only) reaches 0.28.
    - That is still a quarter of the 1.5 gate. The deflated Sharpe is 0.12, against a luck benchmark of 0.72
      after 19 trials.
- **Batch 5 (C5-012…014), registered:** the same volatility scaling (C5-010's ATR₁₄₄₀ / ATR_F, clipped to 0.5–2),
  applied to the rest of the trend family.
  - **C5-012:** an equal blend of the dollar consensus and per-pair A3 trend, then volatility-managed.
  - **C5-013:** per-pair A3 trend, volatility-managed.
  - **C5-014:** a dollar consensus of the 1/3/12-month momentum blend (A2), volatility-managed.
  - All three reuse C5-010's grid (fast ATR 120–960 bars × band 0.1–0.5); all fourteen cycle-5 selftests pass.
  - **Result:** all fail gate (b).
    - **C5-013** (volatility-managed per-pair A3) reaches **0.33** (WFE 1.03, drawdown 7.9 %). It has the largest edge
      before costs so far, about +$155k, but pays $27k of financing, and USDCAD and EURJPY lose.
    - **C5-012** (the blend) lands between its parents at 0.28.
    - **C5-014** (dollar consensus of A2's sign-based momentum) is negative (−0.15). Its sign signals flip the whole
      dollar book often: 1,164 fills a year cost $62k of spread, and the kill switch fired.
  - **Lesson:** volatility management adds about 0.1 to either form of trend (dollar 0.25 → 0.36, per pair 0.25 →
    0.33). Every trend book earns in the same splits (2014–16 and 2020–22, OOS Sharpe about +1) and loses in
    2018–20 (about −0.6). On these pairs the trend family looks capped near 0.35.
- **Batch 6 (C5-015…017), registered:** three new directions, written while batch 5 ran.
  - **C5-015:** a faster dollar trend. Every A3 EWMA span is multiplied by 1/8, 1/4 or 1/2, and the walk-forward
    chooses the factor. Hourly mean reversion (C5-001) lost because 1–7-day deviations kept running, which is trend
    at horizons shorter than A3's fastest speed.
  - **C5-016:** Donchian / Turtle breakout per pair: enter on a close beyond the N-bar high or low (N = 5–40 days),
    exit on the opposite N/2-bar extreme. It is flat in ranges, unlike A3.
  - **C5-017:** volatility-managed dollar carry (Lustig, Roussanov & Verdelhan 2014): long the foreign currencies
    against the USD when their average known rate is above the US rate, short otherwise. A positive result would be
    the second source an ensemble needs.
  - **Tests:** the multi-pair signal-parity test caught one bug in C5-017 before registration. EURJPY's target was
    NaN rather than 0 during the volatility warm-up; it is fixed and all selftests pass.
  - **Result:** all fail.
    - **The fast dollar trend (C5-015)** reaches 0.26. The walk-forward chose the slowest speed on offer (A3 × 1/2)
      in all five splits, and it trails C5-010 (A3 speed, 0.36). Faster trend is worse, not better.
    - **The Donchian breakout (C5-016)** is negative (−0.10) and negative in-sample in every split. It makes only
      about +$30k before costs; $46k of spread and $27k of financing sink it, and the kill switch fired.
    - **The dollar carry (C5-017)** loses heavily (−0.33, about −$77k before costs). It was long the foreign
      currencies through the 2014–15 dollar rally, because their rates were higher, and the kill switch fired in
      split 2. It also fails gate (a) with 5 fills a year.
  - **Lesson:** after 25 trials, slower trend is the only thing that earns. Batch 7 tests a slower trend, and a
    fundamental signal that is not a price trend: the change in interest-rate differentials.
- **Batch 7 (C5-018…019), registered:**
  - **C5-018:** interest-rate momentum (Dahlquist & Hasseltoft 2020; Brooks 2017). Long a pair when its known rate
    differential rose by more than θ points over the last 3–12 months, short when it fell, volatility-managed like
    C5-010. It trades the change in rates rather than their level (carry), and uses no prices.
  - **C5-019:** a slower dollar trend, with A3's spans × 1, 1.5 or 2. C5-015's walk-forward chose the slowest speed
    on offer every time.
  - **Harness:** the circuit-breaker test's scenario used constant rates, so a rate-change signal never held a
    position there. Its rate table now has a rising EUR rate. Carry rules still go long, and a mutation that
    disables the daily halt is still caught. All nineteen cycle-5 selftests and `--check-data` pass.
  - **Result:** both fail.
    - **The slow dollar trend (C5-019)** reaches **0.41**, the best of cycle 5 (WFE 1.78, drawdown 5.1 %, about
      +$133k before costs). The walk-forward chose speed 1 three times, 1.5 once and 2 once, so the gain over C5-010
      (0.36) is mostly the grid, not a clear preference for slower trend. The deflated Sharpe is 0.13, against a
      luck benchmark of 0.74 after 27 trials.
    - **Rate momentum (C5-018)** loses (−0.28, about −$67k before costs). It trades 9 times a year (gate a), and the
      kill switch fired in split 3.
  - **Lesson:** 27 trials, best 0.41. Price trend is the only source with an edge on these pairs, and it is worth
    about 0.4–0.5 before costs. Carry, rate momentum, breakouts, shocks, mean reversion and relative value are all
    negative before costs.
- **Batch 8 (C5-020…021), registered together:** value and momentum (Asness, Moskowitz & Pedersen 2013). Their
  central result is that value and momentum are negatively correlated in every asset class, so a combination beats
  either one alone.
  - **C5-020:** currency value, volatility-managed. Long the currencies that fell most against the others over 3
    years, short those that rose most (dollar-neutral rank weights).
    - This is a price-only proxy. Proper value needs CPI levels, and this environment's network policy blocks the
      CPI sources (FRED, OECD).
    - It uses 3 years rather than AMP's 5 because the data starts in 2005.
  - **C5-021:** 50/50 of the dollar trend (C5-010's construction) and C5-020's value, volatility-managed. It is
    registered before either has run, so the blend is not chosen after seeing value's result.
  - **Harness:** in the one-pair circuit-breaker scenario, a blend of opposite signals first holds a position too
    small (0.04× leverage) to gap cleanly. The test now gaps the first held position with at least 0.3× leverage.
    The daily-halt mutation is still caught, on C5-021 and C5-010. All twenty-one cycle-5 selftests pass.
  - **Result:** both fail.
    - **Value alone (C5-020)** reaches 0.03. It does earn before costs, about +$88k. But the rank weights jump
      whenever two currencies swap places: 1,344 entries and exits cost $59k of spread, plus $26k of financing.
    - **Value + trend (C5-021)** reaches 0.25, below trend alone, because the value leg's churn costs $33k.
    - Value's daily returns are nearly uncorrelated with trend (−0.06 to −0.10), not the −0.5 AMP report.
  - **Lesson:** value is the second source with an edge before costs, and it is independent of trend. The rank
    weights waste it. Batch 9 keeps the signal and replaces the ranks with smooth weights.
- **Batch 9 (C5-022…023), registered together:**
  - **C5-022:** the same 3-year value signal, with dollar-neutral cross-sectional z-score weights
    (−(x − mean) / 2σ) instead of ranks. A small move in prices now means a small trade, which the band absorbs.
  - **C5-023:** 50/50 of the dollar trend and C5-022's smooth value, volatility-managed.
  - Both selftests pass (signal parity max difference 5e-14; the harness is unchanged since batch 8's full run).
  - **Result:** both fail gate (b), but the combination is the best result of cycle 5.
    - **Smooth value (C5-022)** reaches 0.28, up from 0.03. Spread falls from $59k to $8k, and it is positive in four
      of five OOS splits, including 2016–20 when trend lost.
    - **Value + trend (C5-023)** reaches **0.49**, with drawdown 3.1 %, WFE 1.55 and 1,013 fills a year. Gates (a),
      (c), (d) and (e) pass; only (b) fails. The deflated Sharpe is 0.18, against a luck benchmark of 0.77 after 31
      trials.
  - **Lesson:** this is what two nearly uncorrelated sources should give. With trend at about 0.41 and value at
    0.28, the best mix is about √(0.41² + 0.28²) ≈ 0.50. To reach 1.5 from sources of that size would take about
    nine independent ones. Across 31 trials only two sources on these five pairs have an edge before costs.
  - **Checkpoint:** you chose to keep searching the five pairs with the 1.5 gate, knowing that each extra trial
    raises the deflated-Sharpe luck benchmark and that a pass would more likely be luck.
- **Batch 10 (C5-024…025), registered together:** a third leg for the blend.
  - **C5-024:** cross-sectional momentum (A5, Menkhoff et al. 2012) with C5-022's smooth z-score weights over 1–12
    months, volatility-managed. A5's rank weights made about +$23k before costs and paid $55k of spread. The
    momentum is dollar-neutral, so it should overlap less with the dollar trend than per-pair trend does.
  - **C5-025:** an equal three-way blend of the dollar trend, smooth value and smooth momentum, volatility-managed.
  - Both selftests pass; the harness is unchanged since batch 8's full run.
  - **Result:** both fail.
    - **Smooth momentum (C5-024)** is negative (−0.17). Even with smooth weights, the 1–6-month z-scores keep moving:
      2,810 fills a year cost $40k of spread, against about +$25k before costs.
    - **The three-way blend (C5-025)** reaches 0.29, below value + trend (0.49), because the momentum leg adds cost
      and no edge.
  - **Lesson:** cross-sectional momentum has no usable edge on five currencies, so C5-023 remains the best.
- **Batch 11 (C5-026), registered:** cross-sectional carry, the canonical currency carry factor (HML_FX,
  Lustig, Roussanov & Verdelhan 2011).
  - Long the highest-yielding currencies, short the lowest, with dollar-neutral z-score weights on the known rates,
    volatility-managed. It differs from the per-pair carry (A4) and the dollar carry (C5-017), which both lost: its
    return comes from the gap between high and low yielders, not from the USD's direction.
  - It runs alone. A blend is registered only if a leg earns, so failed legs do not inflate the trial count.
  - The selftest passes, including the rate look-ahead canaries.
  - **Result:** fail (−0.29). Its in-sample Sharpe is negative in every split. It earns $12k of financing but loses
    about $73k on price: high yielders fell against low yielders over 2006–22.
  - **Lesson:** every canonical currency factor has now been tested on these five pairs.
    - Time-series trend and value have an edge before costs.
    - Carry (per pair, dollar and cross-sectional), cross-sectional momentum and rate momentum do not.
    - Neither do hourly mean reversion, relative value, shocks and breakouts.
    - From here the search can only refine C5-023, which is closer to data mining. The deflated Sharpe accounts for
      that.
- **Batch 12 (C5-027), registered:** C5-023 refined. Its two fixed choices go into the walk-forward grid in a
  single trial: the trend speed (A3 spans × 1, 1.5 or 2; C5-019 sometimes preferred slower) and the value horizon
  (2, 3 or 4 years; AMP use 5, and 3 was set by the data length).
  - At speed 1 and a 3-year horizon it reproduces C5-023's signals exactly.
  - The selftest passes.
  - **Result:** fail gate (b), but the best of cycle 5 at **0.56** (drawdown 2.3 %, WFE 1.71, 1,101 fills a year).
    - The walk-forward chose the 4-year value horizon in four of five splits, and the fastest volatility scale (120
      bars) in all five. Both are at the edge of the grid.
    - The deflated Sharpe is 0.22, against a luck benchmark of 0.78 after 35 trials.
- **Batch 13 (C5-028…029), registered:**
  - **C5-028:** dollar value, a new leg. Sell the dollar after it has risen against the other four currencies over
    3–4 years, buy it after it has fallen. It is value's counterpart to the dollar-consensus trend and should run
    against it.
  - **C5-029:** C5-027 unchanged, with its grid extended past the two edges it chose: value horizon 4 or 5 years
    (AMP's 5), fast ATR of 60, 120 or 240 bars, trend speed ×1 or ×2.
    - A 5-year horizon starts trading only in 2010, which handicaps it in split 1's in-sample window.
  - Both selftests pass.
  - **Result:** both fail.
    - **Dollar value (C5-028)** looks good in-sample (0.49) but is negative out of sample (−0.14). The kill switch
      fired in split 2, during the 2014–15 dollar rally.
    - **The extended grid (C5-029)** reaches 0.50, below C5-027's 0.56. The 5-year horizon was never chosen, and
      moving the fast ATR either way off 120 bars made it worse.
  - **Lesson:** refinements of the value + trend book now move it by about ±0.05, which is within noise. There are
    no principled hypotheses left for these five pairs.
  - **Checkpoint 2:** asked again, you chose to keep fishing on the five pairs. From here the trials are a
    **data-mining phase** and are labelled as such. They are still logged, registered before they run and deflated.
    A Stage-1 pass would still have to pass the untouched 2023 holdout.
- **Batch 14 (C5-030…031), data-mining phase:**
  - **C5-030:** value + trend with the trend weight (1/3, 1/2, 2/3), value horizon (3, 4 years) and fast ATR (120,
    240) in the grid.
  - **C5-031:** value + trend with the value leg averaged over the 3- and 4-year horizons, instead of choosing one.
  - Both selftests pass.
  - **Result:** both fail gate (b).
    - **C5-031** is the new best at **0.59** (drawdown 2.3 %, WFE 2.04): averaging the value horizons beats choosing
      one.
    - **C5-030** reaches 0.45. The walk-forward flipped between trend weights 2/3 and 1/3, which costs more than it
      gains.
    - The deflated Sharpe of C5-031 is 0.22, against a luck benchmark of 0.82 after 39 trials.
- **Batch 15 (C5-032…033), data-mining phase:** more horizon averaging, the one change that helped.
  - **C5-032:** C5-031 with the trend leg also averaged, over A3 speeds ×1 and ×2.
  - **C5-033:** the value leg averaged over 2, 3 and 4 years, with the band grid moved up to 0.3–0.5.
  - Both selftests pass.
  - **Result:** both fail at 0.53, below C5-031's 0.59. Averaging the trend leg, or adding the 2-year value
    horizon, does not help. Every value + trend variant since C5-023 lands between 0.45 and 0.59, which is noise
    around one underlying book.
- **Batch 16 (C5-034…035), data-mining phase:** the volatility scale, the one ingredient that helped trend, varied
  on C5-031.
  - **C5-034:** Moreira & Muir's variance form, (ATR₁₄₄₀ / ATR_F)² clipped to 0.25–4.
  - **C5-035:** no scaling, the control. Given C5-031's own scale, the shared code reproduces C5-031 exactly.
  - Both selftests pass.
  - **Result:** both fail gate (b).
    - **Variance scaling (C5-034)** reaches 0.597, level with C5-031 (0.593) and the best of cycle 5 by a hair
      (drawdown 2.3 %, WFE 1.79).
    - **No scaling (C5-035)** reaches 0.50, so volatility management adds about 0.1 to the blend, as it did to trend.
- **Batch 17 (C5-036), data-mining phase:** a better value signal.
  - AMP's value uses the real exchange rate, which needs CPI data that the network policy blocks. The Fisher relation
    gives a proxy: inflation differentials track interest differentials. So each currency's 3- and 4-year spot
    change is adjusted by its average known rate differential against the USD. That is the same as long-horizon
    reversal of its excess return.
  - Blended 50/50 with the dollar trend and variance-scaled, as in C5-034.
  - **Dropped before registration:** a value-only companion. In the one-pair parity test, its two horizons can
    cancel exactly (two currencies give weights of ±0.5), so it never traded there. Nothing was run on real data.
  - The selftest passes, including the rate look-ahead canaries.
  - **Result:** fail at 0.58 (drawdown 2.1 %, WFE 1.51), in the same 0.53–0.60 range as the nominal-value blends.
    The real-rate adjustment does not move it.
- **Batch 18 (C5-037), data-mining phase, hindsight:** JPY is removed from the value leg of C5-034; the trend leg
  keeps USDJPY.
  - USDJPY lost in every value + trend blend (−8 to −15 pips a trade), although trend alone made money on it. The
    value leg kept buying the yen while it fell in 2012–15 and 2021–22.
  - This is pair selection after seeing the results, the kind of choice most likely to fail out of sample. It is
    labelled as such in the module and the log. The deflated Sharpe counts it, and only the 2023 holdout could
    confirm it.
  - The selftest passes.
  - **Result:** fail at 0.47, worse than C5-034's 0.60.
    - Without the yen, USDJPY now earns (+87 pips a trade, trend only), but the other three pairs fall from 12–17 to
      3–4 pips a trade.
    - In the cross-section, the yen's value position was hedging the others. The per-pair losses that motivated the
      change were misleading.
- **Batch 19 (C5-038…039), data-mining phase:** the trend kernel inside C5-034.
  - **C5-038:** a capped linear response, clip(z/2, −1, 1), instead of A3's response, which fades the strongest
    trends.
  - **C5-039:** A3's normalisation windows halved, to a 32-day price σ and a 126-day signal σ.
  - Given A3's own kernel, the new trend code reproduces C5-034 to 1e-16. Both selftests pass.
  - **Result:** both fail gate (b).
    - **Halved windows (C5-039)** is the new best at **0.68** (drawdown 2.0 %, WFE 1.78). Its OOS Sharpe is
      positive in four of five splits.
    - **The linear response (C5-038)** reaches 0.53.
    - The deflated Sharpe of C5-039 is 0.22, against a luck benchmark of 0.90 after 47 trials.
- **Batch 20 (C5-040), data-mining phase:** the normalisation-window scale goes into the walk-forward grid (A3's
  windows × 1/4, 1/3 or 1/2), rather than me fixing another scale after seeing C5-039.
  - At scale 1/2 it reproduces C5-039 to 1e-16.
  - The selftest passes.

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
