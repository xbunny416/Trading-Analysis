# Cycle 3 pre-registration

Committed and pushed **before any backtest on the extended data**. Nothing below changes after results are seen.
The git history is the audit trail: this file's commit precedes every entry in `trials.log` and `holdout.log`.

## Why pre-register

The extended MT5 exports cover 2019-12-02 → 2023-12-29 (4.1 years, EURUSD GBPUSD USDJPY USDCAD EURJPY). About half
of that span, 2020-11-30 → 2022-12-30, is exactly the data cycle 2 mined with 7 adaptive trials. EURUSD
2019-12 → 2020-05 was seen in cycle 1 (OANDA source). An adaptive loop over this data would be data mining.
This protocol removes every researcher degree of freedom that could react to results, and saves one year of
never-seen data for the final verdict.

## Data

| Pair | SHA-256 of `data/mt5/<PAIR>.csv` (first 16 hex) |
|---|---|
| EURUSD | `bc4b9cfd06f8c3dc` |
| GBPUSD | `01e555bc13d01094` |
| USDJPY | `1af85499df842f9b` |
| USDCAD | `6f93d394bc220851` |
| EURJPY | `99467048dc35ee9f` |

* **Development period:** every bar before 2023-01-01 00:00 UTC (2019-12-02 → 2022-12-30, 3.08 years; meets the
  original ≥ 3-year rule).
* **Holdout:** 2023-01-01 → 2023-12-29 UTC. **Locked.** Stage-1 runs load data only up to the cutoff, and a test
  asserts it.
* **Costs and engine:** unchanged from cycle 2.
  * NautilusTrader 1.221.
  * Broker per-bar spread (floor 1.0 pip) + 0.5 pip slippage per fill.
  * Orders fill at the next hour's open quote.
  * USD margin account; risk limits: daily loss 2.5 % → 24 h halt, drawdown 10 % → permanent halt.

## Hypotheses: fixed list, fixed order, no additions, no edits

Each module is the cycle-2 `strategy.py` of the listed commit, copied byte for byte. H1–H4 additionally end with an
appended 19-line compatibility shim, so their per-pair signal functions accept the `{pair: frame}` universe. No
existing line is removed or changed (`git diff <commit>:strategy.py hypotheses/<file>` shows only additions).

The code fingerprint is `sha256(harness.py + backtest.py + hypothesis file)[:12]` at this commit.

| ID | File | From commit | Idea | Code fingerprint |
|---|---|---|---|---|
| H1 | `hypotheses/h1_shock_chandelier.py` | `0350c08` | range-shock entry + chandelier stop | `c85d762d51eb` |
| H2 | `hypotheses/h2_high_conviction_shock.py` | `7d66a4b` | H1, large shocks only | `f185d1c47bbd` |
| H3 | `hypotheses/h3_donchian_breakout.py` | `3951388` | multi-day Donchian breakout, stop-and-reverse | `4f592bd2773c` |
| H4 | `hypotheses/h4_shock_wide_stops.py` | `c7f2519` | H1 with wider stops | `459f7c21ed0e` |
| H5 | `hypotheses/h5_currency_strength.py` | `e90f080` | currency-strength trend, zero-cross | `97dfde7f975d` |
| H6 | `hypotheses/h6_currency_strength_band.py` | `cded45d` | currency strength with a hysteresis band | `819eaa225508` |
| H7 | `hypotheses/h7_parsimonious_shock.py` | `992e3e7` | shock momentum, stop tied to the threshold | `fe640a3dcfbc` |

* All seven run **in this order, whatever the outcomes**. The harness refuses any other order.
* This uses 7 of the 8-trial budget; slot 8 is not used.
* A later fingerprint different from the table (e.g. a harness bug fix) will be visible in the logs and must be
  explained in the README.

## Stage 1: the mandate's gates (development period)

`python harness.py --hypothesis Hn`, unchanged cycle-2 method:

* 5 rolling splits, 70 % in-sample / 30 % out-of-sample, over the development period.
* A 4×4 grid per hypothesis. Plateau selection (best 3×3-neighbourhood mean Sharpe) among combinations with more
  than 100 in-sample trades/year.
* One stitched Nautilus OOS run.
* Gates, all required:

| Gate | Requirement |
|---|---|
| (a) | OOS trades/year > 100 |
| (b) | OOS Sharpe ≥ 1.5 (daily returns, √252) |
| (c) | OOS max drawdown < 12 % |
| (d) | WFE = OOS Sharpe / mean IS Sharpe ≥ 0.60 |
| (e) | perturbation look-ahead test (vectorised + Nautilus, mid-bar cuts, canaries) |

* Plus every integrity test: parity vs the vectorised reference, edge cases, circuit breakers, friction,
  complexity audit, holdout lock.

**Caveat, stated before the results:** Stage-1 out-of-sample segments (≈ 2020-11 → 2022-12) overlap cycle 2's mined
period. A Stage-1 pass is therefore necessary but not independent evidence.

## Stage 2: unbiased confirmation (holdout)

`python harness.py --hypothesis Hn --holdout`.

* **Eligibility:** only for a hypothesis with a logged Stage-1 PASS under the same code fingerprint.
* **Exactly once:** a second attempt is refused.
* **Method:** the development split geometry (window W, step 0.3 W, in-sample 0.7 W) continues into 2023.
  * Parameters are re-selected on each preceding in-sample window by the same plateau rule.
  * The stitched run starts from a fresh account.
* **Gates:** (a)–(d) above, evaluated on 2023. The integrity tests rerun; the leak test carries over from
  Stage 1 (same code).

## Verdict

* A hypothesis **PASSES** only if Stage 1 **and** Stage 2 both pass.
* All seven Stage-1 results and every Stage-2 result are reported.
* Seven hypotheses mean seven chances; this is disclosed next to any pass.
* A pull request to `main` is opened only if some hypothesis passes both stages.
