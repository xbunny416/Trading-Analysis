# Trading-Analysis

This repository just for my own coding trading workspace.

## FX factor research: cycle 4 (published strategies, pre-registered, 2005 → 2022, NautilusTrader)

**Status: pre-registered, runs pending.** [`PREREGISTRATION.md`](PREREGISTRATION.md) fixes everything before any
backtest on real data:
- the six hypotheses and their order;
- the data splice and costs;
- overnight financing;
- the gates and the holdout rules.

Results will be added here once the runs finish.

| ID | Strategy (paper) |
|---|---|
| A1 | Time-series momentum (Moskowitz, Ooi & Pedersen 2012) |
| A2 | 1/3/12-month trend blend (Hurst, Ooi & Pedersen 2017) |
| A3 | Multi-speed EWMA crossover (Baz et al. 2015, Man AHL) |
| A4 | Carry, from interest-rate differentials (Koijen, Moskowitz, Pedersen & Vrugt 2018) |
| A5 | Cross-sectional currency momentum (Menkhoff, Sarno, Schmeling & Schrimpf 2012) |
| A6 | Equal blend of A2, A4 and A5 (Asness, Moskowitz & Pedersen 2013) |

- **Data:**
  - OANDA 1-minute history (2005 → 2019-11) spliced with your MT5 exports (2019-12 → 2022-12). USD/JPY before the
    splice is derived as EUR/JPY ÷ EUR/USD.
  - Interest rates from OECD monthly 3-month rates (to 2020-06), then central-bank policy rates.
  - 2023 is a locked holdout.
- **Costs:** your broker's spreads (by pair and hour in the OANDA era), a 1-pip floor, 0.5 pip slippage per fill,
  and overnight financing at the rate differential less a 0.5 % p.a. mark-up.
- **Gates:** unchanged from the mandate. Every fill counts toward the 100/year trade gate.
- **Expectation, stated in advance:** the literature puts these factors at roughly 0.3–0.9 Sharpe on FX, so the 1.5
  gate is unlikely to be met.

### Reproduce

```bash
pip install -r requirements.txt                 # nautilus_trader==1.221.0, pandas, pyarrow
# MT5 exports at data/mt5/<PAIR>.csv for EURUSD GBPUSD USDJPY USDCAD EURJPY (git-ignored)
python harness.py --check-data                  # splice + rate-table checks, no backtest
python harness.py --hypothesis A1 --selftest    # synthetic integrity tests, ~1 min, never a trial
python harness.py --hypothesis A1               # Stage 1 (A1..A6 in order); fetches OANDA history on first use
python harness.py --hypothesis A1 --holdout     # Stage 2, only after a Stage-1 pass, once
```

### Files

| Path | Purpose |
|---|---|
| `PREREGISTRATION.md` | Cycle-4 protocol, data and rate hashes, code fingerprints; published before any run. |
| `hypotheses/academic_base.py` | Shared engine: sizing, no-trade-band execution, breakers, Nautilus strategy, vectorised reference, signal kernels. |
| `hypotheses/a1..a6_*.py` | The six pre-registered signals. |
| `backtest.py` | Nautilus plumbing, OANDA + MT5 splice, rate table, overnight-financing module. |
| `harness.py` | Stage 1 / Stage 2 walk-forward, gates, integrity tests, trial and holdout guards. |
| `data/rates/` | OECD short-term rates and central-bank policy decisions (committed). |
| `research/cycle1..3/` | Earlier cycles: logs, reports and write-ups. |

<details>
<summary>Earlier cycles (all failed the gates honestly)</summary>

* **Cycle 1:** custom engine, OANDA EURUSD 2015 → 2020, 8 adaptive trials; best OOS Sharpe 0.32.
* **Cycle 2:** NautilusTrader, your 5-pair MT5 data 2020-11 → 2022-12, 7 adaptive trials; the best (0.66) proved
  fragile.
* **Cycle 3:** the 7 cycle-2 hypotheses pre-registered and re-tested on 2019-12 → 2022-12; all failed Stage 1 (best
  0.63). The 2023 holdout was never used. See [`research/cycle3/README.md`](research/cycle3/README.md).
</details>
