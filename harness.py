#!/usr/bin/env python3
"""
harness.py - cycle 5 walk-forward evaluation and integrity harness, running every backtest in NautilusTrader.

Cycle 5 is an adaptive search (README, cycle 5): every look at real data is a logged trial with an immutable ID
(C5-001, ...), committed before its run; each report carries a deflated Sharpe ratio over all trials on this data;
a Stage-1 pass unlocks the 2023 holdout once, and only a holdout pass counts. Cycle 4 (PREREGISTRATION.md, A1..A6,
A3F) is archived in research/cycle4; its modules remain as a library.

Each stage exits with status 0 only if every gate passes on the 5-pair portfolio:

    a) out-of-sample trades per year      > 100   (every fill counts: entries, exits, flips, rebalances)
    b) out-of-sample Sharpe ratio         >= 1.5
    c) out-of-sample max drawdown         < 12 %
    d) walk-forward efficiency (OOS/IS)   > 0.50
    e) perturbation look-ahead test       passes

and every integrity check passes (data validation, complexity audit, Nautilus-vs-reference parity, edge-case
tests). Any failure prints the exact reason and exits with status 1.

Usage
-----
    python harness.py --hypothesis C5-001 --selftest   synthetic-data tests only (never a trial)
    python harness.py --hypothesis C5-001              Stage 1: WFA on the development period (logs a trial)
    python harness.py --hypothesis C5-001 --holdout    Stage 2: once, only after a logged Stage-1 pass
    python harness.py --hypothesis C5-001 --data-dir D Stage 1 on other MT5 exports, no splice (never logged)
    python harness.py --check-data                 splice and rate-table checks on the real data (no backtest)

Data: OANDA 1-minute mids 2005-01 -> 2019-11 spliced with the user's MT5 M15 exports 2019-12 -> 2023-12 for
EURUSD GBPUSD USDJPY USDCAD EURJPY (USD/JPY before the splice = EUR/JPY / EUR/USD), plus interest rates for
overnight financing and carry (data/rates). Development = everything before 2023-01-01 UTC; the first 18 months
are warm-up only. Holdout = 2023. Stage-1 runs never load holdout bars or holdout rates.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

import backtest as B
import hypotheses as H

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "cycle5"
ARCHIVE_RESULTS = (ROOT / "research" / "cycle4" / "results",)   # earlier trials on the same data (for the DSR)
TRIALS_LOG = ROOT / "trials.log"
HOLDOUT_LOG = ROOT / "holdout.log"
CODE_FILES = ("harness.py", "backtest.py", "hypotheses/academic_base.py")   # + the selected hypothesis module
DEV_END = pd.Timestamp("2023-01-01", tz="UTC")   # development data strictly before, holdout at or after
BURN_IN = pd.DateOffset(months=18)               # warm-up only: the split geometry starts after it

S = None        # the selected hypothesis module (bound by use_hypothesis)
HYP_ID = None


def use_hypothesis(hid: str) -> None:
    global S, HYP_ID
    S, HYP_ID = H.load(hid), hid

# ----------------------------------------------------------------------------- gates
MIN_TRADES_PER_YEAR = 100.0
MIN_OOS_SHARPE = 1.5
MAX_OOS_DRAWDOWN = 0.12
MIN_WFE = 0.50                # gate (d) is strict: WFE > 0.50
MAX_TRIALS = None             # cycle 5: no cap (your decision); every trial is logged and counted in the DSR

# ----------------------------------------------------------------------------- WFA
N_SPLITS = 5
IS_FRACTION = 0.70
MIN_DATA_YEARS = 3.0          # the original mandate
TRADING_DAYS_PER_YEAR = 252
WORKERS = 4

# ----------------------------------------------------------------------------- limits
MAX_TUNABLE_PARAMS = 4
MAX_INDICATORS = 3
CALENDAR_PATTERN = re.compile(
    r"\.(hour|minute|dayofweek|day_of_week|weekday|isoweekday|day_name|month|quarter|"
    r"dayofyear|day_of_year|is_month_end|is_month_start|week|weekofyear)\b"
)
HOUR = B.ONE_HOUR
YEAR_NS = 365.25 * 86_400 * 1e9

# synthetic rate tables for the unit tests (never used for performance)
SYN_RATES = B.RateTable.synthetic(start="2020-01-01", months=72, seed=7, switch="2022-01-01")
_CONST = pd.date_range("2019-01-01", periods=96, freq="MS", tz="UTC")
FIXED_RATES = B.RateTable({c: pd.Series(v, index=_CONST) for c, v in
                           {"EUR": 3.0, "USD": 1.0, "GBP": 2.0, "JPY": -0.1, "CAD": 1.5}.items()})


# =============================================================================
# Data
# =============================================================================
def data_provenance(pairs=B.PAIRS, data_dir: Path = B.DATA_DIR, m15: dict | None = None) -> dict:
    """SHA-256 prefixes: the MT5 exports, the rate files, and (given the loaded frames) the frames themselves."""
    files = {p: Path(data_dir) / f"{p}.csv" for p in pairs}
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"missing MT5 exports: {missing} (expected data/mt5/<PAIR>.csv)")
    out = {f"mt5_{p}": hashlib.sha256(f.read_bytes()).hexdigest()[:16] for p, f in files.items()}
    for f in sorted(B.RATES_DIR.glob("*.csv")):
        out[f"rates_{f.stem}"] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    out["oanda_commit"] = B.OANDA_COMMIT[:12]
    for p, f in (m15 or {}).items():
        out[f"frame_{p}"] = hashlib.sha256(pd.util.hash_pandas_object(f, index=True).to_numpy().tobytes()).hexdigest()[:16]
    return out


def validate_data(m15: dict[str, pd.DataFrame]) -> tuple[list[str], dict]:
    errors, info = [], {}
    for p, f in m15.items():
        idx = f.index
        years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 86400)
        info[p] = {"first": str(idx[0]), "last": str(idx[-1]), "m15_bars": int(len(f)), "years": round(years, 3),
                   "spread_pips_median": round(float(f["spread_pips"].median()), 2),
                   "spread_pips_p99": round(float(f["spread_pips"].quantile(0.99)), 2)}
        if years < MIN_DATA_YEARS:
            errors.append(f"{p}: data span {years:.2f}y < required {MIN_DATA_YEARS}y")
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            errors.append(f"{p}: index not strictly increasing")
        px = f[["open", "high", "low", "close"]]
        if not np.isfinite(px.to_numpy()).all() or (px <= 0).any().any():
            errors.append(f"{p}: non-finite or non-positive prices")
        if (f["high"] < px[["open", "close"]].max(axis=1)).any() or (f["low"] > px[["open", "close"]].min(axis=1)).any():
            errors.append(f"{p}: OHLC inconsistency")
    return errors, info


# =============================================================================
# Metrics
# =============================================================================
def perf_metrics(equity: pd.Series, start_nav: float, fills: pd.DataFrame, trades: pd.DataFrame,
                 t0: int, t1: int) -> dict:
    """Sharpe from trading-day returns; `trades_per_year` counts every fill (gate a), round trips are reported."""
    years = (t1 - t0) / YEAR_NS
    if len(equity) == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "fills": int(len(fills)),
                "trades_per_year": round(len(fills) / years, 2) if years > 0 else 0.0, "round_trips": 0,
                "years": round(years, 3)}
    day = S.trading_day_ids(equity.index - pd.Timedelta(hours=1))
    daily = equity.groupby(day).last()
    rets = daily.pct_change()
    rets.iloc[0] = daily.iloc[0] / start_nav - 1.0
    sd = rets.std(ddof=1)
    sharpe = float(rets.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0
    curve = np.concatenate([[start_nav], equity.to_numpy()])
    dd = curve / np.maximum.accumulate(curve) - 1.0
    total = float(curve[-1] / start_nav - 1.0)
    closed = trades[trades["exit_reason"] != "open"]
    out = {
        "sharpe": round(sharpe, 4),
        "total_return": round(total, 5),
        "cagr": round((1 + total) ** (1 / years) - 1, 5) if years > 0 and total > -1 else None,
        "ann_vol": round(float(sd * math.sqrt(TRADING_DAYS_PER_YEAR)), 5) if sd == sd else None,
        "max_drawdown": round(float(-dd.min()), 5),
        "years": round(years, 3),
        "fills": int(len(fills)),
        "trades_per_year": round(len(fills) / years, 2) if years > 0 else 0.0,
        "round_trips": int(len(trades)),
        "round_trips_per_year": round(len(trades) / years, 2) if years > 0 else 0.0,
    }
    if len(fills):
        out["fills_by_role"] = {k: int(v) for k, v in fills["role"].value_counts().items()}
    if len(closed):
        pips = closed["pnl_pips"].to_numpy(dtype=float)
        out["win_rate"] = round(float((pips > 0).mean()), 4)
        out["avg_trade_pips_net"] = round(float(pips.mean()), 3)
        out["per_pair"] = {p: {"trades": int(len(g)), "avg_pips": round(float(g["pnl_pips"].mean()), 2),
                               "win_rate": round(float((g["pnl_pips"] > 0).mean()), 3)}
                           for p, g in closed.groupby("pair")}
    return out


def slice_metrics(res: B.RunResult, t0: int, t1: int) -> dict:
    """Metrics of one segment of a continuous run: equity in (t0, t1], fills and trades entered in [t0, t1)."""
    ts = res.equity.index.as_unit("ns").asi8
    before = res.equity[ts <= t0]
    start_nav = float(before.iloc[-1]) if len(before) else res.start_nav
    eq = res.equity[(ts > t0) & (ts <= t1)]
    fl = res.fills[(res.fills["ts"] >= t0) & (res.fills["ts"] < t1)]
    tr = res.trades[(res.trades["entry_ts"] >= t0) & (res.trades["entry_ts"] < t1)]
    return perf_metrics(eq, start_nav, fl, tr, t0, t1)


# =============================================================================
# Walk-forward analysis (every backtest is a NautilusTrader run)
# =============================================================================
def _geometry_start(opens: np.ndarray, burn_in) -> float:
    return float((pd.Timestamp(int(opens[0]), tz="UTC") + burn_in).value) if burn_in is not None else float(opens[0])


def make_splits(index: pd.DatetimeIndex, n_splits: int = N_SPLITS, is_frac: float = IS_FRACTION,
                burn_in=BURN_IN) -> list[dict]:
    """Rolling windows of equal length W over [first bar + burn-in, last close]; IS = 70% W, OOS = next 30% W;
    OOS segments tile the tail. The burn-in only provides indicator warm-up history.

    Boundaries are decision times: a window starting at bar i starts deciding at bar i's open (= close of i-1).
    """
    opens = index.as_unit("ns").asi8
    t0, t1 = _geometry_start(opens, burn_in), float(opens[-1] + HOUR)
    width = (t1 - t0) / (1 + (n_splits - 1) * (1 - is_frac))
    step = width * (1 - is_frac)

    def at(v: float) -> int:
        k = int(np.searchsorted(opens, v, side="left"))
        return int(opens[k]) if k < len(opens) else int(t1)

    out = []
    for k in range(n_splits):
        a = t0 + k * step
        b = a + width * is_frac
        c = a + width if k < n_splits - 1 else t1
        out.append({"split": k + 1, "is": (at(a), at(b)), "oos": (at(b), at(c) if k < n_splits - 1 else int(t1))})
    return out


def grid_params(base: S.StrategyParams):
    keys = list(S.PARAM_GRID)
    values = [list(S.PARAM_GRID[k]) for k in keys]
    combos = [dataclasses.replace(base, **dict(zip(keys, c))) for c in itertools.product(*values)]
    return keys, values, combos


_WORKER: dict = {}


def load_market(pairs, data_dir: Path, end: pd.Timestamp | None, official: bool) -> tuple[dict, B.RateTable]:
    """Official runs: the spliced OANDA + MT5 history; other data dirs: those MT5 exports only. Rates are cut at
    `end` like the prices."""
    m15 = (B.load_spliced_universe(pairs, end=end, mt5_dir=data_dir) if official
           else B.load_universe(pairs, data_dir, end=end))
    rates = B.RateTable.load(end=end)
    return m15, rates


def _worker_init(pairs: tuple[str, ...], data_dir: str, end: str | None, hid: str, official: bool) -> None:
    import logging
    logging.disable(logging.CRITICAL)
    use_hypothesis(hid)
    m15, rates = load_market(pairs, Path(data_dir), pd.Timestamp(end) if end else None, official)
    _WORKER["market"] = B.Market(m15, rates=rates)


def _worker_run(task: tuple) -> dict:
    params, start, end = task
    p = S.StrategyParams(**params)
    mk = _WORKER["market"]
    st = S.PortfolioTrendStrategy([S.Segment(start, end, p)], mk.pairs)
    res = mk.run(st, start, end)
    m = perf_metrics(res.equity, res.start_nav, res.fills, res.trades, start, end)
    m["desyncs"] = st.desyncs
    m["financing_usd"] = round(mk.financing.total_usd, 2) if mk.financing is not None else 0.0
    return m


def select_params(is_rows: list[dict], values: list[list]) -> dict:
    """Plateau selection: score = mean IS Sharpe over the 3^k grid neighbourhood; only combos trading
    more than MIN_TRADES_PER_YEAR in-sample are eligible (if none are, all are)."""
    shape = [len(v) for v in values]
    sharpe = np.array([r["metrics"]["sharpe"] for r in is_rows]).reshape(shape)
    eligible = [r["metrics"]["trades_per_year"] > MIN_TRADES_PER_YEAR for r in is_rows]
    if not any(eligible):
        eligible = [True] * len(is_rows)
    best, best_score = None, -np.inf
    for flat, r in enumerate(is_rows):
        pos = np.unravel_index(flat, shape)
        score = float(np.mean(sharpe[tuple(slice(max(0, i - 1), i + 2) for i in pos)]))
        r["plateau_score"] = round(score, 4)
        if eligible[flat] and score > best_score + 1e-12:
            best, best_score = r, score
    return best


def make_holdout_splits(index: pd.DatetimeIndex, dev_end: pd.Timestamp = DEV_END,
                        n_splits: int = N_SPLITS, is_frac: float = IS_FRACTION, burn_in=BURN_IN) -> list[dict]:
    """Continue the development split geometry (window W, step 0.3W, IS = 0.7W) into the holdout."""
    opens = index.as_unit("ns").asi8
    dev = opens[opens < dev_end.value]
    t0, t_dev = _geometry_start(dev, burn_in), float(dev[-1] + HOUR)
    width = (t_dev - t0) / (1 + (n_splits - 1) * (1 - is_frac))
    step = width * (1 - is_frac)
    t_end = float(opens[-1] + HOUR)

    def at(v: float) -> int:
        k = int(np.searchsorted(opens, v, side="left"))
        return int(opens[k]) if k < len(opens) else int(t_end)

    out, a, k = [], float(opens[np.searchsorted(opens, dev_end.value)]), 1
    while a < t_end:
        c = min(a + step, t_end)
        out.append({"split": k, "is": (at(a - width * is_frac), at(a)), "oos": (at(a), at(c) if c < t_end else int(t_end))})
        a, k = c, k + 1
    return [sp for sp in out if sp["oos"][1] > sp["oos"][0]]


def make_rolling_splits(index: pd.DatetimeIndex, max_oos_days: float, is_frac: float, start_ns: int,
                        oos_days: float | None = None) -> list[dict]:
    """Fast rolling (not anchored) walk-forward: OOS windows of equal length, at most `max_oos_days`, tile
    [start, last close]; each IS window is the stretch immediately before its OOS window, is_frac : 1 - is_frac
    of it in length. With `oos_days` given (the holdout), that window length is continued and the last window
    may be short. Boundaries snap to bar opens."""
    opens = index.as_unit("ns").asi8
    t_end = int(opens[-1] + HOUR)
    day = 86_400_000_000_000

    def at(v: float) -> int:
        k = int(np.searchsorted(opens, v, side="left"))
        return int(opens[k]) if k < len(opens) else t_end

    a0 = at(start_ns)
    span = (t_end - a0) / day
    if oos_days is None:
        oos_days = span / math.ceil(span / max_oos_days)
    step, is_len = oos_days * day, oos_days * day * is_frac / (1 - is_frac)
    n = math.ceil(span / oos_days - 1e-9)
    bounds = [a0] + [at(a0 + k * step) for k in range(1, n)] + [t_end]
    return [{"split": k + 1, "is": (at(bounds[k] - is_len), bounds[k]), "oos": (bounds[k], bounds[k + 1]),
             "oos_days": oos_days} for k in range(n) if bounds[k + 1] > bounds[k]]


def wfa_splits(index: pd.DatetimeIndex) -> list[dict]:
    """The development walk-forward: the standard 5 splits, or - for a hypothesis that sets WFA_MAX_OOS_DAYS
    (Addendum 2) - the fast rolling geometry over the same stitched OOS span."""
    std = make_splits(index)
    if not hasattr(S, "WFA_MAX_OOS_DAYS"):
        return std
    return make_rolling_splits(index, S.WFA_MAX_OOS_DAYS, IS_FRACTION, std[0]["oos"][0])


def holdout_wfa_splits(index: pd.DatetimeIndex) -> list[dict]:
    """The development geometry continued into the holdout."""
    if not hasattr(S, "WFA_MAX_OOS_DAYS"):
        return make_holdout_splits(index)
    opens = index.as_unit("ns").asi8
    dev = wfa_splits(index[opens < DEV_END.value])
    return make_rolling_splits(index, S.WFA_MAX_OOS_DAYS, IS_FRACTION,
                               int(opens[np.searchsorted(opens, DEV_END.value)]), oos_days=dev[0]["oos_days"])


def min_wfe() -> float:
    return float(getattr(S, "WFA_MIN_WFE", MIN_WFE))


def trial_guard(hid: str, fp: str, trials: list[dict], active=None) -> str | None:
    """Cycle-5 rules for an official Stage-1 run; returns the refusal reason, or None.
    - only cycle-5 IDs run (earlier cycles are archived);
    - an ID is immutable: once logged, it can only be reproduced with identical code (not logged again);
    - no trial cap (MAX_TRIALS None), otherwise the cap applies."""
    if hid not in (H.CYCLE5 if active is None else active):
        return f"{hid} belongs to an archived cycle (reproduce at commit c447b91)"
    same_id = [t for t in trials if t["hyp"] == hid]
    if any(t["code"] == fp for t in same_id):
        return None                                             # reproduction: runs, logs nothing
    if same_id:
        return f"{hid} is already logged with other code ({same_id[-1]['code']}); register the change as a new ID"
    if MAX_TRIALS is not None and len(trials) >= MAX_TRIALS:
        return f"trial budget exhausted ({len(trials)}/{MAX_TRIALS})"
    return None


_EULER_GAMMA = 0.5772156649015329


def daily_returns(equity: pd.Series, start_nav: float) -> np.ndarray:
    """Trading-day returns of an equity curve (the series perf_metrics' Sharpe uses)."""
    if len(equity) == 0:
        return np.zeros(0)
    day = S.trading_day_ids(equity.index - pd.Timedelta(hours=1))
    daily = equity.groupby(day).last()
    r = daily.pct_change()
    r.iloc[0] = daily.iloc[0] / start_nav - 1.0
    return r.to_numpy(dtype=float)


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max of n_trials zero-skill Sharpe estimates] (Bailey & Lopez de Prado 2014), same units as var_sharpe."""
    if n_trials < 2 or not var_sharpe > 0:
        return 0.0
    z = statistics.NormalDist().inv_cdf
    return math.sqrt(var_sharpe) * ((1 - _EULER_GAMMA) * z(1 - 1 / n_trials)
                                    + _EULER_GAMMA * z(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: np.ndarray, trial_sharpes_annual: list[float]) -> dict:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): the probability that the true Sharpe exceeds the
    best Sharpe expected from len(trial_sharpes_annual) zero-skill trials, given the returns' length, skew and
    kurtosis. Per-trading-day units inside; the benchmark is also reported annualised."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(trial_sharpes_annual)
    if len(r) < 30 or not r.std(ddof=1) > 0:
        return {"dsr": None, "n_trials": n}
    sr = r.mean() / r.std(ddof=1)
    c = r - r.mean()
    m2 = (c ** 2).mean()
    skew, kurt = (c ** 3).mean() / m2 ** 1.5, (c ** 4).mean() / m2 ** 2
    var_trials = float(np.var(np.asarray(trial_sharpes_annual, dtype=float), ddof=1)) / TRADING_DAYS_PER_YEAR \
        if n >= 2 else 0.0
    sr0 = expected_max_sharpe(n, var_trials)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    cdf = statistics.NormalDist().cdf
    return {"dsr": round(cdf((sr - sr0) * math.sqrt(len(r) - 1) / denom), 4),
            "psr_vs_zero": round(cdf(sr * math.sqrt(len(r) - 1) / denom), 4),
            "sharpe_benchmark_annual": round(sr0 * math.sqrt(TRADING_DAYS_PER_YEAR), 4),
            "n_trials": n, "skew": round(float(skew), 3), "kurtosis": round(float(kurt), 3), "days": int(len(r))}


def all_trial_sharpes() -> list[float]:
    """OOS Sharpe of every logged trial on this development data: archived cycle-4 reports + cycle-5 reports."""
    out = []
    for folder in ARCHIVE_RESULTS + (RESULTS_DIR,):
        for f in sorted(Path(folder).glob("*.json")) if Path(folder).exists() else []:
            if f.stem.endswith("_holdout"):
                continue
            d = json.loads(f.read_text())
            if d.get("stage") == 1 and d.get("trial", {}).get("counts_as_trial"):
                out.append(float(d["metrics"]["oos_sharpe"]))
    return out


def run_wfa(market: B.Market, pool: ProcessPoolExecutor, splits: list[dict] | None = None) -> dict:
    keys, values, combos = grid_params(S.DEFAULT_PARAMS)
    splits = splits or make_splits(market.index)
    tasks = [(dataclasses.asdict(p), *sp["is"]) for sp in splits for p in combos]
    results = list(pool.map(_worker_run, tasks))
    rows, chosen, plan = [], [], []
    for k, sp in enumerate(splits):
        is_rows = [{"params": dataclasses.asdict(p), "metrics": results[k * len(combos) + j]}
                   for j, p in enumerate(combos)]
        best = select_params(is_rows, values)
        p_best = S.StrategyParams(**best["params"])
        chosen.append(p_best)
        plan.append(S.Segment(*sp["oos"], p_best))
        rows.append({"split": sp["split"], "is_rows": is_rows, "best": best})
    st = S.PortfolioTrendStrategy(plan, market.pairs)
    oos_res = market.run(st, plan[0].start_ns, plan[-1].end_ns)
    oos = perf_metrics(oos_res.equity, oos_res.start_nav, oos_res.fills, oos_res.trades, plan[0].start_ns,
                       plan[-1].end_ns)
    oos["financing_usd"] = round(market.financing.total_usd, 2) if market.financing is not None else 0.0
    oos["cost_usd_spread_slippage"] = round(spread_cost_usd(oos_res.fills, market), 2)
    split_rows = []
    for sp, r, seg in zip(splits, rows, plan):
        seg_m = slice_metrics(oos_res, seg.start_ns, seg.end_ns)
        is_sh = r["best"]["metrics"]["sharpe"]
        split_rows.append({
            "split": sp["split"],
            "is_period": [str(pd.Timestamp(t, tz="UTC")) for t in sp["is"]],
            "oos_period": [str(pd.Timestamp(t, tz="UTC")) for t in sp["oos"]],
            "chosen_params": r["best"]["params"],
            "is_plateau_score": r["best"]["plateau_score"],
            "is": r["best"]["metrics"],
            "oos": seg_m,
            "wfe": round(seg_m["sharpe"] / is_sh, 4) if is_sh > 0 else None,
            "is_grid": [{**{k: x["params"][k] for k in keys}, "sharpe": x["metrics"]["sharpe"],
                         "trades_per_year": x["metrics"]["trades_per_year"]} for x in r["is_rows"]],
        })
    is_mean = float(np.mean([r["is"]["sharpe"] for r in split_rows]))
    return {"splits": split_rows, "chosen": chosen, "plan": plan, "oos_result": oos_res, "oos": oos,
            "oos_desyncs": st.desyncs, "is_desyncs": int(sum(m.get("desyncs", 0) for m in results)),
            "is_sharpe_mean": round(is_mean, 4),
            "wfe": round(oos["sharpe"] / is_mean, 4) if is_mean > 0 else None}


def spread_cost_usd(fills: pd.DataFrame, market: B.Market) -> float:
    """Spread + slippage paid: each fill's distance from the mid of the hour it filled in, in USD (attribution)."""
    total = 0.0
    for pair, g in fills.groupby("pair"):
        h = market.h1[pair]
        opens = h.index.as_unit("ns").asi8
        k = np.clip(np.searchsorted(opens, g["ts"].to_numpy() - B.ONE_MS), 0, len(opens) - 1)
        mid = h["open"].to_numpy()[k]
        cost = np.abs(g["units"].to_numpy()) * np.abs(g["px"].to_numpy() - mid)
        if pair.endswith("USD"):
            conv = np.ones(len(g))
        elif pair.startswith("USD"):
            conv = 1.0 / g["px"].to_numpy()
        else:
            quote_usd = "USD" + pair[3:]
            conv = np.full(len(g), np.nan)
            if quote_usd in market.h1:
                hq = market.h1[quote_usd]
                kq = np.clip(np.searchsorted(hq.index.as_unit("ns").asi8, g["ts"].to_numpy() - B.ONE_MS, side="right") - 1,
                             0, len(hq) - 1)
                conv = 1.0 / hq["open"].to_numpy()[kq]
        total += float(np.nansum(cost * conv))
    return total


# =============================================================================
# Integrity tests
# =============================================================================
def synthetic_m15(pair: str = "EURUSD", days: int = 150, seed: int = 0, ann_vol: float = 0.08,
                  start: str = "2021-03-01", spread_points: float = 16.0, price: float = 1.15,
                  vol_regimes: bool = False, jump_prob: float = 0.0005) -> pd.DataFrame:
    """UTC M15 bid bars with weekend gaps, regime drift and rare fat-tail jumps (optionally volatility regimes
    too). Unit tests only - never used for performance."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    idx = idx[idx.dayofweek < 5]  # the *test generator* skips weekends; strategy code never sees calendars
    n, sub = len(idx), 3
    sig = ann_vol / math.sqrt(6240 * 4 * sub)
    if vol_regimes:
        sig = sig * np.repeat(rng.choice([0.4, 1.0, 2.5], size=n // 2400 + 1), 2400 * sub)[: n * sub]
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 400 + 1), 400 * sub)[: n * sub]
    jumps = np.where(rng.random(n * sub) < jump_prob, rng.choice([-25.0, 25.0], n * sub), 0.0) * sig
    path = price * np.exp(np.cumsum(regime * sig * 0.15 + sig * rng.standard_normal(n * sub) + jumps)).reshape(n, sub)
    # like real MT5 data, a bar opens near (not exactly at) the previous close, so fill timing is observable
    sig_open = sig[::sub] if np.ndim(sig) else sig
    open_ = np.concatenate([[price], path[:-1, -1]]) * np.exp(0.3 * sig_open * rng.standard_normal(n))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, path.max(axis=1)),
                         "low": np.minimum(open_, path.min(axis=1)), "close": path[:, -1],
                         "spread": spread_points}, index=idx)


def m15_from_hourly_close(close_h: np.ndarray, start: str, spread_points: float = 16.0) -> pd.DataFrame:
    """Deterministic M15 bars that walk linearly from each hourly close to the next (breaker scenarios)."""
    idx = pd.date_range(start, periods=len(close_h) * 4, freq="15min", tz="UTC")
    prev = np.concatenate([[close_h[0]], close_h[:-1]])
    steps = np.linspace(0.25, 1.0, 4)
    closes = (prev[:, None] + (close_h - prev)[:, None] * steps[None, :]).ravel()
    opens = np.concatenate([[close_h[0]], closes[:-1]])
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.00002,
                         "low": np.minimum(opens, closes) * 0.99998, "close": closes, "spread": spread_points},
                        index=idx)


def _market(frames: dict[str, pd.DataFrame], rates: B.RateTable | None = SYN_RATES) -> B.Market:
    return B.Market({p: B.clean_m15(f, p) for p, f in frames.items()}, rates=rates)


def _syn_days(extra_bars: int = 2000, p=None) -> int:
    """Calendar days of weekday-only synthetic data covering the warm-up of `p` plus `extra_bars` hourly bars."""
    bars = S.warmup_bars(p or S.DEFAULT_PARAMS) + extra_bars
    return int(math.ceil(bars / 24 * 7 / 5)) + 3


def _decision_times(market: B.Market) -> np.ndarray:
    return market.index.as_unit("ns").asi8 + HOUR


def _plan_over(market: B.Market, p: S.StrategyParams, skip: int = 100) -> list[S.Segment]:
    closes = _decision_times(market)
    return [S.Segment(int(closes[skip]), int(closes[-1]), p)]


def parity(frames_or_market, plan_fn=None) -> dict:
    """Nautilus run vs the vectorised reference on one USD-quoted pair: orders, fill prices, equity (financing
    included)."""
    mk = frames_or_market if isinstance(frames_or_market, B.Market) else _market(frames_or_market)
    pair = mk.pairs[0]
    plan = plan_fn(mk) if plan_fn else _plan_over(mk, S.DEFAULT_PARAMS)
    st = S.PortfolioTrendStrategy(plan, mk.pairs)
    res = mk.run(st, plan[0].start_ns, plan[-1].end_ns)
    n_book = len(mk.financing.bookings) if mk.financing is not None else 0
    fin_usd = mk.financing.total_usd if mk.financing is not None else 0.0
    ref = S.reference_backtest(mk.h1[pair], plan, rates=mk.rates, pair=pair)
    o_n = res.orders[["ts", "units", "role", "reason"]].reset_index(drop=True)
    o_r = ref["orders"][["ts", "units", "role", "reason"]].reset_index(drop=True)
    same_orders = o_n.equals(o_r)
    n_f = len(res.fills)
    k = min(n_f, len(ref["orders"]))
    px_diff = (float(np.max(np.abs(res.fills["px"].to_numpy()[:k] - ref["orders"]["px"].to_numpy()[:k])))
               if k else 0.0) if n_f == len(ref["orders"].dropna(subset=["px"])) else math.inf
    same_idx = res.equity.index.equals(ref["equity"].index)
    diff = (float(np.max(np.abs(res.equity.to_numpy() - ref["equity"].to_numpy()))) if len(res.equity) else 0.0) \
        if same_idx else math.inf
    # Nautilus books realised P&L and financing in whole cents; the reference keeps full floats for P&L ->
    # allow 1 cent per fill and per financing booking. A run without fills proves nothing, so it fails.
    ok = bool(same_orders and px_diff < 1e-9 and same_idx and diff <= 0.01 * max(1, n_f + n_book)
              and st.desyncs == 0 and n_f > 0)
    mk.dispose()
    return {"passed": ok, "orders": int(len(o_n)), "fills": n_f, "orders_identical": bool(same_orders),
            "fills_by_role": {k: int(v) for k, v in res.fills["role"].value_counts().items()},
            "financing_bookings": n_book, "financing_usd": round(fin_usd, 2),
            "max_fill_px_diff": px_diff, "max_equity_abs_diff_usd": diff, "desyncs": st.desyncs}


class _RoundTripProbe(Strategy):
    """Buys `units` of `pair` at one bar close and sells them at the next; records the USD balance. Like the
    strategies, it sends each order on the pair's next quote (the next bar's open) with zero latency."""

    def __init__(self, pairs, pair, units, start_ns):
        super().__init__(StrategyConfig(strategy_id="Probe-001"))
        self.pairs, self.pair, self.units = pairs, pair, units
        self.warmup_start_ns = start_ns
        self.k, self.ts, self.balances = 0, -1, []
        self.pending = None

    def on_start(self):
        for p in self.pairs:
            self.subscribe_bars(B.bar_type(p))
        self.subscribe_quote_ticks(B.instrument_id(self.pair))

    def queue(self, side):
        self.pending = side

    def on_quote_tick(self, tick):
        if self.pending is not None:
            self.submit_order(self.order_factory.market(B.instrument_id(self.pair), self.pending,
                                                        Quantity.from_int(abs(self.units))))
            self.pending = None

    def on_bar(self, bar):
        if bar.ts_event == self.ts:
            return
        self.ts = bar.ts_event
        self.k += 1
        self.balances.append(self.portfolio.account(B.SIM).balance_total(USD).as_double())
        if self.k in (3, 4):
            self.queue(OrderSide.BUY if self.k == 3 else OrderSide.SELL)

    def result(self):
        return self.balances


def friction_tests() -> dict:
    """Exact round-trip costs through Nautilus: USD-quoted pair, and a JPY cross converted to USD."""
    out = {}
    flat_usd = synthetic_m15("EURUSD", days=10, ann_vol=0.0, spread_points=16.0, price=1.1)
    mk = _market({"EURUSD": flat_usd}, rates=None)
    idx = mk.index.as_unit("ns").asi8
    bal = []
    for _ in range(2):  # twice: the second run exercises the engine-reset path
        pr = _RoundTripProbe(mk.pairs, "EURUSD", 100_000, int(idx[0]))
        bal.append(mk.run(pr, int(idx[0]), int(idx[40]) + HOUR))
    mk.dispose()
    exp = -100_000 * (1.6 + 2 * B.SLIPPAGE_PIPS) * 1e-4
    got = [b[-1] - b[0] for b in bal]
    out["usd_pair"] = {"passed": all(abs(g - exp) < 0.011 for g in got), "cost_usd": got, "expected": exp}

    frames = {"USDJPY": synthetic_m15("USDJPY", days=10, ann_vol=0.0, spread_points=15.0, price=110.0),
              "EURJPY": synthetic_m15("EURJPY", days=10, ann_vol=0.0, spread_points=20.0, price=120.0)}
    mk = _market(frames, rates=None)
    idx = mk.index.as_unit("ns").asi8
    got = []
    for _ in range(2):
        pr = _RoundTripProbe(mk.pairs, "EURJPY", 100_000, int(idx[0]))
        b = mk.run(pr, int(idx[0]), int(idx[40]) + HOUR)
        got.append(b[-1] - b[0])
    mk.dispose()
    exp = -100_000 * (2.0 + 2 * B.SLIPPAGE_PIPS) * 0.01 / 110.0
    out["jpy_cross_to_usd"] = {"passed": all(abs(g - exp) < 0.02 for g in got), "cost_usd": got, "expected": exp}
    return out


class _HoldProbe(_RoundTripProbe):
    """Opens `units` of `pair` at the close of bar `k_in` and closes the position at the close of bar `k_out`."""

    def __init__(self, pairs, pair, units, start_ns, k_in, k_out):
        super().__init__(pairs, pair, units, start_ns)
        self.k_in, self.k_out = k_in, k_out

    def on_bar(self, bar):
        if bar.ts_event == self.ts:
            return
        self.ts = bar.ts_event
        self.k += 1
        self.balances.append(self.portfolio.account(B.SIM).balance_total(USD).as_double())
        if self.k in (self.k_in, self.k_out):
            opening = self.k == self.k_in
            self.queue(OrderSide.BUY if (self.units > 0) == opening else OrderSide.SELL)


def financing_tests() -> dict:
    """Overnight financing through Nautilus against hand calculations on flat prices with fixed rates
    (EUR 3 %, USD 1 %, JPY -0.1 %, mark-up 0.5 %): long and short EUR/USD over a weekend, and EUR/JPY converted
    to USD. Rolls are counted independently from the fill times: every 17:00 New York in [c_in, o_out]."""
    out = {}
    rolls = B.roll_times_ns()
    k_in, k_out = 3, 3 + 24 * 8                       # held for 8 x 24 hourly bars (weekdays), spans a weekend

    def run(frames, pair, units, spread_pips, mid, quote_usd):
        mk = _market(frames, rates=FIXED_RATES)
        idx = mk.index.as_unit("ns").asi8
        got = []
        for _ in range(2):                            # twice: the second run exercises the engine-reset path
            pr = _HoldProbe(mk.pairs, pair, units, int(idx[0]), k_in, k_out)
            b = mk.run(pr, int(idx[0]), int(idx[k_out + 5]) + HOUR)
            got.append((b[-1] - b[0], len(mk.financing.bookings), mk.financing.total_usd))
        # position held from the open fill of bar k_in (0-based index k_in) to the open fill of bar k_out
        c_in, o_out = int(idx[k_in]) + HOUR, int(idx[k_out])
        n_rolls = int(np.searchsorted(rolls, o_out, side="right") - np.searchsorted(rolls, c_in, side="left"))
        diff = FIXED_RATES.accrual(pair[:3], [c_in])[0] - FIXED_RATES.accrual(pair[3:], [c_in])[0]
        per_day = mid * (units * diff / 100.0 - abs(units) * B.FIN_MARKUP) / 365.0 * quote_usd
        spread = -abs(units) * (spread_pips + 2 * B.SLIPPAGE_PIPS) * B.pip_size(pair) * quote_usd
        exp_fin = n_rolls * per_day
        mk.dispose()
        ok = all(abs(fin - exp_fin) <= 0.01 * nb + 1e-9 and abs(bal - (exp_fin + spread)) <= 0.01 * (nb + 2)
                 for bal, nb, fin in got) and got[0] == got[1] and n_rolls >= 8
        return {"passed": bool(ok), "rolls": n_rolls, "expected_financing_usd": round(exp_fin, 2),
                "engine_financing_usd": [round(g[2], 2) for g in got], "bookings": got[0][1],
                "expected_balance_change_usd": round(exp_fin + spread, 2),
                "engine_balance_change_usd": [round(g[0], 2) for g in got]}

    flat = {"EURUSD": synthetic_m15("EURUSD", days=14, ann_vol=0.0, spread_points=16.0, price=1.1)}
    out["long_eurusd"] = run(flat, "EURUSD", 1_000_000, 1.6, 1.1, 1.0)
    out["short_eurusd"] = run(flat, "EURUSD", -1_000_000, 1.6, 1.1, 1.0)
    out["opposite_signs"] = {"passed": bool(out["long_eurusd"]["expected_financing_usd"] > 0 >
                                            out["short_eurusd"]["expected_financing_usd"])}
    frames = {"USDJPY": synthetic_m15("USDJPY", days=14, ann_vol=0.0, spread_points=15.0, price=110.0),
              "EURJPY": synthetic_m15("EURJPY", days=14, ann_vol=0.0, spread_points=20.0, price=120.0)}
    out["long_eurjpy_to_usd"] = run(frames, "EURJPY", 100_000, 2.0, 120.0, 1.0 / 110.0)
    none = synthetic_m15("EURUSD", days=14, ann_vol=0.0, price=1.1)
    mk = _market({"EURUSD": none}, rates=FIXED_RATES)
    idx = mk.index.as_unit("ns").asi8
    b = mk.run(_HoldProbe(mk.pairs, "EURUSD", 0, int(idx[0]), -1, -1), int(idx[0]), int(idx[100]) + HOUR)
    out["flat_book_no_charge"] = {"passed": bool(len(mk.financing.bookings) == 0 and b[-1] == b[0])}
    mk.dispose()
    return out


def fill_timing_test() -> dict:
    """Multi-pair execution. Every fill must be at its pair's OWN next-open quote after the decision (ask for buys,
    bid for sells) - never at a stale quote - even when pairs have different gaps (random missing bars and a
    two-day holiday in which only USDCAD trades), and the strategy's bookkeeping must never desync from the venue.
    (Nautilus processes due orders after each data point, so a naive design fills every pair but the first at
    the previous close, and decides late when a close is incomplete.)"""
    p = dataclasses.replace(S.DEFAULT_PARAMS, **{k: v[0] for k, v in S.PARAM_GRID.items()})
    days = _syn_days(4000, p)
    px = {"EURUSD": 1.15, "GBPUSD": 1.3, "USDJPY": 110.0, "USDCAD": 1.3, "EURJPY": 125.0}
    rng = np.random.default_rng(41)
    frames = {}
    for i, (pair, price) in enumerate(px.items()):
        f = synthetic_m15(pair, days=days, seed=40 + i, price=price, ann_vol=0.12, vol_regimes=True)
        f = f[rng.random(len(f)) > 0.03]                                    # random missing M15 bars
        frames[pair] = f
    hol = frames["USDCAD"].index[int(len(frames["USDCAD"]) * 0.8)]
    for pair in frames:                                                     # holiday: only USDCAD trades
        if pair != "USDCAD":
            f = frames[pair]
            frames[pair] = f[(f.index < hol) | (f.index >= hol + pd.Timedelta(days=2))]
    mk = _market(frames)
    plan = _plan_over(mk, p)
    st = S.PortfolioTrendStrategy(plan, mk.pairs)
    res = mk.run(st, plan[0].start_ns, plan[-1].end_ns)
    bad, n = [], 0
    for pair, fl in res.fills.groupby("pair"):
        h = mk.h1[pair]
        opens = h.index.as_unit("ns").asi8
        dec = np.sort(res.orders.loc[res.orders["pair"] == pair, "ts"].to_numpy())
        for f in fl.itertuples(index=False):
            n += 1
            j = int(np.searchsorted(opens, f.ts - B.ONE_MS))
            d = dec[np.searchsorted(dec, f.ts, side="right") - 1]            # latest decision before the fill
            first = int(np.searchsorted(opens, d, side="left"))              # the pair's first bar opening after it
            want = h["open_ask"].iloc[j] if f.units > 0 else h["open_bid"].iloc[j]
            if not (j < len(opens) and opens[j] + B.ONE_MS == f.ts and j == first and abs(f.px - want) < 1e-9):
                bad.append((pair, str(pd.Timestamp(f.ts, tz="UTC")), f.px, float(want)))
    # pairs the strategy can trade here (A5 never trades the EURJPY cross, C5-007 only its two legs)
    feats = S.compute_features(mk.h1, p)
    tradeable = {k for k, f in feats.items() if (f["signal"].fillna(0.0) != 0).any()}
    mk.dispose()
    filled = set(res.fills["pair"])
    ok = (not bad and n >= 10 and st.desyncs == 0 and len(filled) >= min(4, len(tradeable))
          and ("USDCAD" in filled or "USDCAD" not in tradeable))
    return {"passed": bool(ok), "fills_checked": n, "pairs_filled": sorted(filled), "tradeable": sorted(tradeable),
            "roles": sorted(set(res.fills["role"])), "desyncs": st.desyncs, "bad_fills": bad[:5], "n_bad": len(bad)}


def signal_parity_multi() -> dict:
    """Multi-pair: at every decision, the signal the event-driven strategy trades for each pair with a bar at
    that close equals the vectorised signal (unshifted feature row of that bar), including cross-sectional,
    consensus and carry signals. Single-pair parity cannot see cross-pair logic; this can."""
    p = S.DEFAULT_PARAMS
    px = {"EURUSD": 1.15, "GBPUSD": 1.3, "USDJPY": 110.0, "USDCAD": 1.3, "EURJPY": 125.0}
    days = _syn_days(1500)
    rng = np.random.default_rng(61)
    frames = {}
    for i, (pair, price) in enumerate(px.items()):
        f = synthetic_m15(pair, days=days, seed=60 + i, price=price, ann_vol=0.1)
        frames[pair] = f[rng.random(len(f)) > 0.02]                          # a few missing bars
    mk = _market(frames)
    plan = _plan_over(mk, p)
    rec: list[tuple[int, dict]] = []
    now = {"ts": None}

    class Recorder(S.PortfolioTrendStrategy):
        def _decide_all(self, ts):
            now["ts"] = ts
            super()._decide_all(ts)

    st = Recorder(plan, mk.pairs)
    for sig in st._signals.values():                 # record exactly what the decision uses (after on_close)
        def wrapped(orig=sig.signals):
            out = orig()
            rec.append((now["ts"], dict(out)))
            return out
        sig.signals = wrapped
    mk.run(st, plan[0].start_ns, plan[-1].end_ns)
    feats = S.compute_features(mk.h1, p)
    bars = {k: f.index.as_unit("ns").asi8 for k, f in feats.items()}
    vec = {k: f["signal"].to_numpy(dtype=float) for k, f in feats.items()}
    worst, n, bad = 0.0, 0, []
    for ts, sig in rec:
        for k in mk.pairs:
            j = int(np.searchsorted(bars[k], ts - HOUR))
            if j >= len(bars[k]) or bars[k][j] != ts - HOUR:
                continue                                                     # no bar of k closes at ts
            a, b = float(sig.get(k, math.nan)), float(vec[k][j])
            n += 1
            if (a != a) != (b != b) or (a == a and abs(a - b) > 1e-6):
                bad.append((k, str(pd.Timestamp(ts, tz="UTC")), a, b))
            elif a == a:
                worst = max(worst, abs(a - b))
    mk.dispose()
    nonzero = sum(1 for _, sig in rec for v in sig.values() if v == v and v != 0)
    return {"passed": bool(not bad and n > 1000 and nonzero >= 10), "compared": n, "nonzero_signals": nonzero,
            "max_abs_diff": worst, "n_bad": len(bad), "bad": bad[:5]}


def rolling_split_tests() -> dict:
    """The fast rolling walk-forward (make_rolling_splits): OOS windows of `oos_days` (< half a year) tile
    [start, end) without gaps or overlaps; each IS window is the `is_days` immediately before its OOS window
    (rolling, not anchored); every boundary is a bar open; the holdout version continues the same geometry from
    DEV_END and never puts an OOS bar before it."""
    idx = pd.date_range("2005-01-03", "2023-12-29", freq="h", tz="UTC")
    idx = idx[~B.market_closed(idx)]
    start = pd.Timestamp("2011-10-02 21:00", tz="UTC")
    dev = idx[idx < DEV_END]
    sp = make_rolling_splits(dev, 152, IS_FRACTION, start.value)
    opens = set(dev.as_unit("ns").asi8.tolist()) | {int(dev[-1].value + HOUR)}
    day = 86_400e9
    oos_days = [(b - a) / day for a, b in (x["oos"] for x in sp)]
    is_days = [(b - a) / day for a, b in (x["is"] for x in sp)]
    ok = (sp[0]["oos"][0] == int(dev[np.searchsorted(dev.asi8, start.value)].value)
          and all(sp[k]["oos"][1] == sp[k + 1]["oos"][0] for k in range(len(sp) - 1))
          and sp[-1]["oos"][1] == int(dev[-1].value + HOUR) and all(x["is"][1] == x["oos"][0] for x in sp)
          and max(oos_days) < 152 + 3 and min(oos_days) > 140
          and all(abs(i / o - IS_FRACTION / (1 - IS_FRACTION)) < 0.05 for i, o in zip(is_days, oos_days))
          and all(v in opens for x in sp for v in (*x["is"], *x["oos"])))
    first_ho = int(idx[np.searchsorted(idx.asi8, DEV_END.value)].value)
    ho = make_rolling_splits(idx, 152, IS_FRACTION, first_ho, oos_days=sp[0]["oos_days"])
    ho_ok = (ho[0]["oos"][0] == first_ho and all(x["is"][1] == x["oos"][0] for x in ho)
             and ho[0]["is"][0] < DEV_END.value and ho[-1]["oos"][1] == int(idx[-1].value + HOUR)
             and all(abs(x["oos_days"] - sp[0]["oos_days"]) < 1e-9 for x in ho))
    return {"passed": bool(ok and ho_ok), "n_splits": len(sp), "oos_days_min_max": [round(min(oos_days), 1),
            round(max(oos_days), 1)], "is_days_min_max": [round(min(is_days), 1), round(max(is_days), 1)],
            "holdout_splits": len(ho)}


def execution_rule_tests() -> dict:
    """Table-driven checks of the rebalancing rule and the signed target sizing."""
    A = __import__("hypotheses.academic_base", fromlist=["x"])
    cases = [((0, 0, 0.2), None), ((0, 5000, 0.2), (5000, "ENTRY")), ((5000, 0, 0.2), (-5000, "EXIT")),
             ((5000, -3000, 0.5), (-8000, "FLIP")), ((-3000, 1000, 0.99), (4000, "FLIP")),
             ((10000, 11000, 0.2), None), ((10000, 13000, 0.2), (3000, "REBAL")), ((10000, 12000, 0.2), None),
             ((10000, 8000, 0.2), (-2000, "REBAL")), ((-10000, -7000, 0.3), (3000, "REBAL")),
             ((-10000, -9000, 0.0), (1000, "REBAL")), ((-10000, -10000, 0.0), None)]
    bad = [(args, want, A.rebalance_order(*args)) for args, want in cases if A.rebalance_order(*args) != want]
    lim = A.RiskLimits()
    t = {"nan": A.target_units(math.nan, 1e6, 0.001, 1.1, 1.0, lim), "zero": A.target_units(0.0, 1e6, 0.001, 1.1, 1.0, lim),
         "long": A.target_units(1.0, 1e6, 0.001, 1.1, 1.0, lim), "short": A.target_units(-1.0, 1e6, 0.001, 1.1, 1.0, lim),
         "half": A.target_units(0.5, 1e6, 0.001, 1.1, 1.0, lim), "over": A.target_units(3.0, 1e6, 0.001, 1.1, 1.0, lim)}
    full = int(1e6 * A.RISK_PCT / (0.001 * A.ATR_MULT) // 1000 * 1000)
    sizing_ok = (t["nan"] == 0 and t["zero"] == 0 and t["long"] == full == -t["short"] and t["over"] == full
                 and t["half"] == int(1e6 * A.RISK_PCT * 0.5 / (0.001 * A.ATR_MULT) // 1000 * 1000))
    return {"passed": bool(not bad and sizing_ok), "rule_mismatches": [str(b) for b in bad], "targets": t}


def _positions_by_bar(res: B.RunResult, closes: np.ndarray) -> np.ndarray:
    """Units held during each hourly bar (fills land at the bar's open quote)."""
    units = np.zeros(len(closes))
    if len(res.fills):
        fts = res.fills["ts"].to_numpy()
        cum = np.cumsum(res.fills["units"].to_numpy())
        k = np.searchsorted(fts, closes - HOUR + B.ONE_MS, side="right") - 1
        units = np.where(k >= 0, cum[np.clip(k, 0, None)], 0.0)
    return units


def circuit_breaker_test() -> dict:
    """A down-trend turning into an oscillating up-trend with periodic up-jumps, so trend and reversion rules
    both take positions; gap the market AGAINST the first position held through a bar near the equity peak
    (down if long, up if short), sized off the leverage actually held (legitimate because positions are causal).
    A 5 % loss must trip the 24-hour halt, a 15 % loss the permanent kill switch."""
    warm = S.warmup_bars(S.DEFAULT_PARAMS) + 50        # flat warm-up, then down 400 bars, then up with jumps
    n, turn = warm + 1600, warm + 400
    t = np.arange(n, dtype=float)
    drift = np.cumsum(np.where(t < warm, 0.0, np.where(t < turn, -0.0001, 0.0001)))
    jumps = 0.004 * np.floor(np.maximum(t - turn, 0.0) / 97.0)
    close = 1.10 * np.exp(drift + 0.001 * np.sin(2 * np.pi * t / 40.0) + jumps)
    base = m15_from_hourly_close(close, "2021-03-01")
    mk = _market({"EURUSD": base}, rates=FIXED_RATES)   # EUR carries more than USD: carry rules go long
    closes = _decision_times(mk)
    p = S.DEFAULT_PARAMS
    plan = [S.Segment(int(closes[50]), int(closes[-1]), p)]
    pre = mk.run(S.PortfolioTrendStrategy(plan, mk.pairs), plan[0].start_ns, plan[-1].end_ns)
    mk.dispose()
    held = _positions_by_bar(pre, closes)
    eq = pre.equity.reindex(pd.to_datetime(closes, utc=True)).ffill().fillna(B.STARTING_NAV).to_numpy()
    near_peak = eq >= np.maximum.accumulate(np.maximum(eq, B.STARTING_NAV)) * 0.98
    # a position in bar k-1 still held unchanged through the open of bar k (the gap bar), with the account
    # within 2 % of its peak so that a 5 % loss trips the daily halt and not the 10 % kill switch
    holds = [j for j in range(warm, len(closes) - 300)
             if held[j - 1] != 0 and held[j] == held[j - 1] and near_peak[j - 1]]
    if not holds:
        return {"passed": False, "reason": "scenario never held a position"}
    k = holds[0]
    side = 1.0 if held[k - 1] > 0 else -1.0
    eq_prev = float(pre.equity[pre.equity.index.as_unit("ns").asi8 <= closes[k - 1]].iloc[-1])
    lev = abs(held[k - 1]) * mk.h1["EURUSD"]["close"].iloc[k - 1] / eq_prev
    if lev < 0.3:
        return {"passed": False, "reason": f"leverage {lev:.3f} too small for a clean gap scenario"}
    gap_start = mk.index[k]

    def gapped(loss: float) -> B.RunResult:
        d = base.copy()
        f = 1.0 - side * loss / lev                    # against the position
        rows = d.index >= gap_start
        for c in ("open", "high", "low", "close"):
            d.loc[rows, c] = d.loc[rows, c] * f
        m = _market({"EURUSD": d}, rates=FIXED_RATES)
        r = m.run(S.PortfolioTrendStrategy(plan, m.pairs), plan[0].start_ns, plan[-1].end_ns)
        m.dispose()
        return r

    t_k = int(closes[k])               # the close of the gap bar: breaker fires here, acts at the next open
    halt = gapped(0.05)
    o = halt.orders
    after = o[(o["ts"] > t_k) & (o["ts"] <= t_k + 24 * HOUR)]
    halt_ok = (halt.n_halts >= 1 and not ((o["ts"] == t_k) & (o["role"] == "EXIT") & (o["reason"] == "kill")).any()
               and ((o["ts"] == t_k) & (o["reason"] == "halt")).any() and len(after) == 0
               and (o[o["ts"] > t_k + 24 * HOUR]["role"] == "ENTRY").any())
    kill = gapped(0.15)
    ko = kill.orders
    kill_ok = kill.killed and ((ko["ts"] == t_k) & (ko["reason"] == "kill")).any() and not (ko["ts"] > t_k).any()
    return {"passed": bool(halt_ok and kill_ok), "gap_bar": str(gap_start), "position_side": int(side),
            "leverage_before_gap": round(float(lev), 3),
            "daily_halt_ok": bool(halt_ok), "max_dd_kill_ok": bool(kill_ok)}


def edge_case_tests() -> dict:
    p, lim = S.DEFAULT_PARAMS, S.RiskLimits()
    res: dict[str, dict] = {}
    cases = [(1e6, 0.0, 1.1, 1.0), (1e6, -1.0, 1.1, 1.0), (1e6, math.nan, 1.1, 1.0), (1e6, math.inf, 1.1, 1.0),
             (0.0, 0.001, 1.1, 1.0), (-5.0, 0.001, 1.1, 1.0), (math.nan, 0.001, 1.1, 1.0), (1e6, 0.001, 0.0, 1.0),
             (1e6, 0.001, math.nan, 1.0), (1e6, 0.001, 1.1, None), (1e6, 0.001, 1.1, 0.0), (1e6, 1e-300, 1.1, 1.0)]
    sizes = [S.RiskManager.position_size(n, a, px, q, p, lim) for n, a, px, q in cases]
    ok = (all(isinstance(s, int) and s >= 0 for s in sizes) and all(s == 0 for s in sizes[:-1])
          and sizes[-1] * 1.1 / 1e6 <= lim.max_leverage and sizes[-1] % lim.lot == 0)
    res["zero_division_sizing"] = {"passed": bool(ok), "sizes": sizes}

    flat = synthetic_m15("EURUSD", days=40, ann_vol=0.0, price=1.1)
    mk = _market({"EURUSD": flat})
    plan = _plan_over(mk, p)
    r = mk.run(S.PortfolioTrendStrategy(plan, mk.pairs), plan[0].start_ns, plan[-1].end_ns)
    mk.dispose()
    res["flat_prices_zero_atr"] = {"passed": bool(len(r.orders) == 0 and (r.equity == B.STARTING_NAV).all())}

    syn = synthetic_m15("EURUSD", days=_syn_days(), seed=3)
    rng = np.random.default_rng(3)
    dirty = syn.drop(syn.index[rng.random(len(syn)) < 0.05]).copy()
    bad = rng.choice(len(dirty), 60, replace=False)
    dirty.iloc[bad[:30], dirty.columns.get_loc("close")] = np.nan
    dirty.iloc[bad[30:40], dirty.columns.get_loc("open")] = -1.0
    dirty.iloc[bad[40:50], dirty.columns.get_loc("high")] = dirty["low"].to_numpy()[bad[40:50]] * 0.99
    dirty.iloc[bad[50:], dirty.columns.get_loc("spread")] = np.nan
    dirty = pd.concat([dirty, dirty.iloc[200:205]])
    clean = B.clean_m15(dirty, "EURUSD")
    par = parity({"EURUSD": dirty})
    res["missing_and_corrupt_bars"] = {"passed": bool(par["passed"] and len(clean) < len(syn) and
                                                      clean.index.is_monotonic_increasing and not clean.index.has_duplicates),
                                       "rows_in": int(len(dirty)), "rows_clean": int(len(clean)), "parity": par}

    spk = synthetic_m15("EURUSD", days=_syn_days(3000), seed=4)
    k = len(spk) - 1500 * 4                        # the spike comes after the warm-up, 1,500 hours before the end
    for c in ("open", "high", "low", "close"):
        spk.iloc[k:, spk.columns.get_loc(c)] = spk[c].to_numpy()[k:] * 1.08
    spk.iloc[k, spk.columns.get_loc("high")] *= 1.05
    spk.iloc[k, spk.columns.get_loc("low")] *= 0.95
    mk = _market({"EURUSD": spk})
    plan = _plan_over(mk, p)
    r = mk.run(S.PortfolioTrendStrategy(plan, mk.pairs), plan[0].start_ns, plan[-1].end_ns)
    closes = _decision_times(mk)
    held = _positions_by_bar(r, closes)
    eq = r.equity.reindex(pd.to_datetime(closes, utc=True)).ffill().fillna(B.STARTING_NAV).to_numpy()
    lev = np.abs(held) * mk.h1["EURUSD"]["close"].to_numpy() / eq
    par = parity(mk)
    res["volatility_spike"] = {"passed": bool(np.isfinite(r.equity.to_numpy()).all() and lev.max() <= lim.max_leverage * 1.2
                                              and par["passed"]), "max_leverage_seen": round(float(lev.max()), 3),
                               "parity": par}
    seg_frames = {"EURUSD": synthetic_m15("EURUSD", days=_syn_days(3000), seed=9)}
    alt = dataclasses.replace(p, **{k: v[0] for k, v in S.PARAM_GRID.items()})

    def three_segments(mk: B.Market) -> list[S.Segment]:
        c = _decision_times(mk)
        w = S.warmup_bars(p)
        a, b, d, e = (int(c[k]) for k in (100, w + (len(c) - w) // 3, w + 2 * (len(c) - w) // 3, len(c) - 1))
        return [S.Segment(a, b, p), S.Segment(b, d, alt), S.Segment(d, e, p)]

    res["parity_segmented"] = parity(seg_frames, plan_fn=three_segments)
    act = alt                                       # the most active corner of the grid, volatility regimes
    act_frames = {"EURUSD": synthetic_m15("EURUSD", days=_syn_days(6000, act), seed=21, vol_regimes=True)}
    res["parity_active"] = parity(act_frames, plan_fn=lambda mk: _plan_over(mk, act))
    # data starting long before the run's warm-up start: recursive indicators must be seeded at the same bar
    w = S.warmup_bars(p)
    late_frames = {"EURUSD": synthetic_m15("EURUSD", days=_syn_days(w + 3000), seed=31, vol_regimes=True)}
    res["parity_late_start"] = parity(late_frames, plan_fn=lambda mk: [
        S.Segment(int(_decision_times(mk)[int(1.6 * w) + 200]), int(_decision_times(mk)[-1]), p)])
    res["fill_timing_multi_pair"] = fill_timing_test()
    roles = set()
    for r in (res["missing_and_corrupt_bars"]["parity"], res["volatility_spike"]["parity"], res["parity_segmented"],
              res["parity_active"], res["parity_late_start"]):
        roles |= set(r.get("fills_by_role", {}))
    need = {"ENTRY"} | ({"REBAL"} if "REBAL" in res["fill_timing_multi_pair"]["roles"] else set())
    res["parity_coverage"] = {"passed": bool(need <= roles and roles & {"FLIP", "EXIT"}),
                              "roles_exercised": sorted(roles), "required": sorted(need) + ["FLIP|EXIT"]}
    res["circuit_breakers"] = circuit_breaker_test()
    res["friction"] = friction_tests()
    res["financing"] = financing_tests()
    res["execution_rule"] = execution_rule_tests()
    res["rolling_splits"] = rolling_split_tests()
    res["signal_parity_multi_pair"] = signal_parity_multi()
    return res


# ---------------------------------------------------------------- look-ahead tests
def _perturb_frames(frames: dict[str, pd.DataFrame], cut: pd.Timestamp, rng) -> dict[str, pd.DataFrame]:
    """Scramble every M15 bar at or after `cut` in every pair (same factor per row keeps OHLC consistent)."""
    out = {}
    for p, f in frames.items():
        d = f.copy()
        rows = d.index >= cut
        n = int(rows.sum())
        fac = np.exp(np.cumsum(rng.normal(0.0, 0.003, n))) * rng.choice([0.97, 1.0, 1.03], n)
        for c in ("open", "high", "low", "close"):
            d.loc[rows, c] = d.loc[rows, c].to_numpy() * fac
        d.loc[rows, "high"] = d.loc[rows, "high"].to_numpy() * np.where(rng.random(n) < 0.05, 1.02, 1.0)
        d.loc[rows, "spread_pips"] = rng.uniform(1.0, 8.0, n)
        out[p] = d
    return out


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.shape == b.shape and all(
        np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True) for c in a.columns)


def signal_leak_violations(frame_fn, h1s: dict[str, pd.DataFrame], p, points, strict: bool,
                           seed: int = 11) -> list[int]:
    """Vectorised signal frames (one per pair, possibly cross-sectional) must not change up to bar t when
    every pair's bars from t+1 (strict: from t) on are scrambled."""
    rng = np.random.default_rng(seed)
    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in h1s.values()))))
    base = frame_fn(h1s, p)
    bad = []
    for t in points:
        cut = index[t] if strict else index[t + 1]
        alt_in = {}
        for k, f in h1s.items():
            d = f.copy()
            rows = d.index >= cut
            n = int(rows.sum())
            fac = np.exp(np.cumsum(rng.normal(0, 0.004, n))) * rng.choice([0.9, 1.0, 1.1], n)
            for c in ("open", "high", "low", "close"):
                d.loc[rows, c] = d.loc[rows, c].to_numpy() * fac
            alt_in[k] = d
        alt = frame_fn(alt_in, p)
        if not all(_frames_equal(base[k].loc[: index[t]], alt[k].loc[: index[t]]) for k in h1s):
            bad.append(int(t))
    return bad


def nautilus_leak_violations(frames: dict[str, pd.DataFrame], p: S.StrategyParams, points: list[int],
                             mid_bar: bool, seed: int = 13, rates: B.RateTable | None = SYN_RATES) -> list[int]:
    """Every decision (order) and equity mark stamped at or before a cut time c must not move when all
    M15 data starting at or after c is scrambled.

    mid_bar=False: c = the open of bar t+1 (= the close of bar t)
    mid_bar=True : c = 15/30/45 minutes into hourly bar t - catches a decision that is stamped early but
                   secretly uses the rest of the hour (e.g. a bar published at its open instead of its close)
    """
    rng = np.random.default_rng(seed)
    mk = B.Market(frames, rates=rates)
    closes = _decision_times(mk)
    plan = [S.Segment(int(closes[100]), int(closes[-1]), p)]
    base = mk.run(S.PortfolioTrendStrategy(plan, mk.pairs), plan[0].start_ns, plan[-1].end_ns)
    opens = mk.index
    mk.dispose()
    if len(base.orders) == 0:
        return [-1]                                   # nothing traded: the test would be vacuous
    bad = []
    for t in points:
        cut = opens[t] + pd.Timedelta(minutes=int(rng.choice([15, 30, 45]))) if mid_bar else opens[t + 1]
        upto = int(cut.value)
        m2 = B.Market(_perturb_frames(frames, cut, rng), rates=rates)
        alt = m2.run(S.PortfolioTrendStrategy(plan, m2.pairs), plan[0].start_ns, plan[-1].end_ns)
        m2.dispose()
        o1 = base.orders[base.orders["ts"] <= upto].reset_index(drop=True)
        o2 = alt.orders[alt.orders["ts"] <= upto].reset_index(drop=True)
        e1 = base.equity[base.equity.index.as_unit("ns").asi8 <= upto]
        e2 = alt.equity[alt.equity.index.as_unit("ns").asi8 <= upto]
        if not (o1.equals(o2) and e1.equals(e2)):
            bad.append(int(t))
    return bad


def run_leak_suite(frames: dict[str, pd.DataFrame], p: S.StrategyParams, n_points: int = 6, seed: int = 5,
                   rates: B.RateTable | None = SYN_RATES) -> dict:
    mk = B.Market(frames, rates=rates)
    h1s = mk.h1
    n = len(mk.index)
    mk.dispose()
    rng = np.random.default_rng(seed)
    points = sorted(int(x) for x in rng.integers(max(130, S.warmup_bars(p) + 5), n - 5, n_points))
    out = {
        "points": points,
        "signals_perturb_t+1": signal_leak_violations(S.signal_frame, h1s, p, points, strict=False),
        "signals_perturb_t": signal_leak_violations(S.signal_frame, h1s, p, points, strict=True),
        "nautilus_perturb_from_bar_boundary": nautilus_leak_violations(frames, p, points, mid_bar=False, rates=rates),
        "nautilus_perturb_from_mid_bar": nautilus_leak_violations(frames, p, points, mid_bar=True, rates=rates),
    }
    future_leak = lambda d, q: {k: v.shift(-2) for k, v in S.signal_frame(d, q).items()}  # row t reads t+1
    same_bar_leak = lambda d, q: S.compute_features(d, q)                                 # row t reads bar t
    out["canary_future_detected"] = len(signal_leak_violations(future_leak, h1s, p, points, strict=False)) > 0
    out["canary_same_bar_detected"] = len(signal_leak_violations(same_bar_leak, h1s, p, points, strict=True)) > 0
    out["passed"] = (all(len(v) == 0 for k, v in out.items() if "perturb" in k)
                     and out["canary_future_detected"] and out["canary_same_bar_detected"])
    return out


def rate_lag_tests(frames: dict[str, pd.DataFrame], rates: B.RateTable, p: S.StrategyParams, n_cuts: int = 4,
                   seed: int = 17) -> dict:
    """Rates enter signals only once public. For cut times X (month starts): scrambling every rate value that
    becomes known at or after X must leave (1) the known-rate series of every pair and (2) every signal row
    before X unchanged. Canaries: the same with monthly averages treated as known during their own month
    (a same-month look-ahead) must be caught - by the data layer always, by the signals if the hypothesis
    uses carry."""
    rng = np.random.default_rng(seed)
    mk = B.Market(frames, rates=rates)
    h1s = {k: f.drop(columns="carry") for k, f in mk.h1.items()}
    grid = mk.index.as_unit("ns").asi8 + HOUR
    mk.dispose()
    months = pd.date_range(mk.index[0].normalize() + pd.offsets.MonthBegin(2), mk.index[-1], freq="MS")
    monthly_era = [m for m in months if rates.switch is None or m < rates.switch]
    picks = rng.choice(len(monthly_era), min(n_cuts, len(monthly_era)), replace=False)
    cuts = sorted({monthly_era[int(i)] for i in picks} |
                  set(months[months >= rates.switch][:1] if rates.switch is not None else []))

    def with_carry(table: B.RateTable) -> dict[str, pd.DataFrame]:
        return {k: f.assign(carry=table.known_diff(k, f.index.as_unit("ns").asi8 + HOUR)) for k, f in h1s.items()}

    def signals_before(table: B.RateTable, cut: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {k: f.loc[f.index < cut] for k, f in S.signal_frame(with_carry(table), p).items()}

    known_bad, signal_bad, canary_known, canary_signal = [], [], [], []
    base_signals = {}
    for cut in cuts:
        alt = rates.perturbed(cut, np.random.default_rng(int(rng.integers(1 << 30))))
        before = grid < cut.value
        same = all(np.array_equal(rates.known_diff(k, grid[before]), alt.known_diff(k, grid[before]), equal_nan=True)
                   for k in frames)
        if not same:
            known_bad.append(str(cut))
        leaky, leaky_alt = rates.unlagged(), alt.unlagged()
        canary_known.append(not all(np.array_equal(leaky.known_diff(k, grid[before]), leaky_alt.known_diff(k, grid[before]),
                                                   equal_nan=True) for k in frames))
        base_signals = signals_before(rates, cut)
        alt_signals = signals_before(alt, cut)
        if not all(_frames_equal(base_signals[k], alt_signals[k]) for k in frames):
            signal_bad.append(str(cut))
        if S.USES_CARRY:
            a, b = signals_before(leaky, cut), signals_before(leaky_alt, cut)
            canary_signal.append(not all(_frames_equal(a[k], b[k]) for k in frames))
    in_monthly = [c for c, cut in zip(canary_known, cuts) if rates.switch is None or cut < rates.switch]
    # a policy decision effective on day D is public only from D + 1 (00:00 UTC): in the policy era the rate known
    # at t must be the rate in force at t - 1 day (checked at midday of every decision day and of the day after)
    policy_ok = True
    if rates.switch is not None:
        day = 24 * HOUR
        for c, ev in rates.events.items():
            ev = ev[ev.index > rates.switch + pd.Timedelta(days=1)]
            t = np.concatenate([ev.index.asi8 + 12 * HOUR, ev.index.asi8 + day + 12 * HOUR])
            policy_ok &= bool(np.array_equal(rates.known(c, t), rates.accrual(c, t - day), equal_nan=True))
    out = {"cuts": [str(c.date()) for c in cuts], "policy_decisions_known_next_day": bool(policy_ok),
           "known_rates_changed_before_cut": known_bad,
           "signals_changed_before_cut": signal_bad, "canary_known_detected": bool(in_monthly and all(in_monthly)),
           "uses_carry": bool(S.USES_CARRY)}
    if S.USES_CARRY:
        in_monthly_s = [c for c, cut in zip(canary_signal, cuts) if rates.switch is None or cut < rates.switch]
        out["canary_signal_detected"] = bool(in_monthly_s and all(in_monthly_s))
    out["passed"] = bool(not known_bad and not signal_bad and out["canary_known_detected"] and policy_ok
                         and out.get("canary_signal_detected", True))
    return out


def complexity_audit() -> dict:
    fields = [f.name for f in dataclasses.fields(S.StrategyParams)]
    src = "".join(f.read_text() for f in [H.path(HYP_ID), H.BASE_FILE] + H.depends(HYP_ID))
    hits = sorted({m.group(0) for m in CALENDAR_PATTERN.finditer(src)})
    grid_ok = set(S.PARAM_GRID) <= set(fields) and set(S.FEATURE_PARAMS) <= set(fields)
    overrides = [a for a in ("WFA_MAX_OOS_DAYS", "WFA_MIN_WFE") if HYP_ID in H.CYCLE5 and hasattr(S, a)]
    ok = (len(fields) <= MAX_TUNABLE_PARAMS and len(S.INDICATORS) <= MAX_INDICATORS and not hits and grid_ok
          and not overrides)
    return {"passed": bool(ok), "tunable_parameters": fields, "indicators": list(S.INDICATORS),
            "optimised_in_wfa": list(S.PARAM_GRID), "calendar_filter_hits": hits,
            "protocol_overrides": overrides}


# =============================================================================
# Trials and holdout logs
# =============================================================================
def code_fingerprint(hid: str) -> str:
    h = hashlib.sha256()
    for f in [ROOT / name for name in CODE_FILES] + [H.path(hid)] + H.depends(hid):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


_ENTRY = re.compile(r"^=== (TRIAL|HOLDOUT) (\d+) \| (\S+) \| hyp ([A-Z][\w-]*) \| code (\w+) \| (PASS|FAIL)")


def _entries(path: Path, kind: str) -> list[dict]:
    if not path.exists():
        return []
    return [{"n": int(m.group(2)), "hyp": m.group(4), "code": m.group(5), "status": m.group(6)}
            for line in path.read_text().splitlines() if (m := _ENTRY.match(line)) and m.group(1) == kind]


def logged_trials() -> list[dict]:
    return _entries(TRIALS_LOG, "TRIAL")


def logged_holdouts() -> list[dict]:
    return _entries(HOLDOUT_LOG, "HOLDOUT")


def append_entry(path: Path, kind: str, n: int, fp: str, report: dict) -> None:
    o = report["metrics"]
    lines = [
        f"=== {kind} {n} | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | hyp {HYP_ID} | code {fp} | "
        f"{report['status']} ===",
        f"hypothesis : {S.HYPOTHESIS}",
        f"grid       : {json.dumps(S.PARAM_GRID)}  fixed: "
        f"{json.dumps({k: v for k, v in dataclasses.asdict(S.DEFAULT_PARAMS).items() if k not in S.PARAM_GRID})}",
        f"data       : {report['period']} | {', '.join(f'{p} {h}' for p, h in report['provenance'].items())}",
        f"metrics    : oos_sharpe={o['oos_sharpe']} is_sharpe_mean={o['is_sharpe_mean']} wfe={o['wfe']} "
        f"oos_max_dd={o['oos_max_drawdown']} oos_trades_per_year={o['oos_trades_per_year']} "
        f"oos_total_return={o['oos_total_return']} leak_test={'PASS' if o['leak_test_passed'] else 'FAIL'}",
    ]
    if o.get("oos_deflated_sharpe"):
        ds = o["oos_deflated_sharpe"]
        lines.append(f"deflated   : dsr={ds.get('dsr')} n_trials={ds.get('n_trials')} "
                     f"benchmark_sharpe={ds.get('sharpe_benchmark_annual')} psr_vs_zero={ds.get('psr_vs_zero')}")
    for r in report["splits"]:
        lines.append(f"split {r['split']}    : OOS {r['oos_period'][0][:10]}..{r['oos_period'][1][:10]} "
                     f"params={ {k: r['chosen_params'][k] for k in S.PARAM_GRID} } IS_sh={r['is']['sharpe']} "
                     f"OOS_sh={r['oos']['sharpe']} OOS_dd={r['oos']['max_drawdown']} "
                     f"OOS_tr/yr={r['oos']['trades_per_year']}")
    lines += [f"failure    : {f}" for f in report["failures"]]
    with path.open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")


# =============================================================================
# Main
# =============================================================================
def protocol_tests() -> dict:
    """Cycle-5 protocol: deflated Sharpe behaviour, the trial guard, and gate (d) strictness."""
    out = {}
    rng = np.random.default_rng(3)
    # E[max of N standard normals]: the formula vs Monte Carlo
    mc = float(np.mean(rng.standard_normal((4000, 100)).max(axis=1)))
    em = expected_max_sharpe(100, 1.0)
    r = rng.normal(0.0008, 0.01, 2520)                          # ~1.27 annual Sharpe, 10 years
    sr_ann = r.mean() / r.std(ddof=1) * math.sqrt(252)
    one = deflated_sharpe(r, [sr_ann])
    few = deflated_sharpe(r, [sr_ann, 0.3, -0.2, 0.1])
    many = deflated_sharpe(r, [sr_ann] + list(rng.normal(0, 0.3, 99)))
    wide = deflated_sharpe(r, [sr_ann] + list(rng.normal(0, 0.9, 99)))
    at_bench = deflated_sharpe(np.concatenate([r, -r]) + r.mean(), [sr_ann])   # zero skew case
    out["deflated_sharpe"] = {
        "passed": bool(abs(em - mc) / mc < 0.03 and one["dsr"] == one["psr_vs_zero"]
                       and one["dsr"] >= few["dsr"] >= many["dsr"] >= wide["dsr"] and one["dsr"] > 0.99
                       and at_bench["dsr"] is not None),
        "expected_max_formula_vs_mc": [round(em, 3), round(mc, 3)],
        "dsr_n1_n4_n100_n100wide": [one["dsr"], few["dsr"], many["dsr"], wide["dsr"]]}
    logged = [{"n": 1, "hyp": "C5-001", "code": "aaa", "status": "FAIL"}]
    act = ("C5-001", "C5-002")
    cases = {"new_id": trial_guard("C5-002", "bbb", logged, act) is None,
             "reproduce_same_code": trial_guard("C5-001", "aaa", logged, act) is None,
             "changed_code_same_id": trial_guard("C5-001", "ccc", logged, act) is not None,
             "archived_id": trial_guard("A3", "ddd", logged, act) is not None}
    out["trial_guard"] = {"passed": all(cases.values()), **cases}
    base = {"oos": {"trades_per_year": 500, "sharpe": 2.0, "max_drawdown": 0.05}, "is_sharpe_mean": 4.0}
    g = {w: _gate_failures({**base, "wfe": w}) for w in (0.5, 0.51, None)}
    out["gate_d_strict"] = {"passed": bool(g[0.5] and not g[0.51] and g[None]),
                            "wfe_0.50": g[0.5], "wfe_0.51": g[0.51]}
    return out


def selftest() -> dict:
    days = _syn_days(1500)
    syn = {"EURUSD": B.clean_m15(synthetic_m15("EURUSD", days=days, seed=1), "EURUSD"),
           "USDJPY": B.clean_m15(synthetic_m15("USDJPY", days=days, seed=2, price=110.0, spread_points=15.0), "USDJPY")}
    long_days = max(days, 420)                      # reaches SYN_RATES' policy era (2022-01) for every hypothesis
    syn_long = {"EURUSD": B.clean_m15(synthetic_m15("EURUSD", days=long_days, seed=1), "EURUSD"),
                "USDJPY": B.clean_m15(synthetic_m15("USDJPY", days=long_days, seed=2, price=110.0, spread_points=15.0),
                                      "USDJPY")}
    return {"complexity": complexity_audit(), "protocol": protocol_tests(), "edge_cases": edge_case_tests(),
            "leak_synthetic": run_leak_suite(syn, S.DEFAULT_PARAMS, n_points=4),
            "rate_lag_synthetic": rate_lag_tests(syn_long, SYN_RATES, S.DEFAULT_PARAMS)}


def _all_passed(d) -> bool:
    if isinstance(d, dict):
        if d.get("passed") is False:
            return False
        return all(_all_passed(v) for v in d.values() if isinstance(v, dict))
    return True


def _failed_names(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            if v.get("passed") is False:
                out.append(prefix + k)
            out += _failed_names(v, prefix + k + ".")
    return out


def _gate_failures(wfa: dict) -> list[str]:
    o, out = wfa["oos"], []
    if not o["trades_per_year"] > MIN_TRADES_PER_YEAR:
        out.append(f"(a) OOS fills/year {o['trades_per_year']} <= {MIN_TRADES_PER_YEAR:g}")
    if not o["sharpe"] >= MIN_OOS_SHARPE:
        out.append(f"(b) OOS Sharpe {o['sharpe']} < {MIN_OOS_SHARPE}")
    if not o["max_drawdown"] < MAX_OOS_DRAWDOWN:
        out.append(f"(c) OOS max drawdown {o['max_drawdown']:.2%} >= {MAX_OOS_DRAWDOWN:.0%}")
    if wfa["wfe"] is None:
        out.append(f"(d) WFE undefined: mean IS Sharpe {wfa['is_sharpe_mean']} <= 0")
    elif not wfa["wfe"] > min_wfe():
        out.append(f"(d) WFE {wfa['wfe']} <= {min_wfe():.4g} (OOS Sharpe {o['sharpe']} / IS Sharpe "
                   f"{wfa['is_sharpe_mean']})")
    return out


def _report(stage: int, status: str, failures: list[str], trial: dict, provenance: dict, data_info: dict,
            period: str, wfa: dict, tests: dict, leak_passed: bool, t_start: float) -> dict:
    o = wfa["oos"]
    A = __import__("hypotheses.academic_base", fromlist=["x"])
    return {
        "stage": stage, "status": status, "failures": failures, "trial": trial, "hypothesis_id": HYP_ID,
        "hypothesis": S.HYPOTHESIS, "period": period,
        "engine": {"nautilus_trader": __import__("nautilus_trader").__version__, "venue": "SIM NETTING MARGIN USD",
                   "latency_ns": 1, "fees": 0, "min_spread_pips": B.MIN_SPREAD_PIPS,
                   "slippage_pips_per_fill": B.SLIPPAGE_PIPS, "financing_markup": B.FIN_MARKUP,
                   "risk_pct": A.RISK_PCT, "atr_mult": A.ATR_MULT, "atr_bars": A.ATR_BARS},
        "provenance": provenance, "data": data_info,
        "config": {"param_grid": S.PARAM_GRID, "defaults": dataclasses.asdict(S.DEFAULT_PARAMS),
                   "limits": dataclasses.asdict(S.RiskLimits()), "n_splits": len(wfa["splits"]),
                   "is_fraction": IS_FRACTION, "wfa_max_oos_days": getattr(S, "WFA_MAX_OOS_DAYS", None),
                   "min_wfe": min_wfe(),
                   "burn_in_months": 18, "min_data_years": MIN_DATA_YEARS, "dev_end": str(DEV_END)},
        "metrics": {
            "oos_sharpe": o["sharpe"], "is_sharpe_mean": wfa["is_sharpe_mean"], "wfe": wfa["wfe"],
            "oos_max_drawdown": o["max_drawdown"], "oos_trades_per_year": o["trades_per_year"],
            "oos_fills": o.get("fills"), "oos_fills_by_role": o.get("fills_by_role"),
            "oos_round_trips_per_year": o.get("round_trips_per_year"),
            "oos_total_return": o["total_return"], "oos_cagr": o.get("cagr"), "oos_ann_vol": o.get("ann_vol"),
            "oos_financing_usd": o.get("financing_usd"), "oos_spread_slippage_usd": o.get("cost_usd_spread_slippage"),
            "oos_win_rate": o.get("win_rate"), "oos_avg_trade_pips_net": o.get("avg_trade_pips_net"),
            "oos_years": o["years"], "oos_halts_daily_loss": wfa["oos_result"].n_halts,
            "oos_killed_max_dd": wfa["oos_result"].killed, "oos_per_pair": o.get("per_pair"),
            "leak_test_passed": leak_passed,
        },
        "splits": wfa["splits"], "tests": tests, "runtime_seconds": round(time.time() - t_start, 1),
    }


def _pool(pairs, data_dir: Path, end: pd.Timestamp | None, official: bool) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                               initargs=(pairs, str(data_dir), str(end) if end is not None else None, HYP_ID,
                                         official))


def _finish(report: dict, name: str, wfa: dict, market: B.Market) -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_DIR / f"{name}_equity.csv").write_text(wfa["oos_result"].equity.rename("equity").to_csv())
    slim = {**report, "splits": [{k: v for k, v in r.items() if k != "is_grid"} for r in report["splits"]]}
    print(json.dumps(slim, indent=2, default=str))
    market.dispose()
    if report["failures"]:
        print(f"\nHARNESS FAIL ({name}):", file=sys.stderr)
        for f in report["failures"]:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\nHARNESS PASS ({name})", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------- real-data checks
def splice_checks(data_dir: Path = B.DATA_DIR, overlap_end: pd.Timestamp = pd.Timestamp("2020-06-01", tz="UTC"),
                  max_median_pips: float = 1.0) -> dict:
    """OANDA vs MT5 on their overlap (the first MT5 bar -> 2020-05): hourly MID closes must agree to a median
    |diff| < 1 pip per pair (USD/JPY derived from EUR/JPY and EUR/USD); also the gap at the splice point, the
    spread profile's development-only window, and the rate-table hand-over (per currency, the median monthly
    |OECD 3-month - policy rate| over 2020-01..06 must be <= 0.3 pp)."""
    out, ok = {}, True
    mt5 = B.load_universe(B.PAIRS, data_dir, end=overlap_end)
    splice = max(f.index[0] for f in mt5.values())
    dev = B.load_universe(B.PAIRS, data_dir, end=DEV_END)
    start = splice - pd.Timedelta(days=30)
    mins = {q: B.load_oanda_minutes(B._OANDA_NAME[q], overlap_end, start=start)
            for q in ("EURUSD", "GBPUSD", "USDCAD", "EURJPY")}
    mins["USDJPY"] = B.derive_cross(mins["EURJPY"], mins["EURUSD"])
    for p in B.PAIRS:
        prof = B.mt5_spread_profile(dev[p])
        o = B.hourly_frame(B.minutes_to_m15(mins[p][mins[p].index >= splice], p, prof), p)
        m = B.hourly_frame(mt5[p], p)
        j = o[["close", "high", "low"]].join(m[["close", "high", "low"]], lsuffix="_o", rsuffix="_m", how="inner")
        d = (j["close_o"] - j["close_m"]) / B.pip_size(p)
        shifted = ((j["close_o"].shift(1) - j["close_m"]) / B.pip_size(p)).abs().median()
        med = float(d.abs().median())
        before = mins[p][mins[p].index < splice]
        gap_h = (splice - before.index[-1]).total_seconds() / 3600 if len(before) else math.inf
        row_ok = med < max_median_pips and len(j) > 1000 and gap_h <= 72 and shifted > 3 * med
        ok &= row_ok
        out[p] = {"passed": bool(row_ok), "overlap_hours": int(len(j)), "median_abs_diff_pips": round(med, 3),
                  "p95_abs_diff_pips": round(float(d.abs().quantile(0.95)), 3), "mean_diff_pips": round(float(d.mean()), 3),
                  "median_abs_diff_one_hour_shifted_pips": round(float(shifted), 3),
                  "median_range_ratio_oanda_vs_mt5": round(float(((j["high_o"] - j["low_o"]) /
                                                                 (j["high_m"] - j["low_m"])).median()), 3),
                  "gap_at_splice_hours": round(gap_h, 2),
                  "spread_profile_pips_by_utc_hour": [round(float(x), 2) for x in prof]}
    # the spread profile must ignore everything from PROFILE_END on
    f = dev["EURUSD"].copy()
    later = f.iloc[-500:].copy()
    later.index = later.index + pd.DateOffset(years=2)
    later["spread_pips"] = 99.0
    same = B.mt5_spread_profile(pd.concat([f, later])).equals(B.mt5_spread_profile(f))
    out["spread_profile_development_only"] = {"passed": bool(same)}
    rates = B.RateTable.load()
    months = pd.date_range("2020-01-01", "2020-06-01", freq="MS", tz="UTC")
    diffs = {}
    for c in ("USD", "EUR", "GBP", "JPY", "CAD"):
        oecd = rates.monthly[c].reindex(months)
        ev = rates.events[c]
        pol = []
        for m in months:
            days = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D", tz="UTC")
            k = np.searchsorted(ev.index.asi8, days.asi8, side="right") - 1
            pol.append(float(np.mean(ev.to_numpy()[k])))
        diffs[c] = [None if math.isnan(a) else round(float(a - b), 3) for a, b in zip(oecd.to_numpy(), pol)]
    # a transcription error is a persistent level gap; single stress months (March-April 2020 LIBOR-OIS) are not
    medians = {c: float(np.median([abs(x) for x in v if x is not None])) for c, v in diffs.items()}
    rates_ok = max(medians.values()) <= 0.3
    out["policy_vs_oecd_2020H1"] = {"passed": bool(rates_ok), "oecd_minus_policy_pp": diffs,
                                    "median_abs_pp": {c: round(m, 3) for c, m in medians.items()},
                                    "max_abs_pp": max(abs(x) for v in diffs.values() for x in v if x is not None)}
    out["passed"] = bool(ok and same and rates_ok)
    return out


def run_stage1(data_dir: Path, official: bool, t_start: float) -> int:
    fp = code_fingerprint(HYP_ID)
    trials = logged_trials()
    rerun = next((t for t in trials if t["code"] == fp and t["hyp"] == HYP_ID), None)
    if official:
        refusal = trial_guard(HYP_ID, fp, trials)
        if refusal:
            print(f"REFUSED: {refusal}", file=sys.stderr)
            return 1

    failures: list[str] = []
    m15, rates = load_market(B.PAIRS, data_dir, DEV_END, official)
    provenance = data_provenance(data_dir=data_dir, m15=m15)
    data_errors, data_info = validate_data(m15)
    failures += [f"data: {e}" for e in data_errors]
    market = B.Market(m15, rates=rates)

    tests = selftest()
    rate_rows_after = sum(int((s.index >= DEV_END).sum()) for d in (rates.monthly, rates.events) for s in d.values())
    tests["holdout_locked"] = {"passed": bool(market.index.max() < DEV_END and rate_rows_after == 0 and
                                              all(f.index.max() < DEV_END for f in m15.values())),
                               "last_bar_loaded": str(market.index.max()), "rate_rows_on_or_after_dev_end": rate_rows_after}
    if official:
        tests["splice"] = splice_checks(data_dir)
    with _pool(market.pairs, data_dir, DEV_END, official) as pool:
        wfa = run_wfa(market, pool, wfa_splits(market.index))

    first = market.index[0]
    span = pd.Timedelta(hours=int(S.warmup_bars(wfa["chosen"][0]) * 1.5) + 24 * 120)
    leak_frames = {p: f.loc[: first + span] for p, f in m15.items()}
    tests["leak"] = run_leak_suite(leak_frames, wfa["chosen"][0], rates=rates)
    tests["rate_lag_real"] = rate_lag_tests(leak_frames, rates, wfa["chosen"][0])
    hand_over = {p: f.loc["2019-01-01":"2021-12-31"] for p, f in m15.items()}   # OECD -> policy rates, 2020-07
    tests["rate_lag_real_policy_handover"] = rate_lag_tests(hand_over, rates, wfa["chosen"][0])
    eur = B.Market({"EURUSD": m15["EURUSD"]}, rates=rates)
    tests["parity_real_eurusd"] = parity(eur, plan_fn=lambda mk: [
        S.Segment(sg.start_ns, sg.end_ns, sg.params) for sg in wfa["plan"]])
    tests["no_order_desyncs"] = {"passed": wfa["oos_desyncs"] == 0 and wfa["is_desyncs"] == 0,
                                 "oos": wfa["oos_desyncs"], "is": wfa["is_desyncs"]}

    failures += _gate_failures(wfa)
    if not tests["leak"]["passed"]:
        failures.append(f"(e) perturbation leak test failed: {json.dumps(tests['leak'])}")
    failures += [f"integrity: {n} failed" for n in _failed_names({k: v for k, v in tests.items() if k != "leak"})]
    trial = {"code_fingerprint": fp, "rerun_of_trial": rerun["n"] if rerun else None,
             "trial_number": (rerun["n"] if rerun else len(trials) + 1) if official else None,
             "budget": MAX_TRIALS, "counts_as_trial": official, "data_dir": str(data_dir)}
    period = (f"development {market.index[0]:%Y-%m-%d} -> {market.index[-1]:%Y-%m-%d} "
              f"(split geometry from {pd.Timestamp(wfa['splits'][0]['is_period'][0]):%Y-%m-%d})")
    report = _report(1, "PASS" if not failures else "FAIL", failures, trial, provenance, data_info, period, wfa,
                     tests, tests["leak"]["passed"], t_start)
    prior = all_trial_sharpes()
    report["metrics"]["oos_deflated_sharpe"] = deflated_sharpe(
        daily_returns(wfa["oos_result"].equity, wfa["oos_result"].start_nav),
        prior + ([] if rerun else [wfa["oos"]["sharpe"]]))
    if not official:
        failed = {n: v for n, v in tests.items() if isinstance(v, dict) and not _all_passed(v)}
        print(json.dumps({**{k: report[k] for k in ("status", "failures", "trial", "metrics")}, "failed_tests": failed},
                         indent=2, default=str))
        print(f"\nNOT A TRIAL: custom data dir {data_dir}", file=sys.stderr)
        market.dispose()
        return 0 if not failures else 1
    if rerun is None:
        append_entry(TRIALS_LOG, "TRIAL", len(trials) + 1, fp, report)
    return _finish(report, HYP_ID, wfa, market)


def run_stage2(data_dir: Path, official: bool, t_start: float) -> int:
    fp = code_fingerprint(HYP_ID)
    if HYP_ID not in H.CYCLE5:
        print(f"REFUSED: {HYP_ID} belongs to an archived cycle (reproduce at commit c447b91)", file=sys.stderr)
        return 1
    if not official:
        print("REFUSED: the holdout is only evaluated on the official data (data/mt5)", file=sys.stderr)
        return 1
    passed = [t for t in logged_trials() if t["hyp"] == HYP_ID and t["code"] == fp and t["status"] == "PASS"]
    if not passed:
        print(f"REFUSED: {HYP_ID} (code {fp}) has no logged Stage-1 PASS; the holdout stays locked", file=sys.stderr)
        return 1
    if any(h["code"] == fp for h in logged_holdouts()):
        print(f"REFUSED: the holdout was already used for {HYP_ID} (code {fp}); it is evaluated exactly once",
              file=sys.stderr)
        return 1

    failures: list[str] = []
    m15, rates = load_market(B.PAIRS, data_dir, None, official)
    provenance = data_provenance(data_dir=data_dir, m15=m15)
    data_errors, data_info = validate_data(m15)
    failures += [f"data: {e}" for e in data_errors]
    market = B.Market(m15, rates=rates)
    tests = selftest()
    splits = holdout_wfa_splits(market.index)
    with _pool(market.pairs, data_dir, None, official) as pool:
        wfa = run_wfa(market, pool, splits)
    tests["no_order_desyncs"] = {"passed": wfa["oos_desyncs"] == 0 and wfa["is_desyncs"] == 0,
                                 "oos": wfa["oos_desyncs"], "is": wfa["is_desyncs"]}
    failures += _gate_failures(wfa)
    failures += [f"integrity: {n} failed" for n in _failed_names(tests)]
    trial = {"code_fingerprint": fp, "stage1_trial": passed[-1]["n"], "counts_as_trial": True,
             "data_dir": str(data_dir)}
    period = f"holdout {pd.Timestamp(splits[0]['oos'][0], tz='UTC'):%Y-%m-%d} -> {market.index[-1]:%Y-%m-%d}"
    report = _report(2, "PASS" if not failures else "FAIL", failures, trial, provenance, data_info, period, wfa,
                     tests, True, t_start)   # leak test: same code fingerprint passed it in Stage 1
    append_entry(HOLDOUT_LOG, "HOLDOUT", len(logged_holdouts()) + 1, fp, report)
    return _finish(report, f"{HYP_ID}_holdout", wfa, market)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hypothesis", choices=tuple(H.REGISTRY), help="trial id (cycle 5: C5-xxx)")
    ap.add_argument("--selftest", action="store_true", help="synthetic tests only (no real-data metrics)")
    ap.add_argument("--holdout", action="store_true", help="Stage 2: evaluate a Stage-1 pass once on 2023")
    ap.add_argument("--check-data", action="store_true", help="splice and rate checks on the real data only")
    ap.add_argument("--data-dir", default=str(B.DATA_DIR), help="folder with <PAIR>.csv MT5 exports")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    if args.check_data:
        out = splice_checks(data_dir)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out["passed"] else 1
    if not args.hypothesis:
        ap.error("--hypothesis is required")
    use_hypothesis(args.hypothesis)
    official = data_dir == B.DATA_DIR.resolve()
    t_start = time.time()

    if args.selftest:
        tests = selftest()
        ok = _all_passed(tests)
        print(json.dumps({"mode": "selftest", "hypothesis": HYP_ID, "status": "PASS" if ok else "FAIL",
                          "failed": _failed_names(tests), "tests": tests}, indent=2, default=str))
        return 0 if ok else 1
    if args.holdout:
        return run_stage2(data_dir, official, t_start)
    return run_stage1(data_dir, official, t_start)


if __name__ == "__main__":
    sys.exit(main())
