#!/usr/bin/env python3
"""
harness.py - cycle 2 walk-forward evaluation and integrity harness, running every backtest in NautilusTrader.

The process exits with status 0 only if every gate passes on the 5-pair portfolio:

    a) out-of-sample trades per year      > 100
    b) out-of-sample Sharpe ratio         >= 1.5
    c) out-of-sample max drawdown         < 12 %
    d) walk-forward efficiency (OOS/IS)   >= 0.60
    e) perturbation look-ahead test       passes

and every integrity check passes (data validation, complexity audit, Nautilus-vs-reference parity, edge-case
tests). Any failure prints the exact reason and exits with status 1.

Usage
-----
    python harness.py               full evaluation on data/mt5/<PAIR>.csv (logs a trial in trials.log)
    python harness.py --selftest    synthetic-data tests only (no real-data metrics, not a trial)
    python harness.py --data-dir D  evaluate MT5 exports in D instead (reported, never logged as a trial)

Data: the user's MT5 M15 exports (bid OHLC + spread), 2020-11-30 -> 2022-12-30, EURUSD GBPUSD USDJPY USDCAD
EURJPY. The span is 2.1 years; the 3-year minimum of cycle 1 was relaxed to 2 years by the user for cycle 2.
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
import strategy as S

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TRIALS_LOG = ROOT / "trials.log"
CODE_FILES = ("strategy.py", "harness.py", "backtest.py")

# ----------------------------------------------------------------------------- gates
MIN_TRADES_PER_YEAR = 100.0
MIN_OOS_SHARPE = 1.5
MAX_OOS_DRAWDOWN = 0.12
MIN_WFE = 0.60
MAX_TRIALS = 8

# ----------------------------------------------------------------------------- WFA
N_SPLITS = 5
IS_FRACTION = 0.70
MIN_DATA_YEARS = 2.0          # user decision for cycle 2 (cycle 1 required 3.0)
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


# =============================================================================
# Data
# =============================================================================
def data_provenance(pairs=B.PAIRS, data_dir: Path = B.DATA_DIR) -> dict:
    files = {p: Path(data_dir) / f"{p}.csv" for p in pairs}
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(f"missing MT5 exports: {missing} (expected data/mt5/<PAIR>.csv)")
    return {p: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for p, f in files.items()}


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
def perf_metrics(equity: pd.Series, start_nav: float, trades: pd.DataFrame, t0: int, t1: int) -> dict:
    years = (t1 - t0) / YEAR_NS
    if len(equity) == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "trades": 0, "trades_per_year": 0.0,
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
        "trades": int(len(trades)),
        "trades_per_year": round(len(trades) / years, 2) if years > 0 else 0.0,
    }
    if len(closed):
        pips = closed["pnl_pips"].to_numpy(dtype=float)
        out["win_rate"] = round(float((pips > 0).mean()), 4)
        out["avg_trade_pips_net"] = round(float(pips.mean()), 3)
        out["per_pair"] = {p: {"trades": int(len(g)), "avg_pips": round(float(g["pnl_pips"].mean()), 2),
                               "win_rate": round(float((g["pnl_pips"] > 0).mean()), 3)}
                           for p, g in closed.groupby("pair")}
    return out


def slice_metrics(res: B.RunResult, t0: int, t1: int) -> dict:
    """Metrics of one segment of a continuous run: equity in (t0, t1], trades entered in [t0, t1)."""
    ts = res.equity.index.as_unit("ns").asi8
    before = res.equity[ts <= t0]
    start_nav = float(before.iloc[-1]) if len(before) else res.start_nav
    eq = res.equity[(ts > t0) & (ts <= t1)]
    tr = res.trades[(res.trades["entry_ts"] >= t0) & (res.trades["entry_ts"] < t1)]
    return perf_metrics(eq, start_nav, tr, t0, t1)


# =============================================================================
# Walk-forward analysis (every backtest is a NautilusTrader run)
# =============================================================================
def make_splits(index: pd.DatetimeIndex, n_splits: int = N_SPLITS, is_frac: float = IS_FRACTION) -> list[dict]:
    """Rolling windows of equal length W; IS = 70% W, OOS = next 30% W; OOS segments tile the tail.

    Boundaries are decision times: a window starting at bar i starts deciding at bar i's open (= close of i-1).
    """
    opens = index.as_unit("ns").asi8
    t0, t1 = float(opens[0]), float(opens[-1] + HOUR)
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


def _worker_init(pairs: tuple[str, ...], data_dir: str) -> None:
    import logging
    logging.disable(logging.CRITICAL)
    _WORKER["market"] = B.Market(B.load_universe(pairs, Path(data_dir)))


def _worker_run(task: tuple) -> dict:
    params, start, end = task
    p = S.StrategyParams(**params)
    mk = _WORKER["market"]
    st = S.PortfolioTrendStrategy([S.Segment(start, end, p)], mk.pairs)
    res = mk.run(st, start, end)
    m = perf_metrics(res.equity, res.start_nav, res.trades, start, end)
    m["desyncs"] = st.desyncs
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


def run_wfa(market: B.Market, pool: ProcessPoolExecutor) -> dict:
    keys, values, combos = grid_params(S.DEFAULT_PARAMS)
    splits = make_splits(market.index)
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
    oos = perf_metrics(oos_res.equity, oos_res.start_nav, oos_res.trades, plan[0].start_ns, plan[-1].end_ns)
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


# =============================================================================
# Integrity tests
# =============================================================================
def synthetic_m15(pair: str = "EURUSD", days: int = 150, seed: int = 0, ann_vol: float = 0.08,
                  start: str = "2021-03-01", spread_points: float = 16.0, price: float = 1.15) -> pd.DataFrame:
    """UTC M15 bid bars with weekend gaps and regime drift. Unit tests only - never used for performance."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    idx = idx[idx.dayofweek < 5]  # the *test generator* skips weekends; strategy code never sees calendars
    n, sub = len(idx), 3
    sig = ann_vol / math.sqrt(6240 * 4 * sub)
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 400 + 1), 400 * sub)[: n * sub]
    path = price * np.exp(np.cumsum(regime * sig * 0.15 + sig * rng.standard_normal(n * sub))).reshape(n, sub)
    # like real MT5 data, a bar opens near (not exactly at) the previous close, so fill timing is observable
    open_ = np.concatenate([[price], path[:-1, -1]]) * np.exp(0.3 * sig * rng.standard_normal(n))
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


def _market(frames: dict[str, pd.DataFrame]) -> B.Market:
    return B.Market({p: B.clean_m15(f, p) for p, f in frames.items()})


def _decision_times(market: B.Market) -> np.ndarray:
    return market.index.as_unit("ns").asi8 + HOUR


def _plan_over(market: B.Market, p: S.StrategyParams, skip: int = 100) -> list[S.Segment]:
    closes = _decision_times(market)
    return [S.Segment(int(closes[skip]), int(closes[-1]), p)]


def parity(frames_or_market, plan_fn=None) -> dict:
    """Nautilus run vs the vectorised reference on one USD-quoted pair: orders, fill prices, equity."""
    mk = frames_or_market if isinstance(frames_or_market, B.Market) else _market(frames_or_market)
    pair = mk.pairs[0]
    plan = plan_fn(mk) if plan_fn else _plan_over(mk, S.DEFAULT_PARAMS)
    st = S.PortfolioTrendStrategy(plan, mk.pairs)
    res = mk.run(st, plan[0].start_ns, plan[-1].end_ns)
    ref = S.reference_backtest(mk.h1[pair], plan)
    o_n = res.orders[["ts", "units", "role", "reason"]].reset_index(drop=True)
    o_r = ref["orders"][["ts", "units", "role", "reason"]].reset_index(drop=True)
    same_orders = o_n.equals(o_r)
    n_f = len(res.fills)
    px_diff = float(np.max(np.abs(res.fills["px"].to_numpy() - ref["orders"]["px"].to_numpy()[:n_f]))) if n_f else 0.0
    same_idx = res.equity.index.equals(ref["equity"].index)
    diff = (float(np.max(np.abs(res.equity.to_numpy() - ref["equity"].to_numpy()))) if len(res.equity) else 0.0) \
        if same_idx else math.inf
    # Nautilus books realised P&L in whole cents; the reference keeps full floats -> allow 1 cent per fill
    ok = bool(same_orders and px_diff < 1e-9 and same_idx and diff <= 0.01 * max(1, n_f) and st.desyncs == 0)
    mk.dispose()
    return {"passed": ok, "orders": int(len(o_n)), "fills": n_f, "orders_identical": bool(same_orders),
            "max_fill_px_diff": px_diff, "max_equity_abs_diff_usd": diff, "desyncs": st.desyncs}


class _RoundTripProbe(Strategy):
    """Buys `units` of `pair` at one bar close and sells them at the next; records the USD balance."""

    def __init__(self, pairs, pair, units, start_ns):
        super().__init__(StrategyConfig(strategy_id="Probe-001"))
        self.pairs, self.pair, self.units = pairs, pair, units
        self.warmup_start_ns = start_ns
        self.k, self.ts, self.balances = 0, -1, []

    def on_start(self):
        for p in self.pairs:
            self.subscribe_bars(B.bar_type(p))

    def on_bar(self, bar):
        if bar.ts_event == self.ts:
            return
        self.ts = bar.ts_event
        self.k += 1
        self.balances.append(self.portfolio.account(B.SIM).balance_total(USD).as_double())
        if self.k in (3, 4):
            side = OrderSide.BUY if self.k == 3 else OrderSide.SELL
            self.submit_order(self.order_factory.market(B.instrument_id(self.pair), side,
                                                        Quantity.from_int(self.units)))

    def result(self):
        return self.balances


def friction_tests() -> dict:
    """Exact round-trip costs through Nautilus: USD-quoted pair, and a JPY cross converted to USD."""
    out = {}
    flat_usd = synthetic_m15("EURUSD", days=10, ann_vol=0.0, spread_points=16.0, price=1.1)
    mk = _market({"EURUSD": flat_usd})
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
    mk = _market(frames)
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
    """A down-trend turning into an oscillating up-trend with periodic up-jumps (so any trend rule goes long
    at some point); gap the market down right after a bar in which the strategy is long, sized off the
    leverage it actually holds (legitimate because positions are causal)."""
    n, turn = 1600, 400
    t = np.arange(n, dtype=float)
    drift = np.cumsum(np.where(t < turn, -0.0001, 0.0001))
    jumps = 0.004 * np.floor(np.maximum(t - turn, 0.0) / 97.0)
    close = 1.10 * np.exp(drift + 0.004 * np.sin(2 * np.pi * t / 40.0) + jumps)
    base = m15_from_hourly_close(close, "2021-03-01")
    mk = _market({"EURUSD": base})
    closes = _decision_times(mk)
    p = S.DEFAULT_PARAMS
    plan = [S.Segment(int(closes[50]), int(closes[-1]), p)]
    pre = mk.run(S.PortfolioTrendStrategy(plan, mk.pairs), plan[0].start_ns, plan[-1].end_ns)
    mk.dispose()
    held = _positions_by_bar(pre, closes)
    # long in bar k-1 and still holding the same position through the open of bar k (the gap bar)
    longs = [j for j in range(turn, len(closes) - 300) if held[j - 1] > 0 and held[j] == held[j - 1]]
    if not longs:
        return {"passed": False, "reason": "scenario never went long"}
    k = longs[0]
    eq_prev = float(pre.equity[pre.equity.index.as_unit("ns").asi8 <= closes[k - 1]].iloc[-1])
    lev = held[k - 1] * mk.h1["EURUSD"]["close"].iloc[k - 1] / eq_prev
    gap_start = mk.index[k]

    def gapped(loss: float) -> B.RunResult:
        d = base.copy()
        f = 1.0 - loss / lev
        rows = d.index >= gap_start
        for c in ("open", "high", "low", "close"):
            d.loc[rows, c] = d.loc[rows, c] * f
        m = _market({"EURUSD": d})
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
    return {"passed": bool(halt_ok and kill_ok), "gap_bar": str(gap_start), "leverage_before_gap": round(float(lev), 3),
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

    syn = synthetic_m15("EURUSD", days=150, seed=3)
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

    spk = synthetic_m15("EURUSD", days=150, seed=4)
    k = len(spk) // 2
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
    seg_frames = {"EURUSD": synthetic_m15("EURUSD", days=200, seed=9)}
    alt = dataclasses.replace(p, **{k: v[0] for k, v in S.PARAM_GRID.items()})

    def three_segments(mk: B.Market) -> list[S.Segment]:
        c = _decision_times(mk)
        a, b, d, e = (int(c[k]) for k in (100, len(c) // 3, 2 * len(c) // 3, len(c) - 1))
        return [S.Segment(a, b, p), S.Segment(b, d, alt), S.Segment(d, e, p)]

    res["parity_segmented"] = parity(seg_frames, plan_fn=three_segments)
    res["circuit_breakers"] = circuit_breaker_test()
    res["friction"] = friction_tests()
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
                             mid_bar: bool, seed: int = 13) -> list[int]:
    """Every decision (order) and equity mark stamped at or before a cut time c must not move when all
    M15 data starting at or after c is scrambled.

    mid_bar=False: c = the open of bar t+1 (= the close of bar t)
    mid_bar=True : c = 15/30/45 minutes into hourly bar t - catches a decision that is stamped early but
                   secretly uses the rest of the hour (e.g. a bar published at its open instead of its close)
    """
    rng = np.random.default_rng(seed)
    mk = B.Market(frames)
    closes = _decision_times(mk)
    plan = [S.Segment(int(closes[100]), int(closes[-1]), p)]
    base = mk.run(S.PortfolioTrendStrategy(plan, mk.pairs), plan[0].start_ns, plan[-1].end_ns)
    opens = mk.index
    mk.dispose()
    bad = []
    for t in points:
        cut = opens[t] + pd.Timedelta(minutes=int(rng.choice([15, 30, 45]))) if mid_bar else opens[t + 1]
        upto = int(cut.value)
        m2 = B.Market(_perturb_frames(frames, cut, rng))
        alt = m2.run(S.PortfolioTrendStrategy(plan, m2.pairs), plan[0].start_ns, plan[-1].end_ns)
        m2.dispose()
        o1 = base.orders[base.orders["ts"] <= upto].reset_index(drop=True)
        o2 = alt.orders[alt.orders["ts"] <= upto].reset_index(drop=True)
        e1 = base.equity[base.equity.index.as_unit("ns").asi8 <= upto]
        e2 = alt.equity[alt.equity.index.as_unit("ns").asi8 <= upto]
        if not (o1.equals(o2) and e1.equals(e2)):
            bad.append(int(t))
    return bad


def run_leak_suite(frames: dict[str, pd.DataFrame], p: S.StrategyParams, n_points: int = 6, seed: int = 5) -> dict:
    mk = B.Market(frames)
    h1s = mk.h1
    n = len(mk.index)
    mk.dispose()
    rng = np.random.default_rng(seed)
    points = sorted(int(x) for x in rng.integers(max(130, S.warmup_bars(p) + 5), n - 5, n_points))
    out = {
        "points": points,
        "signals_perturb_t+1": signal_leak_violations(S.signal_frame, h1s, p, points, strict=False),
        "signals_perturb_t": signal_leak_violations(S.signal_frame, h1s, p, points, strict=True),
        "nautilus_perturb_from_bar_boundary": nautilus_leak_violations(frames, p, points, mid_bar=False),
        "nautilus_perturb_from_mid_bar": nautilus_leak_violations(frames, p, points, mid_bar=True),
    }
    future_leak = lambda d, q: {k: v.shift(-2) for k, v in S.signal_frame(d, q).items()}  # row t reads t+1
    same_bar_leak = lambda d, q: S.compute_features(d, q)                                 # row t reads bar t
    out["canary_future_detected"] = len(signal_leak_violations(future_leak, h1s, p, points, strict=False)) > 0
    out["canary_same_bar_detected"] = len(signal_leak_violations(same_bar_leak, h1s, p, points, strict=True)) > 0
    out["passed"] = (all(len(v) == 0 for k, v in out.items() if "perturb" in k)
                     and out["canary_future_detected"] and out["canary_same_bar_detected"])
    return out


def complexity_audit() -> dict:
    fields = [f.name for f in dataclasses.fields(S.StrategyParams)]
    src = (ROOT / "strategy.py").read_text()
    hits = sorted({m.group(0) for m in CALENDAR_PATTERN.finditer(src)})
    grid_ok = set(S.PARAM_GRID) <= set(fields) and set(S.FEATURE_PARAMS) <= set(fields)
    ok = len(fields) <= MAX_TUNABLE_PARAMS and len(S.INDICATORS) <= MAX_INDICATORS and not hits and grid_ok
    return {"passed": bool(ok), "tunable_parameters": fields, "indicators": list(S.INDICATORS),
            "optimised_in_wfa": list(S.PARAM_GRID), "calendar_filter_hits": hits}


# =============================================================================
# Trials log
# =============================================================================
def code_fingerprint() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:12]


def logged_trials() -> list[dict]:
    if not TRIALS_LOG.exists():
        return []
    pat = re.compile(r"^=== TRIAL (\d+) \| (\S+) \| code (\w+) \| (PASS|FAIL)")
    return [{"n": int(m.group(1)), "code": m.group(3)} for line in TRIALS_LOG.read_text().splitlines()
            if (m := pat.match(line))]


def append_trial(n: int, fp: str, report: dict) -> None:
    o = report["metrics"]
    lines = [
        f"=== TRIAL {n} | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | code {fp} | "
        f"{report['status']} ===",
        f"hypothesis : {S.HYPOTHESIS}",
        f"grid       : {json.dumps(S.PARAM_GRID)}  fixed: "
        f"{json.dumps({k: v for k, v in dataclasses.asdict(S.DEFAULT_PARAMS).items() if k not in S.PARAM_GRID})}",
        f"data       : {', '.join(f'{p} {h}' for p, h in report['provenance'].items())}",
        f"metrics    : oos_sharpe={o['oos_sharpe']} is_sharpe_mean={o['is_sharpe_mean']} wfe={o['wfe']} "
        f"oos_max_dd={o['oos_max_drawdown']} oos_trades_per_year={o['oos_trades_per_year']} "
        f"oos_total_return={o['oos_total_return']} leak_test={'PASS' if report['tests']['leak']['passed'] else 'FAIL'}",
    ]
    for r in report["splits"]:
        lines.append(f"split {r['split']}    : OOS {r['oos_period'][0][:10]}..{r['oos_period'][1][:10]} "
                     f"params={ {k: r['chosen_params'][k] for k in S.PARAM_GRID} } IS_sh={r['is']['sharpe']} "
                     f"OOS_sh={r['oos']['sharpe']} OOS_dd={r['oos']['max_drawdown']} "
                     f"OOS_tr/yr={r['oos']['trades_per_year']}")
    lines += [f"failure    : {f}" for f in report["failures"]]
    with TRIALS_LOG.open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")


# =============================================================================
# Main
# =============================================================================
def selftest() -> dict:
    syn = {"EURUSD": B.clean_m15(synthetic_m15("EURUSD", days=90, seed=1), "EURUSD"),
           "USDJPY": B.clean_m15(synthetic_m15("USDJPY", days=90, seed=2, price=110.0, spread_points=15.0), "USDJPY")}
    return {"complexity": complexity_audit(), "edge_cases": edge_case_tests(),
            "leak_synthetic": run_leak_suite(syn, S.DEFAULT_PARAMS, n_points=4)}


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="synthetic tests only (no real-data metrics)")
    ap.add_argument("--data-dir", default=str(B.DATA_DIR), help="folder with <PAIR>.csv MT5 exports")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    counts_as_trial = data_dir == B.DATA_DIR.resolve()
    t_start = time.time()

    if args.selftest:
        tests = selftest()
        ok = _all_passed(tests)
        print(json.dumps({"mode": "selftest", "status": "PASS" if ok else "FAIL", "failed": _failed_names(tests),
                          "tests": tests}, indent=2, default=str))
        return 0 if ok else 1

    fp = code_fingerprint()
    trials = logged_trials()
    rerun = next((t for t in trials if t["code"] == fp), None)
    if counts_as_trial and rerun is None and len(trials) >= MAX_TRIALS:
        print(json.dumps({"status": "FAIL", "failures": [
            f"trial budget exhausted: {len(trials)} of {MAX_TRIALS} trials already logged in trials.log"]}, indent=2))
        return 1

    failures: list[str] = []
    provenance = data_provenance(data_dir=data_dir)
    m15 = B.load_universe(data_dir=data_dir)
    data_errors, data_info = validate_data(m15)
    failures += [f"data: {e}" for e in data_errors]
    market = B.Market(m15)

    tests = selftest()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_worker_init,
                             initargs=(market.pairs, str(data_dir))) as pool:
        wfa = run_wfa(market, pool)

    first = market.index[0]
    leak_frames = {p: f.loc[: first + pd.Timedelta(days=75)] for p, f in m15.items()}
    tests["leak"] = run_leak_suite(leak_frames, wfa["chosen"][0])
    eur = B.Market({"EURUSD": m15["EURUSD"]})
    tests["parity_real_eurusd"] = parity(eur, plan_fn=lambda mk: [
        S.Segment(sg.start_ns, sg.end_ns, sg.params) for sg in wfa["plan"]])
    tests["no_order_desyncs"] = {"passed": wfa["oos_desyncs"] == 0 and wfa["is_desyncs"] == 0,
                                 "oos": wfa["oos_desyncs"], "is": wfa["is_desyncs"]}

    o = wfa["oos"]
    if not o["trades_per_year"] > MIN_TRADES_PER_YEAR:
        failures.append(f"(a) OOS trades/year {o['trades_per_year']} <= {MIN_TRADES_PER_YEAR:g}")
    if not o["sharpe"] >= MIN_OOS_SHARPE:
        failures.append(f"(b) OOS Sharpe {o['sharpe']} < {MIN_OOS_SHARPE}")
    if not o["max_drawdown"] < MAX_OOS_DRAWDOWN:
        failures.append(f"(c) OOS max drawdown {o['max_drawdown']:.2%} >= {MAX_OOS_DRAWDOWN:.0%}")
    if wfa["wfe"] is None:
        failures.append(f"(d) WFE undefined: mean IS Sharpe {wfa['is_sharpe_mean']} <= 0")
    elif not wfa["wfe"] >= MIN_WFE:
        failures.append(f"(d) WFE {wfa['wfe']} < {MIN_WFE} (OOS Sharpe {o['sharpe']} / IS Sharpe "
                        f"{wfa['is_sharpe_mean']})")
    if not tests["leak"]["passed"]:
        failures.append(f"(e) perturbation leak test failed: {json.dumps(tests['leak'])}")
    failures += [f"integrity: {n} failed" for n in _failed_names({k: v for k, v in tests.items() if k != "leak"})]

    status = "PASS" if not failures else "FAIL"
    report = {
        "status": status,
        "failures": failures,
        "trial": {"code_fingerprint": fp, "rerun_of_trial": rerun["n"] if rerun else None,
                  "trial_number": (rerun["n"] if rerun else len(trials) + 1) if counts_as_trial else None,
                  "budget": MAX_TRIALS, "counts_as_trial": counts_as_trial, "data_dir": str(data_dir)},
        "hypothesis": S.HYPOTHESIS,
        "engine": {"nautilus_trader": __import__("nautilus_trader").__version__, "venue": "SIM NETTING MARGIN USD",
                   "latency_ns": 1, "fees": 0, "min_spread_pips": B.MIN_SPREAD_PIPS,
                   "slippage_pips_per_fill": B.SLIPPAGE_PIPS},
        "provenance": provenance,
        "data": data_info,
        "config": {"param_grid": S.PARAM_GRID, "defaults": dataclasses.asdict(S.DEFAULT_PARAMS),
                   "limits": dataclasses.asdict(S.RiskLimits()), "n_splits": N_SPLITS, "is_fraction": IS_FRACTION,
                   "min_data_years": MIN_DATA_YEARS},
        "metrics": {
            "oos_sharpe": o["sharpe"], "is_sharpe_mean": wfa["is_sharpe_mean"], "wfe": wfa["wfe"],
            "oos_max_drawdown": o["max_drawdown"], "oos_trades_per_year": o["trades_per_year"],
            "oos_total_return": o["total_return"], "oos_cagr": o.get("cagr"), "oos_ann_vol": o.get("ann_vol"),
            "oos_win_rate": o.get("win_rate"), "oos_avg_trade_pips_net": o.get("avg_trade_pips_net"),
            "oos_years": o["years"], "oos_halts_daily_loss": wfa["oos_result"].n_halts,
            "oos_killed_max_dd": wfa["oos_result"].killed, "oos_per_pair": o.get("per_pair"),
            "leak_test_passed": tests["leak"]["passed"],
        },
        "splits": wfa["splits"],
        "tests": tests,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    if not counts_as_trial:
        print(json.dumps({k: report[k] for k in ("status", "failures", "trial", "metrics")}, indent=2, default=str))
        print(f"\nNOT A TRIAL: custom data dir {data_dir}", file=sys.stderr)
        market.dispose()
        return 0 if not failures else 1
    if rerun is None:
        append_trial(len(trials) + 1, fp, report)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "wfa_latest.json").write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_DIR / "wfa_oos_equity.csv").write_text(wfa["oos_result"].equity.rename("equity").to_csv())
    slim = {**report, "splits": [{k: v for k, v in r.items() if k != "is_grid"} for r in report["splits"]]}
    print(json.dumps(slim, indent=2, default=str))
    market.dispose()
    if failures:
        print("\nHARNESS FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nHARNESS PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
