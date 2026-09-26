#!/usr/bin/env python3
"""
harness.py - walk-forward evaluation and integrity harness for ``strategy.py``.

The process exits with status 0 only if every gate passes:

    a) out-of-sample trades per year      > 100
    b) out-of-sample Sharpe ratio         >= 1.5
    c) out-of-sample max drawdown         < 12 %
    d) walk-forward efficiency (OOS/IS)   >= 0.60
    e) perturbation look-ahead test       passes

and every integrity check passes (data validation, complexity audit, edge-case
tests, vectorised/event-driven parity). Any failure prints the exact reason and
exits with status 1.

Usage
-----
    python harness.py                 full evaluation on real EURUSD H1 data (logs a trial)
    python harness.py --selftest      synthetic-data unit tests only (no performance, no trial)
    python harness.py --data my.csv   evaluate an OHLC CSV of your own (time,open,high,low,close)

Data
----
Default source: OANDA EUR/USD 1-minute mid candles (UTC) published in the public
GitHub repository FutureSharks/financial-data (pinned commit below), resampled to
1-hour bars and cached under ``data/``. The series ends on 2020-05-14.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import strategy as S

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TRIALS_LOG = ROOT / "trials.log"

# ----------------------------------------------------------------------------- gates
MIN_TRADES_PER_YEAR = 100.0
MIN_OOS_SHARPE = 1.5
MAX_OOS_DRAWDOWN = 0.12
MIN_WFE = 0.60
MAX_TRIALS = 8

# ----------------------------------------------------------------------------- WFA
N_SPLITS = 5
IS_FRACTION = 0.70
MIN_DATA_YEARS = 3.0
TRADING_DAYS_PER_YEAR = 252
DEFAULT_START = "2015-01-01"

# ----------------------------------------------------------------------------- limits
MAX_TUNABLE_PARAMS = 4
MAX_INDICATORS = 3
# attribute names that would betray a calendar / time-of-day filter in strategy code
CALENDAR_PATTERN = re.compile(
    r"\.(hour|minute|dayofweek|day_of_week|weekday|isoweekday|day_name|month|quarter|"
    r"dayofyear|day_of_year|is_month_end|is_month_start|week|weekofyear)\b"
)

# ----------------------------------------------------------------------------- source
SOURCE_REPO = "https://github.com/FutureSharks/financial-data"
SOURCE_COMMIT = "7ba1d404aa8b0e1c0f71321acebadcbfb9bcca8d"
OANDA_SUBDIR = "pyfinancialdata/data/currencies/oanda"
PAIR_TO_OANDA = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "AUDUSD": "AUD_USD",
    "USDCAD": "USD_CAD",
    "AUDJPY": "AUD_JPY",
}


# =============================================================================
# Data
# =============================================================================
def pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)


def ensure_source(source_dir: Path, instrument: str, years: list[int]) -> None:
    """Sparse, blob-less checkout of just the monthly files we need, pinned to SOURCE_COMMIT."""
    base = source_dir / OANDA_SUBDIR / instrument
    if all((base / str(y)).is_dir() for y in years):
        return
    if not (source_dir / ".git").exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        r = _git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", SOURCE_REPO, str(source_dir))
        if r.returncode != 0:
            raise RuntimeError(f"git clone of {SOURCE_REPO} failed: {r.stderr.strip()}")
    patterns = [f"{OANDA_SUBDIR}/{instrument}/{y}/*" for y in years]
    r = _git("sparse-checkout", "set", "--no-cone", *patterns, cwd=source_dir)
    if r.returncode != 0:
        raise RuntimeError(f"git sparse-checkout failed: {r.stderr.strip()}")
    r = _git("checkout", SOURCE_COMMIT, cwd=source_dir)
    if r.returncode != 0:
        _git("fetch", "--depth", "1", "--filter=blob:none", "origin", SOURCE_COMMIT, cwd=source_dir)
        r = _git("checkout", SOURCE_COMMIT, cwd=source_dir)
        if r.returncode != 0:
            raise RuntimeError(f"git checkout {SOURCE_COMMIT} failed: {r.stderr.strip()}")


def build_hourly_from_minutes(source_dir: Path, instrument: str, years: list[int]) -> pd.DataFrame:
    base = source_dir / OANDA_SUBDIR / instrument
    files = sorted(f for y in years for f in (base / str(y)).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no minute files under {base}")
    cols = ["time", "open", "high", "low", "close", "volume"]
    m = pd.concat((pd.read_csv(f, usecols=cols) for f in files), ignore_index=True)
    m["time"] = pd.to_datetime(m["time"], utc=True)
    m = m.drop_duplicates("time", keep="last").set_index("time").sort_index()
    h = m.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return h.dropna(subset=["open", "high", "low", "close"])


def load_market_data(pair: str, start: str, end: str | None, data_path: str | None,
                     source_dir: Path) -> tuple[pd.DataFrame, dict]:
    if data_path:
        raw = pd.read_csv(data_path)
        raw.columns = [c.strip().lower() for c in raw.columns]
        tcol = next(c for c in ("time", "datetime", "date", "timestamp") if c in raw.columns)
        raw[tcol] = pd.to_datetime(raw[tcol], utc=True)
        raw = raw.set_index(tcol)
        provenance = {"source": f"user file {data_path}"}
    else:
        instrument = PAIR_TO_OANDA.get(pair.upper())
        if instrument is None:
            raise ValueError(f"pair {pair} not in bundled source; supported: {sorted(PAIR_TO_OANDA)}")
        y0 = pd.Timestamp(start).year
        y1 = pd.Timestamp(end).year if end else 2020
        cache = DATA_DIR / f"{pair.upper()}_H1_oanda_{y0}_{y1}.csv.gz"
        if cache.exists():
            raw = pd.read_csv(cache, index_col=0)
            raw.index = pd.to_datetime(raw.index, utc=True)
        else:
            years = list(range(y0, y1 + 1))
            ensure_source(source_dir, instrument, years)
            raw = build_hourly_from_minutes(source_dir, instrument, years)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            raw.to_csv(cache, compression="gzip")
        provenance = {
            "source": "OANDA mid candles, 1-minute, resampled to 1-hour (UTC)",
            "repository": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "instrument": instrument,
            "cache": str(cache.relative_to(ROOT)),
        }
    raw = raw.loc[pd.Timestamp(start, tz="UTC"):]
    if end:
        raw = raw.loc[: pd.Timestamp(end, tz="UTC")]
    df = S.clean_ohlc(raw)
    provenance["sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(df[["open", "high", "low", "close"]], index=True).values.tobytes()
    ).hexdigest()
    return df, provenance


def validate_data(df: pd.DataFrame, raw_rows: int | None = None) -> tuple[list[str], dict]:
    errors: list[str] = []
    idx = df.index
    years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 86400) if len(idx) > 1 else 0.0
    gaps = pd.Series(idx[1:] - idx[:-1])
    info = {
        "first_bar": str(idx[0]) if len(idx) else None,
        "last_bar": str(idx[-1]) if len(idx) else None,
        "bars": int(len(df)),
        "years": round(years, 3),
        "bars_per_year": round(len(df) / years, 1) if years > 0 else None,
        "gaps_over_1h_excl_weekends": int(((gaps > pd.Timedelta("1h")) & (gaps < pd.Timedelta("36h"))).sum()),
        "max_gap_hours": float(gaps.max() / pd.Timedelta("1h")) if len(gaps) else None,
    }
    if years < MIN_DATA_YEARS:
        errors.append(f"data span {years:.2f}y < required {MIN_DATA_YEARS}y")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        errors.append("index not strictly increasing")
    px = df[["open", "high", "low", "close"]]
    if not np.isfinite(px.to_numpy()).all() or (px <= 0).any().any():
        errors.append("non-finite or non-positive prices after cleaning")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any() or (df["low"] > df[["open", "close"]].min(axis=1)).any():
        errors.append("OHLC inconsistency (high/low do not bracket open/close)")
    return errors, info


# =============================================================================
# Metrics
# =============================================================================
def perf_metrics(res: "S.BacktestResult", initial_nav: float, pip: float) -> dict:
    nav = res.nav
    if len(nav) == 0:
        return {"sharpe": 0.0, "trades": 0, "trades_per_year": 0.0, "max_drawdown": 0.0}
    day = S.trading_day_ids(nav.index)
    daily = nav.groupby(day).last()
    rets = daily.pct_change()
    rets.iloc[0] = daily.iloc[0] / initial_nav - 1.0
    sd = rets.std(ddof=1)
    sharpe = float(rets.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR)) if sd > 0 else 0.0
    years = ((nav.index[-1] - nav.index[0]).total_seconds() + 3600.0) / (365.25 * 86400)
    curve = np.concatenate([[initial_nav], nav.to_numpy()])
    dd = curve / np.maximum.accumulate(curve) - 1.0
    total = float(curve[-1] / initial_nav - 1.0)
    tr = res.trades
    n_tr = int(len(tr))
    out = {
        "sharpe": round(sharpe, 4),
        "total_return": round(total, 5),
        "cagr": round(float((1 + total) ** (1 / years) - 1), 5) if years > 0 and total > -1 else None,
        "ann_vol": round(float(sd * math.sqrt(TRADING_DAYS_PER_YEAR)), 5) if sd == sd else None,
        "max_drawdown": round(float(-dd.min()), 5),
        "years": round(years, 3),
        "trades": n_tr,
        "trades_per_year": round(n_tr / years, 2) if years > 0 else 0.0,
        "exposure": round(float((res.units != 0).mean()), 4),
        "halts_daily_loss": int(res.n_halts),
        "killed_max_dd": bool(res.killed),
    }
    if n_tr:
        pips = tr["pnl_pips"].to_numpy()
        wins, losses = pips[pips > 0].sum(), -pips[pips < 0].sum()
        out.update({
            "win_rate": round(float((pips > 0).mean()), 4),
            "avg_trade_pips_net": round(float(pips.mean()), 3),
            "profit_factor": round(float(wins / losses), 3) if losses > 0 else None,
            "avg_bars_held": round(float(tr["bars_held"].mean()), 1),
        })
    return out


# =============================================================================
# Walk-forward analysis
# =============================================================================
def make_splits(index: pd.DatetimeIndex, n_splits: int = N_SPLITS, is_frac: float = IS_FRACTION) -> list[dict]:
    """Rolling windows of equal length W; IS = 70% W, OOS = next 30% W; OOS segments tile the tail."""
    ns = index.as_unit("ns").asi8
    t0, t1 = float(ns[0]), float(ns[-1] + 3_600_000_000_000)
    width = (t1 - t0) / (1 + (n_splits - 1) * (1 - is_frac))
    step = width * (1 - is_frac)
    splits = []
    for k in range(n_splits):
        a = t0 + k * step
        b = a + width * is_frac
        c = a + width if k < n_splits - 1 else t1
        lo, mid, hi = (int(np.searchsorted(ns, v, side="left")) for v in (a, b, c))
        splits.append({"split": k + 1, "is": (lo, mid), "oos": (mid, hi)})
    return splits


def grid_params(base: S.StrategyParams) -> tuple[list[str], list[list], list[S.StrategyParams]]:
    keys = list(S.PARAM_GRID)
    values = [list(S.PARAM_GRID[k]) for k in keys]
    combos = [dataclasses.replace(base, **dict(zip(keys, c))) for c in itertools.product(*values)]
    return keys, values, combos


class SignalCache:
    """Signal frames depend only on S.FEATURE_PARAMS; compute each once on the full history.

    Frames are causal (row t uses bars <= t-1, proven by the perturbation test), so computing
    them over the whole series and slicing a window leaks nothing into that window.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._c: dict[tuple, pd.DataFrame] = {}

    def get(self, p: S.StrategyParams) -> pd.DataFrame:
        key = tuple(getattr(p, f) for f in S.FEATURE_PARAMS)
        if key not in self._c:
            self._c[key] = S.signal_frame(self.df, p)
        return self._c[key]


def select_params(is_rows: list[dict], values: list[list]) -> dict:
    """Plateau selection: score = mean IS Sharpe over the 3^k grid neighbourhood.

    Only combos that trade often enough in-sample are eligible (if none are, all are).
    """
    shape = [len(v) for v in values]
    sharpe = np.array([r["metrics"]["sharpe"] for r in is_rows]).reshape(shape)
    eligible = [r["metrics"]["trades_per_year"] > MIN_TRADES_PER_YEAR for r in is_rows]
    if not any(eligible):
        eligible = [True] * len(is_rows)
    best, best_score = None, -np.inf
    for flat, r in enumerate(is_rows):
        pos = np.unravel_index(flat, shape)
        sl = tuple(slice(max(0, i - 1), i + 2) for i in pos)
        score = float(np.mean(sharpe[sl]))
        r["plateau_score"] = round(score, 4)
        if eligible[flat] and score > best_score + 1e-12:
            best, best_score = r, score
    return best


def run_wfa(df: pd.DataFrame, cfg: S.ExecutionConfig) -> dict:
    keys, values, combos = grid_params(S.DEFAULT_PARAMS)
    cache = SignalCache(df)
    splits = make_splits(df.index)
    state = None
    oos_navs, oos_units, oos_trades, split_rows = [], [], [], []
    chosen_by_split = []
    for sp in splits:
        lo, hi = sp["is"]
        is_rows = []
        for p in combos:
            r = S.backtest_vectorized(df, p, cfg, start=lo, end=hi, sig=cache.get(p))
            is_rows.append({"params": dataclasses.asdict(p), "metrics": perf_metrics(r, cfg.initial_nav, cfg.pip)})
        best = select_params(is_rows, values)
        p_best = S.StrategyParams(**best["params"])
        chosen_by_split.append(p_best)
        olo, ohi = sp["oos"]
        seg = S.backtest_vectorized(df, p_best, cfg, start=olo, end=ohi, state=state, sig=cache.get(p_best))
        seg_start_nav = state["nav"] if state else cfg.initial_nav
        state = seg.final_state
        oos_navs.append(seg.nav)
        oos_units.append(seg.units)
        oos_trades.append(seg.trades)
        seg_m = perf_metrics(seg, seg_start_nav, cfg.pip)
        is_sh = best["metrics"]["sharpe"]
        split_rows.append({
            "split": sp["split"],
            "is_period": [str(df.index[lo]), str(df.index[hi - 1])],
            "oos_period": [str(df.index[olo]), str(df.index[ohi - 1])],
            "chosen_params": best["params"],
            "is_plateau_score": best["plateau_score"],
            "is": best["metrics"],
            "oos": seg_m,
            "wfe": round(seg_m["sharpe"] / is_sh, 4) if is_sh > 0 else None,
            "is_grid": [{**{k: r["params"][k] for k in keys}, "sharpe": r["metrics"]["sharpe"],
                         "trades_per_year": r["metrics"]["trades_per_year"]} for r in is_rows],
        })
    stitched = S.BacktestResult(
        nav=pd.concat(oos_navs), units=pd.concat(oos_units),
        trades=pd.concat(oos_trades, ignore_index=True),
        final_state=state, n_halts=int(state["n_halts"]), killed=bool(state["killed"]),
    )
    oos = perf_metrics(stitched, cfg.initial_nav, cfg.pip)
    is_mean = float(np.mean([r["is"]["sharpe"] for r in split_rows]))
    return {
        "splits": split_rows,
        "chosen": chosen_by_split,
        "split_bounds": [sp["oos"] for sp in splits],
        "stitched": stitched,
        "oos": oos,
        "is_sharpe_mean": round(is_mean, 4),
        "wfe": round(oos["sharpe"] / is_mean, 4) if is_mean > 0 else None,
    }


# =============================================================================
# Integrity tests
# =============================================================================
def synthetic_ohlc(years: float = 2.0, seed: int = 0, ann_vol: float = 0.08, trend: bool = True) -> pd.DataFrame:
    """Hourly OHLC with weekend gaps, regime-switching drift. Unit tests only - never used for performance."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=int(years * 365.25 * 24), freq="1h", tz="UTC")
    idx = idx[idx.dayofweek < 5]  # the *test generator* skips weekends; strategy code never sees calendars
    n, sub = len(idx), 4
    sig = ann_vol / math.sqrt(6240 * sub)
    drift = np.zeros(n * sub)
    if trend:
        regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 300 + 1), 300 * sub)[: n * sub]
        drift = regime * sig * 0.15
    path = 1.15 * np.exp(np.cumsum(drift + sig * rng.standard_normal(n * sub))).reshape(n, sub)
    open_ = np.concatenate([[1.15], path[:-1, -1]])
    high = np.maximum(open_, path.max(axis=1))
    low = np.minimum(open_, path.min(axis=1))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": path[:, -1],
                         "volume": rng.integers(50, 500, n).astype(float)}, index=idx)


def _perturb(df: pd.DataFrame, first: int, rng: np.random.Generator) -> pd.DataFrame:
    """Scramble every bar from position `first` on (same factor per row keeps OHLC consistent)."""
    d = df.copy()
    n = len(d) - first
    f = np.exp(np.cumsum(rng.normal(0.0, 0.004, n))) * rng.choice([0.9, 1.0, 1.1], n)
    for c in ("open", "high", "low", "close"):
        d.iloc[first:, d.columns.get_loc(c)] = d[c].to_numpy()[first:] * f
    spike = rng.random(n) < 0.05
    hi = d["high"].to_numpy().copy()
    hi[first:][spike] *= 1.03
    d["high"] = hi
    return d


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.shape == b.shape and all(
        np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True) for c in a.columns
    )


def leak_violations(frame_fn, df: pd.DataFrame, p: S.StrategyParams, points: list[int], strict: bool,
                    seed: int = 11) -> list[int]:
    """Return the bars t at which rows <= t of frame_fn changed when future data was altered.

    strict=False: perturb bars t+1.. (the task's required test)
    strict=True : perturb bar t itself (all of O/H/L/C) and everything after
    """
    rng = np.random.default_rng(seed)
    base = frame_fn(df, p)
    bad = []
    for t in points:
        d2 = _perturb(df, t if strict else t + 1, rng)
        alt = frame_fn(d2, p)
        if not _frames_equal(base.iloc[: t + 1], alt.iloc[: t + 1]):
            bad.append(int(t))
    return bad


def position_leak_violations(engine: str, df: pd.DataFrame, p: S.StrategyParams, cfg: S.ExecutionConfig,
                             points: list[int], strict: bool, seed: int = 13) -> list[int]:
    """Same test on the full simulation: units held in bar t and NAV up to bar t-1 must not move."""
    rng = np.random.default_rng(seed)
    run = (lambda d: S.backtest_vectorized(d, p, cfg)) if engine == "vectorized" else \
          (lambda d: S.EventDrivenBacktester(d, p, cfg).run())
    base = run(df)
    bad = []
    for t in points:
        alt = run(_perturb(df, t if strict else t + 1, rng))
        nav_upto = t if strict else t + 1
        ok = (np.array_equal(base.units.to_numpy()[: t + 1], alt.units.to_numpy()[: t + 1])
              and np.array_equal(base.nav.to_numpy()[:nav_upto], alt.nav.to_numpy()[:nav_upto]))
        if not ok:
            bad.append(int(t))
    return bad


def run_leak_suite(df: pd.DataFrame, p: S.StrategyParams, cfg: S.ExecutionConfig, n_points: int = 8,
                   seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    lo = S.warmup_bars(p) + 5
    points = sorted(int(x) for x in rng.integers(lo, len(df) - 5, n_points))
    sig_fn = S.signal_frame
    out = {
        "points": points,
        "signals_perturb_t+1": leak_violations(sig_fn, df, p, points, strict=False),
        "signals_perturb_t": leak_violations(sig_fn, df, p, points, strict=True),
        "vectorized_positions_perturb_t+1": position_leak_violations("vectorized", df, p, cfg, points, False),
        "vectorized_positions_perturb_t": position_leak_violations("vectorized", df, p, cfg, points, True),
        "event_positions_perturb_t+1": position_leak_violations("event", df, p, cfg, points, False),
        "event_positions_perturb_t": position_leak_violations("event", df, p, cfg, points, True),
    }
    # canaries: prove the test can see leaks. Both deliberately-leaky frames MUST be flagged.
    future_leak = lambda d, q: S.signal_frame(d, q).shift(-2)   # row t reads bar t+1
    same_bar_leak = lambda d, q: S.compute_features(d, q)        # row t reads bar t's close
    out["canary_future_detected"] = len(leak_violations(future_leak, df, p, points, strict=False)) > 0
    out["canary_same_bar_detected"] = len(leak_violations(same_bar_leak, df, p, points, strict=True)) > 0
    out["passed"] = (all(len(v) == 0 for k, v in out.items() if "perturb" in k)
                     and out["canary_future_detected"] and out["canary_same_bar_detected"])
    return out


def parity(df_raw: pd.DataFrame, p: S.StrategyParams, cfg: S.ExecutionConfig,
           segments: list[tuple[int, int]] | None = None, params: list[S.StrategyParams] | None = None) -> dict:
    """Vectorised research path vs event-driven production path must agree bar by bar."""
    df = S.clean_ohlc(df_raw)
    segments = segments or [(0, len(df))]
    params = params or [p] * len(segments)
    sv = se = None
    nav_v, nav_e, u_v, u_e, n_v, n_e = [], [], [], [], 0, 0
    for (a, b), q in zip(segments, params):
        rv = S.backtest_vectorized(df, q, cfg, start=a, end=b, state=sv)
        re_ = S.EventDrivenBacktester(df_raw, q, cfg).run(start=a, end=b, state=se)
        sv, se = rv.final_state, re_.final_state
        nav_v.append(rv.nav.to_numpy()); nav_e.append(re_.nav.to_numpy())
        u_v.append(rv.units.to_numpy()); u_e.append(re_.units.to_numpy())
        n_v += len(rv.trades); n_e += len(re_.trades)
    nv, ne, uv, ue = map(np.concatenate, (nav_v, nav_e, u_v, u_e))
    same_len = len(nv) == len(ne)
    max_nav_rel = float(np.max(np.abs(nv / ne - 1))) if same_len and len(nv) else float("inf")
    dir_match = same_len and np.array_equal(np.sign(uv), np.sign(ue))
    units_close = same_len and np.allclose(uv, ue, rtol=1e-9, atol=1e-9)
    ok = bool(same_len and dir_match and units_close and max_nav_rel < 1e-9 and n_v == n_e)
    return {"passed": ok, "bars": int(len(nv)), "trades_vectorized": n_v, "trades_event": n_e,
            "max_nav_rel_diff": max_nav_rel, "direction_match": bool(dir_match)}


def edge_case_tests(cfg: S.ExecutionConfig) -> dict:
    p = S.DEFAULT_PARAMS
    res: dict[str, dict] = {}

    # 1) zero-division / degenerate inputs in position sizing
    cases = [(1e6, 0.0, 1.1), (1e6, -1.0, 1.1), (1e6, float("nan"), 1.1), (1e6, float("inf"), 1.1),
             (0.0, 0.001, 1.1), (-5.0, 0.001, 1.1), (float("nan"), 0.001, 1.1), (1e6, 0.001, 0.0),
             (1e6, 0.001, float("nan")), (1e6, 1e-300, 1.1)]
    sizes = [S.RiskManager.position_size(n, a, px, p, cfg) for n, a, px in cases]
    lev_tiny_atr = sizes[-1] * 1.1 / 1e6
    ok = (all(math.isfinite(s) and s >= 0 for s in sizes) and all(s == 0 for s in sizes[:-1])
          and lev_tiny_atr <= cfg.max_leverage + 1e-9)
    res["zero_division_sizing"] = {"passed": bool(ok), "sizes": sizes, "leverage_at_tiny_atr": lev_tiny_atr}

    # 2) flat prices -> ATR == 0 -> no trades, NAV untouched
    idx = pd.date_range("2019-01-01", periods=3000, freq="1h", tz="UTC")
    flat = pd.DataFrame({"open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1}, index=idx)
    rv, re_ = S.backtest_vectorized(flat, p, cfg), S.EventDrivenBacktester(flat, p, cfg).run()
    ok = (len(rv.trades) == 0 and len(re_.trades) == 0 and np.all(rv.nav.to_numpy() == cfg.initial_nav)
          and np.all(re_.nav.to_numpy() == cfg.initial_nav))
    res["flat_prices_zero_atr"] = {"passed": bool(ok)}

    # 3) missing bars, NaN rows, duplicates, corrupt rows
    syn = synthetic_ohlc(1.5, seed=3)
    rng = np.random.default_rng(3)
    dirty = syn.drop(syn.index[rng.random(len(syn)) < 0.07]).copy()
    nan_rows = rng.choice(len(dirty), 40, replace=False)
    dirty.iloc[nan_rows, dirty.columns.get_loc("close")] = np.nan
    dirty.iloc[nan_rows[:5], dirty.columns.get_loc("open")] = -1.0
    bad_hl = rng.choice(len(dirty), 5, replace=False)
    dirty.iloc[bad_hl, dirty.columns.get_loc("high")] = dirty["low"].to_numpy()[bad_hl] * 0.99
    dirty = pd.concat([dirty, dirty.iloc[100:103]])  # duplicated timestamps, out of order
    clean = S.clean_ohlc(dirty)
    par = parity(dirty, p, cfg)
    rv = S.backtest_vectorized(clean, p, cfg)
    ok = (par["passed"] and np.isfinite(rv.nav.to_numpy()).all() and clean.index.is_monotonic_increasing
          and not clean.index.has_duplicates and len(clean) < len(syn))
    res["missing_and_corrupt_bars"] = {"passed": bool(ok), "rows_in": int(len(dirty)),
                                       "rows_clean": int(len(clean)), "parity": par}

    # 4) extreme volatility spike: 8% gap + 10% range bar mid-sample
    spk = synthetic_ohlc(1.5, seed=4)
    k = len(spk) // 2
    for c in ("open", "high", "low", "close"):
        spk.iloc[k:, spk.columns.get_loc(c)] = spk[c].to_numpy()[k:] * 1.08
    spk.iloc[k, spk.columns.get_loc("high")] *= 1.05
    spk.iloc[k, spk.columns.get_loc("low")] *= 0.95
    rv = S.backtest_vectorized(spk, p, cfg)
    lev = np.abs(rv.units.to_numpy()) * spk["close"].to_numpy() / rv.nav.to_numpy()
    par = parity(spk, p, cfg)
    ok = bool(np.isfinite(rv.nav.to_numpy()).all() and lev.max() <= cfg.max_leverage * 1.2 and par["passed"])
    res["volatility_spike"] = {"passed": ok, "max_leverage_seen": round(float(lev.max()), 3), "parity": par}

    # 5) circuit breakers (deterministic trend then gap-down sized off the realised leverage)
    res["circuit_breakers"] = _circuit_breaker_test(cfg)

    # 6) friction: round trip at an unchanged mid costs exactly spread + 2 x slippage
    ex = S.ExecutionHandler(cfg, cash=1_000_000.0)
    ex.execute(100_000.0, 1.1, ts=0)
    ex.execute(-100_000.0, 1.1, ts=1)
    expected = -100_000.0 * (cfg.spread_pips + 2 * cfg.slippage_pips) * cfg.pip
    got = ex.cash - 1_000_000.0
    res["friction_round_trip"] = {"passed": abs(got - expected) < 1e-6, "cost": got, "expected": expected}

    # 7) parity with state carried across segments (as used by the stitched OOS run)
    syn2 = synthetic_ohlc(2.0, seed=9)
    n = len(syn2)
    segs = [(n // 5, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)]
    alt = dataclasses.replace(p, **{k: v[0] for k, v in S.PARAM_GRID.items()})
    res["parity_segmented"] = parity(syn2, p, cfg, segments=segs, params=[p, alt, p])
    return res


def _circuit_breaker_test(cfg: S.ExecutionConfig) -> dict:
    """Oscillating up-trend; gap the market down right after a bar in which the strategy is long.

    The gap is sized off the leverage the strategy actually holds, which is legitimate only because
    positions are causal: re-pricing bars >= k cannot change the position held in bar k-1.
    """
    p = S.DEFAULT_PARAMS
    n = 2400
    idx = pd.date_range("2019-03-04", periods=n, freq="1h", tz="UTC")
    t = np.arange(n, dtype=float)
    close = 1.10 * np.exp(0.0001 * t + 0.004 * np.sin(2 * np.pi * t / 40.0))
    open_ = np.concatenate([[close[0]], close[:-1]])
    base = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0002,
                         "low": np.minimum(open_, close) * 0.9998, "close": close}, index=idx)
    pre = S.backtest_vectorized(base, p, cfg)
    pu = pre.units.to_numpy()
    longs = [j for j in range(S.warmup_bars(p) + 50, n - 600) if pu[j - 1] > 0]
    if not longs:
        return {"passed": False, "reason": "scenario did not produce a long position before the gap"}
    k = longs[0]
    lev = pu[k - 1] * close[k - 1] / pre.nav.iloc[k - 1]

    def gapped(nav_loss: float) -> pd.DataFrame:
        d = base.copy()
        f = 1.0 - nav_loss / lev
        for c in ("open", "high", "low", "close"):
            d.iloc[k:, d.columns.get_loc(c)] = d[c].to_numpy()[k:] * f
        return d

    halt = S.backtest_vectorized(gapped(0.05), p, cfg)          # ~5% day loss -> 24h halt
    hu, hn = halt.units.to_numpy(), halt.nav.to_numpy()
    ev = S.EventDrivenBacktester(gapped(0.05), p, cfg).run()
    dd_at_gap = hn[k] / max(cfg.initial_nav, hn[: k + 1].max()) - 1.0
    halt_ok = (halt.n_halts >= 1 and dd_at_gap > -cfg.max_drawdown_limit   # halt, not the kill switch
               and np.all(hu[k + 1: k + 25] == 0)                           # flat for 24 hourly bars
               and np.any(hu[k + 25:] != 0)                                 # ...then allowed to trade again
               and np.array_equal(hu, ev.units.to_numpy()))
    kill = S.backtest_vectorized(gapped(0.15), p, cfg)          # ~15% loss -> permanent stop
    kill_ok = kill.killed and np.all(kill.units.to_numpy()[k + 1:] == 0)
    return {"passed": bool(halt_ok and kill_ok), "leverage_before_gap": round(float(lev), 3),
            "daily_halt_ok": bool(halt_ok), "max_dd_kill_ok": bool(kill_ok)}


def complexity_audit() -> dict:
    fields = [f.name for f in dataclasses.fields(S.StrategyParams)]
    src = (ROOT / "strategy.py").read_text()
    calendar_hits = sorted({m.group(0) for m in CALENDAR_PATTERN.finditer(src)})
    grid_ok = set(S.PARAM_GRID) <= set(fields) and set(S.FEATURE_PARAMS) <= set(fields)
    ok = len(fields) <= MAX_TUNABLE_PARAMS and len(S.INDICATORS) <= MAX_INDICATORS and not calendar_hits and grid_ok
    return {"passed": bool(ok), "tunable_parameters": fields, "indicators": list(S.INDICATORS),
            "optimised_in_wfa": list(S.PARAM_GRID), "calendar_filter_hits": calendar_hits}


# =============================================================================
# Trials log
# =============================================================================
def code_fingerprint() -> str:
    h = hashlib.sha256()
    for name in ("strategy.py", "harness.py"):
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:12]


def logged_trials() -> list[dict]:
    if not TRIALS_LOG.exists():
        return []
    pat = re.compile(r"^=== TRIAL (\d+) \| (\S+) \| code (\w+) \| (PASS|FAIL)")
    return [{"n": int(m.group(1)), "code": m.group(3), "result": m.group(4)}
            for line in TRIALS_LOG.read_text().splitlines() if (m := pat.match(line))]


def append_trial(n: int, fp: str, report: dict) -> None:
    o, s = report["metrics"], report["splits"]
    lines = [
        f"=== TRIAL {n} | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} | code {fp} | "
        f"{report['status']} ===",
        f"hypothesis : {S.HYPOTHESIS}",
        f"grid       : {json.dumps(S.PARAM_GRID)}  fixed: "
        f"{json.dumps({k: v for k, v in dataclasses.asdict(S.DEFAULT_PARAMS).items() if k not in S.PARAM_GRID})}",
        f"data       : {report['data']['first_bar']} -> {report['data']['last_bar']} "
        f"({report['data']['bars']} bars, sha256 {report['provenance']['sha256'][:12]})",
        f"metrics    : oos_sharpe={o['oos_sharpe']} is_sharpe_mean={o['is_sharpe_mean']} wfe={o['wfe']} "
        f"oos_max_dd={o['oos_max_drawdown']} oos_trades_per_year={o['oos_trades_per_year']} "
        f"oos_total_return={o['oos_total_return']} leak_test={'PASS' if report['tests']['leak']['passed'] else 'FAIL'}",
    ]
    for r in s:
        lines.append(
            f"split {r['split']}    : OOS {r['oos_period'][0][:10]}..{r['oos_period'][1][:10]} "
            f"params={ {k: r['chosen_params'][k] for k in S.PARAM_GRID} } IS_sh={r['is']['sharpe']} "
            f"OOS_sh={r['oos']['sharpe']} OOS_dd={r['oos']['max_drawdown']} OOS_tr/yr={r['oos']['trades_per_year']}"
        )
    for f in report["failures"]:
        lines.append(f"failure    : {f}")
    with TRIALS_LOG.open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")


# =============================================================================
# Main
# =============================================================================
def selftest(cfg: S.ExecutionConfig) -> dict:
    syn = synthetic_ohlc(1.0, seed=1)
    return {
        "complexity": complexity_audit(),
        "edge_cases": edge_case_tests(cfg),
        "leak_synthetic": run_leak_suite(syn, S.DEFAULT_PARAMS, cfg, n_points=6),
    }


def _all_passed(d) -> bool:
    if isinstance(d, dict):
        if "passed" in d and not d["passed"]:
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
    ap.add_argument("--selftest", action="store_true", help="synthetic unit tests only (no real-data metrics)")
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--data", default=None, help="optional OHLC CSV instead of the bundled source")
    ap.add_argument("--source-dir", default=str(DATA_DIR / "_src" / "financial-data"))
    args = ap.parse_args(argv)

    t_start = time.time()
    cfg = S.ExecutionConfig(pip=pip_size(args.pair))

    if args.selftest:
        tests = selftest(cfg)
        ok = _all_passed(tests)
        print(json.dumps({"mode": "selftest", "status": "PASS" if ok else "FAIL",
                          "failed": _failed_names(tests), "tests": tests}, indent=2, default=str))
        return 0 if ok else 1

    fp = code_fingerprint()
    trials = logged_trials()
    rerun = next((t for t in trials if t["code"] == fp), None)
    if rerun is None and len(trials) >= MAX_TRIALS:
        print(json.dumps({"status": "FAIL", "failures": [
            f"trial budget exhausted: {len(trials)} of {MAX_TRIALS} trials already logged in trials.log"]}, indent=2))
        return 1

    failures: list[str] = []
    df, provenance = load_market_data(args.pair, args.start, args.end, args.data, Path(args.source_dir))
    data_errors, data_info = validate_data(df)
    failures += [f"data: {e}" for e in data_errors]

    tests = selftest(cfg)
    wfa = run_wfa(df, cfg)

    # real-data leak test (on a 1-year slice for speed) with the first split's chosen parameters
    leak_slice = df.iloc[: min(len(df), 6500)]
    tests["leak"] = run_leak_suite(leak_slice, wfa["chosen"][0], cfg)
    # end-to-end parity of the stitched OOS run (vectorised vs event-driven, state carried)
    tests["parity_real_oos"] = parity(df, wfa["chosen"][0], cfg, segments=wfa["split_bounds"], params=wfa["chosen"])

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
    for name in _failed_names({k: v for k, v in tests.items() if k != "leak"}):
        failures.append(f"integrity: {name} failed")

    status = "PASS" if not failures else "FAIL"
    report = {
        "status": status,
        "failures": failures,
        "trial": {"code_fingerprint": fp, "rerun_of_trial": rerun["n"] if rerun else None,
                  "trial_number": rerun["n"] if rerun else len(trials) + 1, "budget": MAX_TRIALS},
        "hypothesis": S.HYPOTHESIS,
        "provenance": provenance,
        "data": data_info,
        "config": {"execution": dataclasses.asdict(S.ExecutionConfig(pip=cfg.pip)),
                   "cost_per_fill_pips": cfg.cost_per_fill / cfg.pip,
                   "param_grid": S.PARAM_GRID, "defaults": dataclasses.asdict(S.DEFAULT_PARAMS),
                   "n_splits": N_SPLITS, "is_fraction": IS_FRACTION},
        "metrics": {
            "oos_sharpe": o["sharpe"], "is_sharpe_mean": wfa["is_sharpe_mean"], "wfe": wfa["wfe"],
            "oos_max_drawdown": o["max_drawdown"], "oos_trades_per_year": o["trades_per_year"],
            "oos_total_return": o["total_return"], "oos_cagr": o["cagr"], "oos_ann_vol": o["ann_vol"],
            "oos_win_rate": o.get("win_rate"), "oos_profit_factor": o.get("profit_factor"),
            "oos_avg_trade_pips_net": o.get("avg_trade_pips_net"), "oos_years": o["years"],
            "oos_halts_daily_loss": o["halts_daily_loss"], "oos_killed_max_dd": o["killed_max_dd"],
            "leak_test_passed": tests["leak"]["passed"],
        },
        "splits": wfa["splits"],
        "tests": tests,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    if rerun is None:
        append_trial(len(trials) + 1, fp, report)
    RESULTS_DIR.mkdir(exist_ok=True)
    slim = {**report, "splits": [{k: v for k, v in r.items() if k != "is_grid"} for r in report["splits"]]}
    (RESULTS_DIR / "wfa_latest.json").write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_DIR / "wfa_oos_equity.csv").write_text(wfa["stitched"].nav.rename("nav").to_csv())
    print(json.dumps(slim, indent=2, default=str))
    if failures:
        print("\nHARNESS FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nHARNESS PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
