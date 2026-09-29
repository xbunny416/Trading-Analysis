"""
C5-036 - value + trend with a Fisher-adjusted ("real") value leg, variance-scaled as C5-034 (cycle 5, data-mining
phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated. AMP's
value uses the REAL exchange rate, which needs CPI data this environment cannot download. With equal real rates
(Fisher), the inflation differential is roughly the interest differential, so the real appreciation of currency c
against USD over a window of Y years is approximately

    q_c = ln(S_c,now / S_c,then) + mean(r_c - r_USD over the window) / 100 * Y

which is also c's cumulative excess return (spot plus carry). The rates are the known ones (the `carry` column).

    s_value = 0.5 * (smooth value weights on q over 3 years + over 4 years)       (-z / 2, dollar-neutral)
    s_trend = C5-006's dollar consensus of A3
    s       = (0.5 * s_trend + 0.5 * s_value) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)   (C5-034's variance scale)

Registration note: a value-only companion trial was dropped before registration because in a one-pair universe (the
single-pair parity tests) its two horizons can cancel exactly and it never trades there; nothing was run on real data.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score) with the
rate differential as its real-rate adjustment, ATR. Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_031_value_horizon_blend as C31
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_022_smooth_value import z_weights
from hypotheses.c5_031_value_horizon_blend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                   VALUE_LAGS, StrategyParams, combine, history_bars, warmup_bars)
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale

HYPOTHESIS = ("C5-036 value + trend with a Fisher-adjusted value leg (3/4-year excess-return reversal, smooth "
              "weights), variance-scaled as C5-034; data-mining phase.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)",
              "past returns (cross-sectional z-score, real-rate adjusted)", "ATR")
USES_CARRY = True
DEPENDS = ("c5_034_value_trend_variance_scaled.py", "c5_031_value_horizon_blend.py") + C31.DEPENDS
BARS_PER_YEAR = 252 * A.BARS_PER_DAY


class RollingMean:
    """Mean of the last n values pushed; NaN unless all n are known (pandas rolling(n).mean() semantics)."""

    def __init__(self, n: int):
        self.n, self.q, self.sum, self.nans = n, deque(), 0.0, 0

    def push(self, v: float) -> float:
        self.q.append(v)
        if v == v:
            self.sum += v
        else:
            self.nans += 1
        if len(self.q) > self.n:
            old = self.q.popleft()
            if old == old:
                self.sum -= old
            else:
                self.nans -= 1
        return self.sum / self.n if len(self.q) == self.n and self.nans == 0 else math.nan


def real_weights(perf: dict[str, float]) -> dict[str, float]:
    """Value weights (-z / 2) on real appreciation; NaN for all if any is unknown."""
    return z_weights(perf)


# ============================================================================= vectorised value
def value_vec(frames) -> dict[str, list[pd.Series]]:
    """Per pair, the value signal for each horizon in VALUE_LAGS."""
    legs = {p: A.usd_leg(p) for p in frames if A.usd_leg(p) is not None}
    index = A.universe_index(frames)
    out = {p: [] for p in frames}
    for lag in VALUE_LAGS:
        q = {"USD": np.zeros(len(index))}
        for p, (ccy, d) in legs.items():
            f = frames[p]
            spot = d * np.log(f["close"] / f["close"].shift(lag))
            carry = f["carry"] if "carry" in f.columns else pd.Series(np.nan, index=f.index)
            cm = (d * carry).rolling(lag).mean()
            q[ccy] = (spot + cm / 100.0 * (lag / BARS_PER_YEAR)).reindex(index).ffill().to_numpy(dtype=float)
        ccys = list(q)
        rows = [real_weights({c: float(q[c][i]) for c in ccys}) for i in range(len(index))]
        w = {c: pd.Series([r[c] for r in rows], index=index) for c in ccys}
        for p, f in frames.items():
            leg = A.usd_leg(p)
            out[p].append((leg[1] * w[leg[0]]).reindex(f.index) if leg else pd.Series(0.0, index=f.index))
    return out


# ============================================================================= event-driven value state
class RealValueState:
    """Per-pair rolling carry means for each horizon (fed from the pair's own bars, like the vectorised path)."""

    def __init__(self, pairs):
        self.legs = {q: A.usd_leg(q) for q in pairs if A.usd_leg(q) is not None}
        self.cm = {q: [RollingMean(lag) for lag in VALUE_LAGS] for q in self.legs}
        self.last = {q: [math.nan] * len(VALUE_LAGS) for q in self.legs}

    def update(self, md, pair: str) -> None:
        if pair in self.legs:
            d = self.legs[pair][1]
            self.last[pair] = [r.push(d * md.carry[pair]) for r in self.cm[pair]]

    def weights(self, md) -> list[dict[str, float]]:
        """Pair signals for each horizon."""
        out = []
        for i, lag in enumerate(VALUE_LAGS):
            q = {"USD": 0.0}
            for p, (ccy, d) in self.legs.items():
                c, c0 = md.close(p), md.past_close(p, lag)
                spot = d * math.log(c / c0) if c0 == c0 else math.nan
                q[ccy] = spot + self.last[p][i] / 100.0 * (lag / BARS_PER_YEAR)
            w = real_weights(q)
            out.append({p: (A.usd_leg(p)[1] * w[A.usd_leg(p)[0]] if A.usd_leg(p) else 0.0) for p in md.pairs})
        return out


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    v = value_vec(frames)
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        a3, a4 = (x.to_numpy(dtype=float) for x in v[k])
        s = np.array([combine(t, x, y) for t, x, y in zip(g["signal"].to_numpy(dtype=float), a3, a4)])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C6.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}
        self._rv = RealValueState(self.pairs)

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))
        self._rv.update(self.md, pair)

    def signals(self) -> dict[str, float]:
        trend = super().signals()
        v3, v4 = self._rv.weights(self.md)
        return {k: combine(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS)
                for k in trend}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
