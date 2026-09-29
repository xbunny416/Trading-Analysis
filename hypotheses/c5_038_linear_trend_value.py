"""
C5-038 - value + trend with a capped LINEAR trend response instead of A3's fading response (cycle 5, data-mining
phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
A3's response u(z) = z exp(-z^2/4) / 0.89 peaks at |z| = sqrt(2) and fades for stronger trends; a capped linear
response keeps full size in the strongest trends. The book is otherwise C5-034's.

    u_k    = clip(z_k / 2, -1, 1)               (A3: z_k exp(-z_k^2 / 4) / 0.89); z_k, speeds and windows as A3
    s_trend = C5-006's dollar consensus of mean_k u_k
    s_value = 0.5 * (smooth value weights over 3 years + over 4 years)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)

`make_book(response, price_std_bars, signal_std_bars)` builds the book for any trend kernel; C5-039 reuses it.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses import c5_034_value_trend_variance_scaled as C34
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                   VALUE_LAGS, StrategyParams, combine)
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale

HYPOTHESIS = ("C5-038 value + trend with a capped linear trend response clip(z/2, -1, 1) instead of A3's fading "
              "response; otherwise C5-034; data-mining phase.")
INDICATORS = C34.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_034_value_trend_variance_scaled.py",) + C34.DEPENDS


def linear_response(z: float) -> float:
    return min(max(z / 2.0, -1.0), 1.0)


def make_book(response, price_std_bars: int, signal_std_bars: int):
    """(warmup_bars, history_bars, compute_features, Signals) for value + trend with the given trend kernel."""

    def warmup_bars(p) -> int:
        return max(price_std_bars + signal_std_bars, max(VALUE_LAGS), A.ATR_BARS, p.fast_bars) + 2

    def history_bars(p) -> int:
        return max(VALUE_LAGS) + 1

    def pair_trend(c: pd.Series) -> pd.Series:
        c_np = c.to_numpy(dtype=float)
        sd = c.rolling(price_std_bars).std()
        us = []
        for s_days, l_days in A3.SPEEDS_DAYS:
            x = pd.Series(A.ewma(c_np, A3.alpha(s_days)) - A.ewma(c_np, A3.alpha(l_days)), index=c.index)
            q = (x / sd).where(sd > 0)
            qs = q.rolling(signal_std_bars).std()
            z = (q / qs).where(qs > 0).to_numpy(dtype=float)
            us.append(np.array([response(v) if v == v else math.nan for v in z]))
        return pd.Series((us[0] + us[1] + us[2]) / 3.0, index=c.index)

    def dollar_trend(frames) -> dict[str, pd.Series]:
        trend = {k: pair_trend(f["close"]) for k, f in frames.items()}
        index = A.universe_index(frames)
        legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
        if legs:
            cols = pd.DataFrame({k: -d * trend[k].reindex(index).ffill() for k, (_, d) in legs.items()})
            s_usd = cols.mean(axis=1, skipna=False)
        else:
            s_usd = pd.Series(math.nan, index=index)
        return {k: ((-A.usd_leg(k)[1] * s_usd).reindex(f.index) if A.usd_leg(k) is not None
                    else pd.Series(0.0, index=f.index)) for k, f in frames.items()}

    def compute_features(data, p):
        """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
        single, frames = A.features_dict(data)
        trend = dollar_trend(frames)
        v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
        out = {}
        for k, f in frames.items():
            g = A.base_features(f)
            fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
            m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
            s = np.array([combine(t, a, b) for t, a, b in zip(trend[k].to_numpy(dtype=float),
                                                               v3[k].to_numpy(dtype=float),
                                                               v4[k].to_numpy(dtype=float))])
            g["signal"] = s * m
            out[k] = g
        return A.finish(single, out)

    class Signals(A.Signals):
        """A3's event-driven trend state (A3.Signals.update, line for line) with the given kernel."""

        def __init__(self, p, pairs, md):
            super().__init__(p, pairs, md)
            self._fast = {q: [A.EWMA(A3.alpha(s)) for s, _ in A3.SPEEDS_DAYS] for q in self.pairs}
            self._slow = {q: [A.EWMA(A3.alpha(l)) for _, l in A3.SPEEDS_DAYS] for q in self.pairs}
            self._price_sd = {q: A.RollingStd(price_std_bars) for q in self.pairs}
            self._q_sd = {q: [A.RollingStd(signal_std_bars) for _ in A3.SPEEDS_DAYS] for q in self.pairs}
            self._u = {q: math.nan for q in self.pairs}
            self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

        def update(self, pair: str) -> None:
            c = self.md.close(pair)
            sd = self._price_sd[pair].update(c)
            us = []
            for fast, slow, q_sd in zip(self._fast[pair], self._slow[pair], self._q_sd[pair]):
                x = fast.update(c) - slow.update(c)
                if not sd > 0:
                    us.append(math.nan)
                    continue
                qs = q_sd.update(x / sd)
                us.append(response((x / sd) / qs) if qs > 0 else math.nan)
            self._u[pair] = (us[0] + us[1] + us[2]) / 3.0
            self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], c)

        def signals(self) -> dict[str, float]:
            trend = C6.dollar_signals(dict(self._u))
            v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
            return {k: combine(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS)
                    for k in self.pairs}

    return warmup_bars, history_bars, compute_features, Signals


warmup_bars, history_bars, compute_features, Signals = make_book(linear_response, A3.PRICE_STD_BARS,
                                                                 A3.SIGNAL_STD_BARS)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
