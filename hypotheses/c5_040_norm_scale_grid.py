"""
C5-040 - value + trend with the trend normalisation windows chosen in the walk-forward: A3's 63/252-day windows
scaled by 1/4, 1/3 or 1/2 (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-039 (0.68, the best) halved A3's normalisation windows. Rather than fixing another scale after seeing that, the
walk-forward chooses among 1/4, 1/3 and 1/2 per split. The book is otherwise C5-034's.

    price std window  = 63 days x scale,  signal std window = 252 days x scale     (scale in 1/4, 1/3, 1/2)
    s_trend = C5-006's dollar consensus of A3's response on those windows
    s_value = 0.5 * (smooth value weights over 3 years + over 4 years)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (3): norm_scale, fast_bars, band (band fixed at 0.3, C5-039's choice in every split).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses import c5_038_linear_trend_value as C38
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import VALUE_LAGS, combine
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale

HYPOTHESIS = ("C5-040 value + trend with the trend normalisation windows (A3's 63/252 days x 1/4, 1/3 or 1/2) "
              "chosen in the walk-forward; otherwise C5-034; data-mining phase.")
INDICATORS = C38.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_038_linear_trend_value.py",) + C38.DEPENDS


@dataclass(frozen=True)
class StrategyParams:
    norm_scale: float = 0.5        # multiplier on A3's normalisation windows
    fast_bars: int = 240           # fast ATR period in hourly bars
    band: float = 0.3              # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        if not (0 < self.norm_scale <= 1 and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2
                and 0 <= self.band < 1):
            raise ValueError("norm_scale in (0, 1], fast_bars an int >= 2, band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"norm_scale": [0.25, 1 / 3, 0.5], "fast_bars": [60, 120, 240]}
FEATURE_PARAMS = ("norm_scale", "fast_bars")


def windows(p) -> tuple[int, int]:
    return (int(round(63 * p.norm_scale)) * A.BARS_PER_DAY, int(round(252 * p.norm_scale)) * A.BARS_PER_DAY)


def warmup_bars(p: StrategyParams) -> int:
    return max(sum(windows(p)), max(VALUE_LAGS), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return max(VALUE_LAGS) + 1


# ============================================================================= research path (vectorised)
def pair_trend(c: pd.Series, price_std_bars: int, signal_std_bars: int) -> pd.Series:
    """A3's signal on the given normalisation windows; row t uses closes <= t."""
    c_np = c.to_numpy(dtype=float)
    sd = c.rolling(price_std_bars).std()
    us = []
    for s_days, l_days in A3.SPEEDS_DAYS:
        x = pd.Series(A.ewma(c_np, A3.alpha(s_days)) - A.ewma(c_np, A3.alpha(l_days)), index=c.index)
        q = (x / sd).where(sd > 0)
        qs = q.rolling(signal_std_bars).std()
        z = (q / qs).where(qs > 0)
        us.append(z * np.exp(-z * z / 4.0) / 0.89)
    return (us[0] + us[1] + us[2]) / 3.0


def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    pw, sw = windows(p)
    trend = {k: pair_trend(f["close"], pw, sw) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * trend[k].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        leg = A.usd_leg(k)
        t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([combine(a, b, c) for a, b, c in zip(t, v3[k].to_numpy(dtype=float), v4[k].to_numpy(dtype=float))])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A3.Signals):
    """A3's event-driven state with the windows replaced; value and the variance scale as C5-034."""

    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        pw, sw = windows(p)
        self._price_sd = {q: A.RollingStd(pw) for q in self.pairs}
        self._q_sd = {q: [A.RollingStd(sw) for _ in A3.SPEEDS_DAYS] for q in self.pairs}
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        trend = C6.dollar_signals(super().signals())
        v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
        return {k: combine(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS)
                for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
