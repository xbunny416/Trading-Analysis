"""
C5-049 - C5-040 with ONE portfolio-level variance scale instead of one per pair (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Moreira & Muir scale a factor PORTFOLIO by its recent variance. The books so far scale each pair by its own ATR ratio,
which adds pair-level noise to a book that is mostly one dollar bet plus a dollar-neutral value tilt. This uses a
single scalar, the mean over the four USD pairs of (ATR_1440 / ATR_F)^2, each clipped to 0.25-4, for every pair.

    m_t = mean over EURUSD, GBPUSD, USDJPY, USDCAD of clip((ATR_1440 / ATR_F)^2, 0.25, 4)    (latest bar of each pair)
    s   = (0.5 * s_trend + 0.5 * s_value) * m_t          s_trend, s_value as C5-040

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (3): norm_scale, fast_bars, band (band fixed at 0.3).
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
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import VALUE_LAGS, combine
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, history_bars, warmup_bars, windows)

HYPOTHESIS = ("C5-049 C5-040 with one portfolio-level variance scale (mean of the USD pairs' clipped ATR-ratio "
              "squares) for every pair; data-mining phase.")
INDICATORS = C40.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_040_norm_scale_grid.py",) + C40.DEPENDS


def mean_scale(values: list[float]) -> float:
    if not values or any(v != v for v in values):
        return math.nan
    return sum(values) / len(values)


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    pw, sw = windows(p)
    trend = {k: C40.pair_trend(f["close"], pw, sw) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    base = {k: A.base_features(f) for k, f in frames.items()}
    per_pair = {}
    for k, f in frames.items():
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        per_pair[k] = pd.Series([scale(a, b, POWER, LIMITS) for a, b in zip(base[k]["atr"].to_numpy(dtype=float), fast)],
                                index=f.index)
    if legs:
        cols = pd.DataFrame({k: -d * trend[k].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
        sc = {k: per_pair[k].reindex(index).ffill().to_numpy(dtype=float) for k in legs}
        m_u = pd.Series([mean_scale([sc[k][i] for k in legs]) for i in range(len(index))], index=index)
    else:
        s_usd = pd.Series(math.nan, index=index)
        m_u = pd.Series(math.nan, index=index)
    v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
    out = {}
    for k, f in frames.items():
        g = base[k]
        leg = A.usd_leg(k)
        t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
        m = m_u.reindex(f.index).to_numpy(dtype=float)
        s = np.array([combine(a, b, c) for a, b, c in zip(t, v3[k].to_numpy(dtype=float), v4[k].to_numpy(dtype=float))])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C40.Signals):
    def signals(self) -> dict[str, float]:
        trend = C6.dollar_signals(A3.Signals.signals(self))
        v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
        legs = [k for k in self.pairs if A.usd_leg(k) is not None]
        m = mean_scale([scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS) for k in legs])
        return {k: combine(trend[k], v3[k], v4[k]) * m for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
