"""
C5-034 - C5-031 (value + trend, value averaged over 3 and 4 years) with Moreira & Muir's variance scaling instead of
the volatility ratio (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Volatility management is the one ingredient that has helped every book it touched (+0.1 on trend). Moreira & Muir
(2017) scale by the inverse of recent VARIANCE; C5-010 onwards used the volatility ratio. This uses theirs:

    m = clip((ATR_1440 / ATR_F)^2, 0.25, 4)     (C5-010: power 1, clip 0.5-2)
    s = (0.5 * s_trend + 0.5 * s_value) * m      s_trend: C5-006's dollar consensus; s_value: C5-031's 3/4-year value

`make_book(power, limits)` builds the module's features and signals for any power, so C5-035 (no scaling) reuses it.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses import c5_031_value_horizon_blend as C31
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                   VALUE_LAGS, StrategyParams, combine, history_bars, warmup_bars)

HYPOTHESIS = ("C5-034 C5-031 with Moreira-Muir variance scaling (ATR ratio squared, clipped 0.25-4) instead of the "
              "volatility ratio; data-mining phase.")
INDICATORS = C31.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_031_value_horizon_blend.py",) + C31.DEPENDS
POWER, LIMITS = 2.0, (0.25, 4.0)


def scale(slow: float, fast: float, power: float, limits: tuple[float, float]) -> float:
    if not (slow > 0 and fast > 0):
        return math.nan
    return min(max((slow / fast) ** power, limits[0]), limits[1])


def make_book(power: float, limits: tuple[float, float]):
    """(compute_features, Signals) for C5-031's value + trend book with the given volatility scale."""

    def compute_features(data, p: StrategyParams):
        """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
        single, frames = A.features_dict(data)
        trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
        v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
        out = {}
        for k, f in frames.items():
            g = trend[k].copy()
            fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
            m = np.array([scale(a, b, power, limits) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
            s = np.array([combine(t, a, b) for t, a, b in zip(g["signal"].to_numpy(dtype=float),
                                                               v3[k].to_numpy(dtype=float),
                                                               v4[k].to_numpy(dtype=float))])
            g["signal"] = s * m
            out[k] = g
        return A.finish(single, out)

    class Signals(C6.Signals):
        def __init__(self, p, pairs, md):
            super().__init__(p, pairs, md)
            self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

        def update(self, pair: str) -> None:
            super().update(pair)
            self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

        def signals(self) -> dict[str, float]:
            trend = super().signals()
            v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
            return {k: combine(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, power, limits)
                    for k in trend}

    return compute_features, Signals


compute_features, Signals = make_book(POWER, LIMITS)


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
