"""
C5-047 - C5-040 with a wider variance-scale clip, 0.1-10 instead of 0.25-4 (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Volatility management is the ingredient that has helped every book (+0.1). The variance scale (ATR_1440 / ATR_F)^2 is
clipped at 0.25-4; a wider clip lets it de-risk harder in volatility spikes (upscaling is limited anyway, as the
engine caps |s| at 1). Built with C5-046's make_book at C5-040's 50/50 weights.

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (3): norm_scale, fast_bars, band (band fixed at 0.3).
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_046_trend_weighted_blend as C46
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, history_bars, warmup_bars)

HYPOTHESIS = "C5-047 C5-040 with the variance-scale clip widened to 0.1-10; data-mining phase."
INDICATORS = C46.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_046_trend_weighted_blend.py",) + C46.DEPENDS
LIMITS_WIDE = (0.1, 10.0)

compute_features, Signals = C46.make_book(0.5, LIMITS_WIDE)


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
