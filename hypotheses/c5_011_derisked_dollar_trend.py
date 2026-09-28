"""
C5-011 - dollar trend that only de-risks when volatility jumps (cycle 5, family 3).

The one-sided sibling of C5-010, logged as its own trial: the same scale ATR_1440 / ATR_F, but capped at 1, so
exposure is cut when recent volatility exceeds the 60-day level and never raised above C5-006's.

    s = s_C5-006 * clip(ATR_1440 / ATR_F, 0.0, 1.0)            (hypotheses/c5_010_volmanaged_dollar_trend.py)

Indicators (3): EWMA crossover, price standard deviation, ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_010_volmanaged_dollar_trend as C10
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                       StrategyParams, history_bars, warmup_bars)

HYPOTHESIS = ("C5-011 de-risk-only dollar trend: C5-006's USD consensus scaled by min(1, ATR_1440 / ATR_F), cutting "
              "exposure when recent volatility exceeds the 60-day level.")
INDICATORS = C10.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a3_ewma_crossover.py")
LIMITS = (0.0, 1.0)


def compute_features(data, p: StrategyParams):
    return C10.compute_features(data, p, limits=LIMITS)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C10.Signals):
    LIMITS = LIMITS


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
