"""
C5-045 - value + trend with only the signal normalisation window halved (63 / 126 days) (cycle 5, data-mining
phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-039 halved both of A3's normalisation windows (63 -> 32 days of price standard deviation, 252 -> 126 days of
signal standard deviation) and scored 0.68. This and C5-044 halve one window each, to see which one matters. The
book is otherwise C5-034's (built with C5-038's make_book, A3's response).

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_038_linear_trend_value as C38
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                   StrategyParams)

HYPOTHESIS = ("C5-045 value + trend with only the signal normalisation window halved (63-day price std, "
              "126-day signal std); otherwise C5-034; data-mining phase.")
INDICATORS = C38.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_038_linear_trend_value.py",) + C38.DEPENDS
PRICE_STD_BARS = 63 * A.BARS_PER_DAY
SIGNAL_STD_BARS = 126 * A.BARS_PER_DAY

warmup_bars, history_bars, compute_features, Signals = C38.make_book(A3.response, PRICE_STD_BARS, SIGNAL_STD_BARS)


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
