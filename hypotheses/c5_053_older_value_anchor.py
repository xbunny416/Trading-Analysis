"""
C5-053 - C5-050 with the value anchor moved half a year older: the mean log price from 4.5 to 3.5 years ago
(cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
AMP anchor value on the average price 4.5-5.5 years ago; the data (from 2005) cannot support that and a 3-5 year
window, so C5-050 used 4-3 years ago. This moves the one-year window half a year towards AMP's (signal from mid-2009).
Built with C5-051's make_book. Indicators (3) and tunable parameters (3, band fixed at 0.3) as C5-050.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_051_wide_value_anchor as C51
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams)

HYPOTHESIS = ("C5-053 C5-050 with the value anchor = mean log price from 4.5 to 3.5 years ago; data-mining phase.")
INDICATORS = C51.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_051_wide_value_anchor.py",) + C51.DEPENDS
ANCHOR_LAG = int(3.5 * C51.YEAR)
ANCHOR_BARS = C51.YEAR

warmup_bars, history_bars, compute_features, Signals = C51.make_book(ANCHOR_LAG, ANCHOR_BARS)


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
