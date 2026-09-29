"""
C5-029 - C5-027 with its grid extended past the two edges the walk-forward chose (cycle 5, family 5: ensemble,
refinement).

Rationale (batch 12): C5-027 (0.56) picked the longest value horizon on offer (4 years) in four of five splits and the
fastest volatility scale (ATR_F = 120 bars) in all five. A choice at the edge of a grid says the optimum may lie
beyond it, so this offers 4 or 5 years (5 is AMP's published horizon) and ATR_F of 60, 120 or 240 bars, with the
trend speed at 1 or 2. The model is C5-027's, unchanged.

Disclosure: a 5-year horizon needs data from 2005 + 5 years, so it starts trading in 2010; split 1's in-sample window
(2006-07 -> 2011-10) then has about 1.75 years of positions, which biases that split's selection against it.
Indicators (3) and tunable parameters (4, band fixed at 0.2) as C5-027.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_027_value_trend_tuned import (FEATURE_PARAMS, INDICATORS, StrategyParams,  # noqa: F401
                                                 compute_features, history_bars, warmup_bars)
from hypotheses.c5_027_value_trend_tuned import Signals as C27Signals
from hypotheses import c5_027_value_trend_tuned as C27

HYPOTHESIS = ("C5-029 value + trend at the grid edges: C5-027 unchanged, with the value horizon (4, 5 years), the "
              "fast ATR (60, 120, 240 bars) and the trend speed (x1, x2) chosen in the walk-forward.")
USES_CARRY = False
DEPENDS = ("c5_027_value_trend_tuned.py",) + C27.DEPENDS
DEFAULT_PARAMS = StrategyParams(speed=1.0, value_years=4, fast_bars=120, band=0.2)
PARAM_GRID = {"speed": [1.0, 2.0], "value_years": [4, 5], "fast_bars": [60, 120, 240]}


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C27Signals):
    pass


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
