"""
C5-035 - C5-031 (value + trend, value averaged over 3 and 4 years) with NO volatility scaling (cycle 5, data-mining
phase; the control for C5-034).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated. The
volatility scale (ATR_1440 / ATR_F) was added to trend in batch 4 and carried into every blend since; this measures
what it adds to the value + trend book by removing it (m = 1 once the fast ATR is ready, so the warm-up and the
walk-forward grid are C5-031's).

    s = 0.5 * s_trend + 0.5 * s_value            s_trend: C5-006's dollar consensus; s_value: C5-031's 3/4-year value

Sizing (the engine's 1,440-bar ATR), execution, costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars (only delays the start, as in C5-031), band.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_031_value_horizon_blend as C31
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import (FEATURE_PARAMS, StrategyParams,  # noqa: F401
                                                   history_bars, warmup_bars)
from hypotheses.c5_034_value_trend_variance_scaled import make_book

HYPOTHESIS = "C5-035 C5-031 without volatility scaling (the control for C5-034); data-mining phase."
INDICATORS = C31.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_034_value_trend_variance_scaled.py", "c5_031_value_horizon_blend.py") + C31.DEPENDS
DEFAULT_PARAMS = StrategyParams(fast_bars=120, band=0.2)
PARAM_GRID = {"band": [0.1, 0.2, 0.3, 0.5]}

compute_features, Signals = make_book(0.0, (1.0, 1.0))


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
