"""
C5-009 - large-shock reversal: fade hourly shocks for a few bars (cycle 5, family 1b).

The mirror image of C5-008, logged as its own trial (and counted in the deflated Sharpe). Rationale: C5-001 showed
FX mean reversion loses at 1-7 days, but an overshoot on a single hour (news, thin liquidity) may partly retrace
within hours.

    shock at bar t  <=>  |C_t - C_{t-1}| > k * ATR_{t-1};   on a shock: position = -sign(C_t - C_{t-1}) for `hold` bars

The detector is C5-008's (hypotheses/c5_008_shock_follow.py, unchanged) with the direction reversed.
Indicators (2): price change, ATR. Tunable parameters (3): k, hold, band (grid: k x hold).
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_008_shock_follow as C8
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_008_shock_follow import StrategyParams, warmup_bars, history_bars  # noqa: F401

HYPOTHESIS = ("C5-009 large-shock reversal: after an hourly close-to-close move beyond k x ATR, hold the opposite "
              "direction for `hold` bars.")
INDICATORS = C8.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_008_shock_follow.py",)
DEFAULT_PARAMS = StrategyParams(k=3.0, hold=3)
PARAM_GRID = {"k": [2.5, 3.0, 3.5, 4.0], "hold": [1, 3, 6, 12]}
FEATURE_PARAMS = ("k", "hold")


def compute_features(data, p: StrategyParams):
    return C8.compute_features(data, p, direction=-1)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C8.Signals):
    DIRECTION = -1


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
