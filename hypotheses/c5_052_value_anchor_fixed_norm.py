"""
C5-052 - C5-050 with the trend normalisation scale fixed at 1/2 instead of chosen in the walk-forward (cycle 5,
data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-043/048 showed that which normalisation scales the grid offers moves the result by 0.15-0.2 (plateau selection
lands on different scales). C5-050 (0.72, the best) chose 1/4, 1/4, 1/2, 1/3, 1/2. This fixes the scale at 1/2 (C5-039's
choice) and lets the walk-forward choose only the fast ATR and the band, removing that source of selection noise.
Indicators (3) as C5-050. Tunable parameters (3): norm_scale (fixed at 1/2), fast_bars, band.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_050_averaged_value_anchor as C50
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_050_averaged_value_anchor import (FEATURE_PARAMS, INDICATORS, StrategyParams,  # noqa: F401
                                                     compute_features, history_bars, warmup_bars)

HYPOTHESIS = ("C5-052 C5-050 with the normalisation scale fixed at 1/2; fast ATR (60, 120, 240) and band (0.1, 0.2, "
              "0.3) chosen in the walk-forward; data-mining phase.")
USES_CARRY = False
DEPENDS = ("c5_050_averaged_value_anchor.py",) + C50.DEPENDS
DEFAULT_PARAMS = StrategyParams(norm_scale=0.5, fast_bars=240, band=0.3)
PARAM_GRID = {"fast_bars": [60, 120, 240], "band": [0.1, 0.2, 0.3]}


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C50.Signals):
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
