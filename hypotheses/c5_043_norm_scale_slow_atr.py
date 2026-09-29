"""
C5-043 - C5-040 unchanged, with the fast-ATR grid moved past its edge (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-040 (0.69, the best) chose ATR_F = 240 bars, the largest on offer, in four of five splits (and 60 in the last).
This offers 240, 480 or 960 bars, with the normalisation-window scale at 1/4 or 1/2 (the two C5-040 used).
Indicators (3) and tunable parameters (3, band fixed at 0.3) as C5-040.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_040_norm_scale_grid import (FEATURE_PARAMS, INDICATORS, StrategyParams,  # noqa: F401
                                               compute_features, history_bars, warmup_bars)

HYPOTHESIS = ("C5-043 C5-040 unchanged with the fast ATR (240, 480, 960 bars) and normalisation scale (1/4, 1/2) "
              "chosen in the walk-forward; data-mining phase.")
USES_CARRY = False
DEPENDS = ("c5_040_norm_scale_grid.py",) + C40.DEPENDS
DEFAULT_PARAMS = StrategyParams(norm_scale=0.5, fast_bars=480, band=0.3)
PARAM_GRID = {"norm_scale": [0.25, 0.5], "fast_bars": [240, 480, 960]}


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C40.Signals):
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
