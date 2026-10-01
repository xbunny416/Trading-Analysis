"""
C5-055 - C5-052 with the fast-ATR and band grids moved up: ATR_F 240, 360 or 480 bars; band 0.3, 0.4 or 0.5
(cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-052 (0.75, the best) chose ATR_F = 240 (the largest on offer) in every split and band 0.3 (the largest) in three
of five. This offers values above both edges. The model is C5-052's, unchanged.
Indicators (3) as C5-050. Tunable parameters (3): norm_scale (fixed at 1/2), fast_bars, band.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_052_value_anchor_fixed_norm as C52
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_050_averaged_value_anchor import (FEATURE_PARAMS, INDICATORS, StrategyParams,  # noqa: F401
                                                     compute_features, history_bars, warmup_bars)

HYPOTHESIS = ("C5-055 C5-052 unchanged with fast ATR (240, 360, 480) and band (0.3, 0.4, 0.5) chosen in the "
              "walk-forward; data-mining phase.")
USES_CARRY = False
DEPENDS = ("c5_052_value_anchor_fixed_norm.py",) + C52.DEPENDS
DEFAULT_PARAMS = StrategyParams(norm_scale=0.5, fast_bars=360, band=0.4)
PARAM_GRID = {"fast_bars": [240, 360, 480], "band": [0.3, 0.4, 0.5]}


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C52.Signals):
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
