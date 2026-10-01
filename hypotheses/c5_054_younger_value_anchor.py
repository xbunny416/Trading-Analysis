"""
C5-054 - C5-052 with the value anchor moved half a year younger: the mean log price from 3.5 to 2.5 years ago
(cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Moving C5-050's anchor half a year older (C5-053) dropped it from 0.72 to 0.48. This is the same shift in the other
direction, on C5-052's set-up (normalisation scale fixed at 1/2), so the two together show how sensitive the best book
is to where the anchor sits. Built with C5-051's make_book.
Indicators (3) as C5-050. Tunable parameters (3): norm_scale (fixed at 1/2), fast_bars, band.
"""
from __future__ import annotations

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_051_wide_value_anchor as C51
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_040_norm_scale_grid import FEATURE_PARAMS, StrategyParams  # noqa: F401

HYPOTHESIS = ("C5-054 C5-052 with the value anchor = mean log price from 3.5 to 2.5 years ago; data-mining phase.")
INDICATORS = C51.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_051_wide_value_anchor.py",) + C51.DEPENDS
DEFAULT_PARAMS = StrategyParams(norm_scale=0.5, fast_bars=240, band=0.3)
PARAM_GRID = {"fast_bars": [60, 120, 240], "band": [0.1, 0.2, 0.3]}
ANCHOR_LAG = int(2.5 * C51.YEAR)
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
