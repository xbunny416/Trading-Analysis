"""
C5-019 - slow, volatility-managed dollar trend: C5-015's construction with A3's spans scaled UP (cycle 5, family 3).

Rationale (batch 6): C5-015 offered spans of A3 x 1/8, 1/4 and 1/2, and the walk-forward picked the slowest (1/2) in
all five splits; it still trailed C5-010 (A3's own speed, 0.36). Cycle 4's slowest trend (A2's 12-month blend) was
also its best per-pair TSMOM. This asks whether the dollar trend improves at A3's speed or slower: spans x 1, 1.5
or 2, i.e. up to (16, 48), (32, 96), (64, 192) days. The normalisation windows (63 and 252 days) stay A3's.

    s_pair = A3's formula with speeds (8, 24), (16, 48), (32, 96) days x speed     (c5_015_fast_dollar_trend.py)
    s_USD  = C5-006's dollar consensus of s_pair over the four USD pairs (EURJPY flat)
    s      = s_USD * clip(ATR_1440 / ATR_F, 0.5, 2.0)                             (C5-010's volatility scale)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover, price standard deviation, ATR (two speeds).
Tunable parameters (3): speed, fast_bars, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_015_fast_dollar_trend as C15
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-019 slow volatility-managed dollar trend: C5-015's construction with every A3 EWMA span scaled "
              "by `speed` (1, 1.5 or 2), chosen in the walk-forward.")
INDICATORS = C15.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_015_fast_dollar_trend.py",) + C15.DEPENDS


@dataclass(frozen=True)
class StrategyParams:
    speed: float = 1.5             # multiplier on A3's EWMA spans (1 = A3)
    fast_bars: int = 480           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (1 <= self.speed <= 4 and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2
                and 0 <= self.band < 1):
            raise ValueError("speed must be in [1, 4], fast_bars an int >= 2 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"speed": [1.0, 1.5, 2.0], "fast_bars": [120, 480, 960], "band": [0.2, 0.5]}
FEATURE_PARAMS = ("speed", "fast_bars")

warmup_bars = C15.warmup_bars
history_bars = C15.history_bars
compute_features = C15.compute_features


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(C15.Signals):
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
