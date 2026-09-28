"""
C5-010 - volatility-managed dollar trend (Moreira & Muir 2017, "Volatility-managed portfolios", JF 72(4)) on C5-006.

Rationale (batches 1-3): trend is the only source with a real gross edge, and the dollar consensus (C5-006) is its
most efficient form (0.25 Sharpe, 6.2 % drawdown). Moreira & Muir find that scaling a factor by the inverse of its
recent variance raises its Sharpe, because volatility clusters while returns barely predict it. The engine sizes by
a slow 1,440-bar ATR; this sizes by a fast one instead (within limits).

    s_C5-006 = dollar-trend consensus signal                    (hypotheses/c5_006_dollar_trend.py, unchanged)
    m        = clip(ATR_1440 / ATR_F, 0.5, 2.0)                 (ATR_F: Wilder ATR of hourly bars, period F)
    s        = s_C5-006 * m        (the engine caps |s| at 1 in sizing, so scaling up is limited by the signal)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover, price standard deviation, ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-010 volatility-managed dollar trend: C5-006's USD consensus scaled by ATR_1440 / ATR_F "
              "(clipped 0.5-2), i.e. sized by recent rather than 60-day volatility (Moreira-Muir 2017).")
INDICATORS = A3.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_006_dollar_trend.py", "a3_ewma_crossover.py")
SCALE_LIMITS = (0.5, 2.0)


@dataclass(frozen=True)
class StrategyParams:
    fast_bars: int = 240           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1):
            raise ValueError("fast_bars must be an int >= 2 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"fast_bars": [120, 240, 480, 960], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("fast_bars",)


def warmup_bars(p: StrategyParams) -> int:
    return max(C6.warmup_bars(C6.StrategyParams(band=p.band)), p.fast_bars + 2)


def history_bars(p: StrategyParams) -> int:
    return C6.history_bars(C6.StrategyParams(band=p.band))


def vol_scale(slow: float, fast: float, limits=SCALE_LIMITS) -> float:
    if not (slow > 0 and fast > 0):
        return math.nan
    return min(max(slow / fast, limits[0]), limits[1])


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams, limits=SCALE_LIMITS):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    base = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    out = {}
    for k, f in frames.items():
        g = base[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b, limits) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = g["signal"].to_numpy(dtype=float) * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C6.Signals):
    LIMITS = SCALE_LIMITS

    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        return {k: v * vol_scale(self.md.atr(k), self._fast_atr[k].value, self.LIMITS)
                for k, v in super().signals().items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
