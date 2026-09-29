"""
C5-041 - the dollar trend ALONE with C5-040's normalisation-window grid and variance scaling (cycle 5, data-mining
phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Shorter normalisation windows lifted the value + trend book from 0.60 to 0.68-0.69 (C5-039/040). This isolates the
trend leg: if the gain comes from better trend timing, the trend-only book (C5-010/019: 0.36-0.41) should improve
too; if it only comes from how the legs combine, it should not.

    s = C5-040's dollar trend (A3 response, windows 63/252 days x norm_scale) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover, price standard deviation, ATR. Tunable parameters (3): norm_scale, fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, windows)

HYPOTHESIS = ("C5-041 dollar trend alone with the normalisation-window scale in the grid and variance scaling "
              "(C5-040's trend leg); data-mining phase.")
INDICATORS = A3.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_040_norm_scale_grid.py",) + C40.DEPENDS


def warmup_bars(p: StrategyParams) -> int:
    return max(sum(windows(p)), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    pw, sw = windows(p)
    trend = {k: C40.pair_trend(f["close"], pw, sw) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * trend[k].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        leg = A.usd_leg(k)
        t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = t * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C40.Signals):
    def signals(self) -> dict[str, float]:
        trend = C6.dollar_signals(A3.Signals.signals(self))
        return {k: trend[k] * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS) for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
