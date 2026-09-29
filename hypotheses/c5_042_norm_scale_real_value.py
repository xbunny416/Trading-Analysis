"""
C5-042 - C5-040 (value + trend, normalisation-window scale in the grid) with C5-036's Fisher-adjusted ("real")
value leg (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Combines the two value + trend changes that each scored at least level with its base: the real-rate-adjusted value
leg (C5-036, 0.58 vs C5-034's 0.60) and the faster trend normalisation (C5-040, 0.69).

    s_trend = C5-040's dollar trend (windows 63/252 days x norm_scale)
    s_value = C5-036's real value (3/4-year excess-return reversal, smooth weights)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score) with the
rate differential as its real-rate adjustment, ATR. Tunable parameters (3): norm_scale, fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_036_real_value_trend as C36
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import combine
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, history_bars, warmup_bars, windows)

HYPOTHESIS = ("C5-042 C5-040 with C5-036's Fisher-adjusted value leg (real-rate value + faster trend "
              "normalisation); data-mining phase.")
INDICATORS = C36.INDICATORS
USES_CARRY = True
DEPENDS = ("c5_040_norm_scale_grid.py", "c5_036_real_value_trend.py") + C40.DEPENDS


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
    v = C36.value_vec(frames)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        leg = A.usd_leg(k)
        t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        a3, a4 = (x.to_numpy(dtype=float) for x in v[k])
        g["signal"] = np.array([combine(x, y, z) for x, y, z in zip(t, a3, a4)]) * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C40.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._rv = C36.RealValueState(self.pairs)

    def update(self, pair: str) -> None:
        super().update(pair)
        self._rv.update(self.md, pair)

    def signals(self) -> dict[str, float]:
        trend = C6.dollar_signals(A3.Signals.signals(self))
        v3, v4 = self._rv.weights(self.md)
        return {k: combine(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS)
                for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
