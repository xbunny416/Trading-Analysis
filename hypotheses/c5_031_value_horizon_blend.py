"""
C5-031 - value + trend with the value leg averaged over the 3- and 4-year horizons (cycle 5, data-mining phase).

Context: see C5-030 (the user chose to keep searching the five pairs with variations; every trial is logged and
deflated). C5-027/029 picked a single value horizon per split (4 years mostly, 3 or 2 at times); averaging the two
leading horizons instead of choosing one removes that selection and usually makes a factor smoother. The trend leg
runs at A3's speed; the walk-forward chooses only the fast ATR and the band.

    s_value = 0.5 * (C5-022 value weights over 3 years + over 4 years)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip(ATR_1440 / ATR_F, 0.5, 2.0)    s_trend: C5-006's dollar consensus

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-031 value + trend with the value leg averaged over the 3- and 4-year horizons; fast ATR and band "
              "chosen in the walk-forward; data-mining phase.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_024_smooth_xs_momentum.py", "c5_022_smooth_value.py", "c5_020_xs_value.py",
           "c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a3_ewma_crossover.py")
VALUE_LAGS = (3 * 252 * A.BARS_PER_DAY, 4 * 252 * A.BARS_PER_DAY)


@dataclass(frozen=True)
class StrategyParams:
    fast_bars: int = 120           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1):
            raise ValueError("fast_bars must be an int >= 2 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"fast_bars": [60, 120, 240], "band": [0.1, 0.2, 0.3]}
FEATURE_PARAMS = ("fast_bars",)


def warmup_bars(p: StrategyParams) -> int:
    return max(C6.warmup_bars(C6.StrategyParams(band=p.band)), max(VALUE_LAGS), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return max(VALUE_LAGS) + 1


def combine(trend: float, v3: float, v4: float) -> float:
    return 0.5 * trend + 0.5 * (0.5 * v3 + 0.5 * v4)


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([combine(t, a, b) for t, a, b in zip(g["signal"].to_numpy(dtype=float),
                                                           v3[k].to_numpy(dtype=float), v4[k].to_numpy(dtype=float))])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C6.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        trend = super().signals()
        v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
        return {k: combine(trend[k], v3[k], v4[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value)
                for k in trend}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
