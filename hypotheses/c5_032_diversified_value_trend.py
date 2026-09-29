"""
C5-032 - value + trend with BOTH legs averaged over horizons: trend over A3 speeds x1 and x2, value over 3 and 4
years (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-031 (0.59) showed that averaging the value leg over its two leading horizons beats choosing one. The trend leg
had the same choice in C5-027/029 (speed x1 mostly, x2 at times); this averages it too.

    s_trend = 0.5 * (C5-006-style dollar consensus of A3 at speed x1 + at speed x2)     (C5-015's pair_trend)
    s_value = 0.5 * (C5-022 value weights over 3 years + over 4 years)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip(ATR_1440 / ATR_F, 0.5, 2.0)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_015_fast_dollar_trend as C15
from hypotheses import c5_019_slow_dollar_trend as C19
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses import c5_027_value_trend_tuned as C27
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-032 value + trend with both legs horizon-averaged: dollar trend over A3 speeds x1 and x2, value "
              "over 3 and 4 years; fast ATR and band chosen in the walk-forward; data-mining phase.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_027_value_trend_tuned.py", "c5_019_slow_dollar_trend.py") + C27.DEPENDS
SPEEDS = (1.0, 2.0)
VALUE_LAGS = (3 * 252 * A.BARS_PER_DAY, 4 * 252 * A.BARS_PER_DAY)


@dataclass(frozen=True)
class StrategyParams:
    fast_bars: int = 120           # fast ATR period in hourly bars
    band: float = 0.3              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1):
            raise ValueError("fast_bars must be an int >= 2 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"fast_bars": [60, 120, 240], "band": [0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("fast_bars",)


def warmup_bars(p: StrategyParams) -> int:
    return max(A3.PRICE_STD_BARS + A3.SIGNAL_STD_BARS, max(VALUE_LAGS), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return max(VALUE_LAGS) + 1


def combine(t1: float, t2: float, v3: float, v4: float) -> float:
    return 0.5 * (0.5 * t1 + 0.5 * t2) + 0.5 * (0.5 * v3 + 0.5 * v4)


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    t1, t2 = (C27.dollar_trend_vec(frames, sp) for sp in SPEEDS)
    v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        cols = [x[k].to_numpy(dtype=float) for x in (t1, t2, v3, v4)]
        g["signal"] = np.array([combine(*r) for r in zip(*cols)]) * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    """Two C5-015-style trend states (speeds x1, x2) share the market data; value comes from the close history."""

    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._trend = [C15.Signals(C19.StrategyParams(speed=sp, fast_bars=p.fast_bars, band=p.band), pairs, md)
                       for sp in SPEEDS]
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        for t in self._trend:
            A3.Signals.update(t, pair)          # the A3 state only; the fast ATR is kept once, here
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        t1, t2 = (C6.dollar_signals(A3.Signals.signals(t)) for t in self._trend)
        v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
        return {k: combine(t1[k], t2[k], v3[k], v4[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value)
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
