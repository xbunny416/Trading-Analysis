"""
C5-030 - value + trend with the blend weight chosen in the walk-forward (cycle 5, data-mining phase).

Context: after 37 trials the user chose to keep searching the five pairs with parameter and blend variations, knowing
that this is data mining; every trial is logged and its deflated Sharpe counts all trials. C5-023/027 fixed the
blend at 50/50 in signal units. Here the trend weight is 1/3, 1/2 or 2/3, with the value horizon (3 or 4 years) and
the fast ATR (120 or 240 bars); the trend leg runs at A3's speed (C5-027's choice in 3 of 5 splits).

    s = (w * s_trend + (1 - w) * s_value) * clip(ATR_1440 / ATR_F, 0.5, 2.0)
    s_trend = C5-006's dollar consensus of A3;   s_value = C5-022's smooth value weights over value_years

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (4): w_trend, value_years, fast_bars, band (band fixed at 0.2, not in the grid).
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

HYPOTHESIS = ("C5-030 value + trend with the trend weight (1/3, 1/2, 2/3), value horizon (3, 4 years) and fast ATR "
              "(120, 240) chosen in the walk-forward; data-mining phase.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_024_smooth_xs_momentum.py", "c5_022_smooth_value.py", "c5_020_xs_value.py",
           "c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a3_ewma_crossover.py")


@dataclass(frozen=True)
class StrategyParams:
    w_trend: float = 0.5           # weight of the trend leg; value gets 1 - w_trend
    value_years: int = 4           # value horizon (252 trading days a year)
    fast_bars: int = 120           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        ok = (0 <= self.w_trend <= 1 and isinstance(self.value_years, (int, np.integer)) and self.value_years >= 1
              and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1)
        if not ok:
            raise ValueError("w_trend in [0, 1], value_years an int >= 1, fast_bars an int >= 2, band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"w_trend": [1 / 3, 0.5, 2 / 3], "value_years": [3, 4], "fast_bars": [120, 240]}
FEATURE_PARAMS = ("w_trend", "value_years", "fast_bars")


def value_bars(p) -> int:
    return p.value_years * 252 * A.BARS_PER_DAY


def warmup_bars(p: StrategyParams) -> int:
    return max(C6.warmup_bars(C6.StrategyParams(band=p.band)), value_bars(p), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return value_bars(p) + 1


def combine(w: float, trend: float, value: float) -> float:
    return w * trend + (1.0 - w) * value


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    value = C24.xs_weights_vec(frames, value_bars(p), -1.0)
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([combine(p.w_trend, t, v) for t, v in zip(g["signal"].to_numpy(dtype=float),
                                                                value[k].to_numpy(dtype=float))])
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
        value = C24.xs_weights_ev(self.md, value_bars(self.p), -1.0)
        return {k: combine(self.p.w_trend, trend[k], value[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value)
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
