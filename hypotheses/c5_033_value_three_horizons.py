"""
C5-033 - value + trend with the value leg averaged over 2, 3 and 4 years (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
C5-031 (0.59) averaged value over 3 and 4 years; C5-027/029 also chose 2 years in one split. This averages all three
horizons, with the trend leg at A3's speed and the band grid moved up (C5-031 chose band 0.3 in 3 of 5 splits).

    s_value = (C5-022 value weights over 2 + 3 + 4 years) / 3
    s       = (0.5 * s_trend + 0.5 * s_value) * clip(ATR_1440 / ATR_F, 0.5, 2.0)    s_trend: C5-006's dollar consensus

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses import c5_031_value_horizon_blend as C31
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale
from hypotheses.c5_031_value_horizon_blend import FEATURE_PARAMS, StrategyParams  # noqa: F401

HYPOTHESIS = ("C5-033 value + trend with the value leg averaged over 2, 3 and 4 years; fast ATR and band chosen in "
              "the walk-forward; data-mining phase.")
INDICATORS = C31.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_031_value_horizon_blend.py",) + C31.DEPENDS
VALUE_LAGS = tuple(y * 252 * A.BARS_PER_DAY for y in (2, 3, 4))
DEFAULT_PARAMS = StrategyParams(fast_bars=120, band=0.3)
PARAM_GRID = {"fast_bars": [60, 120, 240], "band": [0.3, 0.5]}


def warmup_bars(p: StrategyParams) -> int:
    return max(C6.warmup_bars(C6.StrategyParams(band=p.band)), max(VALUE_LAGS), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return max(VALUE_LAGS) + 1


def combine(trend: float, values: tuple[float, ...]) -> float:
    return 0.5 * trend + 0.5 * (sum(values) / len(values))


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    vals = [C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS]
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        cols = [v[k].to_numpy(dtype=float) for v in vals]
        g["signal"] = np.array([combine(t, vs) for t, vs in zip(g["signal"].to_numpy(dtype=float), zip(*cols))]) * m
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
        vals = [C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS]
        return {k: combine(trend[k], tuple(v[k] for v in vals)) * vol_scale(self.md.atr(k), self._fast_atr[k].value)
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
