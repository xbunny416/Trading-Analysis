"""
C5-025 - three-way blend: dollar trend + smooth value + smooth cross-sectional momentum, volatility-managed (cycle 5,
family 5: ensemble).

Rationale (batch 9): the value + trend blend (C5-023, 0.49) reached what its two nearly uncorrelated legs predict
(about sqrt(0.41^2 + 0.28^2)). A higher Sharpe needs another leg. Cross-sectional momentum is dollar-neutral, so it
should overlap less with the dollar trend than per-pair trend does; C5-024 tests it alone, and this blend is
registered with it, before either runs.

    s_trend = C5-006's dollar consensus of A3                                   (c5_006_dollar_trend.py)
    s_value = C5-022's smooth 3-year value weights                             (c5_022_smooth_value.py)
    s_mom   = C5-024's smooth L-month cross-sectional momentum weights         (c5_024_smooth_xs_momentum.py)
    s       = (s_trend + s_value + s_mom) / 3 * clip(ATR_1440 / ATR_F, 0.5, 2.0)     (C5-010's volatility scale)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score, at the
value and momentum horizons), ATR (two speeds). Tunable parameters (3): lookback_months, fast_bars, band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_022_smooth_value as C22
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale
from hypotheses.c5_024_smooth_xs_momentum import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                  StrategyParams, lag_bars)

HYPOTHESIS = ("C5-025 trend + value + momentum: equal blend of the dollar-trend consensus (C5-006), smooth 3-year "
              "value (C5-022) and smooth L-month cross-sectional momentum (C5-024), scaled by C5-010's ATR ratio.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_024_smooth_xs_momentum.py", "c5_022_smooth_value.py", "c5_020_xs_value.py",
           "c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a3_ewma_crossover.py")


def warmup_bars(p: StrategyParams) -> int:
    return max(C6.warmup_bars(C6.StrategyParams(band=p.band)), C22.VALUE_BARS + 2, C24.warmup_bars(p))


def history_bars(p: StrategyParams) -> int:
    return max(C22.VALUE_BARS, lag_bars(p)) + 1


def blend(trend: float, value: float, mom: float) -> float:
    return (trend + value + mom) / 3.0


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    value = C22.value_weights_vec(frames)
    mom = C24.xs_weights_vec(frames, lag_bars(p), 1.0)
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([blend(t, v, u) for t, v, u in zip(g["signal"].to_numpy(dtype=float),
                                                         value[k].to_numpy(dtype=float),
                                                         mom[k].to_numpy(dtype=float))])
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
        value = C22.value_weights_ev(self.md)
        mom = C24.xs_weights_ev(self.md, lag_bars(self.p), 1.0)
        return {k: blend(trend[k], value[k], mom[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value)
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
