"""
C5-046 - C5-040 with the trend weight fixed at 2/3 (cycle 5, data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
The value + trend blends weight the legs 50/50 in signal units. On their own the faster-normalised trend leg scores
0.49 (C5-041) and the smooth value leg 0.28 (C5-022), and they are nearly uncorrelated; weights proportional to the
legs' Sharpe ratios put about 0.64 on trend. C5-030 let the walk-forward pick the weight and it flip-flopped, so the
weight is fixed here (at 2/3) rather than tuned.

`make_book(w_trend, limits)` builds C5-040's book with any fixed trend weight and variance-scale clip; C5-047 reuses it.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (3): norm_scale, fast_bars, band (band fixed at 0.3).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_031_value_horizon_blend import VALUE_LAGS
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, history_bars, warmup_bars, windows)

HYPOTHESIS = "C5-046 C5-040 with the trend weight fixed at 2/3 (value 1/3); data-mining phase."
INDICATORS = C40.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_040_norm_scale_grid.py",) + C40.DEPENDS
W_TREND = 2.0 / 3.0


def make_book(w_trend: float, limits: tuple[float, float]):
    """(compute_features, Signals) for C5-040's book with a fixed trend weight and variance-scale clip."""

    def blend(t: float, v3: float, v4: float) -> float:
        return w_trend * t + (1.0 - w_trend) * (0.5 * v3 + 0.5 * v4)

    def compute_features(data, p):
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
        v3, v4 = (C24.xs_weights_vec(frames, lag, -1.0) for lag in VALUE_LAGS)
        out = {}
        for k, f in frames.items():
            g = A.base_features(f)
            leg = A.usd_leg(k)
            t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
            fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
            m = np.array([scale(a, b, POWER, limits) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
            s = np.array([blend(a, b, c) for a, b, c in zip(t, v3[k].to_numpy(dtype=float),
                                                             v4[k].to_numpy(dtype=float))])
            g["signal"] = s * m
            out[k] = g
        return A.finish(single, out)

    class Signals(C40.Signals):
        def signals(self) -> dict[str, float]:
            trend = C6.dollar_signals(A3.Signals.signals(self))
            v3, v4 = (C24.xs_weights_ev(self.md, lag, -1.0) for lag in VALUE_LAGS)
            return {k: blend(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, limits)
                    for k in self.pairs}

    return compute_features, Signals


compute_features, Signals = make_book(W_TREND, LIMITS)


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
