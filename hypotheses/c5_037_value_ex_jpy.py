"""
C5-037 - C5-034 (value + trend, value over 3/4 years, variance-scaled) with JPY removed from the value leg
(cycle 5, data-mining phase; HINDSIGHT selection).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
Disclosure: this change was chosen after seeing C5-031/034/036's per-pair results, where USDJPY lost in every
value + trend blend (the value leg kept buying the yen while it fell in 2012-15 and 2021-22) although trend alone made
money on it. It is pair selection by hindsight, the kind of choice most likely to fail out of sample; the deflated
Sharpe counts it, and only the untouched 2023 holdout could confirm it.

    s_value = 0.5 * (smooth value weights over 3 years + over 4 years), cross-section USD, EUR, GBP, CAD (no JPY);
              USDJPY's value signal is 0
    s_trend = C5-006's dollar consensus of A3 (all four USD pairs, unchanged)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)          (C5-034's scale)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_034_value_trend_variance_scaled as C34
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_022_smooth_value import z_weights
from hypotheses.c5_031_value_horizon_blend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                   VALUE_LAGS, StrategyParams, combine, history_bars, warmup_bars)
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale

HYPOTHESIS = ("C5-037 C5-034 with JPY removed from the value leg (hindsight pair selection after USDJPY lost in "
              "every value + trend blend); data-mining phase.")
INDICATORS = C34.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_034_value_trend_variance_scaled.py",) + C34.DEPENDS
EXCLUDE = frozenset({"JPY"})


def _legs(pairs):
    return {p: A.usd_leg(p) for p in pairs if A.usd_leg(p) is not None and A.usd_leg(p)[0] not in EXCLUDE}


def value_vec(frames, lag: int) -> dict[str, pd.Series]:
    legs = _legs(frames)
    index = A.universe_index(frames)
    perf = {"USD": np.zeros(len(index))}
    for p, (ccy, d) in legs.items():
        c = frames[p]["close"]
        perf[ccy] = (d * np.log(c / c.shift(lag))).reindex(index).ffill().to_numpy(dtype=float)
    ccys = list(perf)
    rows = [z_weights({c: float(perf[c][i]) for c in ccys}) for i in range(len(index))]
    w = {c: pd.Series([r[c] for r in rows], index=index) for c in ccys}
    return {p: ((legs[p][1] * w[legs[p][0]]).reindex(f.index) if p in legs else pd.Series(0.0, index=f.index))
            for p, f in frames.items()}


def value_ev(md, lag: int) -> dict[str, float]:
    legs = _legs(md.pairs)
    perf = {"USD": 0.0}
    for p, (ccy, d) in legs.items():
        c, c0 = md.close(p), md.past_close(p, lag)
        perf[ccy] = d * math.log(c / c0) if c0 == c0 else math.nan
    w = z_weights(perf)
    return {p: (legs[p][1] * w[legs[p][0]] if p in legs else 0.0) for p in md.pairs}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    v3, v4 = (value_vec(frames, lag) for lag in VALUE_LAGS)
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
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
        v3, v4 = (value_ev(self.md, lag) for lag in VALUE_LAGS)
        return {k: combine(trend[k], v3[k], v4[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS)
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
