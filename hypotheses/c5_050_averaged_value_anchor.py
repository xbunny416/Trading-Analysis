"""
C5-050 - C5-040 with the value anchor averaged over 3-4 years ago instead of two point anchors (cycle 5,
data-mining phase).

Context: the user chose to keep searching the five pairs with variations; every trial is logged and deflated.
AMP measure value against the AVERAGE price 4.5-5.5 years ago, which removes the noise of a single anchor hour.
C5-031/040 averaged the weights of two point anchors (3 and 4 years ago); this uses the mean log price over the whole
window from 4 years to 3 years ago as one anchor.

    x_c    = dir * (ln C_t - mean(ln C over bars t - 4y + 1 ... t - 3y))       (x_USD = 0; dir as the pair's USD leg)
    s_value = -z(x) / 2 (dollar-neutral smooth weights, C5-022's z_weights)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip((ATR_1440 / ATR_F)^2, 0.25, 4)     s_trend as C5-040

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
from hypotheses import c5_036_real_value_trend as C36
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_022_smooth_value import z_weights
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, windows)

HYPOTHESIS = ("C5-050 C5-040 with the value anchor = mean log price over 4 to 3 years ago (AMP-style averaged "
              "anchor) instead of two point anchors; data-mining phase.")
INDICATORS = C40.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_040_norm_scale_grid.py", "c5_036_real_value_trend.py") + C40.DEPENDS
ANCHOR_LAG = 3 * 252 * A.BARS_PER_DAY          # newest bar of the anchor window
ANCHOR_BARS = 1 * 252 * A.BARS_PER_DAY         # window length (3 to 4 years ago)


def warmup_bars(p: StrategyParams) -> int:
    return max(sum(windows(p)), ANCHOR_LAG + ANCHOR_BARS, A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return ANCHOR_LAG + 1


def blend(t: float, v: float) -> float:
    return 0.5 * t + 0.5 * v


# ============================================================================= research path (vectorised)
def value_vec(frames) -> dict[str, pd.Series]:
    legs = {p: A.usd_leg(p) for p in frames if A.usd_leg(p) is not None}
    index = A.universe_index(frames)
    x = {"USD": np.zeros(len(index))}
    for p, (ccy, d) in legs.items():
        lc = np.log(frames[p]["close"])
        anchor = lc.shift(ANCHOR_LAG).rolling(ANCHOR_BARS).mean()
        x[ccy] = (d * (lc - anchor)).reindex(index).ffill().to_numpy(dtype=float)
    ccys = list(x)
    rows = [z_weights({c: float(x[c][i]) for c in ccys}) for i in range(len(index))]
    w = {c: pd.Series([r[c] for r in rows], index=index) for c in ccys}
    return {p: ((legs[p][1] * w[legs[p][0]]).reindex(f.index) if p in legs else pd.Series(0.0, index=f.index))
            for p, f in frames.items()}


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
    v = value_vec(frames)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        leg = A.usd_leg(k)
        t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = np.array([blend(a, b) for a, b in zip(t, v[k].to_numpy(dtype=float))]) * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C40.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._legs = {q: A.usd_leg(q) for q in self.pairs if A.usd_leg(q) is not None}
        self._anchor = {q: C36.RollingMean(ANCHOR_BARS) for q in self._legs}
        self._anchor_val = {q: math.nan for q in self._legs}

    def update(self, pair: str) -> None:
        super().update(pair)
        if pair in self._legs:
            c0 = self.md.past_close(pair, ANCHOR_LAG)
            self._anchor_val[pair] = self._anchor[pair].push(math.log(c0) if c0 == c0 else math.nan)

    def signals(self) -> dict[str, float]:
        trend = C6.dollar_signals(A3.Signals.signals(self))
        x = {"USD": 0.0}
        for k, (ccy, d) in self._legs.items():
            x[ccy] = d * (math.log(self.md.close(k)) - self._anchor_val[k])
        w = z_weights(x)
        value = {k: (self._legs[k][1] * w[self._legs[k][0]] if k in self._legs else 0.0) for k in self.pairs}
        return {k: blend(trend[k], value[k]) * scale(self.md.atr(k), self._fast_atr[k].value, POWER, LIMITS)
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
