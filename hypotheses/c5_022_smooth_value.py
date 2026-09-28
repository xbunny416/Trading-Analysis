"""
C5-022 - currency value with smooth (z-score) weights, volatility-managed (cycle 5, family 5 building block).

Rationale (batch 8): C5-020's 3-year value signal earned about +$88k before costs, nearly uncorrelated with trend,
but its rank weights jump by a full step whenever two currencies swap places: 1,344 entries and exits cost $59k of
spread. This keeps the signal (the 3-year performance of USD, EUR, GBP, JPY, CAD against each other) and replaces
the ranks with weights that move continuously with it, so small changes cause small trades that the band absorbs.

    x_c = ln(price of currency c in USD now / 3 years ago)      (x_USD = 0; 756 trading days of hourly bars)
    w_c = -(x_c - mean(x)) / (2 * std(x))                        (cross-sectional z-score, population std; |w| <= 1)
    EURUSD, GBPUSD: s = w_EUR, w_GBP;   USDJPY, USDCAD: s = -w_JPY, -w_CAD;   EURJPY: 0
    s *= clip(ATR_1440 / ATR_F, 0.5, 2.0)                                     (C5-010's volatility scale)

The weights sum to zero across the five currencies (dollar-neutral), like A5's and C5-020's.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): past returns (cross-sectional z-score), ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale
from hypotheses.c5_020_xs_value import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID, VALUE_BARS,  # noqa: F401
                                        StrategyParams, history_bars, warmup_bars)

HYPOTHESIS = ("C5-022 currency value with smooth weights: C5-020's 3-year value signal as dollar-neutral "
              "cross-sectional z-score weights instead of ranks; scaled by C5-010's ATR ratio.")
INDICATORS = ("past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_020_xs_value.py", "c5_010_volmanaged_dollar_trend.py")


def z_weights(perf: dict[str, float]) -> dict[str, float]:
    """-(x - mean) / (2 std) per currency; NaN for all if any input is NaN or the cross-section is flat."""
    vals = list(perf.values())
    n = len(vals)
    if n < 2 or any(v != v for v in vals):
        return {c: math.nan for c in perf}
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / n)
    return {c: (-(v - mu) / (2.0 * sd) if sd > 0 else math.nan) for c, v in perf.items()}


def value_weights_vec(frames) -> dict[str, pd.Series]:
    legs = {p: A.usd_leg(p) for p in frames if A.usd_leg(p) is not None}
    index = A.universe_index(frames)
    perf = {"USD": np.zeros(len(index))}
    for p, (ccy, d) in legs.items():
        c = frames[p]["close"]
        perf[ccy] = (d * np.log(c / c.shift(VALUE_BARS))).reindex(index).ffill().to_numpy(dtype=float)
    ccys = list(perf)
    rows = [z_weights({c: float(perf[c][i]) for c in ccys}) for i in range(len(index))]
    w = {c: pd.Series([r[c] for r in rows], index=index) for c in ccys}
    out = {}
    for p, f in frames.items():
        leg = A.usd_leg(p)
        out[p] = (leg[1] * w[leg[0]]).reindex(f.index) if leg else pd.Series(0.0, index=f.index)
    return out


def value_weights_ev(md) -> dict[str, float]:
    legs = {p: A.usd_leg(p) for p in md.pairs if A.usd_leg(p) is not None}
    perf = {"USD": 0.0}
    for p, (ccy, d) in legs.items():
        c, c0 = md.close(p), md.past_close(p, VALUE_BARS)
        perf[ccy] = d * math.log(c / c0) if c0 == c0 else math.nan
    w = z_weights(perf)
    return {p: (A.usd_leg(p)[1] * w[A.usd_leg(p)[0]] if A.usd_leg(p) else 0.0) for p in md.pairs}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    v = value_weights_vec(frames)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = v[k].to_numpy(dtype=float) * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        return {k: v * vol_scale(self.md.atr(k), self._fast_atr[k].value)
                for k, v in value_weights_ev(self.md).items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
