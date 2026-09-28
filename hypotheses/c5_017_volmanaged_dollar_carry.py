"""
C5-017 - volatility-managed dollar carry (Lustig, Roussanov & Verdelhan 2014, "Countercyclical currency risk premia",
JFE 111(3); Moreira & Muir 2017) (cycle 5, family 5 building block).

Rationale (batches 1-5): cycle 4's per-pair carry (A4) lost, and it ran against trend (daily correlation -0.6). LRV's
dollar carry is a different, time-series bet: hold every foreign currency against the USD when foreign rates are on
average above the US rate, and the USD against all of them otherwise. It is one position on the USD, like the dollar
trend (C5-010), and is volatility-managed the same way. A positive result would be the second independent source an
ensemble needs; a negative one closes carry on this universe.

    a     = mean over EURUSD, GBPUSD, USDJPY, USDCAD of (dir * d_pair)     d = known r_base - r_quote (percent p.a.)
            dir = +1 for xxxUSD, -1 for USDxxx, so dir * d = r_foreign - r_USD
    s_FX  = sign(a)                                  (+1: long the foreign currencies; NaN until every rate is known)
    s     = dir * s_FX * clip(ATR_1440 / ATR_F, 0.5, 2.0) for each USD pair; EURJPY flat

Rates are those public at the time (OECD monthly from the next month, policy decisions from the next day), as in A4.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): rate differential, ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                       StrategyParams, vol_scale)

HYPOTHESIS = ("C5-017 volatility-managed dollar carry (Lustig-Roussanov-Verdelhan 2014): long the foreign "
              "currencies against the USD when their average known rate is above the US rate, short otherwise; "
              "scaled by C5-010's ATR ratio.")
INDICATORS = ("rate differential", "ATR")
USES_CARRY = True
DEPENDS = ("c5_010_volmanaged_dollar_trend.py",)


def warmup_bars(p: StrategyParams) -> int:
    return max(A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


def dollar_carry_sign(foreign_minus_us: list[float]) -> float:
    """sign of the average (rounded to 1e-9 so both paths agree at exact ties); NaN if any rate is unknown."""
    if not foreign_minus_us or any(v != v for v in foreign_minus_us):
        return math.nan
    a = round(float(sum(foreign_minus_us)) / len(foreign_minus_us), 9)
    return float((a > 0) - (a < 0))


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    base = {k: A.base_features(f) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = {k: (d * base[k]["carry"].reindex(index).ffill()).to_numpy(dtype=float) for k, (_, d) in legs.items()}
        s_fx = pd.Series([dollar_carry_sign([cols[k][i] for k in legs]) for i in range(len(index))], index=index)
    else:
        s_fx = pd.Series(math.nan, index=index)
    out = {}
    for k, f in frames.items():
        g = base[k]
        leg = A.usd_leg(k)
        if leg is None:
            g["signal"] = 0.0                  # EURJPY: always flat
            out[k] = g
            continue
        s = (leg[1] * s_fx).reindex(g.index).to_numpy(dtype=float)
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}
        self._legs = {q: A.usd_leg(q) for q in self.pairs if A.usd_leg(q) is not None}

    def update(self, pair: str) -> None:
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        s_fx = dollar_carry_sign([d * self.md.carry[k] for k, (_, d) in self._legs.items()])
        return {k: (self._legs[k][1] * s_fx * vol_scale(self.md.atr(k), self._fast_atr[k].value)
                    if k in self._legs else 0.0) for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
