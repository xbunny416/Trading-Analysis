"""
C5-028 - dollar value: long-horizon reversal of the USD against the basket, volatility-managed (the time-series
counterpart of AMP 2013's currency value) (cycle 5, family 5 building block).

Rationale (batches 8-12): cross-sectional value (dollar-neutral) is the second source with an edge, and the 4-year
horizon is the walk-forward's choice (C5-027). The USD's own long-horizon valuation is a separate bet: sell the dollar
after it has risen against the other four currencies for years, buy it after it has fallen. It is the value analogue
of the dollar-consensus trend and should run against it, so it is a candidate leg for the blend.

    x_c   = ln(price of currency c in USD now / L years ago)        (c = EUR, GBP, JPY, CAD; L in 3, 4 years)
    f     = -sign(mean_c x_c)          (+1: the foreign currencies fell against USD, so buy them; NaN until known)
    s     = dir * f * clip(ATR_1440 / ATR_F, 0.5, 2.0) for each USD pair (dir = +1 xxxUSD, -1 USDxxx); EURJPY 0

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): past returns (USD against the basket), ATR (two speeds).
Tunable parameters (3): value_years, fast_bars, band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale
from hypotheses.c5_017_volmanaged_dollar_carry import dollar_carry_sign

HYPOTHESIS = ("C5-028 dollar value: buy the four foreign currencies against USD after they fell over L years, sell "
              "them after they rose (sign of their mean L-year log return, reversed); scaled by C5-010's ATR ratio.")
INDICATORS = ("past returns (USD against the basket)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_017_volmanaged_dollar_carry.py", "c5_010_volmanaged_dollar_trend.py")


@dataclass(frozen=True)
class StrategyParams:
    value_years: int = 4           # horizon (252 trading days a year)
    fast_bars: int = 960           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.value_years, (int, np.integer)) and self.value_years >= 1
                and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1):
            raise ValueError("value_years an int >= 1, fast_bars an int >= 2, band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"value_years": [3, 4], "fast_bars": [120, 960], "band": [0.2, 0.5]}
FEATURE_PARAMS = ("value_years", "fast_bars")


def value_bars(p) -> int:
    return p.value_years * 252 * A.BARS_PER_DAY


def warmup_bars(p: StrategyParams) -> int:
    return max(value_bars(p), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return value_bars(p) + 1


def dollar_value_sign(foreign_perf: list[float]) -> float:
    """-sign(mean foreign log return against USD); the mean is rounded to 1e-9 so both paths agree at ties."""
    return -dollar_carry_sign(foreign_perf)


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    L = value_bars(p)
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    x = {k: (d * np.log(frames[k]["close"] / frames[k]["close"].shift(L))).reindex(index).ffill().to_numpy(dtype=float)
         for k, (_, d) in legs.items()}
    f_usd = pd.Series([dollar_value_sign([x[k][i] for k in legs]) if legs else math.nan for i in range(len(index))],
                      index=index)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        leg = A.usd_leg(k)
        s = (leg[1] * f_usd).reindex(g.index).to_numpy(dtype=float) if leg else np.zeros(len(g))
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
        L = value_bars(self.p)
        perf = []
        for k, (_, d) in self._legs.items():
            c, c0 = self.md.close(k), self.md.past_close(k, L)
            perf.append(d * math.log(c / c0) if c0 == c0 else math.nan)
        f = dollar_value_sign(perf) if self._legs else math.nan
        return {k: ((self._legs[k][1] * f) if k in self._legs else 0.0)
                * vol_scale(self.md.atr(k), self._fast_atr[k].value) for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
