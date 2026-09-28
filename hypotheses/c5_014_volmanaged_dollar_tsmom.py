"""
C5-014 - volatility-managed dollar consensus of time-series momentum (A2's 1/3/12-month blend) (cycle 5, family 3).

Rationale (batch 4): C5-010's structure (dollar consensus + volatility scale) with a different trend definition, the
published 1/3/12-month TSMOM blend instead of EWMA crossovers, to see whether the gain is specific to one signal.

    s_A2  = (sign r_1m + sign r_3m + sign r_12m) / 3 per pair         (hypotheses/a2_tsmom_blend.py, unchanged)
    s_USD = C5-006's dollar consensus applied to s_A2                   (hypotheses/c5_006_dollar_trend.py)
    s     = s_USD per pair * clip(ATR_1440 / ATR_F, 0.5, 2.0)          (C5-010's volatility scale)

Indicators (2): past returns, ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a2_tsmom_blend as A2
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                       StrategyParams, vol_scale)

HYPOTHESIS = ("C5-014 volatility-managed dollar consensus of the A2 1/3/12-month TSMOM blend (C5-010's structure "
              "with a different trend signal).")
INDICATORS = ("past returns", "ATR")
USES_CARRY = False
DEPENDS = ("c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a2_tsmom_blend.py",
           "a3_ewma_crossover.py")


def warmup_bars(p: StrategyParams) -> int:
    return max(A2.warmup_bars(A2.StrategyParams(band=p.band)), p.fast_bars + 2)


def history_bars(p: StrategyParams) -> int:
    return A2.history_bars(A2.StrategyParams(band=p.band))


def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    raw = A2.compute_features(frames, A2.StrategyParams(band=p.band))
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * raw[k]["signal"].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    out = {}
    for k, f in frames.items():
        g = raw[k].copy()
        leg = A.usd_leg(k)
        s = (-leg[1] * s_usd).reindex(g.index).to_numpy(dtype=float) if leg is not None else np.zeros(len(g))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(A2.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        dol = C6.dollar_signals(super().signals())
        return {k: v * vol_scale(self.md.atr(k), self._fast_atr[k].value) for k, v in dol.items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
