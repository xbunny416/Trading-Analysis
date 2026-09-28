"""
C5-012 - volatility-managed blend of the dollar-trend consensus and per-pair trend (cycle 5, family 3).

Rationale (batch 4): the volatility-managed dollar consensus (C5-010) is the best book so far (0.36). The consensus
throws away pair-specific trends (EUR vs GBP vs JPY vs CAD); blending it 50/50 with each pair's own A3 signal keeps
them, and EURJPY gets its own trend back at half weight.

    s_A3   = A3 signal of each pair                               (hypotheses/a3_ewma_crossover.py, unchanged)
    s_USD  = C5-006's dollar consensus of s_A3 per pair          (hypotheses/c5_006_dollar_trend.py)
    s      = (0.5 * s_USD + 0.5 * s_A3) * clip(ATR_1440 / ATR_F, 0.5, 2.0)     (C5-010's volatility scale)

Indicators (3): EWMA crossover, price standard deviation, ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_010_volmanaged_dollar_trend as C10
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                       StrategyParams, history_bars, vol_scale, warmup_bars)

HYPOTHESIS = ("C5-012 volatility-managed 50/50 blend of the dollar-trend consensus and each pair's own A3 trend "
              "(C5-010's scale).")
INDICATORS = A3.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a3_ewma_crossover.py")
WEIGHT_USD = 0.5


def blend(s_usd: float, s_pair: float) -> float:
    return WEIGHT_USD * s_usd + (1.0 - WEIGHT_USD) * s_pair


def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    raw = A3.compute_features(frames, A3.StrategyParams(band=p.band))
    dol = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    out = {}
    for k, f in frames.items():
        g = raw[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([blend(u, v) for u, v in zip(dol[k]["signal"].to_numpy(dtype=float),
                                                   g["signal"].to_numpy(dtype=float))])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


class Signals(A3.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        raw = super().signals()
        dol = C6.dollar_signals(raw)
        return {k: blend(dol[k], raw[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value) for k in raw}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
