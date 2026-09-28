"""
C5-021 - value and momentum combined: 50/50 of the dollar trend and currency value, volatility-managed (Asness,
Moskowitz & Pedersen 2013, "Value and momentum everywhere") (cycle 5, family 5: ensemble).

Rationale: AMP's central result is that value and momentum are negatively correlated within every asset class, so an
equal-weight combination has a far higher Sharpe than either alone. This pairs the best trend construction (C5-010's
dollar consensus of A3, the batch-4 best, at A3's own speed) with C5-020's value signal. Registered together with
C5-020, before either has run, so the combination is not chosen after seeing value's result.

    s_trend = C5-006's dollar consensus of A3                        (hypotheses/c5_006_dollar_trend.py)
    s_value = C5-020's 3-year cross-sectional reversal weights      (hypotheses/c5_020_xs_value.py)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip(ATR_1440 / ATR_F, 0.5, 2.0)     (C5-010's volatility scale)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover (+ price standard deviation, its normaliser), past returns (cross-sectional rank),
ATR (two speeds) - counted as 3: the trend signal, the value rank, ATR. Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_020_xs_value as C20
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_020_xs_value import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                        StrategyParams)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-021 value + momentum (AMP 2013): equal blend of the dollar-trend consensus (C5-006) and 3-year "
              "currency value (C5-020), scaled by C5-010's ATR ratio.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "past returns (cross-sectional rank)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_020_xs_value.py", "c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py",
           "a3_ewma_crossover.py")
WEIGHT_TREND = 0.5


def warmup_bars(p: StrategyParams) -> int:
    return max(C6.warmup_bars(C6.StrategyParams(band=p.band)), C20.warmup_bars(p))


def history_bars(p: StrategyParams) -> int:
    return max(C6.history_bars(C6.StrategyParams(band=p.band)), C20.history_bars(p))


def combine(trend: float, value: float) -> float:
    return WEIGHT_TREND * trend + (1.0 - WEIGHT_TREND) * value


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = C6.compute_features(frames, C6.StrategyParams(band=p.band))
    value = C20.value_weights_vec(frames)
    out = {}
    for k, f in frames.items():
        g = trend[k].copy()
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([combine(t, v) for t, v in zip(g["signal"].to_numpy(dtype=float),
                                                     value[k].to_numpy(dtype=float))])
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
        value = C20.value_weights_ev(self.md)
        return {k: combine(trend[k], value[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value) for k in trend}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
