"""
C5-015 - fast, volatility-managed dollar trend: C5-010 with A3's speeds scaled down (cycle 5, family 3).

Rationale (batches 1-5): hourly mean reversion (C5-001) lost before costs because deviations of 1-7 days from the
mean kept running; that is trend at horizons shorter than A3's fastest speed (8/24 days). Here the walk-forward
chooses how much faster than A3 to run: every EWMA span is multiplied by `speed` (1/8 -> 1/3-day spans ... 1/2).
The normalisation windows (63 and 252 days) and the response function are A3's, unchanged.

    s_pair = A3's formula with speeds (8, 24), (16, 48), (32, 96) days x speed
    s_USD  = C5-006's dollar consensus of s_pair over the four USD pairs (EURJPY flat)
    s      = s_USD * clip(ATR_1440 / ATR_F, 0.5, 2.0)                   (C5-010's volatility scale)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover, price standard deviation, ATR (two speeds).
Tunable parameters (3): speed, fast_bars, band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-015 fast volatility-managed dollar trend: C5-010 with every A3 EWMA span scaled by `speed` "
              "(1/8, 1/4 or 1/2), chosen in the walk-forward.")
INDICATORS = A3.INDICATORS
USES_CARRY = False
DEPENDS = ("c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py", "a3_ewma_crossover.py")


@dataclass(frozen=True)
class StrategyParams:
    speed: float = 0.25            # multiplier on A3's EWMA spans (1 = A3)
    fast_bars: int = 480           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (0 < self.speed <= 1 and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2
                and 0 <= self.band < 1):
            raise ValueError("speed must be in (0, 1], fast_bars an int >= 2 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"speed": [0.125, 0.25, 0.5], "fast_bars": [120, 480, 960], "band": [0.2, 0.5]}
FEATURE_PARAMS = ("speed", "fast_bars")


def speeds(speed: float) -> tuple[tuple[float, float], ...]:
    return tuple((s * speed, l * speed) for s, l in A3.SPEEDS_DAYS)


def warmup_bars(p: StrategyParams) -> int:
    return max(A3.PRICE_STD_BARS + A3.SIGNAL_STD_BARS + 2, p.fast_bars + 2)


def history_bars(p: StrategyParams) -> int:
    return 2


# ============================================================================= research path (vectorised)
def pair_trend(c: pd.Series, speed: float) -> pd.Series:
    """A3's signal (same formulas) with scaled speeds; row t uses closes <= t."""
    c_np = c.to_numpy(dtype=float)
    sd = c.rolling(A3.PRICE_STD_BARS).std()
    us = []
    for s_days, l_days in speeds(speed):
        x = pd.Series(A.ewma(c_np, A3.alpha(s_days)) - A.ewma(c_np, A3.alpha(l_days)), index=c.index)
        q = (x / sd).where(sd > 0)
        qs = q.rolling(A3.SIGNAL_STD_BARS).std()
        z = (q / qs).where(qs > 0)
        us.append(z * np.exp(-z * z / 4.0) / 0.89)
    return (us[0] + us[1] + us[2]) / 3.0


def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    base = {k: A.base_features(f) for k, f in frames.items()}
    for k, f in frames.items():
        base[k]["signal"] = pair_trend(f["close"], p.speed)
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * base[k]["signal"].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    out = {}
    for k, f in frames.items():
        g = base[k]
        leg = A.usd_leg(k)
        s = (-leg[1] * s_usd).reindex(g.index).to_numpy(dtype=float) if leg is not None else np.zeros(len(g))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A3.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        sp = speeds(p.speed)
        self._fast = {q: [A.EWMA(A3.alpha(s)) for s, _ in sp] for q in self.pairs}
        self._slow = {q: [A.EWMA(A3.alpha(l)) for _, l in sp] for q in self.pairs}
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        return {k: v * vol_scale(self.md.atr(k), self._fast_atr[k].value)
                for k, v in C6.dollar_signals(super().signals()).items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
