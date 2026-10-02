"""
C5-057 - trend with pullback entries: enter in the direction of each pair's A3 trend only after a short-term pullback
against it, exit when the trend turns (cycle 5, new family after the holdout diagnostic).

Context: after the 2023 holdout diagnostic the user asked for other, better strategies. Hourly mean reversion (C5-001)
won 60-70 % of its trades but lost on the large counter-trend ones; A3 trend earns but enters wherever the trend
happens to turn. This keeps only the mean-reversion entries that agree with the trend and holds them with the trend:

    u_t  = A3's signal for the pair (published constants)
    z_t  = (C_t - EWMA(C; n)) / std of that deviation over 4n bars           (C5-001's z-score)
    flat  -> long  if u_t > 0.1 and z_t < -z_in      (price dipped in an uptrend)
          -> short if u_t < -0.1 and z_t >  z_in      (price rallied in a downtrend)
    long  -> flat  if u_t < -0.1;   short -> flat if u_t > 0.1               (the trend turned)
    signal s = state in {-1, 0, +1}; all five pairs

The 0.1 dead zone on u is fixed (not tuned). Sizing, execution, costs, financing and breakers: academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), z-score of price vs EWMA, ATR.
Tunable parameters (3): n, z_in, band (band fixed at 0.2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-057 trend with pullback entries: enter with each pair's A3 trend only when the hourly z-score has "
              "pulled back beyond z_in against it, exit when the trend turns; new family after the holdout diagnostic.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "z-score of price vs EWMA", "ATR")
USES_CARRY = False
DEPENDS = ("a3_ewma_crossover.py",)
DEAD_ZONE = 0.1


@dataclass(frozen=True)
class StrategyParams:
    n: int = 72                    # pullback EWMA length in hourly bars (std window 4n)
    z_in: float = 1.5              # pullback depth that triggers an entry
    band: float = 0.2              # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.n, (int, np.integer)) and self.n >= 2 and self.z_in > 0 and 0 <= self.band < 1):
            raise ValueError("n an int >= 2, z_in > 0, band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"n": [24, 72, 168], "z_in": [1.0, 1.5, 2.0]}
FEATURE_PARAMS = ("n", "z_in")


def warmup_bars(p: StrategyParams) -> int:
    return max(A3.warmup_bars(A3.StrategyParams()), 5 * p.n, A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


class Pullback:
    """z-score of price vs its EWMA plus the pullback-entry / trend-exit state (shared by both paths)."""

    def __init__(self, p):
        self.z_in = p.z_in
        self.ewma = A.EWMA(1.0 / p.n)
        self.sd = A.RollingStd(4 * p.n)
        self.state = 0

    def update(self, c: float, u: float) -> int:
        dev = c - self.ewma.update(c)
        sd = self.sd.update(dev)
        z = dev / sd if sd > 0 else math.nan
        if u == u:
            if self.state == 1 and u < -DEAD_ZONE:
                self.state = 0
            elif self.state == -1 and u > DEAD_ZONE:
                self.state = 0
            if self.state == 0 and z == z:
                if u > DEAD_ZONE and z < -self.z_in:
                    self.state = 1
                elif u < -DEAD_ZONE and z > self.z_in:
                    self.state = -1
        return self.state


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = A3.compute_features(frames, A3.StrategyParams())
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        pb = Pullback(p)
        u = trend[k]["signal"].to_numpy(dtype=float)
        g["signal"] = [float(pb.update(c, x)) for c, x in zip(f["close"].to_numpy(dtype=float), u)]
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A3.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._pb = {q: Pullback(p) for q in self.pairs}
        self._s = {q: 0.0 for q in self.pairs}

    def update(self, pair: str) -> None:
        super().update(pair)                         # A3's u for this pair
        self._s[pair] = float(self._pb[pair].update(self.md.close(pair), self._u[pair]))

    def signals(self) -> dict[str, float]:
        return dict(self._s)


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
