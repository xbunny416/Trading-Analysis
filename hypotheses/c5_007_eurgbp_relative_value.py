"""
C5-007 - EUR/GBP relative value: mean reversion of the EURUSD-GBPUSD spread (cycle 5, family 2).

Rationale (batches 1-2): every trend variant nets ~0.25 and single-pair mean reversion loses because the majors
trend against the dollar. The two European currencies share most of their dollar trend; their spread (synthetic
EUR/GBP) removes it and is the classic range-bound cross.

On the union of hourly closes (the other leg at its latest close):
    x_t = ln(EURUSD_t) - ln(GBPUSD_t)                    (= ln EURGBP; ln EURUSD alone in a universe without GBPUSD)
    z_t = (x_t - EWMA(x; n)_t) / std(x - EWMA over the last 4n closes)
    state: flat -> short the spread when z > z_in, long when z < -z_in; exit at the mean (as C5-001's ZState)
    signals: EURUSD = state, GBPUSD = -state, other pairs 0

Sizing (each leg vol-targeted), execution (no-trade band), costs, financing and breakers: academic_base.py.
Indicators (3): EWMA, deviation standard deviation, ATR. Tunable parameters (3): n, z_in, band (grid: n x z_in).
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
from hypotheses.c5_001_ts_meanrev import ZState

HYPOTHESIS = ("C5-007 EUR/GBP relative value: fade z-score extremes of ln(EURUSD) - ln(GBPUSD) vs its n-bar EWMA "
              "(long one leg, short the other), exit at the mean.")
INDICATORS = ("EWMA", "deviation standard deviation", "ATR")
USES_CARRY = False
DEPENDS = ("c5_001_ts_meanrev.py",)
LEG_A, LEG_B = "EURUSD", "GBPUSD"


@dataclass(frozen=True)
class StrategyParams:
    n: int = 72                    # EWMA length in hourly closes (std window = 4n)
    z_in: float = 2.0              # entry threshold on |z|
    band: float = 0.25             # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.n, (int, np.integer)) and self.n >= 2):
            raise ValueError(f"n must be an int >= 2, got {self.n}")
        if not (self.z_in > 0 and 0 <= self.band < 1):
            raise ValueError("z_in must be > 0 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"n": [24, 72, 168, 336], "z_in": [1.5, 2.0, 2.5, 3.0]}
FEATURE_PARAMS = ("n", "z_in")


def warmup_bars(p: StrategyParams) -> int:
    return max(5 * p.n, A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


def spread(a: float, b: float, two_legs: bool) -> float:
    if not two_legs:
        return math.log(a) if a == a else math.nan
    return math.log(a) - math.log(b) if (a == a and b == b) else math.nan


def leg_signals(state: float, pairs, two_legs: bool) -> dict[str, float]:
    return {k: (state if k == LEG_A else (-state if (k == LEG_B and two_legs) else 0.0)) for k in pairs}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    two = LEG_A in frames and LEG_B in frames
    index = A.universe_index(frames)
    st = pd.Series(math.nan, index=index)
    if LEG_A in frames:
        a = frames[LEG_A]["close"].reindex(index).ffill().to_numpy(dtype=float)
        b = frames[LEG_B]["close"].reindex(index).ffill().to_numpy(dtype=float) if two else a
        zs, vals = ZState(p), np.full(len(index), math.nan)
        for i in range(len(index)):
            x = spread(a[i], b[i], two)
            if x == x:
                vals[i] = zs.update(x)[1]
        st = pd.Series(vals, index=index)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        s = st.reindex(f.index)
        g["signal"] = s if k == LEG_A else (-s if (k == LEG_B and two) else 0.0)
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._two = LEG_A in self.pairs and LEG_B in self.pairs
        self._z = ZState(p)
        self._state = math.nan

    def on_close(self, ts: int) -> None:
        if LEG_A not in self.pairs:
            return
        x = spread(self.md.close(LEG_A), self.md.close(LEG_B) if self._two else math.nan, self._two)
        if x == x:
            self._state = float(self._z.update(x)[1])

    def signals(self) -> dict[str, float]:
        return leg_signals(self._state, self.pairs, self._two)


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
