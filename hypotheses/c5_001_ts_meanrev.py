"""
C5-001 - time-series mean reversion on hourly bars (cycle 5, family 1).

Rationale: cycle 4's return sources were all one trend bet (A1-A3 correlate 0.75-0.91) worth ~0.25 net Sharpe;
a higher Sharpe needs an independent source. Short-horizon deviations from a moving average are a candidate that
should be negatively correlated with slow trend.

Per pair, on hourly closes (row t uses bars <= t, decided at the close and filled at the next open):

    dev_t = C_t - EWMA(C; n)_t                     EWMA: y_t = y_{t-1} + (C_t - y_{t-1}) / n
    z_t   = dev_t / std(dev over the last 4n bars)
    state: flat  -> short when z > z_in, long when z < -z_in
           long  -> flat when z >= 0            short -> flat when z <= 0     (exit at the mean)
    signal s = state in {-1, 0, +1}

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
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

HYPOTHESIS = ("C5-001 time-series mean reversion: enter against hourly z-score extremes of price vs its n-bar EWMA "
              "(|z| > z_in), exit at the mean; independent of the cycle-4 trend bet.")
INDICATORS = ("EWMA", "deviation standard deviation", "ATR")
USES_CARRY = False


@dataclass(frozen=True)
class StrategyParams:
    n: int = 48                    # EWMA length in hourly bars (std window = 4n)
    z_in: float = 2.0              # entry threshold on |z|
    band: float = 0.25             # no-trade band for size changes (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.n, (int, np.integer)) and self.n >= 2):
            raise ValueError(f"n must be an int >= 2, got {self.n}")
        if not (self.z_in > 0 and 0 <= self.band < 1):
            raise ValueError("z_in must be > 0 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"n": [24, 48, 96, 168], "z_in": [1.5, 2.0, 2.5, 3.0]}
FEATURE_PARAMS = ("n", "z_in")


def warmup_bars(p: StrategyParams) -> int:
    return max(5 * p.n, A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


class ZState:
    """z-score of price vs its EWMA and the enter-at-extreme / exit-at-mean state (shared by both paths)."""

    def __init__(self, p: StrategyParams):
        self.p = p
        self.ewma = A.EWMA(1.0 / p.n)
        self.sd = A.RollingStd(4 * p.n)
        self.state = 0

    def update(self, c: float) -> tuple[float, int]:
        dev = c - self.ewma.update(c)
        sd = self.sd.update(dev)
        z = dev / sd if sd > 0 else math.nan
        if z == z:
            if self.state == 0:
                self.state = -1 if z > self.p.z_in else (1 if z < -self.p.z_in else 0)
            elif (self.state == 1 and z >= 0) or (self.state == -1 and z <= 0):
                self.state = 0
        return z, self.state


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        zs, st = ZState(p), np.empty(len(f))
        z = np.empty(len(f))
        for i, c in enumerate(f["close"].to_numpy(dtype=float)):
            z[i], st[i] = zs.update(c)
        g["z"] = z
        g["signal"] = st
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._z = {q: ZState(p) for q in self.pairs}
        self._s = {q: 0.0 for q in self.pairs}

    def update(self, pair: str) -> None:
        self._s[pair] = float(self._z[pair].update(self.md.close(pair))[1])

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
