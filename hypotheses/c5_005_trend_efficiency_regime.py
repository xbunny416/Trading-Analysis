"""
C5-005 - trend (A2 1/3/12-month blend) only in trending regimes, measured by Kaufman's efficiency ratio
(cycle 5, family 4).

Rationale (batch 1): trend is the only gross edge, and FX trends at 1-7 days (mean reversion lost before costs). A
trend rule loses in choppy, range-bound markets; holding it only when recent price action is directional should
raise the edge per unit of risk and cut churn.

    ER_t  = |C_t - C_{t-L}| / sum_{i=t-L+1..t} |C_i - C_{i-1}|         (0 = pure chop, 1 = straight line)
    ERn_t = ER_t * sqrt(L)                                              (~1 for a random walk, any L)
    s     = s_A2   if ERn_t > k,   else 0                               (s_A2: hypotheses/a2_tsmom_blend.py)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): past returns, efficiency ratio, ATR. Tunable parameters (3): er_bars, k, band (grid: er_bars x k).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a2_tsmom_blend as A2
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-005 A2 trend blend held only while the normalised efficiency ratio over L bars exceeds k "
              "(directional, not choppy, markets).")
INDICATORS = ("past returns", "efficiency ratio", "ATR")
USES_CARRY = False
DEPENDS = ("a2_tsmom_blend.py",)


@dataclass(frozen=True)
class StrategyParams:
    er_bars: int = 520             # efficiency-ratio window L in hourly bars
    k: float = 1.5                 # threshold on ER * sqrt(L) (random walk ~ 1)
    band: float = 0.2              # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.er_bars, (int, np.integer)) and self.er_bars >= 2):
            raise ValueError(f"er_bars must be an int >= 2, got {self.er_bars}")
        if not (self.k >= 0 and 0 <= self.band < 1):
            raise ValueError("k must be >= 0 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"er_bars": [130, 260, 520, 1040], "k": [1.0, 1.5, 2.0, 2.5]}
FEATURE_PARAMS = ("er_bars", "k")


def warmup_bars(p: StrategyParams) -> int:
    return max(A2.warmup_bars(A2.StrategyParams(band=p.band)), p.er_bars + 2)


def history_bars(p: StrategyParams) -> int:
    return max(A2.history_bars(A2.StrategyParams(band=p.band)), p.er_bars + 1)


class Efficiency:
    """Normalised efficiency ratio of the last L closes (shared by both paths; running sum re-summed every L)."""

    def __init__(self, n: int):
        self.n, self.buf, self.closes = n, deque(maxlen=n), deque(maxlen=n + 1)
        self._s, self._k = 0.0, 0

    def update(self, c: float) -> float:
        if self.closes:
            step = abs(c - self.closes[-1])
            if len(self.buf) == self.n:
                self._s -= self.buf[0]
            self.buf.append(step)
            self._s += step
            self._k += 1
            if self._k % self.n == 0:
                self._s = math.fsum(self.buf)
        self.closes.append(c)
        if len(self.closes) <= self.n or not self._s > 0:
            return math.nan
        return abs(c - self.closes[0]) / self._s * math.sqrt(self.n)


def regime(s: float, ern: float, k: float) -> float:
    if s != s or ern != ern:
        return math.nan
    return s if ern > k else 0.0


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    base = A2.compute_features(frames, A2.StrategyParams(band=p.band))
    out = {}
    for k, f in frames.items():
        g = base[k].copy()
        eff = Efficiency(p.er_bars)
        ern = np.array([eff.update(c) for c in f["close"].to_numpy(dtype=float)])
        g["ern"] = ern
        g["signal"] = [regime(s, e, p.k) for s, e in zip(g["signal"].to_numpy(dtype=float), ern)]
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A2.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._eff = {q: Efficiency(p.er_bars) for q in self.pairs}
        self._ern = {q: math.nan for q in self.pairs}

    def update(self, pair: str) -> None:
        self._ern[pair] = self._eff[pair].update(self.md.close(pair))

    def signals(self) -> dict[str, float]:
        return {k: regime(v, self._ern[k], self.p.k) for k, v in super().signals().items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
