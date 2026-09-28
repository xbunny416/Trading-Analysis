"""
C5-016 - Donchian channel breakout, the Turtle rules (Faith 2007, "Way of the Turtle", system 1; Donchian 1960)
(cycle 5, family 3).

Rationale (batches 1-5): trend is the only source with an edge before costs, and hourly mean reversion lost because
large deviations keep running (C5-001). A3 and its variants hold a position scaled by a smoothed trend strength; the
Turtle rules instead enter only when price breaks out of its recent range and exit on a shorter opposite breakout, so
they are flat in ranges. Their daily-return overlap with A3 is an empirical question this trial answers.

Per pair, at the close of hourly bar t, with channels over the N (entry) and N/2 (exit) bars before t:
    long  -> flat   if C_t < min(L over the exit window)
    short -> flat   if C_t > max(H over the exit window)
    -> long         if C_t > max(H over the entry window)
    -> short        if C_t < min(L over the entry window)
    signal s in {-1, 0, +1}   (0 until N bars of history exist)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): Donchian channel, ATR. Tunable parameters (2): entry_bars, band.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-016 Turtle / Donchian breakout: enter on a close beyond the N-bar high or low, exit on a close "
              "beyond the opposite N/2-bar extreme; volatility-targeted, no-trade band.")
INDICATORS = ("Donchian channel", "ATR")
USES_CARRY = False


@dataclass(frozen=True)
class StrategyParams:
    entry_bars: int = 480          # entry channel in hourly bars (480 = 20 trading days, Turtle system 1)
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.entry_bars, (int, np.integer)) and self.entry_bars >= 4 and 0 <= self.band < 1):
            raise ValueError("entry_bars must be an int >= 4 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"entry_bars": [120, 240, 480, 960], "band": [0.2, 0.5]}
FEATURE_PARAMS = ("entry_bars",)


def warmup_bars(p: StrategyParams) -> int:
    return max(A.ATR_BARS, p.entry_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


class RollingExtreme:
    """Max (sign=+1) or min (sign=-1) of the last n values pushed, O(1) amortised (monotonic deque)."""

    def __init__(self, n: int, sign: int):
        self.n, self.sign, self.q, self.i = n, sign, deque(), 0

    def push(self, v: float) -> None:
        s = self.sign
        while self.q and s * self.q[-1][1] <= s * v:
            self.q.pop()
        self.q.append((self.i, v))
        self.i += 1
        while self.q[0][0] < self.i - self.n:
            self.q.popleft()

    @property
    def value(self) -> float:
        return self.q[0][1] if self.i >= self.n else math.nan


class Breakout:
    """Turtle entry / exit state machine (shared by both paths)."""

    def __init__(self, entry_bars: int):
        n, x = entry_bars, entry_bars // 2
        self.hi_n, self.lo_n = RollingExtreme(n, 1), RollingExtreme(n, -1)
        self.hi_x, self.lo_x = RollingExtreme(x, 1), RollingExtreme(x, -1)
        self.state = 0

    def update(self, h: float, l: float, c: float) -> int:
        hn, ln = self.hi_n.value, self.lo_n.value          # channels over the bars BEFORE this one
        if hn == hn and ln == ln:
            if self.state == 1 and c < self.lo_x.value:
                self.state = 0
            elif self.state == -1 and c > self.hi_x.value:
                self.state = 0
            if c > hn:
                self.state = 1
            elif c < ln:
                self.state = -1
        for w, v in ((self.hi_n, h), (self.hi_x, h), (self.lo_n, l), (self.lo_x, l)):
            w.push(v)
        return self.state


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for key, f in frames.items():
        g = A.base_features(f)
        br = Breakout(p.entry_bars)
        g["signal"] = [float(br.update(h, l, c)) for h, l, c in zip(f["high"].to_numpy(dtype=float),
                                                                    f["low"].to_numpy(dtype=float),
                                                                    f["close"].to_numpy(dtype=float))]
        out[key] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._br = {q: Breakout(p.entry_bars) for q in self.pairs}
        self._s = {q: 0.0 for q in self.pairs}

    def update(self, pair: str) -> None:
        self._s[pair] = float(self._br[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair)))

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
