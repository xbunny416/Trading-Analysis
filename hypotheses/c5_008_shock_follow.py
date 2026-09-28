"""
C5-008 - large-shock follow-through with a bar-count holding period (cycle 5, family 1b).

Rationale: the best idea of cycles 2-3 was momentum after large hourly shocks (H2: 0.63 OOS on 2020-11 -> 2022-12,
but on 2 years of data and with the since-fixed fill timing). This re-tests the idea in its simplest form on the
2005 -> 2022 history (never used for it) with correct execution.

Per pair, on hourly closes:
    shock at bar t  <=>  |C_t - C_{t-1}| > k * ATR_{t-1}          (ATR: the engine's Wilder ATR of hourly bars)
    on a shock: position = sign(C_t - C_{t-1}) for the next `hold` bars (a new shock restarts the clock, either way)
    signal s in {-1, 0, +1}

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): price change, ATR. Tunable parameters (3): k, hold, band (grid: k x hold).
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

HYPOTHESIS = ("C5-008 large-shock follow-through: after an hourly close-to-close move beyond k x ATR, hold its "
              "direction for `hold` bars.")
INDICATORS = ("price change", "ATR")
USES_CARRY = False
DIRECTION = 1                      # +1 follow the shock; C5-009 uses -1 (fade it)


@dataclass(frozen=True)
class StrategyParams:
    k: float = 3.0                 # shock threshold in ATRs
    hold: int = 24                 # holding period in hourly bars
    band: float = 0.25             # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        if not (self.k > 0 and isinstance(self.hold, (int, np.integer)) and self.hold >= 1 and 0 <= self.band < 1):
            raise ValueError("k must be > 0, hold an int >= 1 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"k": [2.5, 3.0, 3.5, 4.0], "hold": [6, 24, 72, 168]}
FEATURE_PARAMS = ("k", "hold")


def warmup_bars(p: StrategyParams) -> int:
    return A.ATR_BARS + 2


def history_bars(p: StrategyParams) -> int:
    return 2


class Shock:
    """Shock detector and holding clock (shared by both paths)."""

    def __init__(self, k: float, hold: int, direction: int):
        self.k, self.hold, self.dir = k, hold, direction
        self.prev_c = self.prev_atr = math.nan
        self.state, self.left = 0, 0

    def update(self, c: float, atr: float) -> int:
        if self.prev_c == self.prev_c and self.prev_atr > 0:
            move = c - self.prev_c
            if abs(move) > self.k * self.prev_atr:
                self.state, self.left = self.dir * (1 if move > 0 else -1), self.hold
            elif self.left > 0:
                self.left -= 1
                if self.left == 0:
                    self.state = 0
        self.prev_c, self.prev_atr = c, atr
        return self.state


def make_shock(p) -> Shock:
    return Shock(p.k, p.hold, DIRECTION)


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams, direction: int = DIRECTION):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for key, f in frames.items():
        g = A.base_features(f)
        sh = Shock(p.k, p.hold, direction)
        g["signal"] = [float(sh.update(c, a)) for c, a in zip(g["close"].to_numpy(dtype=float),
                                                               g["atr"].to_numpy(dtype=float))]
        out[key] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    DIRECTION = DIRECTION

    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._sh = {q: Shock(p.k, p.hold, self.DIRECTION) for q in self.pairs}
        self._s = {q: 0.0 for q in self.pairs}

    def update(self, pair: str) -> None:
        self._s[pair] = float(self._sh[pair].update(self.md.close(pair), self.md.atr(pair)))

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
