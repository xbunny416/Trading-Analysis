"""
C5-003 - A3 (Baz et al. multi-speed EWMA crossover) with a signal-strength threshold (cycle 5, family 3).

Rationale: A3 is the only cycle-4 strategy with a positive net result (OOS Sharpe 0.25). Its gross edge was ~0.35;
costs and financing took the rest, and most of its 5,000+ rebalances happen while the signal is weak. Holding no
position while |s| < theta should cut costs and financing where the forecast is least informative.

    s = s_A3   if |s_A3| >= theta,   else 0          (s_A3 as in hypotheses/a3_ewma_crossover.py, unchanged)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover, price standard deviation, ATR. Tunable parameters (2): theta, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-003 A3 multi-speed EWMA crossover with a signal-strength threshold: no position while "
              "|s| < theta, to cut the cost and financing drag of weak forecasts.")
INDICATORS = A3.INDICATORS
USES_CARRY = False
DEPENDS = ("a3_ewma_crossover.py",)


@dataclass(frozen=True)
class StrategyParams:
    theta: float = 0.3             # minimum |signal| to hold a position
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (0 <= self.theta < 1 and 0 <= self.band < 1):
            raise ValueError("theta and band must be in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"theta": [0.1, 0.2, 0.3, 0.45], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("theta",)


def _a3(p: StrategyParams):
    return A3.StrategyParams(band=p.band)


def warmup_bars(p: StrategyParams) -> int:
    return A3.warmup_bars(_a3(p))


def history_bars(p: StrategyParams) -> int:
    return A3.history_bars(_a3(p))


def threshold(s: float, theta: float) -> float:
    return s if (s != s or abs(s) >= theta) else 0.0


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    feats = A3.compute_features(data, _a3(p))

    def apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        s = g["signal"]
        g["signal"] = s.where(s.isna() | (s.abs() >= p.theta), 0.0)
        return g

    return apply(feats) if isinstance(feats, pd.DataFrame) else {k: apply(g) for k, g in feats.items()}


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A3.Signals):
    def signals(self) -> dict[str, float]:
        return {k: threshold(v, self.p.theta) for k, v in super().signals().items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
