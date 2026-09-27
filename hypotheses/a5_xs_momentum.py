"""
A5 - cross-sectional currency momentum (Menkhoff, Sarno, Schmeling & Schrimpf 2012, "Currency momentum
strategies", JFE 106(3), 660-684; rank weighting as in Asness, Moskowitz & Pedersen 2013).

The five currencies USD, EUR, GBP, JPY and CAD are ranked by their past L-month performance against USD (price
ratio C_t / C_{t-L} of EURUSD, GBPUSD, inverted for USDJPY, USDCAD; USD itself = 1), ties getting average ranks.
Weights are demeaned ranks scaled to [-1, 1]:  w_c = (rank_c - 3) / 2, which sum to zero (dollar-neutral).
Signal per pair, decided at the close of bar t-1 and traded at the open of bar t:

    EURUSD, GBPUSD: s = w_EUR, w_GBP        USDJPY, USDCAD: s = -w_JPY, -w_CAD        EURJPY (a cross): s = 0

MSSS use formation periods of 1, 3, 6, 9 and 12 months; the walk-forward grid is 1, 3, 6, 12.

Sizing and execution (hypotheses/academic_base.py): units = NAV * 0.3 % * |s| / (ATR * 10 * quote->USD),
Wilder ATR over 1,440 hourly bars; trades through the no-trade band `band`; overnight financing is booked by the
venue. Indicators (2): past returns (cross-sectional rank), ATR. Tunable parameters (2): lookback_months, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("A5 cross-sectional currency momentum (Menkhoff-Sarno-Schmeling-Schrimpf 2012): rank USD, EUR, GBP, "
              "JPY, CAD by L-month performance, dollar-neutral rank weights traded through the USD pairs; "
              "volatility-targeted, no-trade band.")
INDICATORS = ("past returns (cross-sectional rank)", "ATR")
USES_CARRY = False


@dataclass(frozen=True)
class StrategyParams:
    """The only tunable numbers (2)."""

    lookback_months: int = 12      # formation period
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.lookback_months, (int, np.integer)) and self.lookback_months >= 1):
            raise ValueError(f"lookback_months must be an int >= 1, got {self.lookback_months}")
        if not 0 <= self.band < 1:
            raise ValueError(f"band must be in [0, 1), got {self.band}")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"lookback_months": [1, 3, 6, 12], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("lookback_months",)


def lookback_bars(p: StrategyParams) -> int:
    return p.lookback_months * A.BARS_PER_MONTH


def warmup_bars(p: StrategyParams) -> int:
    return max(lookback_bars(p), A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return lookback_bars(p) + 1


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    w = A.vec_xs_weights(frames, lookback_bars(p))
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        g["signal"] = w[k]
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def signals(self) -> dict[str, float]:
        return A.ev_xs_weights(self.md, lookback_bars(self.p))


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
