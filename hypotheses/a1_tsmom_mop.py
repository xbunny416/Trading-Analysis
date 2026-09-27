"""
A1 - time-series momentum (Moskowitz, Ooi & Pedersen 2012, "Time series momentum", JFE 104(2), 228-250).

Signal per pair, decided at the close of bar t-1 and traded at the open of bar t:

    s = sign(C_{t-1} - C_{t-1-L})          L = lookback_months * 520 hourly bars

i.e. long a pair whose own past L-month return is positive, short if negative. MOP's headline lookback is 12 months;
their Table 2 also reports 1, 3 and 6 months, which is the walk-forward grid here.

Sizing and execution (hypotheses/academic_base.py): units = NAV * 0.3 % * |s| / (ATR * 10 * quote->USD),
Wilder ATR over 1,440 hourly bars; trades through the no-trade band `band`; overnight financing is booked by the
venue. Indicators (2): past return, ATR. Tunable parameters (2): lookback_months, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("A1 time-series momentum (Moskowitz-Ooi-Pedersen 2012): long a pair whose own past L-month return is "
              "positive, short if negative; volatility-targeted, no-trade band.")
INDICATORS = ("past return", "ATR")
USES_CARRY = False


@dataclass(frozen=True)
class StrategyParams:
    """The only tunable numbers (2)."""

    lookback_months: int = 12      # MOP 2012 headline
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
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        g["signal"] = A.vec_tsmom(f["close"], lookback_bars(p))
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def signals(self) -> dict[str, float]:
        lag = lookback_bars(self.p)
        return {pair: A.ev_tsmom(self.md, pair, lag) for pair in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
