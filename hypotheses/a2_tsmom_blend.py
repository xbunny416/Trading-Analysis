"""
A2 - time-series momentum, multi-horizon blend (Hurst, Ooi & Pedersen 2017, "A century of evidence on
trend-following investing", Journal of Portfolio Management 44(1), 15-29).

Signal per pair, decided at the close of bar t-1 and traded at the open of bar t:

    s = ( sign(C_{t-1} - C_{t-1-L1}) + sign(C_{t-1} - C_{t-1-L3}) + sign(C_{t-1} - C_{t-1-L12}) ) / 3

with L1, L3, L12 = 1, 3 and 12 months (520 hourly bars a month): the paper's equal-weight blend of the 1-, 3- and
12-month trend signals. The horizons are the published ones and are not tuned.

Sizing and execution (hypotheses/academic_base.py): units = NAV * 0.3 % * |s| / (ATR * 10 * quote->USD),
Wilder ATR over 1,440 hourly bars; trades through the no-trade band `band`; overnight financing is booked by the
venue. Indicators (2): past returns, ATR. Tunable parameters (1): band.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("A2 time-series momentum blend (Hurst-Ooi-Pedersen 2017): equal-weight average of the signs of the "
              "1-, 3- and 12-month past returns; volatility-targeted, no-trade band.")
INDICATORS = ("past returns", "ATR")
USES_CARRY = False
HORIZONS_MONTHS = (1, 3, 12)


@dataclass(frozen=True)
class StrategyParams:
    """The only tunable number (1)."""

    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not 0 <= self.band < 1:
            raise ValueError(f"band must be in [0, 1), got {self.band}")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ()
LAGS = tuple(m * A.BARS_PER_MONTH for m in HORIZONS_MONTHS)


def warmup_bars(p: StrategyParams) -> int:
    return max(max(LAGS), A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return max(LAGS) + 1


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        a, b, c = (A.vec_tsmom(f["close"], lag) for lag in LAGS)
        g["signal"] = (a + b + c) / 3.0
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def signals(self) -> dict[str, float]:
        out = {}
        for pair in self.pairs:
            a, b, c = (A.ev_tsmom(self.md, pair, lag) for lag in LAGS)
            out[pair] = (a + b + c) / 3.0
        return out


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
