"""
A6 - multi-factor FX portfolio: equal risk blend of trend, carry and cross-sectional momentum (the diversified
factor portfolio of Asness, Moskowitz & Pedersen 2013, "Value and momentum everywhere", JF 68(3); and of
Koijen, Moskowitz, Pedersen & Vrugt 2018, "Carry", JFE 127(2), which combine carry with momentum).

Signal per pair, decided at the close of bar t-1 and traded at the open of bar t, netted in one position:

    s = ( s_A2 + s_A4 + s_A5 ) / 3
      s_A2 = (sign r_1m + sign r_3m + sign r_12m) / 3               time-series momentum blend (A2)
      s_A4 = sign(r_base - r_quote), rates known at the time        carry with theta = 0 (A4)
      s_A5 = dollar-neutral rank weight from 12-month performance   cross-sectional momentum (A5; 0 for EURJPY)

All sleeve parameters are the published / pre-registered ones; none is tuned.
Sizing and execution (hypotheses/academic_base.py): units = NAV * 0.3 % * |s| / (ATR * 10 * quote->USD),
Wilder ATR over 1,440 hourly bars; trades through the no-trade band `band`; overnight financing is booked by the
venue. Indicators (3): past returns, rate differential, ATR. Tunable parameters (1): band.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("A6 multi-factor FX (Asness-Moskowitz-Pedersen 2013; KMPV 2018): equal blend of the A2 trend blend, "
              "A4 carry (theta = 0) and A5 cross-sectional momentum (12 months), netted per pair; "
              "volatility-targeted, no-trade band.")
INDICATORS = ("past returns", "rate differential", "ATR")
USES_CARRY = True
TREND_LAGS = tuple(m * A.BARS_PER_MONTH for m in (1, 3, 12))
XS_LAG = 12 * A.BARS_PER_MONTH


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


def warmup_bars(p: StrategyParams) -> int:
    return max(max(TREND_LAGS), XS_LAG, A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return max(max(TREND_LAGS), XS_LAG) + 1


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    xs = A.vec_xs_weights(frames, XS_LAG)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        a, b, c = (A.vec_tsmom(f["close"], lag) for lag in TREND_LAGS)
        trend = (a + b + c) / 3.0
        carry = A.vec_carry(g["carry"], 0.0)
        g["signal"] = (trend + carry + xs[k]) / 3.0
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def signals(self) -> dict[str, float]:
        xs = A.ev_xs_weights(self.md, XS_LAG)
        out = {}
        for pair in self.pairs:
            a, b, c = (A.ev_tsmom(self.md, pair, lag) for lag in TREND_LAGS)
            trend = (a + b + c) / 3.0
            carry = A.ev_carry(self.md, pair, 0.0)
            out[pair] = (trend + carry + xs[pair]) / 3.0
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
