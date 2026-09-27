"""
A4 - currency carry, time-series version (Koijen, Moskowitz, Pedersen & Vrugt 2018, "Carry", JFE 127(2), 197-225;
Lustig, Roussanov & Verdelhan 2011, "Common risk factors in currency markets", RFS 24(11)).

For an FX pair the carry of a long position is the interest-rate differential d = r_base - r_quote (the forward
discount). Signal per pair, decided at the close of bar t-1 and traded at the open of bar t:

    s = sign(d)  if |d| > theta,  else 0

where d uses only rates public at that time: an OECD monthly average from the start of the following month, a
central-bank decision from its effective date (the `carry` column built by backtest.RateTable / Market).
The position earns (or pays) that differential through the venue's overnight financing, net of the broker mark-up.

Sizing and execution (hypotheses/academic_base.py): units = NAV * 0.3 % * |s| / (ATR * 10 * quote->USD),
Wilder ATR over 1,440 hourly bars; trades through the no-trade band `band`.
Indicators (2): rate differential, ATR. Tunable parameters (2): theta (percent p.a.), band.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("A4 currency carry (Koijen-Moskowitz-Pedersen-Vrugt 2018; Lustig-Roussanov-Verdelhan 2011): long the "
              "higher-yielding currency of each pair when the known rate differential exceeds theta; "
              "volatility-targeted, no-trade band.")
INDICATORS = ("rate differential", "ATR")
USES_CARRY = True


@dataclass(frozen=True)
class StrategyParams:
    """The only tunable numbers (2)."""

    theta: float = 0.0             # minimum |r_base - r_quote| in percent p.a. to hold a position
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not self.theta >= 0:
            raise ValueError(f"theta must be >= 0, got {self.theta}")
        if not 0 <= self.band < 1:
            raise ValueError(f"band must be in [0, 1), got {self.band}")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"theta": [0.0, 0.25, 0.5, 1.0], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("theta",)


def warmup_bars(p: StrategyParams) -> int:
    return A.ATR_BARS + 2


def history_bars(p: StrategyParams) -> int:
    return 2


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        g["signal"] = A.vec_carry(g["carry"], p.theta)
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def signals(self) -> dict[str, float]:
        return {pair: A.ev_carry(self.md, pair, self.p.theta) for pair in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
