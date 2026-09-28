"""
C5-004 - trend (A2 1/3/12-month blend) that refuses to pay carry (cycle 5, family 3).

Rationale (batch 1): trend is the only source with a real gross edge (~0.33 Sharpe before costs), and overnight
financing - not spread - is its largest drag (about -$30k over the OOS period for A3/C5-003). Positions against the
rate differential pay it every night and are also the ones exposed to carry-unwind reversals.

    s_A2 = (sign r_1m + sign r_3m + sign r_12m) / 3                  (hypotheses/a2_tsmom_blend.py, unchanged)
    d    = r_base - r_quote, as publicly known at the bar's close   (percent p.a.)
    s    = 0      if s_A2 * d < 0 and |d| > kappa                   (would pay more than kappa % a year)
           s_A2   otherwise

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): past returns, rate differential, ATR. Tunable parameters (2): kappa, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import backtest as B
from hypotheses import a2_tsmom_blend as A2
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-004 A2 trend blend without positions that pay more than kappa % a year of carry (the financing "
              "drag and carry-unwind exposure of trend).")
INDICATORS = ("past returns", "rate differential", "ATR")
USES_CARRY = True
DEPENDS = ("a2_tsmom_blend.py",)


@dataclass(frozen=True)
class StrategyParams:
    kappa: float = 0.5             # largest carry (percent p.a.) a position may pay
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (self.kappa >= 0 and 0 <= self.band < 1):
            raise ValueError("kappa must be >= 0 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"kappa": [0.0, 0.5, 1.0, 2.0], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("kappa",)


def warmup_bars(p: StrategyParams) -> int:
    return A2.warmup_bars(A2.StrategyParams(band=p.band))


def history_bars(p: StrategyParams) -> int:
    return A2.history_bars(A2.StrategyParams(band=p.band))


def carry_filter(s: float, d: float, kappa: float) -> float:
    if s != s or d != d:
        return s
    return 0.0 if (s * d < 0 and abs(d) > kappa) else s


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    feats = A2.compute_features(data, A2.StrategyParams(band=p.band))

    def apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        s, d = g["signal"], g["carry"]
        pays = (s * d < 0) & (d.abs() > p.kappa)
        g["signal"] = s.where(~pays, 0.0)
        return g

    return apply(feats) if isinstance(feats, pd.DataFrame) else {k: apply(g) for k, g in feats.items()}


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A2.Signals):
    def signals(self) -> dict[str, float]:
        return {k: carry_filter(v, self.md.carry[k], self.p.kappa) for k, v in super().signals().items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
