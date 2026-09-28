"""
C5-002 - cross-sectional short-term currency reversal (cycle 5, family 2).

Rationale: an independent, market-neutral source. A5 (cross-sectional momentum at 1-12 months) lost money; at short
horizons (days) currencies that moved most are often partly reversed as flows normalise.

Rank USD, EUR, GBP, JPY, CAD by their L-bar performance against USD (price ratio, inverted for USDxxx), average
ranks for ties; dollar-neutral weights w = (rank - 3) / 2. The signal is the REVERSE of A5's:
EURUSD, GBPUSD: s = -w_EUR, -w_GBP;   USDJPY, USDCAD: s = +w_JPY, +w_CAD;   EURJPY (cross): 0.

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): past returns (cross-sectional rank), ATR. Tunable parameters (2): lookback_bars, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-002 cross-sectional short-term reversal: long the currencies that fell most over the last L hourly "
              "bars, short those that rose most (dollar-neutral rank weights, A5 reversed at 1-10 day horizons).")
INDICATORS = ("past returns (cross-sectional rank)", "ATR")
USES_CARRY = False


@dataclass(frozen=True)
class StrategyParams:
    lookback_bars: int = 120       # formation period in hourly bars (24 = 1 day)
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.lookback_bars, (int, np.integer)) and self.lookback_bars >= 2):
            raise ValueError(f"lookback_bars must be an int >= 2, got {self.lookback_bars}")
        if not 0 <= self.band < 1:
            raise ValueError(f"band must be in [0, 1), got {self.band}")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"lookback_bars": [24, 72, 120, 240], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("lookback_bars",)


def warmup_bars(p: StrategyParams) -> int:
    return max(p.lookback_bars, A.ATR_BARS) + 2


def history_bars(p: StrategyParams) -> int:
    return p.lookback_bars + 1


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    w = A.vec_xs_weights(frames, p.lookback_bars)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        g["signal"] = -w[k]
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def signals(self) -> dict[str, float]:
        return {k: -v for k, v in A.ev_xs_weights(self.md, self.p.lookback_bars).items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
