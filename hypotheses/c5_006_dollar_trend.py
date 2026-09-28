"""
C5-006 - dollar-trend consensus: A3's multi-speed EWMA trend averaged into one USD view (cycle 5, family 3).

Rationale (batch 1, cycle 4): all trend variants are one bet and the only gross edge. Much of FX trend is the dollar
trending against everything (2014-15, 2021-22). Averaging A3's per-pair signals into one USD signal cancels
pair-specific noise, and trading it through the four USD pairs drops EURJPY, which only duplicates EUR and JPY
exposure (EURJPY = EURUSD x USDJPY) and lost money in every trend trial.

    s_pair = A3 signal of each USD pair                          (hypotheses/a3_ewma_crossover.py, unchanged)
    s_USD  = mean over EURUSD, GBPUSD, USDJPY, USDCAD of (-dir * s_pair)   dir = +1 for xxxUSD, -1 for USDxxx
    signal = -dir * s_USD for each USD pair (all lean the same way on the dollar);  EURJPY: 0

Disclosure: EURJPY's losses were seen before this design; the exclusion also follows from its redundancy (A5 already
gave it zero weight by construction).
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA crossover, price standard deviation, ATR. Tunable parameters (1): band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-006 dollar-trend consensus: A3's multi-speed EWMA trend averaged over the four USD pairs into one "
              "USD signal, traded through those pairs (EURJPY flat).")
INDICATORS = A3.INDICATORS
USES_CARRY = False
DEPENDS = ("a3_ewma_crossover.py",)


@dataclass(frozen=True)
class StrategyParams:
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not 0 <= self.band < 1:
            raise ValueError(f"band must be in [0, 1), got {self.band}")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ()


def warmup_bars(p: StrategyParams) -> int:
    return A3.warmup_bars(A3.StrategyParams(band=p.band))


def history_bars(p: StrategyParams) -> int:
    return A3.history_bars(A3.StrategyParams(band=p.band))


def dollar_signals(per_pair: dict[str, float]) -> dict[str, float]:
    """Pair signals from the USD consensus of per-pair trend signals (NaN until every USD leg is ready)."""
    legs = {k: A.usd_leg(k) for k in per_pair if A.usd_leg(k) is not None}
    vals = [-d * per_pair[k] for k, (_, d) in legs.items()]
    s_usd = sum(vals) / len(vals) if vals and all(v == v for v in vals) else math.nan
    return {k: (-A.usd_leg(k)[1] * s_usd if A.usd_leg(k) is not None else 0.0) for k in per_pair}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    base = A3.compute_features(frames, A3.StrategyParams(band=p.band))
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * base[k]["signal"].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    out = {}
    for k, g in base.items():
        g = g.copy()
        leg = A.usd_leg(k)
        g["signal"] = (-leg[1] * s_usd).reindex(g.index) if leg is not None else 0.0
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A3.Signals):
    def signals(self) -> dict[str, float]:
        return dollar_signals(super().signals())


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
