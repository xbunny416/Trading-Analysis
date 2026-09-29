"""
C5-027 - value + trend with the walk-forward choosing the trend speed and the value horizon (cycle 5, family 5:
ensemble, refinement of C5-023).

Rationale (batches 7-11): C5-023 (50/50 dollar trend + smooth 3-year value, 0.49) is the best book, and every
canonical currency factor has now been tested; what remains is refining it, which the user chose knowing it is
closer to data mining (the deflated Sharpe accounts for it). Two of C5-023's fixed choices had evidence against them:
the trend leg ran at A3's speed while C5-019's walk-forward sometimes preferred slower spans (0.41 vs 0.36), and the
3-year value horizon was set by the data length (AMP use 5 years). Both go into the grid, in one trial.

    s_trend = C5-015/019's dollar consensus of A3 with every span x speed        (speed in 1, 1.5, 2)
    s_value = C5-022's smooth value weights over value_years                   (2, 3 or 4 years)
    s       = (0.5 * s_trend + 0.5 * s_value) * clip(ATR_1440 / ATR_F, 0.5, 2.0)

Disclosure: a 4-year horizon starts in 2009, leaving split 1's in-sample window about 2.7 years of signal.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (3): EWMA trend (normalised by price standard deviation), past returns (cross-sectional z-score), ATR.
Tunable parameters (4): speed, value_years, fast_bars, band (band fixed at 0.2, not in the grid).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_015_fast_dollar_trend as C15
from hypotheses import c5_024_smooth_xs_momentum as C24
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-027 value + trend, tuned: C5-023's 50/50 blend with the dollar-trend speed (x1, 1.5, 2) and the "
              "smooth value horizon (2, 3, 4 years) chosen in the walk-forward; scaled by C5-010's ATR ratio.")
INDICATORS = ("EWMA trend (normalised by price standard deviation)", "past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_024_smooth_xs_momentum.py", "c5_022_smooth_value.py", "c5_020_xs_value.py",
           "c5_015_fast_dollar_trend.py", "c5_010_volmanaged_dollar_trend.py", "c5_006_dollar_trend.py",
           "a3_ewma_crossover.py")
WEIGHT_TREND = 0.5


@dataclass(frozen=True)
class StrategyParams:
    speed: float = 1.0             # multiplier on A3's EWMA spans
    value_years: int = 3           # value horizon (252 trading days a year)
    fast_bars: int = 960           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        ok = (1 <= self.speed <= 4 and isinstance(self.value_years, (int, np.integer)) and self.value_years >= 1
              and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1)
        if not ok:
            raise ValueError("speed in [1, 4], value_years an int >= 1, fast_bars an int >= 2, band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"speed": [1.0, 1.5, 2.0], "value_years": [2, 3, 4], "fast_bars": [120, 960]}
FEATURE_PARAMS = ("speed", "value_years", "fast_bars")


def value_bars(p) -> int:
    return p.value_years * 252 * A.BARS_PER_DAY


def warmup_bars(p: StrategyParams) -> int:
    return max(A3.PRICE_STD_BARS + A3.SIGNAL_STD_BARS, value_bars(p), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return value_bars(p) + 1


def combine(trend: float, value: float) -> float:
    return WEIGHT_TREND * trend + (1.0 - WEIGHT_TREND) * value


# ============================================================================= research path (vectorised)
def dollar_trend_vec(frames, speed: float) -> dict[str, pd.Series]:
    """C5-015's per-pair trend at `speed`, averaged into C5-006's dollar consensus (EURJPY 0)."""
    trend = {k: C15.pair_trend(f["close"], speed) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * trend[k].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    return {k: ((-A.usd_leg(k)[1] * s_usd).reindex(f.index) if A.usd_leg(k) is not None
                else pd.Series(0.0, index=f.index)) for k, f in frames.items()}


def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    trend = dollar_trend_vec(frames, p.speed)
    value = C24.xs_weights_vec(frames, value_bars(p), -1.0)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        s = np.array([combine(t, v) for t, v in zip(trend[k].to_numpy(dtype=float), value[k].to_numpy(dtype=float))])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C15.Signals):
    """C15.Signals: A3 state at `speed` plus the fast ATR; the blend replaces its output."""

    def signals(self) -> dict[str, float]:
        trend = C6.dollar_signals(A3.Signals.signals(self))
        value = C24.xs_weights_ev(self.md, value_bars(self.p), -1.0)
        return {k: combine(trend[k], value[k]) * vol_scale(self.md.atr(k), self._fast_atr[k].value) for k in trend}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
