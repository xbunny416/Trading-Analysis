"""
C5-018 - interest-rate momentum (monetary-policy momentum), volatility-managed (cycle 5, new family: fundamental
momentum). Dahlquist & Hasseltoft 2020, "Economic momentum and currency returns", JFE 136(1); Brooks 2017, "A half
century of macro momentum", AQR.

Rationale (batches 1-6): the only positive source in 25 trials is price trend, and every trend book wins and loses in
the same regimes. Currencies whose short rate is rising relative to the other side's tend to appreciate over the
following months (the papers' interest-rate / economic momentum), and this signal comes from the rate tables, not
from prices. Unlike carry (the rate LEVEL, which lost in A4 and C5-017), it trades the CHANGE in the differential.

Per pair, at the close of hourly bar t:
    d_t  = the known rate differential r_base - r_quote (percent p.a.; OECD monthly from the next month, policy
           decisions from the next day - the same `carry` column A4 uses)
    dd   = d_t - d_{t-L},  L = lookback_days x 24 bars
    s    = sign(dd) if |dd| > theta else 0            (NaN until both rates are known)
    s   *= clip(ATR_1440 / ATR_F, 0.5, 2.0)           (C5-010's volatility scale)

Disclosure: the rate source switches from OECD 3-month rates to policy rates on 2020-07-01; a window spanning the
switch mixes the two (a level gap of typically 0.1-0.3 points per currency, which partly cancels in d).
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): rate differential, ATR (two speeds). Tunable parameters (4): lookback_days, theta, fast_bars, band
(band fixed at 0.3, not in the grid).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-018 interest-rate momentum (Dahlquist-Hasseltoft 2020; Brooks 2017): long a pair when its known "
              "rate differential rose by more than theta points over the last L days, short when it fell; scaled "
              "by C5-010's ATR ratio.")
INDICATORS = ("rate differential", "ATR")
USES_CARRY = True
DEPENDS = ("c5_010_volmanaged_dollar_trend.py",)


@dataclass(frozen=True)
class StrategyParams:
    lookback_days: int = 126       # change window in trading days (24 hourly bars each)
    theta: float = 0.1             # minimum |change| of the differential, percentage points
    fast_bars: int = 960           # fast ATR period in hourly bars
    band: float = 0.3              # no-trade band (fixed, not in the grid)

    def validate(self) -> "StrategyParams":
        ok = (isinstance(self.lookback_days, (int, np.integer)) and self.lookback_days >= 1 and self.theta >= 0
              and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1)
        if not ok:
            raise ValueError("lookback_days and fast_bars must be positive ints, theta >= 0, band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"lookback_days": [63, 126, 252], "theta": [0.1, 0.25], "fast_bars": [240, 960]}
FEATURE_PARAMS = ("lookback_days", "theta", "fast_bars")


def lag_bars(p: StrategyParams) -> int:
    return p.lookback_days * A.BARS_PER_DAY


def warmup_bars(p: StrategyParams) -> int:
    return max(A.ATR_BARS, lag_bars(p), p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


def rate_momentum(d_now: float, d_then: float, theta: float) -> float:
    dd = float(d_now) - float(d_then)
    if dd != dd:
        return math.nan
    return float((dd > 0) - (dd < 0)) if abs(dd) > theta else 0.0


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    L = lag_bars(p)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        d = g["carry"].to_numpy(dtype=float)
        d_then = g["carry"].shift(L).to_numpy(dtype=float)
        s = np.array([rate_momentum(a, b, p.theta) for a, b in zip(d, d_then)])
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = s * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}
        self._d = {q: deque(maxlen=lag_bars(p) + 1) for q in self.pairs}

    def update(self, pair: str) -> None:
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))
        self._d[pair].append(self.md.carry[pair])

    def signals(self) -> dict[str, float]:
        out = {}
        for k in self.pairs:
            dq = self._d[k]
            s = rate_momentum(dq[-1], dq[0], self.p.theta) if len(dq) == dq.maxlen else math.nan
            out[k] = s * vol_scale(self.md.atr(k), self._fast_atr[k].value)
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
