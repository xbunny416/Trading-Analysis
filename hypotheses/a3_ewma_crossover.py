"""
A3 - multi-speed EWMA crossover trend (Baz, Granger, Harvey, Le Roux & Rattray 2015, "Dissecting investment
strategies in the cross section and time series", Man AHL / SSRN 2695101; the same definition is used by Lim,
Zohren & Roberts 2019).

For each speed k with (S_k, L_k) = (8, 24), (16, 48), (32, 96) days, on hourly closes (24 bars a day):

    x_k = EWMA(C; S_k) - EWMA(C; L_k)          EWMA(n): y_t = y_{t-1} + (C_t - y_{t-1}) / (24 n)
    q_k = x_k / std(C over the last 63 days)
    z_k = q_k / std(q_k over the last 252 days)
    u_k = z_k * exp(-z_k^2 / 4) / 0.89         (response function: strongest at |z| = sqrt(2), fades for extremes)
    s   = (u_1 + u_2 + u_3) / 3                decided at the close of bar t-1, traded at the open of bar t

Every constant is the published one; none is tuned. Sizing and execution (hypotheses/academic_base.py):
units = NAV * 0.3 % * |s| / (ATR * 10 * quote->USD), Wilder ATR over 1,440 hourly bars; trades through the no-trade
band `band`; overnight financing is booked by the venue.
Indicators (3): EWMA crossover, price standard deviation, ATR. Tunable parameters (1): band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("A3 multi-speed EWMA crossover (Baz et al. 2015, Man AHL): average response u(z) of three normalised "
              "EWMA crossovers (8/24, 16/48, 32/96 days); volatility-targeted, no-trade band.")
INDICATORS = ("EWMA crossover", "price standard deviation", "ATR")
USES_CARRY = False
SPEEDS_DAYS = ((8, 24), (16, 48), (32, 96))
PRICE_STD_BARS = 63 * A.BARS_PER_DAY
SIGNAL_STD_BARS = 252 * A.BARS_PER_DAY


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


def alpha(days: int) -> float:
    return 1.0 / (days * A.BARS_PER_DAY)


def warmup_bars(p: StrategyParams) -> int:
    return PRICE_STD_BARS + SIGNAL_STD_BARS + 2


def history_bars(p: StrategyParams) -> int:
    return 2


def response(z: float) -> float:
    return z * math.exp(-z * z / 4.0) / 0.89


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        c = f["close"]
        c_np = c.to_numpy(dtype=float)
        sd = c.rolling(PRICE_STD_BARS).std()
        us = []
        for s_days, l_days in SPEEDS_DAYS:
            x = pd.Series(A.ewma(c_np, alpha(s_days)) - A.ewma(c_np, alpha(l_days)), index=c.index)
            q = (x / sd).where(sd > 0)
            qs = q.rolling(SIGNAL_STD_BARS).std()
            z = (q / qs).where(qs > 0)
            us.append(z * np.exp(-z * z / 4.0) / 0.89)
        g["signal"] = (us[0] + us[1] + us[2]) / 3.0
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast = {q: [A.EWMA(alpha(s)) for s, _ in SPEEDS_DAYS] for q in self.pairs}
        self._slow = {q: [A.EWMA(alpha(l)) for _, l in SPEEDS_DAYS] for q in self.pairs}
        self._price_sd = {q: A.RollingStd(PRICE_STD_BARS) for q in self.pairs}
        self._q_sd = {q: [A.RollingStd(SIGNAL_STD_BARS) for _ in SPEEDS_DAYS] for q in self.pairs}
        self._u = {q: math.nan for q in self.pairs}

    def update(self, pair: str) -> None:
        c = self.md.close(pair)
        sd = self._price_sd[pair].update(c)
        us = []
        for fast, slow, q_sd in zip(self._fast[pair], self._slow[pair], self._q_sd[pair]):
            x = fast.update(c) - slow.update(c)
            if not sd > 0:              # NaN (history too short) or a flat window
                us.append(math.nan)
                continue
            qs = q_sd.update(x / sd)
            us.append(response((x / sd) / qs) if qs > 0 else math.nan)
        self._u[pair] = (us[0] + us[1] + us[2]) / 3.0

    def signals(self) -> dict[str, float]:
        return dict(self._u)


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
