"""
C5-024 - cross-sectional currency momentum with smooth (z-score) weights, volatility-managed (Menkhoff, Sarno,
Schmeling & Schrimpf 2012) (cycle 5, family 5 building block).

Rationale (batch 9 + cycle 4): smooth weights turned value from 0.03 into 0.28 by cutting its rank-swap churn. A5
(cross-sectional momentum, rank weights) had a small edge before costs (about +$23k) and lost $55k to spread. This is
A5 with C5-022's smooth weights: a dollar-neutral trend, so it should overlap less with the dollar-consensus trend
than per-pair trend does, and it is the candidate third source for the value + trend blend.

    x_c = ln(price of currency c in USD now / L months ago)      (x_USD = 0; L in 1, 3, 6, 12 months)
    w_c = +(x_c - mean(x)) / (2 * std(x))                         (cross-sectional z-score, population std)
    EURUSD, GBPUSD: s = w_EUR, w_GBP;   USDJPY, USDCAD: s = -w_JPY, -w_CAD;   EURJPY: 0
    s *= clip(ATR_1440 / ATR_F, 0.5, 2.0)                                      (C5-010's volatility scale)

Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): past returns (cross-sectional z-score), ATR (two speeds).
Tunable parameters (3): lookback_months, fast_bars, band.
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
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale
from hypotheses.c5_022_smooth_value import z_weights

HYPOTHESIS = ("C5-024 cross-sectional currency momentum with smooth weights: A5's L-month performance ranking as "
              "dollar-neutral z-score weights; scaled by C5-010's ATR ratio.")
INDICATORS = ("past returns (cross-sectional z-score)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_022_smooth_value.py", "c5_020_xs_value.py", "c5_010_volmanaged_dollar_trend.py")


@dataclass(frozen=True)
class StrategyParams:
    lookback_months: int = 6       # formation period
    fast_bars: int = 960           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.lookback_months, (int, np.integer)) and self.lookback_months >= 1
                and isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1):
            raise ValueError("lookback_months and fast_bars must be positive ints (fast_bars >= 2), band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"lookback_months": [1, 3, 6, 12], "fast_bars": [240, 960], "band": [0.2, 0.5]}
FEATURE_PARAMS = ("lookback_months", "fast_bars")


def lag_bars(p) -> int:
    return p.lookback_months * A.BARS_PER_MONTH


def warmup_bars(p: StrategyParams) -> int:
    return max(lag_bars(p), A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return lag_bars(p) + 1


def xs_weights_vec(frames, lag: int, sign: float) -> dict[str, pd.Series]:
    """Pair signals from sign * z-scores of the currencies' lag-bar log performance against USD."""
    legs = {p: A.usd_leg(p) for p in frames if A.usd_leg(p) is not None}
    index = A.universe_index(frames)
    perf = {"USD": np.zeros(len(index))}
    for p, (ccy, d) in legs.items():
        c = frames[p]["close"]
        perf[ccy] = (d * np.log(c / c.shift(lag))).reindex(index).ffill().to_numpy(dtype=float)
    ccys = list(perf)
    rows = [z_weights({c: float(perf[c][i]) for c in ccys}) for i in range(len(index))]
    w = {c: pd.Series([-sign * r[c] for r in rows], index=index) for c in ccys}     # z_weights returns -z
    out = {}
    for p, f in frames.items():
        leg = A.usd_leg(p)
        out[p] = (leg[1] * w[leg[0]]).reindex(f.index) if leg else pd.Series(0.0, index=f.index)
    return out


def xs_weights_ev(md, lag: int, sign: float) -> dict[str, float]:
    legs = {p: A.usd_leg(p) for p in md.pairs if A.usd_leg(p) is not None}
    perf = {"USD": 0.0}
    for p, (ccy, d) in legs.items():
        c, c0 = md.close(p), md.past_close(p, lag)
        perf[ccy] = d * math.log(c / c0) if c0 == c0 else math.nan
    w = {c: -sign * v for c, v in z_weights(perf).items()}
    return {p: (A.usd_leg(p)[1] * w[A.usd_leg(p)[0]] if A.usd_leg(p) else 0.0) for p in md.pairs}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    v = xs_weights_vec(frames, lag_bars(p), 1.0)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([vol_scale(a, b) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = v[k].to_numpy(dtype=float) * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._fast_atr = {q: A.WilderATR(p.fast_bars) for q in self.pairs}

    def update(self, pair: str) -> None:
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        return {k: v * vol_scale(self.md.atr(k), self._fast_atr[k].value)
                for k, v in xs_weights_ev(self.md, lag_bars(self.p), 1.0).items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
