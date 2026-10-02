"""
C5-056 - dollar trend measured on EXCESS returns (spot plus carry), as the time-series momentum literature does
(Moskowitz, Ooi & Pedersen 2012; Baz et al. 2015 use futures, whose returns include the interest differential)
(cycle 5, new family after the holdout diagnostic).

Context: after the 2023 holdout diagnostic the user asked for other, better strategies. Every trend book here measured
trend on SPOT prices, while the published trend signals use excess-return (futures) prices; overnight financing has
been a steady drag on every trend book. Measuring trend on the carry-inclusive price aligns the signal with what the
position actually earns.

    TR_t   = ln C_t + sum over bars <= t of known_carry / 100 / 6240        (6,240 hourly bars ~ one year)
    P_t    = exp(TR_t)                                                    (carry-inclusive price index)
    s      = C5-041's dollar trend (A3 response, windows 63/252 days x norm_scale) computed on P instead of C,
             times clip((ATR_1440 / ATR_F)^2, 0.25, 4)                    (ATRs on spot prices, as before)

An unknown rate accrues nothing (spot only). Sizing, execution, costs, financing and breakers: academic_base.py.
Indicators (3): EWMA crossover (on excess-return prices), price standard deviation, ATR.
Tunable parameters (3): norm_scale, fast_bars, band (band fixed at 0.3).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import a3_ewma_crossover as A3
from hypotheses import academic_base as A
from hypotheses import c5_006_dollar_trend as C6
from hypotheses import c5_040_norm_scale_grid as C40
from hypotheses import c5_041_trend_norm_scale as C41
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_034_value_trend_variance_scaled import LIMITS, POWER, scale
from hypotheses.c5_040_norm_scale_grid import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                               StrategyParams, windows)
from hypotheses.c5_041_trend_norm_scale import history_bars, warmup_bars  # noqa: F401

HYPOTHESIS = ("C5-056 dollar trend on excess-return prices (spot + accrued known carry), C5-041's construction; "
              "new family after the holdout diagnostic.")
INDICATORS = ("EWMA crossover (excess-return prices)", "price standard deviation", "ATR")
USES_CARRY = True
DEPENDS = ("c5_041_trend_norm_scale.py",) + C41.DEPENDS
ACCRUAL = 1.0 / 100.0 / 6240.0      # percent p.a. -> fraction per hourly bar


def accrual(carry: float) -> float:
    return carry * ACCRUAL if carry == carry else 0.0


def excess_price(f: pd.DataFrame) -> pd.Series:
    carry = f["carry"] if "carry" in f.columns else pd.Series(np.nan, index=f.index)
    cum = np.cumsum([accrual(float(c)) for c in carry.to_numpy(dtype=float)])
    return pd.Series(np.exp(np.log(f["close"].to_numpy(dtype=float)) + cum), index=f.index)


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    pw, sw = windows(p)
    trend = {k: C40.pair_trend(excess_price(f), pw, sw) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    if legs:
        cols = pd.DataFrame({k: -d * trend[k].reindex(index).ffill() for k, (_, d) in legs.items()})
        s_usd = cols.mean(axis=1, skipna=False)
    else:
        s_usd = pd.Series(math.nan, index=index)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        leg = A.usd_leg(k)
        t = (-leg[1] * s_usd).reindex(f.index).to_numpy(dtype=float) if leg else np.zeros(len(f))
        fast = A.wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy(), p.fast_bars)
        m = np.array([scale(a, b, POWER, LIMITS) for a, b in zip(g["atr"].to_numpy(dtype=float), fast)])
        g["signal"] = t * m
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(C41.Signals):
    """C5-041's state; A3's update (line for line) fed with the carry-inclusive price."""

    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._cum = {q: 0.0 for q in self.pairs}

    def update(self, pair: str) -> None:
        self._cum[pair] += accrual(self.md.carry[pair])
        c = math.exp(math.log(self.md.close(pair)) + self._cum[pair])
        sd = self._price_sd[pair].update(c)
        us = []
        for fast, slow, q_sd in zip(self._fast[pair], self._slow[pair], self._q_sd[pair]):
            x = fast.update(c) - slow.update(c)
            if not sd > 0:
                us.append(math.nan)
                continue
            qs = q_sd.update(x / sd)
            us.append(A3.response((x / sd) / qs) if qs > 0 else math.nan)
        self._u[pair] = (us[0] + us[1] + us[2]) / 3.0
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
