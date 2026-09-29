"""
C5-026 - cross-sectional carry (HML_FX) with smooth z-score weights, volatility-managed (Lustig, Roussanov &
Verdelhan 2011, "Common risk factors in currency markets", RFS 24(11); Koijen, Moskowitz, Pedersen & Vrugt 2018)
(cycle 5, family 5 building block).

Rationale (batches 1-10): carry has been tested per pair (A4: sign of each pair's differential) and as one dollar
bet (C5-017), and both lost. The canonical currency carry factor is neither: it is dollar-neutral and cross-sectional
(long the highest-yielding currencies, short the lowest), and its return comes from the spread between high and low
yielders rather than from the USD's direction. With smooth weights it changes only when rates do (monthly before
2020-07, on decisions after), so it trades little.

    x_c = r_c - r_USD, the known rate differential of currency c against USD   (x_USD = 0; the `carry` column,
          signed by the pair's USD leg, as in C5-017)
    w_c = +(x_c - mean(x)) / (2 * std(x))                                  (cross-sectional z-score, population std)
    EURUSD, GBPUSD: s = w_EUR, w_GBP;   USDJPY, USDCAD: s = -w_JPY, -w_CAD;   EURJPY: 0
    s *= clip(ATR_1440 / ATR_F, 0.5, 2.0)                                   (C5-010's volatility scale)

Rates are those public at the time (OECD monthly from the next month, policy decisions from the next day).
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): rate differential (cross-sectional z-score), ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import (DEFAULT_PARAMS, FEATURE_PARAMS, PARAM_GRID,  # noqa: F401
                                                       StrategyParams, vol_scale)
from hypotheses.c5_022_smooth_value import z_weights

HYPOTHESIS = ("C5-026 cross-sectional carry (HML_FX; Lustig-Roussanov-Verdelhan 2011): dollar-neutral z-score "
              "weights on the currencies' known rates, long high yielders, short low; scaled by C5-010's ATR ratio.")
INDICATORS = ("rate differential (cross-sectional z-score)", "ATR")
USES_CARRY = True
DEPENDS = ("c5_022_smooth_value.py", "c5_020_xs_value.py", "c5_010_volmanaged_dollar_trend.py")


def warmup_bars(p: StrategyParams) -> int:
    return max(A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return 2


def carry_weights(x: dict[str, float]) -> dict[str, float]:
    """+z / 2 per currency (z_weights returns -z / 2); NaN for all if any rate is unknown."""
    return {c: -v for c, v in z_weights(x).items()}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    base = {k: A.base_features(f) for k, f in frames.items()}
    index = A.universe_index(frames)
    legs = {k: A.usd_leg(k) for k in frames if A.usd_leg(k) is not None}
    x = {"USD": np.zeros(len(index))}
    for k, (ccy, d) in legs.items():
        x[ccy] = (d * base[k]["carry"]).reindex(index).ffill().to_numpy(dtype=float)
    ccys = list(x)
    rows = [carry_weights({c: float(x[c][i]) for c in ccys}) for i in range(len(index))]
    w = {c: pd.Series([r[c] for r in rows], index=index) for c in ccys}
    out = {}
    for k, f in frames.items():
        g = base[k]
        leg = A.usd_leg(k)
        s = (leg[1] * w[leg[0]]).reindex(g.index).to_numpy(dtype=float) if leg else np.zeros(len(g))
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
        self._legs = {q: A.usd_leg(q) for q in self.pairs if A.usd_leg(q) is not None}

    def update(self, pair: str) -> None:
        self._fast_atr[pair].update(self.md.high[pair], self.md.low[pair], self.md.close(pair))

    def signals(self) -> dict[str, float]:
        x = {"USD": 0.0}
        for k, (ccy, d) in self._legs.items():
            x[ccy] = d * self.md.carry[k]
        w = carry_weights(x)
        return {k: ((self._legs[k][1] * w[self._legs[k][0]]) if k in self._legs else 0.0)
                * vol_scale(self.md.atr(k), self._fast_atr[k].value) for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
