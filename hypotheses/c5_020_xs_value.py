"""
C5-020 - currency value as long-term cross-sectional reversal, volatility-managed (Asness, Moskowitz & Pedersen 2013,
"Value and momentum everywhere", JF 68(3); Menkhoff, Sarno, Schmeling & Schrimpf 2017, "Currency value", RFS 30(2))
(cycle 5, family 5 building block).

Rationale (batches 1-7): price trend is the only positive source, and a second, negatively correlated one is what
lifts a combined book (AMP: value and momentum correlate about -0.5 in every asset class). Currency value needs price
levels (CPI), which this environment cannot download; AMP and Menkhoff et al. note that over multi-year horizons the
nominal exchange-rate change is dominated by the real one among developed currencies, so this uses the nominal proxy:
the currencies that fell most over the last 3 years are "cheap".

Rank USD, EUR, GBP, JPY, CAD by their 3-year (756 trading days) performance against USD, average ranks for ties;
dollar-neutral weights w = (rank - 3) / 2 (A5's machinery, unchanged). The signal is the REVERSE, as in C5-002:
    EURUSD, GBPUSD: s = -w_EUR, -w_GBP;   USDJPY, USDCAD: s = +w_JPY, +w_CAD;   EURJPY: 0
    s *= clip(ATR_1440 / ATR_F, 0.5, 2.0)                                   (C5-010's volatility scale)

Disclosure: the signal needs 3 years of history, so it starts in 2008 and split 1's in-sample window (2006-07 ->
2011-10) has about 3.7 years of it. The 3-year horizon (AMP use 5) is set by the data, which starts in 2005.
Sizing, execution (no-trade band), costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): past returns (cross-sectional rank), ATR (two speeds). Tunable parameters (2): fast_bars, band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)
from hypotheses.c5_010_volmanaged_dollar_trend import vol_scale

HYPOTHESIS = ("C5-020 currency value (AMP 2013 / Menkhoff et al. 2017, nominal proxy): long the currencies that "
              "fell most against the others over 3 years, short those that rose most (dollar-neutral rank weights); "
              "scaled by C5-010's ATR ratio.")
INDICATORS = ("past returns (cross-sectional rank)", "ATR")
USES_CARRY = False
DEPENDS = ("c5_010_volmanaged_dollar_trend.py",)
VALUE_DAYS = 756                   # 3 years of trading days


@dataclass(frozen=True)
class StrategyParams:
    fast_bars: int = 960           # fast ATR period in hourly bars
    band: float = 0.2              # no-trade band, fraction of the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.fast_bars, (int, np.integer)) and self.fast_bars >= 2 and 0 <= self.band < 1):
            raise ValueError("fast_bars must be an int >= 2 and band in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"fast_bars": [120, 240, 480, 960], "band": [0.1, 0.2, 0.3, 0.5]}
FEATURE_PARAMS = ("fast_bars",)
VALUE_BARS = VALUE_DAYS * A.BARS_PER_DAY


def warmup_bars(p: StrategyParams) -> int:
    return max(VALUE_BARS, A.ATR_BARS, p.fast_bars) + 2


def history_bars(p: StrategyParams) -> int:
    return VALUE_BARS + 1


def value_weights_vec(frames) -> dict[str, pd.Series]:
    return {k: -w for k, w in A.vec_xs_weights(frames, VALUE_BARS).items()}


def value_weights_ev(md) -> dict[str, float]:
    return {k: -v for k, v in A.ev_xs_weights(md, VALUE_BARS).items()}


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    v = value_weights_vec(frames)
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
                for k, v in value_weights_ev(self.md).items()}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
