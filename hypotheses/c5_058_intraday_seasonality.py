"""
C5-058 - intraday FX seasonality: a currency tends to depreciate during its home market's trading hours (Breedon &
Ranaldo 2013, "Intraday patterns in FX returns and order flow", JMCB 45(5)) (cycle 5, after the user approved
time-of-day signals for this effect; hypotheses.TIME_OF_DAY_ALLOWED).

Context: 65 trials found no edge near the 1.5 gate with price, rate and value signals, and the best book lost on the
2023 holdout. The cycle's "no calendar filters" rule excluded the one documented FX effect that is about the clock.
Breedon & Ranaldo find that currencies depreciate in their local trading hours (domestic order flow sells them) and
recover outside them. The rule here is theirs, with the sessions fixed by market convention, not fitted:

    in(c, t) = 1 if the local time of c's centre at t is a weekday between 08:00 and 17:00, else 0
               EUR Frankfurt (Europe/Berlin), GBP London, USD New York, CAD Toronto, JPY Tokyo (DST via zoneinfo)
    s(pair = base/quote, t) = in(quote, t) - in(base, t)          in {-1, 0, +1}
        long the base while only the quote's market is open, short it while only its own market is open

t is the decision time (the close of the bar, i.e. the open of the hour the position is held). USDCAD never trades
(New York and Toronto share hours). Sizing, execution, costs, financing and breakers: hypotheses/academic_base.py.
Indicators (2): trading-session clock, ATR. Tunable parameters (1): band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtest as B
from hypotheses import academic_base as A
from hypotheses.academic_base import (RiskLimits, RiskManager, Segment, round_trips,  # noqa: F401
                                      trading_day, trading_day_ids)

HYPOTHESIS = ("C5-058 intraday FX seasonality (Breedon-Ranaldo 2013): short a currency while only its home market "
              "is open (08:00-17:00 local, weekdays), long it while only the other currency's market is open.")
INDICATORS = ("trading-session clock", "ATR")
USES_CARRY = False
USES_TIME_OF_DAY = True
CENTRES = {"EUR": ZoneInfo("Europe/Berlin"), "GBP": ZoneInfo("Europe/London"), "USD": ZoneInfo("America/New_York"),
           "CAD": ZoneInfo("America/Toronto"), "JPY": ZoneInfo("Asia/Tokyo")}
OPEN_HOUR, CLOSE_HOUR = 8, 17


@dataclass(frozen=True)
class StrategyParams:
    band: float = 0.2              # no-trade band (only ATR-driven resizes; the signal is +-1 or 0)

    def validate(self) -> "StrategyParams":
        if not 0 <= self.band < 1:
            raise ValueError("band must be in [0, 1)")
        return self


DEFAULT_PARAMS = StrategyParams()
PARAM_GRID = {"band": [0.2, 0.5]}
FEATURE_PARAMS = ()


def warmup_bars(p: StrategyParams) -> int:
    return A.ATR_BARS + 2


def history_bars(p: StrategyParams) -> int:
    return 2


def in_session(ccy: str, ts_ns: int) -> int:
    tz = CENTRES.get(ccy)
    if tz is None:
        return 0
    local = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone(tz)
    return int(local.weekday() < 5 and OPEN_HOUR <= local.hour < CLOSE_HOUR)


def session_signal(pair: str, ts_ns: int) -> float:
    base, quote = A.currencies(pair)
    return float(in_session(quote, ts_ns) - in_session(base, ts_ns))


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable."""
    single, frames = A.features_dict(data)
    out = {}
    for k, f in frames.items():
        g = A.base_features(f)
        closes = f.index.as_unit("ns").asi8 + A.HOUR_NS
        g["signal"] = np.array([session_signal(k, int(t)) for t in closes])
        out[k] = g
    return A.finish(single, out)


def signal_frame(data, p: StrategyParams):
    return A.signal_frame_from(compute_features, data, p)


# ============================================================================= event-driven path
class Signals(A.Signals):
    def __init__(self, p, pairs, md):
        super().__init__(p, pairs, md)
        self._ts = None

    def on_close(self, ts: int) -> None:
        self._ts = int(ts)

    def signals(self) -> dict[str, float]:
        if self._ts is None:
            return {k: math.nan for k in self.pairs}
        return {k: session_signal(k, self._ts) for k in self.pairs}


class PortfolioTrendStrategy(A.TargetPortfolioStrategy):
    SIGNALS = Signals
    WARMUP = staticmethod(warmup_bars)
    HISTORY = staticmethod(history_bars)


def warmup_start(plan: list[Segment]) -> int:
    return A.warmup_start(plan, warmup_bars)


def reference_backtest(h1: pd.DataFrame, plan: list[Segment], rates=None, pair: str = "EURUSD",
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    return A.reference_backtest(signal_frame, warmup_bars, pair, h1, plan, rates, limits, nav0)
