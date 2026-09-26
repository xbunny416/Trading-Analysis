"""
backtest.py - NautilusTrader plumbing for the research harness.

Data path (all times UTC)
-------------------------
MT5 M15 export (bid OHLC + spread in points, server clock Europe/Athens)
  -> M15 frame: bid OHLC + spread in pips, index = bar open (UTC)
  -> H1 frame: MID OHLC for signals, plus the executable quotes of each hour
         open_bid / open_ask   first M15 bar of the hour
         close_bid / close_ask last M15 bar of the hour
     with  half_width = max(spread, MIN_SPREAD_PIPS) / 2 + SLIPPAGE_PIPS   around the mid
  -> Nautilus objects per pair
         Bar        "<PAIR>.SIM-1-HOUR-MID-EXTERNAL", ts_event = bar close        (signals)
         QuoteTick  at bar open + 1 ms and bar close - 1 ms                       (execution / marking)

Engine
------
One SIM venue: NETTING, MARGIN account in USD, zero fees (costs are in the quotes), bar_execution off,
and a 1 ns order latency. A market order submitted when an hourly bar closes therefore fills against
the NEXT hour's open quote - "decide at the close of bar t-1, execute at the open of bar t".
Positions are marked at the bid (longs) / ask (shorts), i.e. at liquidation value.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import CAD, EUR, GBP, JPY, USD
from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mt5"
PAIRS = ("EURUSD", "GBPUSD", "USDJPY", "USDCAD", "EURJPY")
SERVER_TZ = "Europe/Athens"          # MT5 server clock: EET/EEST, EU daylight-saving rules
SIM = Venue("SIM")
BASE_CCY = USD
STARTING_NAV = 1_000_000.0
MIN_SPREAD_PIPS = 1.0                # mandate floor for the quoted spread
SLIPPAGE_PIPS = 0.5                  # added on every fill
ONE_MS, ONE_HOUR = 1_000_000, 3_600_000_000_000
_CCY = {"EUR": EUR, "USD": USD, "GBP": GBP, "JPY": JPY, "CAD": CAD}


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def point_size(pair: str) -> float:
    return pip_size(pair) / 10.0


def price_precision(pair: str) -> int:
    """One digit finer than the MT5 quote so mid prices (half points) are exact."""
    return 4 if pair.endswith("JPY") else 6


def instrument_id(pair: str) -> InstrumentId:
    return InstrumentId(Symbol(f"{pair[:3]}/{pair[3:]}"), SIM)


def bar_type(pair: str) -> BarType:
    return BarType.from_str(f"{pair[:3]}/{pair[3:]}.SIM-1-HOUR-MID-EXTERNAL")


def make_instrument(pair: str) -> CurrencyPair:
    prec = price_precision(pair)
    return CurrencyPair(
        instrument_id(pair), Symbol(f"{pair[:3]}/{pair[3:]}"), _CCY[pair[:3]], _CCY[pair[3:]],
        prec, 0, Price(10.0 ** -prec, prec), Quantity.from_int(1), 0, 0,
        lot_size=Quantity.from_int(1000), margin_init=Decimal("0.02"), margin_maint=Decimal("0.02"),
        maker_fee=Decimal(0), taker_fee=Decimal(0),
    )


# ============================================================================= loading
def load_mt5_csv(path: str | Path, pair: str) -> pd.DataFrame:
    """MT5 export -> M15 bid bars with a UTC index (bar open) and the spread in pips."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    return clean_m15(raw, pair)


def clean_m15(raw: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Server time -> UTC; drop missing / non-positive / inconsistent bars and duplicate stamps."""
    d = raw.copy()
    if "time" in d.columns:
        t = pd.to_datetime(d.pop("time"))
        idx = pd.DatetimeIndex(t).tz_localize(SERVER_TZ, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    else:
        idx = pd.DatetimeIndex(d.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    d.index = idx.as_unit("ns")
    d = d[["open", "high", "low", "close", "spread"]].astype(float)
    o, h, l, c, s = (d[k].to_numpy() for k in ("open", "high", "low", "close", "spread"))
    with np.errstate(invalid="ignore"):
        good = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isfinite(s)
                & (l > 0) & (s >= 0) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)))
    d = d[good]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d["spread_pips"] = d.pop("spread") * point_size(pair) / pip_size(pair)
    return d


def hourly_frame(m15: pd.DataFrame, pair: str) -> pd.DataFrame:
    """H1 MID bars + the executable open/close quotes of each hour (spread floor + slippage applied)."""
    pip = pip_size(pair)
    mid = m15[["open", "high", "low", "close"]].add(m15["spread_pips"] * pip / 2.0, axis=0)  # bid + spread/2
    width = np.maximum(m15["spread_pips"], MIN_SPREAD_PIPS) * pip / 2.0 + SLIPPAGE_PIPS * pip
    g = pd.DataFrame({
        "open": mid["open"], "high": mid["high"], "low": mid["low"], "close": mid["close"],
        "open_bid": mid["open"] - width, "open_ask": mid["open"] + width,
        "close_bid": mid["close"] - width, "close_ask": mid["close"] + width,
        "spread_pips": m15["spread_pips"],
    }, index=m15.index)
    h = g.resample("1h", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "open_bid": "first", "open_ask": "first", "close_bid": "last", "close_ask": "last",
        "spread_pips": "first",
    }).dropna(subset=["open"])
    prec = price_precision(pair)
    return h.round({c: prec for c in h.columns if c != "spread_pips"})


def load_universe(pairs=PAIRS, data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """M15 frames for every pair, trimmed to their common span."""
    m15 = {p: load_mt5_csv(data_dir / f"{p}.csv", p) for p in pairs}
    lo = max(f.index[0] for f in m15.values())
    hi = min(f.index[-1] for f in m15.values())
    return {p: f.loc[lo:hi] for p, f in m15.items()}


# ============================================================================= nautilus objects
def nautilus_data(pair: str, h1: pd.DataFrame) -> list:
    iid, bt, prec = instrument_id(pair), bar_type(pair), price_precision(pair)
    size = Quantity.from_int(10**12)
    opens = h1.index.as_unit("ns").asi8
    cols = [h1[c].to_numpy() for c in ("open", "high", "low", "close", "open_bid", "open_ask",
                                        "close_bid", "close_ask")]
    out = []
    one = Quantity.from_int(1)
    for i, t in enumerate(opens.tolist()):
        o, hi, lo, c, ob, oa, cb, ca = (float(col[i]) for col in cols)
        out.append(QuoteTick(iid, Price(ob, prec), Price(oa, prec), size, size, t + ONE_MS, t + ONE_MS))
        out.append(QuoteTick(iid, Price(cb, prec), Price(ca, prec), size, size, t + ONE_HOUR - ONE_MS,
                             t + ONE_HOUR - ONE_MS))
        out.append(Bar(bt, Price(o, prec), Price(hi, prec), Price(lo, prec), Price(c, prec), one,
                       t + ONE_HOUR, t + ONE_HOUR))
    return out


@dataclass
class RunResult:
    equity: pd.Series          # USD equity at every hourly decision time in the window
    orders: pd.DataFrame       # every order the strategy submitted (the decision log)
    fills: pd.DataFrame        # every fill (Nautilus events)
    trades: pd.DataFrame       # round trips
    n_halts: int
    killed: bool
    start_nav: float


class Market:
    """Hourly frames for a set of pairs, loaded once into a reusable Nautilus BacktestEngine."""

    def __init__(self, m15: dict[str, pd.DataFrame]):
        self.pairs = tuple(m15)
        self.h1 = {p: hourly_frame(f, p) for p, f in m15.items()}
        idx = self.h1[self.pairs[0]].index
        for f in self.h1.values():
            idx = idx.union(f.index)
        self.index = idx                                   # union of hourly bar-open times
        self._engine: BacktestEngine | None = None

    def engine(self) -> BacktestEngine:
        if self._engine is None:
            eng = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
            eng.add_venue(SIM, OmsType.NETTING, AccountType.MARGIN, [Money(STARTING_NAV, BASE_CCY)],
                          base_currency=BASE_CCY, default_leverage=Decimal(50), bar_execution=False,
                          latency_model=LatencyModel(base_latency_nanos=1))
            data = []
            for p in self.pairs:
                eng.add_instrument(make_instrument(p))
                data += nautilus_data(p, self.h1[p])
            eng.add_data(data)
            self._engine = eng
        return self._engine

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def run(self, strategy, start_ns: int, end_ns: int) -> RunResult:
        """Run `strategy` (a strategy.PortfolioTrendStrategy) from its warm-up start to `end_ns`."""
        eng = self.engine()
        eng.reset()
        # BacktestEngine.reset() clears the cache's FX conversion table (Cache._xrate_symbols), which is
        # only filled by Cache.add_instrument; without this, JPY/CAD P&L silently never reaches the USD account.
        for p in self.pairs:
            eng.cache.add_instrument(eng.cache.instrument(instrument_id(p)))
        eng.clear_strategies()
        eng.add_strategy(strategy)
        eng.run(start=pd.Timestamp(strategy.warmup_start_ns, tz="UTC"), end=pd.Timestamp(end_ns, tz="UTC"))
        res = strategy.result()
        eng.clear_strategies()
        return res
