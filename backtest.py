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

Cycle 4 additions
-----------------
Spliced history: OANDA 1-minute mids (2005 -> the first MT5 bar) + the MT5 exports after it. OANDA has no
USD/JPY, so it is derived as EUR/JPY / EUR/USD. OANDA-era spreads are the broker's median MT5 spread for
the same pair and UTC hour over the development period (data layer only, never seen by strategy code).

Overnight financing: at every 17:00 New York roll (one per calendar day; a weekend is three rolls), each open
position is credited/debited  qty_signed*mid*(r_base - r_quote)/365 - |qty|*mid*FIN_MARKUP/365  (in the quote
currency, converted to USD). Rates come from ``RateTable`` (OECD monthly 3-month rates to 2020-06, central-bank
policy rates after). Nautilus' own FXRolloverInterestModule is not used: it ignores the position side.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import LatencyModel
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import SimulationModuleConfig
from nautilus_trader.model.currencies import CAD, EUR, GBP, JPY, USD
from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mt5"
RATES_DIR = ROOT / "data" / "rates"
SPLICED_DIR = ROOT / "data" / "spliced"
OANDA_REPO = "https://github.com/FutureSharks/financial-data.git"
OANDA_COMMIT = "7ba1d404aa8b0e1c0f71321acebadcbfb9bcca8d"
OANDA_SOURCE = ROOT.parent / "FutureSharks" / "financial-data"
OANDA_SUBDIR = "pyfinancialdata/data/currencies/oanda"
OANDA_START = pd.Timestamp("2005-01-01", tz="UTC")
PROFILE_END = pd.Timestamp("2023-01-01", tz="UTC")   # spread profile: development-period MT5 bars only
POLICY_SWITCH = pd.Timestamp("2020-07-01", tz="UTC")  # OECD monthly rates before, policy rates from here
FIN_MARKUP = 0.005                   # broker financing mark-up, per annum, charged on long and short positions
ROLL_TZ = "America/New_York"         # FX value-date roll at 17:00 New York
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


def load_universe(pairs=PAIRS, data_dir: Path = DATA_DIR, end: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """M15 frames for every pair, trimmed to their common span; bars at or after `end` (UTC) are never loaded."""
    m15 = {p: load_mt5_csv(data_dir / f"{p}.csv", p) for p in pairs}
    lo = max(f.index[0] for f in m15.values())
    hi = min(f.index[-1] for f in m15.values())
    out = {p: f.loc[lo:hi] for p, f in m15.items()}
    if end is not None:
        out = {p: f[f.index < end] for p, f in out.items()}
    return out


# ============================================================================= OANDA history + splice
_OANDA_NAME = {"EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDCAD": "USD_CAD", "EURJPY": "EUR_JPY"}


def ensure_oanda_source(source: Path = OANDA_SOURCE) -> Path:
    """Sparse, blob-less checkout of the four OANDA pairs, pinned to OANDA_COMMIT."""
    base = source / OANDA_SUBDIR
    if all((base / n / "2019").is_dir() for n in _OANDA_NAME.values()):
        return base

    def git(*args, cwd=None):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")

    if not (source / ".git").exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", OANDA_REPO, str(source))
    git("sparse-checkout", "set", "--no-cone", *[f"{OANDA_SUBDIR}/{n}/*" for n in _OANDA_NAME.values()], cwd=source)
    git("checkout", OANDA_COMMIT, cwd=source)
    return base


def load_oanda_minutes(name: str, end: pd.Timestamp, start: pd.Timestamp = OANDA_START,
                       source: Path = OANDA_SOURCE) -> pd.DataFrame:
    """OANDA 1-minute MID candles (UNIX-time stamps, i.e. UTC) in [start, end)."""
    base = ensure_oanda_source(source) / name
    files = sorted(f for y in range(start.year, end.year + 1) for f in (base / str(y)).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no OANDA minute files under {base}")
    m = pd.concat((pd.read_csv(f, usecols=["time", "open", "high", "low", "close"]) for f in files),
                  ignore_index=True)
    m["time"] = pd.to_datetime(m["time"], utc=True)
    m = m.drop_duplicates("time", keep="last").set_index("time").sort_index().astype(float)
    m.index = m.index.as_unit("ns")
    m = m[(m.index >= start) & (m.index < end)]
    return m[~market_closed(m.index)]


def market_closed(index: pd.DatetimeIndex) -> np.ndarray:
    """Friday 17:00 -> Sunday 17:00 New York: the broker's weekly closure (MT5 has no bars there). OANDA's
    indicative weekend and Sunday pre-open quotes fall in it and are dropped (data layer only)."""
    ny = index.tz_convert(ROLL_TZ)
    wd, hr = ny.dayofweek, ny.hour
    return np.asarray(((wd == 4) & (hr >= 17)) | (wd == 5) | ((wd == 6) & (hr < 17)))


def derive_cross(num: pd.DataFrame, den: pd.DataFrame, max_age: str = "5min") -> pd.DataFrame:
    """num/den per minute of `num` (e.g. USD/JPY = EUR/JPY / EUR/USD): num's OHLC divided by den's latest close
    (at most `max_age` old). The range is num's - den's move inside the minute is ignored."""
    den_c = den["close"].reindex(num.index.union(den.index)).ffill(limit=None)
    age = pd.Series(den.index, index=den.index).reindex(den_c.index).ffill()
    den_c = den_c.where((den_c.index - age) <= pd.Timedelta(max_age)).reindex(num.index)
    out = num[["open", "high", "low", "close"]].div(den_c, axis=0)
    return out.dropna()


def mt5_spread_profile(m15: pd.DataFrame, end: pd.Timestamp = PROFILE_END) -> pd.Series:
    """Median MT5 spread (pips) by UTC hour of day over bars before `end` (a cost model, not a trading rule)."""
    f = m15[m15.index < end]
    prof = f["spread_pips"].groupby(f.index.hour).median()
    return prof.reindex(range(24)).ffill().bfill()


def minutes_to_m15(minutes: pd.DataFrame, pair: str, profile: pd.Series) -> pd.DataFrame:
    """MID minutes -> the cleaned M15 format of ``clean_m15`` (bid OHLC at point precision + spread_pips)."""
    g = minutes.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    spread = profile.to_numpy()[g.index.hour]
    half = spread * pip_size(pair) / 2.0
    digits = 3 if pair.endswith("JPY") else 5
    out = g.sub(half, axis=0).round(digits)
    out["spread_pips"] = spread
    out.index = out.index.as_unit("ns")
    return out


SPLICE_VERSION = "v2-weekly-closure-dropped"


def _profile_key(profiles: dict[str, pd.Series]) -> str:
    h = hashlib.sha256((OANDA_COMMIT + SPLICE_VERSION).encode())
    for p in sorted(profiles):
        h.update(p.encode() + profiles[p].to_numpy().tobytes())
    return h.hexdigest()[:10]


def load_spliced_universe(pairs=PAIRS, end: pd.Timestamp | None = None, mt5_dir: Path = DATA_DIR,
                          cache_dir: Path = SPLICED_DIR) -> dict[str, pd.DataFrame]:
    """OANDA-era M15 (derived from minutes) before the first common MT5 bar, the MT5 bars from it on.

    Bars at or after `end` are never returned. The OANDA part is cached (parquet) per pair and profile.
    """
    mt5 = load_universe(pairs, mt5_dir)
    splice = max(f.index[0] for f in mt5.values())
    profiles = {p: mt5_spread_profile(mt5[p]) for p in pairs}
    key = _profile_key(profiles)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, minutes = {}, {}
    for p in pairs:
        path = cache_dir / f"{p}_oanda_{key}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            old.index = pd.DatetimeIndex(old.index).tz_convert("UTC").as_unit("ns")
        else:
            if p == "USDJPY":
                for q in ("EURJPY", "EURUSD"):
                    if q not in minutes:
                        minutes[q] = load_oanda_minutes(_OANDA_NAME[q], splice)
                mins = derive_cross(minutes["EURJPY"], minutes["EURUSD"])
            else:
                mins = minutes.get(p)
                if mins is None:
                    mins = minutes[p] = load_oanda_minutes(_OANDA_NAME[p], splice)
            old = minutes_to_m15(mins[mins.index < splice], p, profiles[p])
            old.to_parquet(path)
        out[p] = pd.concat([old[old.index < splice], mt5[p]])
    lo = max(f.index[0] for f in out.values())
    out = {p: f[f.index >= lo] for p, f in out.items()}
    if end is not None:
        out = {p: f[f.index < end] for p, f in out.items()}
    return out


def splice_point(pairs=PAIRS, mt5_dir: Path = DATA_DIR) -> pd.Timestamp:
    return max(load_mt5_csv(mt5_dir / f"{p}.csv", p).index[0] for p in pairs)


# ============================================================================= interest rates
_OECD_CCY = {"USA": "USD", "EA19": "EUR", "GBR": "GBP", "JPN": "JPY", "CAN": "CAD"}


class RateTable:
    """Short-term interest rates (percent p.a.) per currency as step functions of UTC time.

    monthly[ccy]: a value per month (index = month start). It is the rate *in force* during that month
                  (accrual) but, being a monthly average, only *known* from the start of the next month.
    events[ccy] : policy-rate decisions (index = effective date, 00:00 UTC). In force from that date; *known* from
                  the next day's 00:00 UTC (some banks announce on the effective day itself, during the day).
    Monthly values apply before `switch`, events from `switch` on.
    """

    def __init__(self, monthly: dict[str, pd.Series], events: dict[str, pd.Series] | None = None,
                 switch: pd.Timestamp | None = None):
        self.monthly = {c: s.sort_index() for c, s in monthly.items()}
        self.events = {c: s.sort_index() for c, s in (events or {}).items()}
        self.switch = switch
        self._steps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, rates_dir: Path = RATES_DIR, end: pd.Timestamp | None = None) -> "RateTable":
        """The committed rate files. Rows dated at or after `end` are never loaded (holdout lock)."""
        o = pd.read_csv(rates_dir / "oecd_short_term.csv")
        monthly = {}
        for loc, g in o.groupby("LOCATION"):
            idx = pd.DatetimeIndex(pd.to_datetime(g["TIME"] + "-01")).tz_localize("UTC")
            s = pd.Series(g["Value"].to_numpy(float), index=idx)
            monthly[_OECD_CCY[loc]] = s[s.index < end] if end is not None else s
        pol = pd.read_csv(rates_dir / "policy_rates.csv")
        events = {}
        for ccy, g in pol.groupby("currency"):
            idx = pd.DatetimeIndex(pd.to_datetime(g["effective_date"])).tz_localize("UTC")
            s = pd.Series(g["rate_pct"].to_numpy(float), index=idx)
            events[ccy] = s[s.index < end] if end is not None else s
        return cls(monthly, events, POLICY_SWITCH)

    @classmethod
    def synthetic(cls, ccys=("USD", "EUR", "GBP", "JPY", "CAD"), start: str = "2020-01-01", months: int = 36,
                  seed: int = 0, switch: str | None = None) -> "RateTable":
        """Random-walk monthly rates (and, after `switch`, random policy events) for unit tests."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range(start, periods=months, freq="MS", tz="UTC")
        monthly, events = {}, {}
        for c in ccys:
            monthly[c] = pd.Series(rng.normal(1.5, 1.5) + np.cumsum(rng.normal(0, 0.35, months)), index=idx)
            ev_idx = pd.DatetimeIndex(sorted(set(idx[0] + pd.to_timedelta(rng.integers(0, months * 30, 12), "D"))))
            events[c] = pd.Series(rng.normal(1.5, 1.5, len(ev_idx)), index=ev_idx)
        return cls(monthly, events, pd.Timestamp(switch, tz="UTC") if switch else None)

    def cut(self, end: pd.Timestamp) -> "RateTable":
        """Drop every row dated at or after `end`."""
        return RateTable({c: s[s.index < end] for c, s in self.monthly.items()},
                         {c: s[s.index < end] for c, s in self.events.items()}, self.switch)

    def perturbed(self, info_cut: pd.Timestamp, rng) -> "RateTable":
        """Scramble every value that becomes KNOWN at or after `info_cut` (leak tests)."""
        monthly = {}
        for c, s in self.monthly.items():
            known_at = s.index + pd.offsets.MonthBegin(1)
            monthly[c] = s.where(known_at < info_cut, s + rng.normal(0, 2.0, len(s)))
        events = {c: s.where(s.index < info_cut, s + rng.normal(0, 2.0, len(s))) for c, s in self.events.items()}
        return RateTable(monthly, events, self.switch)

    def unlagged(self) -> "RateTable":
        """A deliberately leaky copy: monthly averages treated as known during their own month (canary)."""
        t = RateTable(self.monthly, self.events, self.switch)
        t._known_lag = False
        return t

    def digest(self) -> str:
        h = hashlib.sha256()
        for d in (self.monthly, self.events):
            for c in sorted(d):
                h.update(c.encode() + d[c].index.asi8.tobytes() + d[c].to_numpy().tobytes())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ lookup
    _known_lag = True

    def _step(self, ccy: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
        key = (ccy, kind)
        if key not in self._steps:
            m = self.monthly.get(ccy, pd.Series(dtype=float))
            e = self.events.get(ccy, pd.Series(dtype=float))
            m_t = m.index + pd.offsets.MonthBegin(1) if (kind == "known" and self._known_lag) else m.index
            m = pd.Series(m.to_numpy(), index=pd.DatetimeIndex(m_t))
            if kind == "known" and self._known_lag and len(e):
                e = pd.Series(e.to_numpy(), index=e.index + pd.Timedelta(days=1))
            if self.switch is not None:
                m = m[m.index < self.switch]
                e = e[e.index >= self.switch] if len(e) else e
                prior = self.events.get(ccy, pd.Series(dtype=float))
                prior = prior[prior.index < self.switch]
                if len(prior):   # the decision in force when the policy era starts
                    e = pd.concat([pd.Series([prior.iloc[-1]], index=[self.switch]), e])
            else:
                e = e.iloc[:0]
            s = pd.concat([x for x in (m, e) if len(x)]) if len(m) or len(e) else pd.Series(dtype=float)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            self._steps[key] = (s.index.as_unit("ns").asi8, s.to_numpy(dtype=float))
        return self._steps[key]

    def _lookup(self, ccy: str, kind: str, ts) -> np.ndarray:
        t, v = self._step(ccy, kind)
        ts = np.asarray(ts, dtype=np.int64)
        k = np.searchsorted(t, ts, side="right") - 1
        out = np.where(k >= 0, v[np.clip(k, 0, None)] if len(v) else np.nan, np.nan)
        return out.astype(float)

    def accrual(self, ccy: str, ts) -> np.ndarray:
        """Rate in force at `ts` (percent p.a.)."""
        return self._lookup(ccy, "accrual", ts)

    def known(self, ccy: str, ts) -> np.ndarray:
        """Latest rate publicly known at `ts` (percent p.a.)."""
        return self._lookup(ccy, "known", ts)

    def known_diff(self, pair: str, ts) -> np.ndarray:
        """r_base - r_quote as known at `ts` (percent p.a.): the carry of a long position."""
        return self.known(pair[:3], ts) - self.known(pair[3:], ts)


# ============================================================================= financing (rollover)
_ROLLS_NS: np.ndarray | None = None


def roll_times_ns() -> np.ndarray:
    """Every 17:00 New York (DST aware) from 2000 to 2030, as UTC nanoseconds: one per calendar day."""
    global _ROLLS_NS
    if _ROLLS_NS is None:
        local = pd.date_range("2000-01-01 17:00", "2030-12-31 17:00", freq="D")
        _ROLLS_NS = local.tz_localize(ROLL_TZ).tz_convert("UTC").as_unit("ns").asi8
    return _ROLLS_NS


def financing_quote(signed_units: float, mid: float, diff_pct: float, markup: float = FIN_MARKUP) -> float:
    """One day's financing of a position, in the quote currency (positive = credit)."""
    return mid * (signed_units * diff_pct / 100.0 - abs(signed_units) * markup) / 365.0


class FinancingModule(SimulationModule):
    """Books overnight financing into the USD account at each 17:00 New York roll.

    Runs in ``pre_process`` - i.e. when the first data point at or after a roll reaches the venue, before any
    order is filled against it - so it charges the positions actually held through the roll, marked at the
    latest quotes before it. A weekend (no data) is charged three rolls when trading resumes.
    """

    def __init__(self, rates: RateTable, markup: float = FIN_MARKUP):
        super().__init__(SimulationModuleConfig())
        self.rates, self.markup = rates, markup
        self._rolls = roll_times_ns()
        self.reset()

    def reset(self):
        self._k = -1                       # index of the next roll not yet booked (-1: not initialised)
        self._next = -1
        self.bookings: list[tuple[int, float]] = []
        self.total_usd = 0.0

    def process(self, ts_now):
        pass

    def log_diagnostics(self, logger):
        pass

    def pre_process(self, data):
        ts = data.ts_init
        if ts < self._next:
            return
        if self._k < 0:                    # first data point: start booking from the next roll
            self._k = int(np.searchsorted(self._rolls, ts, side="left"))
            if self._rolls[self._k] == ts:
                self._k += 1
            self._next = int(self._rolls[self._k])
            return
        k_end = int(np.searchsorted(self._rolls, ts, side="right"))
        total = 0.0
        for pos in self.cache.positions_open(venue=SIM):
            qty = pos.signed_qty
            if qty == 0:
                continue
            iid = pos.instrument_id
            q = self.cache.quote_tick(iid)
            mid = (q.bid_price.as_double() + q.ask_price.as_double()) / 2.0
            inst = self.cache.instrument(iid)
            base, quote = inst.base_currency.code, inst.quote_currency.code
            amt = 0.0
            for r in self._rolls[self._k:k_end]:
                diff = float(self.rates.accrual(base, [r])[0] - self.rates.accrual(quote, [r])[0])
                if diff == diff:
                    amt += financing_quote(qty, mid, diff, self.markup)
            if quote != "USD":
                amt *= self.cache.get_xrate(SIM, inst.quote_currency, USD, PriceType.MID) or 0.0
            total += amt
        self._k, self._next = k_end, int(self._rolls[k_end])
        total = round(total, 2)
        if total != 0.0:
            self.bookings.append((ts, total))
            self.total_usd += total
            self.exchange.adjust_account(Money(total, USD))


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
    """Hourly frames for a set of pairs, loaded once into a reusable Nautilus BacktestEngine.

    With `rates`, the venue books overnight financing (FinancingModule) and every hourly frame gains a
    `carry` column: r_base - r_quote (percent p.a.) as publicly known at the bar's close.
    """

    def __init__(self, m15: dict[str, pd.DataFrame], rates: RateTable | None = None):
        self.pairs = tuple(m15)
        self.rates = rates
        self.h1 = {p: hourly_frame(f, p) for p, f in m15.items()}
        if rates is not None:
            for p, f in self.h1.items():
                f["carry"] = rates.known_diff(p, f.index.as_unit("ns").asi8 + ONE_HOUR)
        idx = self.h1[self.pairs[0]].index
        for f in self.h1.values():
            idx = idx.union(f.index)
        self.index = idx                                   # union of hourly bar-open times
        self._engine: BacktestEngine | None = None
        self.financing: FinancingModule | None = None

    def engine(self) -> BacktestEngine:
        if self._engine is None:
            eng = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
            self.financing = FinancingModule(self.rates) if self.rates is not None else None
            eng.add_venue(SIM, OmsType.NETTING, AccountType.MARGIN, [Money(STARTING_NAV, BASE_CCY)],
                          base_currency=BASE_CCY, default_leverage=Decimal(50), bar_execution=False,
                          latency_model=LatencyModel(base_latency_nanos=1),
                          modules=[self.financing] if self.financing is not None else None)
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
        if hasattr(strategy, "bind_market"):
            strategy.bind_market(self)
        eng.add_strategy(strategy)
        eng.run(start=pd.Timestamp(strategy.warmup_start_ns, tz="UTC"), end=pd.Timestamp(end_ns, tz="UTC"))
        res = strategy.result()
        eng.clear_strategies()
        return res
