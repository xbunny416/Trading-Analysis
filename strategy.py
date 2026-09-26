"""
strategy.py - long/short hourly trend following for a major FX pair (default EURUSD).

Rules (identical in both implementations below)
-----------------------------------------------
Indicators (2): Donchian channel, ATR.  Tunable parameters (4): entry_n, atr_n, atr_mult, risk_pct.

    DonHi_t = max(High_{t-N..t-1})     DonLo_t = min(Low_{t-N..t-1})       N = entry_n
    ATR_t   = mean(TR_{t-n+1..t}),     TR_t = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|),  n = atr_n

    at the close of bar t-1 (decision), executed at the OPEN of bar t:
        long  entry  : C_{t-1} > DonHi_{t-1}
        short entry  : C_{t-1} < DonLo_{t-1}
        exit / flip  : chandelier stop  stop_long = max_{since entry}(C - k*ATR),  exit if C_{t-1} < stop
                       (mirror for shorts), or reverse on the opposite breakout.       k = atr_mult
        size (units) : NAV_{t-1} * risk_pct / (k * ATR_{t-1}),  capped at max_leverage * NAV / C_{t-1}

Friction: 0.5 pip half-spread + 0.5 pip slippage on every fill (2.0 pips per round trip).
Circuit breakers (checked on each bar close, applied from the next open):
    trading-day loss >= 2.5 %  -> flat and no new positions for 24 hours
    drawdown from peak >= 10 % -> flat, permanently halted

Look-ahead guard: every value used for a decision that takes effect in bar t comes from bars <= t-1.
``signal_frame`` makes this explicit with ``.shift(1)``; the harness proves it by perturbation.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

# ============================================================================= configuration
HYPOTHESIS = (
    "T1 Donchian breakout: enter at next open when the hourly close breaks the prior entry_n-bar "
    "high/low; exit on an atr_mult x ATR chandelier stop or reverse on the opposite breakout; "
    "size = NAV*risk_pct/(atr_mult*ATR)."
)
INDICATORS = ("Donchian channel", "ATR")


@dataclass(frozen=True)
class StrategyParams:
    """The only tunable numbers in the strategy (4)."""

    entry_n: int = 72        # Donchian breakout lookback, bars
    atr_n: int = 24          # ATR lookback, bars
    atr_mult: float = 3.0    # chandelier-stop distance and sizing unit, in ATRs
    risk_pct: float = 0.0025  # fraction of NAV lost if price moves atr_mult*ATR against the position

    def validate(self) -> "StrategyParams":
        if not (isinstance(self.entry_n, (int, np.integer)) and self.entry_n >= 2):
            raise ValueError(f"entry_n must be an int >= 2, got {self.entry_n}")
        if not (isinstance(self.atr_n, (int, np.integer)) and self.atr_n >= 1):
            raise ValueError(f"atr_n must be an int >= 1, got {self.atr_n}")
        if not (self.atr_mult > 0 and 0 < self.risk_pct < 0.05):
            raise ValueError("atr_mult must be > 0 and risk_pct in (0, 0.05)")
        return self


DEFAULT_PARAMS = StrategyParams()
# walk-forward search space (the other parameters stay at their defaults)
PARAM_GRID = {"entry_n": [24, 48, 72, 96, 120], "atr_mult": [2.0, 3.0, 4.0, 5.0]}
# parameters that change the signal frame (lets the harness cache it)
FEATURE_PARAMS = ("entry_n", "atr_n")


@dataclass(frozen=True)
class ExecutionConfig:
    """Fixed friction and risk limits. Mandated by the mandate, not tuned."""

    pip: float = 0.0001
    spread_pips: float = 1.0        # round trip, half paid on each fill
    slippage_pips: float = 0.5      # per fill
    daily_loss_limit: float = 0.025
    halt_hours: float = 24.0
    max_drawdown_limit: float = 0.10
    max_leverage: float = 10.0
    initial_nav: float = 1_000_000.0

    @property
    def cost_per_fill(self) -> float:
        return (0.5 * self.spread_pips + self.slippage_pips) * self.pip


def warmup_bars(p: StrategyParams) -> int:
    return max(p.entry_n, p.atr_n) + 2


# ============================================================================= data helpers
def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """UTC/ns index, sorted, de-duplicated; drops bars with missing, non-positive or inconsistent prices."""
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    missing = {"open", "high", "low", "close"} - set(d.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    idx = pd.DatetimeIndex(d.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    d.index = idx.as_unit("ns")
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in d.columns else [])
    d = d[cols].astype(float)
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    with np.errstate(invalid="ignore"):
        good = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
                & (l > 0) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)))
    d = d[good]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def trading_day_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer id of the FX trading day, which rolls at 17:00 New York time (DST aware)."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    local = idx.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    return local.as_unit("ns").asi8 // 86_400_000_000_000


# ============================================================================= research path (vectorised)
def compute_features(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Indicator values known at the CLOSE of each bar (row t uses bars <= t). Not yet tradeable."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = np.fmax(h - l, np.fmax((h - pc).abs(), (l - pc).abs()))
    return pd.DataFrame(
        {
            "close": c,
            "atr": tr.rolling(p.atr_n).mean(),
            "don_hi": h.rolling(p.entry_n).max().shift(1),  # channel of the N bars *before* bar t
            "don_lo": l.rolling(p.entry_n).min().shift(1),
        },
        index=df.index,
    )


def signal_frame(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Row t = everything known at the OPEN of bar t, i.e. computed from bars <= t-1."""
    s = compute_features(df, p).shift(1)  # <- look-ahead guard
    s["long_entry"] = (s["close"] > s["don_hi"]).astype(float)
    s["short_entry"] = (s["close"] < s["don_lo"]).astype(float)
    return s


@dataclass
class BacktestResult:
    nav: pd.Series            # NAV marked at every bar close in the window
    units: pd.Series          # signed base-currency units held during each bar
    trades: pd.DataFrame
    final_state: dict         # risk state to carry into the next window
    n_halts: int = 0          # daily-loss halts triggered in this run
    killed: bool = False      # max-drawdown stop active at the end of this run


TRADE_COLUMNS = ["entry_time", "exit_time", "direction", "units", "entry_price", "exit_price",
                 "pnl", "pnl_pips", "bars_held", "exit_reason"]


def fresh_state(cfg: ExecutionConfig) -> dict:
    return {"nav": cfg.initial_nav, "peak": cfg.initial_nav, "killed": False, "halt_until": -1,
            "day": None, "day_start_nav": cfg.initial_nav, "n_halts": 0}


def _trade(entry: tuple, exit_ts: int, exit_px: float, exit_i: int, reason: str, pip: float) -> dict:
    ts, d, u, px, i = entry
    return {"entry_time": ts, "exit_time": exit_ts, "direction": d, "units": u, "entry_price": px,
            "exit_price": exit_px, "pnl": u * (exit_px - px), "pnl_pips": d * (exit_px - px) / pip,
            "bars_held": exit_i - i, "exit_reason": reason}


def _trades_frame(rows: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c].astype("int64"), utc=True)
    return t


def backtest_vectorized(df: pd.DataFrame, p: StrategyParams, cfg: ExecutionConfig | None = None,
                        start: int = 0, end: int | None = None, state: dict | None = None,
                        sig: pd.DataFrame | None = None) -> BacktestResult:
    """Research backtest: vectorised signals + one pass over bars for NAV-dependent sizing and breakers.

    `df` must already be cleaned with `clean_ohlc`. Bars [start, end) are traded; earlier bars only
    feed indicators. `state` carries NAV / peak / halts from a previous window.
    """
    cfg = cfg or ExecutionConfig()
    p.validate()
    n = len(df)
    end = n if end is None else end
    if not 0 <= start < end <= n:
        raise ValueError(f"bad window [{start}, {end}) for {n} bars")
    if not np.isfinite(df[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("df contains non-finite prices; run clean_ohlc first")
    sig = signal_frame(df, p) if sig is None else sig

    w = slice(start, end)
    O = df["open"].to_numpy()[w].tolist()
    C = df["close"].to_numpy()[w].tolist()
    CP = sig["close"].to_numpy()[w].tolist()
    A = sig["atr"].to_numpy()[w].tolist()
    LS = (sig["long_entry"].to_numpy()[w] > 0).tolist()
    SS = (sig["short_entry"].to_numpy()[w] > 0).tolist()
    idx = df.index[w]
    TS = idx.as_unit("ns").asi8.tolist()
    DAY = trading_day_ids(idx).tolist()

    st = fresh_state(cfg) if state is None else dict(state)
    nav_prev, peak, killed = st["nav"], st["peak"], st["killed"]
    halt_until, cur_day, day_start = st["halt_until"], st["day"], st["day_start_nav"]
    cash, units, d, stop = nav_prev, 0.0, 0, math.nan
    cost, k, pip = cfg.cost_per_fill, p.atr_mult, cfg.pip
    dl_floor, dd_floor = 1.0 - cfg.daily_loss_limit, 1.0 - cfg.max_drawdown_limit
    halt_ns = int(cfg.halt_hours * 3_600_000_000_000)
    L = end - start
    nav_out, units_out, trades = [0.0] * L, [0.0] * L, []
    entry = None
    halts = 0

    for i in range(L):
        cp, a = CP[i], A[i]
        # (a) fold in the close of bar t-1: stop test, then ratchet the chandelier
        stop_hit = False
        if d == 1:
            if cp < stop:
                stop_hit = True
            else:
                s2 = cp - k * a
                if s2 > stop:
                    stop = s2
        elif d == -1:
            if cp > stop:
                stop_hit = True
            else:
                s2 = cp + k * a
                if s2 < stop:
                    stop = s2
        # (b) target direction for this bar
        if d == 0:
            tgt = 1 if LS[i] else (-1 if SS[i] else 0)
        elif d == 1:
            tgt = -1 if SS[i] else (0 if stop_hit else 1)
        else:
            tgt = 1 if LS[i] else (0 if stop_hit else -1)
        gated = killed or TS[i] <= halt_until
        if gated:
            tgt = 0
        # (c) execute at the open of bar t
        if tgt != d:
            if d != 0:
                reason = ("kill" if killed else "halt") if gated else ("reverse" if tgt == -d else "stop")
                fill = O[i] - d * cost
                cash += units * fill
                trades.append(_trade(entry, TS[i], fill, i, reason, pip))
                units, d, stop = 0.0, 0, math.nan
            if tgt != 0:
                size = RiskManager.position_size(nav_prev, a, cp, p, cfg)
                if size > 0:
                    fill = O[i] + tgt * cost
                    units = tgt * size
                    cash -= units * fill
                    d = tgt
                    stop = cp - tgt * k * a
                    entry = (TS[i], d, units, fill, i)
        # (d) mark at the close of bar t and apply the circuit breakers (effective next open)
        nav = cash + units * C[i]
        if DAY[i] != cur_day:
            cur_day, day_start = DAY[i], nav_prev
        if not killed:
            if TS[i] > halt_until and nav <= day_start * dl_floor:
                halt_until = TS[i] + halt_ns
                halts += 1
            if nav > peak:
                peak = nav
            if nav <= peak * dd_floor:
                killed = True
        nav_out[i], units_out[i] = nav, units
        nav_prev = nav

    if d != 0:  # administrative liquidation at the close of the window's last bar
        fill = C[L - 1] - d * cost
        cash += units * fill
        trades.append(_trade(entry, TS[L - 1], fill, L - 1, "window_end", pip))
        nav = cash
        nav_out[L - 1] = nav_prev = nav
        if not killed and nav <= peak * dd_floor:
            killed = True

    final = {"nav": nav_prev, "peak": peak, "killed": killed, "halt_until": halt_until, "day": cur_day,
             "day_start_nav": day_start, "n_halts": st["n_halts"] + halts}
    return BacktestResult(nav=pd.Series(nav_out, index=idx, name="nav"),
                          units=pd.Series(units_out, index=idx, name="units"),
                          trades=_trades_frame(trades), final_state=final, n_halts=halts, killed=killed)


# ============================================================================= production path (event driven)
@dataclass(frozen=True)
class Bar:
    i: int
    ts: int      # bar open time, ns since epoch (UTC)
    day: int     # trading-day id
    open: float
    high: float
    low: float
    close: float


class MarketData:
    """Owns the cleaned bar history and streams completed bars in time order."""

    def __init__(self, raw: pd.DataFrame):
        self.frame = clean_ohlc(raw)
        self._ts = self.frame.index.asi8
        self._day = trading_day_ids(self.frame.index)
        self._px = self.frame[["open", "high", "low", "close"]].to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    def bars(self, start: int = 0, end: int | None = None) -> Iterator[Bar]:
        end = len(self) if end is None else end
        for i in range(start, end):
            o, h, l, c = self._px[i]
            yield Bar(i, int(self._ts[i]), int(self._day[i]), float(o), float(h), float(l), float(c))


@dataclass(frozen=True)
class SignalSnapshot:
    close: float
    atr: float
    don_hi: float
    don_lo: float
    long_entry: bool
    short_entry: bool


class SignalEngine:
    """Incremental Donchian channel and ATR over completed bars."""

    def __init__(self, p: StrategyParams):
        self.p = p
        self._hi: deque[float] = deque(maxlen=p.entry_n)
        self._lo: deque[float] = deque(maxlen=p.entry_n)
        self._tr: deque[float] = deque(maxlen=p.atr_n)
        self._prev_close: float | None = None

    def update(self, bar: Bar) -> SignalSnapshot:
        full = len(self._hi) == self.p.entry_n
        don_hi = max(self._hi) if full else math.nan  # channel excludes the bar being closed
        don_lo = min(self._lo) if full else math.nan
        h, l, c, pc = bar.high, bar.low, bar.close, self._prev_close
        tr = h - l if pc is None else max(h - l, abs(h - pc), abs(l - pc))
        self._tr.append(tr)
        atr = math.fsum(self._tr) / self.p.atr_n if len(self._tr) == self.p.atr_n else math.nan
        self._hi.append(h)
        self._lo.append(l)
        self._prev_close = c
        return SignalSnapshot(c, atr, don_hi, don_lo, c > don_hi, c < don_lo)


class TrendStrategy:
    """Position state machine: breakout entries, chandelier stop, stop-and-reverse."""

    def __init__(self, p: StrategyParams):
        self.p = p
        self.direction = 0
        self.stop = math.nan
        self.stop_hit = False

    def on_bar_close(self, snap: SignalSnapshot) -> int:
        """Fold in a completed bar; return the desired direction for the next open."""
        d, k = self.direction, self.p.atr_mult
        self.stop_hit = False
        if d == 1:
            if snap.close < self.stop:
                self.stop_hit = True
            else:
                s2 = snap.close - k * snap.atr
                if s2 > self.stop:
                    self.stop = s2
        elif d == -1:
            if snap.close > self.stop:
                self.stop_hit = True
            else:
                s2 = snap.close + k * snap.atr
                if s2 < self.stop:
                    self.stop = s2
        if d == 0:
            return 1 if snap.long_entry else (-1 if snap.short_entry else 0)
        if d == 1:
            return -1 if snap.short_entry else (0 if self.stop_hit else 1)
        return 1 if snap.long_entry else (0 if self.stop_hit else -1)

    def on_entry(self, direction: int, snap: SignalSnapshot) -> None:
        self.direction = direction
        self.stop = snap.close - direction * self.p.atr_mult * snap.atr

    def on_exit(self) -> None:
        self.direction, self.stop = 0, math.nan


class RiskManager:
    """Volatility-targeted sizing, leverage cap, daily-loss halt and max-drawdown kill switch."""

    def __init__(self, p: StrategyParams, cfg: ExecutionConfig, state: dict | None = None):
        s = fresh_state(cfg) if state is None else dict(state)
        self.p, self.cfg = p, cfg
        self.last_nav, self.peak, self.killed = s["nav"], s["peak"], s["killed"]
        self.halt_until, self.day, self.day_start_nav = s["halt_until"], s["day"], s["day_start_nav"]
        self.n_halts_before, self.n_halts = s["n_halts"], 0
        self._halt_ns = int(cfg.halt_hours * 3_600_000_000_000)

    @staticmethod
    def position_size(nav: float, atr: float, price: float, p: StrategyParams, cfg: ExecutionConfig) -> float:
        """Units = NAV * risk% / (ATR * multiplier), capped by leverage; 0 for any degenerate input."""
        if not (nav > 0 and atr > 0 and price > 0):  # also rejects NaN
            return 0.0
        if not (math.isfinite(nav) and math.isfinite(atr) and math.isfinite(price)):
            return 0.0
        denom = p.atr_mult * atr
        if not denom > 0:
            return 0.0
        return min(nav * p.risk_pct / denom, cfg.max_leverage * nav / price)

    def allows_trading(self, ts: int) -> bool:
        return not self.killed and ts > self.halt_until

    def on_bar_close(self, bar: Bar, nav: float) -> None:
        if bar.day != self.day:
            self.day, self.day_start_nav = bar.day, self.last_nav
        if not self.killed:
            if bar.ts > self.halt_until and nav <= self.day_start_nav * (1.0 - self.cfg.daily_loss_limit):
                self.halt_until = bar.ts + self._halt_ns
                self.n_halts += 1
            if nav > self.peak:
                self.peak = nav
            if nav <= self.peak * (1.0 - self.cfg.max_drawdown_limit):
                self.killed = True
        self.last_nav = nav

    def on_liquidation(self, nav: float) -> None:
        if not self.killed and nav <= self.peak * (1.0 - self.cfg.max_drawdown_limit):
            self.killed = True
        self.last_nav = nav

    def state(self) -> dict:
        return {"nav": self.last_nav, "peak": self.peak, "killed": self.killed, "halt_until": self.halt_until,
                "day": self.day, "day_start_nav": self.day_start_nav,
                "n_halts": self.n_halts_before + self.n_halts}


class ExecutionHandler:
    """Fills market orders at the bar open with half-spread + slippage; tracks cash and units."""

    def __init__(self, cfg: ExecutionConfig, cash: float):
        self.cfg = cfg
        self.cash = cash
        self.units = 0.0

    def execute(self, delta_units: float, mid: float, ts: int) -> float:
        side = 1 if delta_units > 0 else -1
        fill = mid + side * self.cfg.cost_per_fill
        self.cash -= delta_units * fill
        self.units += delta_units
        return fill

    def mark(self, mid: float) -> float:
        return self.cash + self.units * mid


@dataclass
class EventDrivenBacktester:
    """Wires MarketData -> SignalEngine -> TrendStrategy -> RiskManager -> ExecutionHandler bar by bar."""

    raw: pd.DataFrame
    p: StrategyParams
    cfg: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self):
        self.p.validate()
        self.data = MarketData(self.raw)

    def run(self, start: int = 0, end: int | None = None, state: dict | None = None) -> BacktestResult:
        p, cfg, md = self.p, self.cfg, self.data
        end = len(md) if end is None else end
        if not 0 <= start < end <= len(md):
            raise ValueError(f"bad window [{start}, {end}) for {len(md)} bars")
        signals, strat = SignalEngine(p), TrendStrategy(p)
        risk = RiskManager(p, cfg, state)
        broker = ExecutionHandler(cfg, cash=risk.last_nav)

        snap = None
        for bar in md.bars(max(0, start - warmup_bars(p)), start):  # warm-up: indicators only
            snap = signals.update(bar)
        proposed = strat.on_bar_close(snap) if snap is not None else 0

        nav_out, units_out, trades = [], [], []
        entry, last = None, None
        for bar in md.bars(start, end):
            j = bar.i - start
            # ---- bar open: act on the decision taken at the previous close
            gated = not risk.allows_trading(bar.ts)
            target = 0 if gated else proposed
            d = strat.direction
            if target != d:
                if d != 0:
                    reason = ("kill" if risk.killed else "halt") if gated else ("reverse" if target == -d else "stop")
                    fill = broker.execute(-broker.units, bar.open, bar.ts)
                    trades.append(_trade(entry, bar.ts, fill, j, reason, cfg.pip))
                    strat.on_exit()
                if target != 0:
                    size = RiskManager.position_size(risk.last_nav, snap.atr, snap.close, p, cfg)
                    if size > 0:
                        fill = broker.execute(target * size, bar.open, bar.ts)
                        strat.on_entry(target, snap)
                        entry = (bar.ts, target, broker.units, fill, j)
            # ---- bar close: mark, risk checks, new signal
            nav = broker.mark(bar.close)
            risk.on_bar_close(bar, nav)
            snap = signals.update(bar)
            proposed = strat.on_bar_close(snap)
            nav_out.append(nav)
            units_out.append(broker.units)
            last = bar

        if strat.direction != 0:  # administrative liquidation at the window's last close
            fill = broker.execute(-broker.units, last.close, last.ts)
            trades.append(_trade(entry, last.ts, fill, last.i - start, "window_end", cfg.pip))
            strat.on_exit()
            nav = broker.mark(last.close)
            nav_out[-1] = nav
            risk.on_liquidation(nav)

        idx = md.index[start:end]
        return BacktestResult(nav=pd.Series(nav_out, index=idx, name="nav"),
                              units=pd.Series(units_out, index=idx, name="units"),
                              trades=_trades_frame(trades), final_state=risk.state(),
                              n_halts=risk.n_halts, killed=risk.killed)
