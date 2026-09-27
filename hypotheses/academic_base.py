"""
academic_base.py - shared engine for the cycle-4 hypotheses A1..A6 (published FX factor strategies).

Every hypothesis produces a signal s_i in [-1, 1] per pair from hourly MID bars (and, for carry, the interest-rate
differential known at the time). This module turns signals into trades the same way for all of them.

Sizing (the mandate's formula, volatility targeting)
    units_i = NAV * RISK_PCT * |s_i| / (ATR_i * ATR_MULT * quote->USD),  sign = sign(s_i)
    ATR_i: Wilder ATR of hourly bars, period ATR_BARS = 1440 (~60 trading days, Moskowitz-Ooi-Pedersen's 60-day
    centre of mass). Whole 1,000-unit lots, capped at max_leverage * NAV of USD notional; 0 if any input is degenerate.
    RISK_PCT = 0.3 %, ATR_MULT = 10 -> about 1.5 % annualised volatility per position at |s| = 1. Fixed, not tuned.

Execution (a no-trade band instead of a rebalancing calendar)
    At every hourly close, with P the position and T the target:
        P == 0, T != 0              -> ENTRY  (T)
        T == 0, P != 0              -> EXIT   (-P)
        sign(T) != sign(P)          -> FLIP   (T - P, one order)
        |T - P| >= band * max(|T|, |P|) -> REBAL (T - P)
    A close is decided as soon as every pair with a bar at that close has reported. Each pair's order is sent
    when that pair's next quote arrives (the open of its next bar) and fills against it - a pair that does not
    trade for a while (holiday) is filled when it reopens. Parameter changes at walk-forward segment
    boundaries do not liquidate: the new parameters' target is simply traded through the same rule.

Risk limits (portfolio level, checked at every hourly close, acted on at the next open):
    trading-day loss >= 2.5 %   -> flatten everything, no new positions for 24 h
    drawdown from peak >= 10 %  -> flatten everything, permanently halted

The classes follow the mandate's split: MarketData (incremental per-pair state), Signals (per hypothesis),
RiskManager (sizing + breakers), and the Nautilus Strategy as Execution. ``reference_backtest`` is an
independent vectorised re-implementation for one USD-quoted pair, used to prove the Nautilus plumbing.

Look-ahead guard: a decision at the close of bar t-1 only sees bars <= t-1 (the vectorised ``signal_frame`` of
each hypothesis shifts by one bar; the Nautilus strategy acts on bar-close events and its orders fill at the next
bar's open quote). Rates enter only through the `carry` column, which the data layer builds from values already
public at each bar's close. The harness proves all of it by perturbation.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

import backtest as B

BARS_PER_DAY = 24                  # hourly bars per trading day
BARS_PER_MONTH = 520               # ~6,240 hourly FX bars a year / 12
ATR_BARS = 1440                    # Wilder ATR period (~60 trading days)
RISK_PCT = 0.003                   # NAV fraction risked per ATR_MULT * ATR move at |s| = 1
ATR_MULT = 10.0
HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True)
class RiskLimits:
    """Fixed by the mandate, never tuned."""

    daily_loss_limit: float = 0.025
    halt_hours: float = 24.0
    max_drawdown_limit: float = 0.10
    max_leverage: float = 10.0    # per position, USD notional / NAV
    lot: int = 1_000              # order sizes are whole micro lots


def currencies(pair: str) -> tuple[str, str]:
    return pair[:3], pair[3:]


def trading_day_ids(index) -> np.ndarray:
    """FX trading day, rolling at 17:00 New York (DST aware), for bar-open timestamps."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    local = idx.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    return local.as_unit("ns").asi8 // 86_400_000_000_000


def trading_day(ts_ns: int) -> int:
    return int(trading_day_ids(pd.DatetimeIndex([pd.Timestamp(ts_ns, tz="UTC")]))[0])


# ============================================================================= numeric kernels
# The recursive kernels below are written once and used by both paths, so the event-driven engine and the
# vectorised reference produce bit-identical ATR / EWMA values.
def true_range(h: float, l: float, prev_close: float | None) -> float:
    if prev_close is None:
        return h - l
    return max(h - l, abs(h - prev_close), abs(l - prev_close))


class WilderATR:
    """ATR_t = ATR_{t-1} + (TR_t - ATR_{t-1}) / n, seeded with the mean of the first n true ranges."""

    def __init__(self, n: int = ATR_BARS):
        self.n, self.value, self._seed, self._prev = n, math.nan, [], None

    def update(self, h: float, l: float, c: float) -> float:
        tr = true_range(h, l, self._prev)
        self._prev = c
        if self._seed is not None:
            self._seed.append(tr)
            if len(self._seed) == self.n:
                self.value = math.fsum(self._seed) / self.n
                self._seed = None
        else:
            self.value = self.value + (tr - self.value) / self.n
        return self.value


def wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = ATR_BARS) -> np.ndarray:
    atr, out = WilderATR(n), np.empty(len(c))
    for i in range(len(c)):
        out[i] = atr.update(float(h[i]), float(l[i]), float(c[i]))
    return out


class EWMA:
    """y_0 = x_0, y_t = y_{t-1} + alpha * (x_t - y_{t-1})."""

    def __init__(self, alpha: float):
        self.alpha, self.value = alpha, math.nan

    def update(self, x: float) -> float:
        self.value = x if self.value != self.value else self.value + self.alpha * (x - self.value)
        return self.value


def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    e, out = EWMA(alpha), np.empty(len(x))
    for i in range(len(x)):
        out[i] = e.update(float(x[i]))
    return out


class RollingStd:
    """Sample standard deviation (ddof=1) of the last n values; running sums, re-summed exactly every n updates."""

    def __init__(self, n: int):
        self.n, self.buf = n, deque(maxlen=n)
        self._s = self._s2 = 0.0
        self._k = 0
        self._shift = None

    def update(self, x: float) -> float:
        if self._shift is None:
            self._shift = x            # centring improves the conditioning of the running sums
        y = x - self._shift
        if len(self.buf) == self.n:
            old = self.buf[0]
            self._s -= old
            self._s2 -= old * old
        self.buf.append(y)
        self._s += y
        self._s2 += y * y
        self._k += 1
        if self._k % self.n == 0:
            self._s = math.fsum(self.buf)
            self._s2 = math.fsum(v * v for v in self.buf)
        return self.value

    @property
    def value(self) -> float:
        m = len(self.buf)
        if m < self.n:
            return math.nan
        var = (self._s2 - self._s * self._s / m) / (m - 1)
        return math.sqrt(var) if var > 0 else 0.0


# ============================================================================= event-driven components
class MarketData:
    """Incremental per-pair state every hypothesis needs: close history, Wilder ATR, the known carry."""

    def __init__(self, pairs: tuple[str, ...], history: int):
        self.pairs = tuple(pairs)
        self.closes = {p: deque(maxlen=history) for p in self.pairs}
        self._atr = {p: WilderATR() for p in self.pairs}
        self.carry = {p: math.nan for p in self.pairs}

    def update(self, pair: str, o: float, h: float, l: float, c: float, carry: float) -> None:
        self._atr[pair].update(h, l, c)
        self.closes[pair].append(c)
        self.carry[pair] = carry

    def atr(self, pair: str) -> float:
        return self._atr[pair].value

    def close(self, pair: str) -> float:
        cl = self.closes[pair]
        return cl[-1] if cl else math.nan

    def past_close(self, pair: str, lag: int) -> float:
        """Close `lag` bars before the latest one (NaN until that much history exists)."""
        cl = self.closes[pair]
        return cl[-1 - lag] if len(cl) > lag else math.nan


class Signals:
    """Per-hypothesis signal logic on the event-driven path (subclassed by each hypothesis)."""

    def __init__(self, p, pairs: tuple[str, ...], md: MarketData):
        self.p, self.pairs, self.md = p, tuple(pairs), md

    def update(self, pair: str) -> None:
        """Fold the pair's latest bar (already in `md`) into any extra state."""

    def signals(self) -> dict[str, float]:
        """Signal per pair in [-1, 1]; NaN while not ready. Called at each hourly close."""
        raise NotImplementedError


class RiskManager:
    """Volatility-targeted sizing, leverage cap, daily-loss halt and max-drawdown kill switch."""

    def __init__(self, limits: RiskLimits, nav0: float):
        self.limits = limits
        self.last_equity = self.peak = self.day_start = nav0
        self.day: int | None = None
        self.halt_until = -1
        self.killed = False
        self.n_halts = 0
        self._halt_ns = int(limits.halt_hours * HOUR_NS)

    @staticmethod
    def position_size(nav: float, atr: float, price: float, quote_to_usd: float | None, p=None,
                      limits: RiskLimits = RiskLimits(), weight: float = 1.0) -> int:
        """Units = NAV * RISK_PCT * |weight| / (ATR * ATR_MULT * quote->USD), whole lots, leverage-capped;
        0 for any degenerate input."""
        vals = (nav, atr, price, quote_to_usd)
        if any(v is None or not math.isfinite(v) or v <= 0 for v in vals):
            return 0
        w = abs(weight)
        if not (math.isfinite(w) and w > 0):
            return 0
        denom = atr * ATR_MULT * quote_to_usd
        if not denom > 0:
            return 0
        units = min(nav * RISK_PCT * min(w, 1.0) / denom, limits.max_leverage * nav / (price * quote_to_usd))
        return int(units // limits.lot) * limits.lot

    def on_close(self, ts_close: int, equity: float) -> None:
        """Called once per hourly close, before any decision taken at that close."""
        day = trading_day(ts_close - HOUR_NS)  # day of the bar that just closed
        if day != self.day:
            self.day, self.day_start = day, self.last_equity
        if not self.killed:
            if ts_close > self.halt_until and equity <= self.day_start * (1 - self.limits.daily_loss_limit):
                self.halt_until = ts_close + self._halt_ns
                self.n_halts += 1
            self.peak = max(self.peak, equity)
            if equity <= self.peak * (1 - self.limits.max_drawdown_limit):
                self.killed = True
        self.last_equity = equity

    def allows_trading(self, ts: int) -> bool:
        return not self.killed and ts > self.halt_until


def target_units(signal: float, nav: float, atr: float, price: float, quote_to_usd: float | None,
                 limits: RiskLimits) -> int:
    """Signed target position for a signal in [-1, 1]; 0 when the signal is missing or zero."""
    if not (signal == signal) or signal == 0:
        return 0
    size = RiskManager.position_size(nav, atr, price, quote_to_usd, None, limits, weight=signal)
    return size if signal > 0 else -size


def rebalance_order(current: int, target: int, band: float) -> tuple[int, str] | None:
    """The execution rule: (signed order units, role) or None when no trade is due."""
    if target == current:
        return None
    if current == 0:
        return target, "ENTRY"
    if target == 0:
        return -current, "EXIT"
    if (target > 0) != (current > 0):
        return target - current, "FLIP"
    if abs(target - current) >= band * max(abs(target), abs(current)):
        return target - current, "REBAL"
    return None


@dataclass(frozen=True)
class Segment:
    start_ns: int            # first decision time (a bar close) traded with `params`
    end_ns: int              # decisions strictly before this time; equity recorded up to it
    params: object


def warmup_start(plan: list[Segment], warmup_bars) -> int:
    """Start of the data needed before the first decision: 1.5 calendar hours per bar covers weekends."""
    bars = max(warmup_bars(s.params) for s in plan)
    return plan[0].start_ns - int((bars * 1.5 + 72) * HOUR_NS)


# ============================================================================= NautilusTrader strategy
class TargetPortfolioStrategy(Strategy):
    """Wires Nautilus market data -> MarketData / Signals -> RiskManager -> banded rebalancing orders.

    Subclasses set SIGNALS (a Signals subclass), WARMUP (bars of history a parameter set needs) and HISTORY
    (closes to keep). One instance trades every pair in `pairs` on the SIM venue; `plan` is a list of contiguous
    segments, each with its own parameters.
    """

    SIGNALS: type[Signals] = Signals
    WARMUP = None
    HISTORY = None

    def __init__(self, plan: list[Segment], pairs: tuple[str, ...], limits: RiskLimits = RiskLimits()):
        super().__init__(StrategyConfig(strategy_id="AcademicFX-001"))
        for s in plan:
            s.params.validate()
        self.plan, self.pairs, self.limits = plan, tuple(pairs), limits
        self.plan_start, self.plan_end = plan[0].start_ns, plan[-1].end_ns
        self.warmup_start_ns = warmup_start(plan, type(self).WARMUP)
        self._pair_of = {B.bar_type(p): p for p in self.pairs}
        uniq = list(dict.fromkeys(s.params for s in plan))
        self._md = MarketData(self.pairs, max(type(self).HISTORY(q) for q in uniq))
        self._signals = {q: self.SIGNALS(q, self.pairs, self._md) for q in uniq}
        self._carry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._expected: dict[int, int] = {}          # bar-close time -> number of pairs with a bar closing then
        self._iid_pair = {B.instrument_id(p): p for p in self.pairs}
        self._pending: dict[str, list] = {}           # pair -> [signed units, role, reason] awaiting its next quote
        self._seen: set[str] = set()
        self._units = {pair: 0 for pair in self.pairs}
        self._risk = RiskManager(limits, B.STARTING_NAV)
        self._ts = -1
        self._seg = -1
        self._equity: list[tuple[int, float]] = []
        self._orders: list[dict] = []
        self._fills: list[dict] = []
        self._role: dict = {}
        self.desyncs = 0

    def bind_market(self, market) -> None:
        """Called by backtest.Market.run: the `carry` column (known rate differential at each bar close) and how
        many pairs have a bar closing at each time (so a close is decided as soon as it is complete)."""
        closes = []
        for p in self.pairs:
            f = market.h1[p]
            t = f.index.as_unit("ns").asi8 + HOUR_NS
            closes.append(t)
            if "carry" in f.columns:
                self._carry[p] = (t, f["carry"].to_numpy(dtype=float))
        u, c = np.unique(np.concatenate(closes), return_counts=True)
        self._expected = dict(zip(u.tolist(), c.tolist()))

    def _carry_at(self, pair: str, ts: int) -> float:
        if pair not in self._carry:
            return math.nan
        t, v = self._carry[pair]
        k = int(np.searchsorted(t, ts))
        return float(v[k]) if k < len(t) and t[k] == ts else math.nan

    # ---------------------------------------------------------------- lifecycle
    def on_start(self) -> None:
        for p in self.pairs:
            self.subscribe_bars(B.bar_type(p))
            self.subscribe_quote_ticks(B.instrument_id(p))

    def on_bar(self, bar) -> None:
        ts = bar.ts_event
        pair = self._pair_of[bar.bar_type]
        if ts != self._ts:
            if self._seen:  # only without bind_market: a close is complete once a later bar arrives
                self._decide_all(self._ts)
            self._on_close_time(ts)
        self._md.update(pair, bar.open.as_double(), bar.high.as_double(), bar.low.as_double(),
                        bar.close.as_double(), self._carry_at(pair, ts))
        for sig in self._signals.values():
            sig.update(pair)
        self._seen.add(pair)
        if len(self._seen) == self._expected.get(ts, len(self.pairs)):
            self._decide_all(ts)

    def on_quote_tick(self, tick) -> None:
        """A pair's first quote after a decision is the open of its next bar: send its order now (zero latency,
        so it fills against exactly this quote)."""
        pair = self._iid_pair.get(tick.instrument_id)
        order = self._pending.pop(pair, None)
        if order is None or order[0] == 0:
            return
        du, role, reason = order
        side = OrderSide.BUY if du > 0 else OrderSide.SELL
        o = self.order_factory.market(B.instrument_id(pair), side, Quantity.from_int(abs(du)))
        self._role[o.client_order_id] = (pair, role, reason)
        self.submit_order(o)

    def _decide_all(self, ts: int) -> None:
        seen, self._seen = self._seen, set()
        if self._seg < 0:
            return
        params = self.plan[self._seg].params
        sig = self._signals[params].signals()
        allowed = self._risk.allows_trading(ts)
        for pair in self.pairs:
            if pair not in seen:
                continue
            target = 0
            if allowed:
                target = target_units(sig.get(pair, math.nan), self._risk.last_equity, self._md.atr(pair),
                                      self._md.close(pair), self._quote_to_usd(pair), self.limits)
            order = rebalance_order(self._units[pair], target, params.band)
            if order is not None:
                du, role = order
                self._submit(pair, du, ts, role, "band" if role == "REBAL" else "signal")
                self._units[pair] = target

    # ---------------------------------------------------------------- per-timestamp work
    def _on_close_time(self, ts: int) -> None:
        self._ts = ts
        self._sync_positions()
        equity = self._equity_usd()
        if self.plan_start < ts <= self.plan_end:
            self._equity.append((ts, equity))
        self._risk.on_close(ts, equity)
        self._seg = next((k for k, s in enumerate(self.plan) if s.start_ns <= ts < s.end_ns), -1)
        if self._seg >= 0 and not self._risk.allows_trading(ts):
            reason = "kill" if self._risk.killed else "halt"
            for pair in self.pairs:
                if self._units[pair] != 0:
                    self._submit(pair, -self._units[pair], ts, "EXIT", reason)
                    self._units[pair] = 0

    def _submit(self, pair: str, signed_units: int, ts: int, role: str, reason: str) -> None:
        """Log the decision and queue the order until the pair's next quote (merged with one still waiting)."""
        self._orders.append({"ts": ts, "pair": pair, "units": signed_units, "role": role, "reason": reason})
        waiting = self._pending.get(pair)
        if waiting is None:
            self._pending[pair] = [signed_units, role, reason]
        else:
            waiting[0] += signed_units
            waiting[1], waiting[2] = role, reason

    def _sync_positions(self) -> None:
        """The venue's net position plus orders still waiting for their quote must equal the intended position;
        any mismatch (e.g. a rejected order) is counted and fixed."""
        for pair in self.pairs:
            net = int(self.portfolio.net_position(B.instrument_id(pair)))
            waiting = self._pending[pair][0] if pair in self._pending else 0
            if net + waiting != self._units[pair]:
                self.desyncs += 1
                self._units[pair] = net + waiting

    def _equity_usd(self) -> float:
        eq = self.portfolio.account(B.SIM).balance_total(USD).as_double()
        for money in self.portfolio.unrealized_pnls(B.SIM).values():
            eq += money.as_double()
        return eq

    def _quote_to_usd(self, pair: str) -> float | None:
        quote_ccy = self.cache.instrument(B.instrument_id(pair)).quote_currency
        return self.cache.get_xrate(B.SIM, quote_ccy, USD)

    def on_order_filled(self, event) -> None:
        pair, role, reason = self._role.get(event.client_order_id, (None, "?", "?"))
        signed = 1 if event.order_side == OrderSide.BUY else -1
        self._fills.append({"ts": event.ts_event, "pair": pair, "units": signed * int(event.last_qty.as_double()),
                            "px": event.last_px.as_double(), "role": role, "reason": reason})

    # ---------------------------------------------------------------- results
    def result(self) -> B.RunResult:
        eq = pd.Series([e for _, e in self._equity], index=pd.to_datetime([t for t, _ in self._equity], utc=True),
                       name="equity", dtype=float)
        orders = pd.DataFrame(self._orders, columns=["ts", "pair", "units", "role", "reason"])
        fills = pd.DataFrame(self._fills, columns=["ts", "pair", "units", "px", "role", "reason"])
        return B.RunResult(equity=eq, orders=orders, fills=fills, trades=round_trips(fills),
                           n_halts=self._risk.n_halts, killed=self._risk.killed, start_nav=B.STARTING_NAV)


def round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """Position episodes per pair: opened by ENTRY (or the opening side of a FLIP), closed by EXIT or FLIP.
    REBAL fills resize an episode; pnl_pips compares the episode's first and last fill prices."""
    rows, open_ = [], {}

    def close(pair, e, f, reason):
        d = 1 if e.units > 0 else -1
        rows.append({"pair": pair, "direction": d, "units": abs(e.units), "entry_ts": e.ts, "entry_px": e.px,
                     "exit_ts": f.ts, "exit_px": f.px, "exit_reason": reason,
                     "pnl_pips": d * (f.px - e.px) / B.pip_size(pair), "hours_held": (f.ts - e.ts) / 3.6e12})

    for f in fills.itertuples(index=False):
        if f.role == "ENTRY":
            open_[f.pair] = f
        elif f.role in ("EXIT", "FLIP") and f.pair in open_:
            close(f.pair, open_.pop(f.pair), f, f.reason if f.role == "EXIT" else "flip")
            if f.role == "FLIP":
                open_[f.pair] = f
    for pair, e in open_.items():
        rows.append({"pair": pair, "direction": 1 if e.units > 0 else -1, "units": abs(e.units),
                     "entry_ts": e.ts, "entry_px": e.px, "exit_ts": np.nan, "exit_px": np.nan,
                     "exit_reason": "open", "pnl_pips": np.nan, "hours_held": np.nan})
    cols = ["pair", "direction", "units", "entry_ts", "entry_px", "exit_ts", "exit_px", "exit_reason",
            "pnl_pips", "hours_held"]
    return pd.DataFrame(rows, columns=cols)


# ============================================================================= vectorised reference (parity)
def reference_backtest(signal_frame, warmup_bars, pair: str, h1: pd.DataFrame, plan: list[Segment], rates=None,
                       limits: RiskLimits = RiskLimits(), nav0: float = B.STARTING_NAV) -> dict:
    """Independent re-implementation for ONE USD-quoted pair: pandas signals, the execution rule written out
    again, fills at the next open quote, and overnight financing booked exactly when the venue books it (the
    first event at or after each 17:00 New York roll, before that event's fills, marked at the last close quote).

    Like the Nautilus run, it sees only bars closing at or after the plan's warm-up start: recursive indicators
    (Wilder ATR, EWMAs) depend on where they were seeded.
    """
    if not pair.endswith("USD"):
        raise ValueError("the reference handles USD-quoted pairs only")
    if plan[-1].end_ns > int(h1.index[-1].value) + HOUR_NS:
        raise ValueError("the plan must end at or before the last bar close")
    h1 = h1[h1.index.as_unit("ns").asi8 + HOUR_NS >= warmup_start(plan, warmup_bars)]
    sigs = {q: signal_frame({pair: h1}, q)[pair] for q in dict.fromkeys(s.params for s in plan)}
    opens = h1.index.as_unit("ns").asi8
    closes = opens + HOUR_NS
    ob, oa, cb, ca = (h1[c].to_numpy() for c in ("open_bid", "open_ask", "close_bid", "close_ask"))
    rolls = B.roll_times_ns()
    risk = RiskManager(limits, nav0)
    units, cash, pending = 0, nav0, None
    orders, equity = [], []
    first_roll = int(np.searchsorted(rolls, int(opens[0]) + B.ONE_MS, side="right"))

    def book(lo: int, hi: int, i_mark: int, include_lo: bool) -> None:
        """Financing for rolls in (lo, hi] ([lo, hi] if include_lo) on the current units, marked at bar
        i_mark's close quote."""
        nonlocal cash
        if units == 0 or rates is None:
            return
        a = int(np.searchsorted(rolls, lo, side="left" if include_lo else "right"))
        b = int(np.searchsorted(rolls, hi, side="right"))
        a = max(a, first_roll)
        if b <= a:
            return
        mid = (cb[i_mark] + ca[i_mark]) / 2.0
        amt = 0.0
        for r in rolls[a:b]:
            diff = float(rates.accrual(pair[:3], [r])[0] - rates.accrual(pair[3:], [r])[0])
            if diff == diff:
                amt += B.financing_quote(units, mid, diff)
        cash += round(amt, 2)

    for i in range(len(h1)):
        # -- open quote of bar i: financing for rolls since the previous event, then the pending fill
        if i > 0:
            book(int(closes[i - 1]), int(opens[i]) + B.ONE_MS, i - 1, include_lo=False)
        if pending is not None:
            du, role, reason, t_dec = pending
            px = oa[i] if du > 0 else ob[i]
            cash -= du * px
            units += du
            orders.append({"ts": t_dec, "units": du, "role": role, "reason": reason, "px": px})
            pending = None
        # -- bar close: a roll exactly at the close is booked on the position held through the bar
        book(int(closes[i]), int(closes[i]), i, include_lo=True)
        t_dec = int(closes[i])
        eq = cash + units * (cb[i] if units > 0 else ca[i])
        if plan[0].start_ns < t_dec <= plan[-1].end_ns:
            equity.append((t_dec, eq))
        risk.on_close(t_dec, eq)
        seg = next((k for k, s in enumerate(plan) if s.start_ns <= t_dec < s.end_ns), -1)
        if seg < 0 or i + 1 >= len(h1):     # plans end at or before the last close (asserted by the harness)
            continue
        if not risk.allows_trading(t_dec):
            if units != 0:
                pending = (-units, "EXIT", "kill" if risk.killed else "halt", t_dec)
            continue
        q = plan[seg].params
        row = sigs[q].iloc[i + 1]                         # row i+1 = known at the open of bar i+1
        s = float(row["signal"])
        target = 0
        if s == s and s != 0:
            size = RiskManager.position_size(risk.last_equity, float(row["atr"]), float(row["close"]), 1.0,
                                             q, limits, weight=s)
            target = size if s > 0 else -size
        if target == units:
            continue
        if units == 0:
            pending = (target, "ENTRY", "signal", t_dec)
        elif target == 0:
            pending = (-units, "EXIT", "signal", t_dec)
        elif (target > 0) != (units > 0):
            pending = (target - units, "FLIP", "signal", t_dec)
        elif abs(target - units) >= q.band * max(abs(target), abs(units)):
            pending = (target - units, "REBAL", "band", t_dec)
    eq = pd.Series([e for _, e in equity], index=pd.to_datetime([t for t, _ in equity], utc=True), dtype=float)
    return {"orders": pd.DataFrame(orders, columns=["ts", "units", "role", "reason", "px"]), "equity": eq}


# ============================================================================= signal kernels
# Each family has a vectorised version (pandas, whole history, row t = known at the close of bar t) and an
# event-driven version (MarketData state at the latest close). Hypotheses combine them; parity proves they agree.
def usd_leg(pair: str) -> tuple[str, int] | None:
    """(currency, +1) for CCYUSD, (currency, -1) for USDCCY: the pair's exposure to CCY against USD."""
    base, quote = currencies(pair)
    if quote == "USD" and base != "USD":
        return base, 1
    if base == "USD" and quote != "USD":
        return quote, -1
    return None


def vec_tsmom(close: pd.Series, lag: int) -> pd.Series:
    """sign(C_t - C_{t-lag}); NaN until `lag` bars of history exist."""
    return np.sign(close - close.shift(lag))


def ev_tsmom(md: MarketData, pair: str, lag: int) -> float:
    c, c0 = md.close(pair), md.past_close(pair, lag)
    return float((c > c0) - (c < c0)) if c0 == c0 else math.nan


def vec_carry(carry: pd.Series, theta: float) -> pd.Series:
    """sign(d) if |d| > theta else 0, d = the known rate differential (percent p.a.); NaN if d is unknown."""
    out = np.where(carry.abs() > theta, np.sign(carry), 0.0)
    return pd.Series(np.where(carry.isna(), np.nan, out), index=carry.index)


def ev_carry(md: MarketData, pair: str, theta: float) -> float:
    d = md.carry[pair]
    if d != d:
        return math.nan
    return float((d > 0) - (d < 0)) if abs(d) > theta else 0.0


def _xs_rank_weights(n: int, ranks):
    return (ranks - (n + 1) / 2.0) / ((n - 1) / 2.0)


def vec_xs_weights(frames: dict[str, pd.DataFrame], lag: int) -> dict[str, pd.Series]:
    """Cross-sectional momentum: rank USD and every currency with a USD leg by its L-bar performance against USD
    (average ranks for ties); weights w = (rank - (n+1)/2) / ((n-1)/2) in [-1, 1], summing to 0. A USD-leg pair
    gets +-w of its currency, a cross 0. Performance is the price ratio C_t/C_{t-L} (inverted for USDCCY)."""
    legs = {p: usd_leg(p) for p in frames if usd_leg(p) is not None}
    index = universe_index(frames)
    perf = {"USD": pd.Series(1.0, index=index)}
    for p, (ccy, d) in legs.items():
        c = frames[p]["close"]
        r = c / c.shift(lag) if d > 0 else c.shift(lag) / c
        perf[ccy] = r.reindex(index).ffill()
    df = pd.DataFrame(perf)
    n = df.shape[1]
    w = _xs_rank_weights(n, df.rank(axis=1, method="average")) if n >= 2 else df * np.nan
    w.loc[df.isna().any(axis=1)] = np.nan
    out = {}
    for p, f in frames.items():
        leg = usd_leg(p)
        out[p] = (leg[1] * w[leg[0]]).reindex(f.index) if leg else pd.Series(0.0, index=f.index)
    return out


def ev_xs_weights(md: MarketData, lag: int) -> dict[str, float]:
    legs = {p: usd_leg(p) for p in md.pairs if usd_leg(p) is not None}
    perf = {"USD": 1.0}
    for p, (ccy, d) in legs.items():
        c, c0 = md.close(p), md.past_close(p, lag)
        perf[ccy] = (c / c0 if d > 0 else c0 / c) if c0 == c0 else math.nan
    n = len(perf)
    ready = n >= 2 and all(v == v for v in perf.values())
    w = {}
    for ccy, v in perf.items():
        rank = 1.0 + sum(x < v for x in perf.values()) + 0.5 * (sum(x == v for x in perf.values()) - 1)
        w[ccy] = _xs_rank_weights(n, rank) if ready else math.nan
    return {p: (usd_leg(p)[1] * w[usd_leg(p)[0]] if usd_leg(p) else 0.0) for p in md.pairs}


# ============================================================================= vectorised helpers
def universe_index(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))


def features_dict(data) -> tuple[bool, dict[str, pd.DataFrame]]:
    """Accept one pair's frame (a one-pair universe) or a {pair: frame} dict."""
    single = isinstance(data, pd.DataFrame)
    return single, ({"BASQUO": data} if single else data)


def finish(single: bool, feats: dict[str, pd.DataFrame]):
    return feats["BASQUO"] if single else feats


def signal_frame_from(compute_features, data, p):
    """Row t = everything known when bar t opens, i.e. computed from bars <= t-1 (same shape as `data`)."""
    feats = compute_features(data, p)
    if isinstance(feats, pd.DataFrame):
        return feats.shift(1)  # <- look-ahead guard
    return shift_signals(feats)


def base_features(f: pd.DataFrame) -> pd.DataFrame:
    """Close and Wilder ATR known at each bar's close (row t uses bars <= t), plus the known carry."""
    out = pd.DataFrame({"close": f["close"],
                        "atr": wilder_atr(f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy())},
                       index=f.index)
    out["carry"] = f["carry"] if "carry" in f.columns else np.nan
    return out


def shift_signals(feats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Row t = everything known when bar t opens, i.e. computed from bars <= t-1."""
    return {k: f.shift(1) for k, f in feats.items()}  # <- look-ahead guard
