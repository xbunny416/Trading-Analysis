"""
strategy.py - cycle 2: long/short hourly currency-strength trend following on a 5-pair FX portfolio,
run in NautilusTrader.

Rules (the same in the Nautilus strategy and the vectorised reference below)
----------------------------------------------------------------------------
Indicators (2): momentum (L-bar log return), ATR.  Tunable parameters (4): mom_n, atr_n, atr_mult, risk_pct.
Signals use hourly MID bars of every pair at the same close (cross-sectional); parameters are shared.

    TR_t = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|),  ATR_t = mean(TR_{t-n+1..t})                    n = atr_n
    m_p  = ln(C_t / C_{t-L}) / (ATR_t / C_t * sqrt(L))        vol-normalised pair momentum            L = mom_n
    S_c  = mean over pairs p containing currency c of (+m_p if c is p's base, -m_p if its quote)
    z_p  = S_base(p) - S_quote(p)                              currency-strength score of pair p

    decided at the close of bar t-1, filled at the open quote of bar t:
        entry  : long when z_p crosses above 0, short when it crosses below 0
        exit   : chandelier stop  S = max_d(S, C - d*k*ATR); exit when d*(C_{t-1} - S) < 0           k = atr_mult
        flip   : reverse on the opposite cross
        size   : units = NAV * risk_pct / (k * ATR * quote->USD), floored to 1,000-unit lots and capped
                 at max_leverage * NAV in USD notional; 0 for any degenerate input

Risk limits (portfolio level, checked at every hourly close, acted on at the next open):
    trading-day loss >= 2.5 %   -> flatten everything, no new positions for 24 h
    drawdown from peak >= 10 %  -> flatten everything, permanently halted

Look-ahead guard: a decision at the close of bar t-1 only sees bars <= t-1 (``signal_frame`` shifts by one
bar; the Nautilus strategy acts on bar-close events and its orders fill at the next bar's open quote).
The harness proves both by perturbation.
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

# ============================================================================= configuration
HYPOTHESIS = (
    "C2-T5 currency-strength trend: score each currency by averaging its vol-normalised mom_n-bar momentum "
    "across every pair that contains it, trade each pair on crosses of (base strength - quote strength), "
    "atr_mult x ATR chandelier stop. Motivation: T1-T4 single-pair signals were weak/fragile; currency-level "
    "averaging cancels pair-specific noise (currency momentum, Menkhoff et al. 2012)."
)
INDICATORS = ("momentum", "ATR")


@dataclass(frozen=True)
class StrategyParams:
    """The only tunable numbers in the strategy (4)."""

    mom_n: int = 168          # momentum lookback L, hourly bars
    atr_n: int = 24           # ATR lookback, hourly bars
    atr_mult: float = 5.0     # chandelier-stop distance and sizing unit, in ATRs
    risk_pct: float = 0.001   # per pair: fraction of NAV lost if price moves atr_mult*ATR against it

    @property
    def stop_mult(self) -> float:
        return self.atr_mult

    def validate(self) -> "StrategyParams":
        for name in ("mom_n", "atr_n"):
            v = getattr(self, name)
            if not (isinstance(v, (int, np.integer)) and v >= 2):
                raise ValueError(f"{name} must be an int >= 2, got {v}")
        if not (self.atr_mult > 0 and 0 < self.risk_pct < 0.05):
            raise ValueError("atr_mult must be > 0 and risk_pct in (0, 0.05)")
        return self


DEFAULT_PARAMS = StrategyParams()
# walk-forward search space (the other parameters stay at their defaults)
PARAM_GRID = {"mom_n": [24, 72, 168, 336], "atr_mult": [3.0, 5.0, 7.0, 10.0]}
# parameters that change the signal frame
FEATURE_PARAMS = ("mom_n", "atr_n")


@dataclass(frozen=True)
class RiskLimits:
    """Fixed by the mandate, never tuned."""

    daily_loss_limit: float = 0.025
    halt_hours: float = 24.0
    max_drawdown_limit: float = 0.10
    max_leverage: float = 10.0    # per position, USD notional / NAV
    lot: int = 1_000              # order sizes are whole micro lots


def warmup_bars(p: StrategyParams) -> int:
    return max(p.mom_n, p.atr_n) + 3


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


# ============================================================================= research path (vectorised)
def compute_features(data, p: StrategyParams):
    """Indicator values known at the CLOSE of each hourly bar (row t uses bars <= t). Not yet tradeable.

    `data` is one pair's frame (treated as a one-pair universe) or a dict {pair: frame}; returns the same shape.
    """
    single = isinstance(data, pd.DataFrame)
    frames = {"BASQUO": data} if single else data
    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
    feats, mom = {}, {}
    for k, f in frames.items():
        h, l, c = f["high"], f["low"], f["close"]
        pc = c.shift(1)
        tr = np.fmax(h - l, np.fmax((h - pc).abs(), (l - pc).abs()))
        atr = tr.rolling(p.atr_n).mean()
        m = (np.log(c / c.shift(p.mom_n)) / (atr / c * math.sqrt(p.mom_n))).where(atr > 0)
        feats[k] = pd.DataFrame({"close": c, "atr": atr}, index=f.index)
        mom[k] = m.reindex(index).ffill()
    strength = {}
    for ccy in sorted({x for k in frames for x in currencies(k)}):
        cols = [mom[k] if currencies(k)[0] == ccy else -mom[k] for k in frames if ccy in currencies(k)]
        strength[ccy] = pd.concat(cols, axis=1).mean(axis=1, skipna=False)
    for k, f in feats.items():
        base, quote = currencies(k)
        f["score"] = (strength[base] - strength[quote]).reindex(f.index)
    return feats["BASQUO"] if single else feats


def signal_frame(data, p: StrategyParams):
    """Row t = everything known when bar t opens, i.e. computed from bars <= t-1 (same shape as `data`)."""
    feats = compute_features(data, p)

    def one(f: pd.DataFrame) -> pd.DataFrame:
        s = f.shift(1)  # <- look-ahead guard
        prev = s["score"].shift(1)
        s["long_entry"] = ((s["score"] > 0) & (prev <= 0)).astype(float)
        s["short_entry"] = ((s["score"] < 0) & (prev >= 0)).astype(float)
        return s

    return one(feats) if isinstance(feats, pd.DataFrame) else {k: one(f) for k, f in feats.items()}


# ============================================================================= engine-agnostic components
@dataclass(frozen=True)
class SignalSnapshot:
    close: float
    atr: float
    long_entry: bool
    short_entry: bool


def snapshot_from_row(row) -> SignalSnapshot:
    """The vectorised signal frame's row t as the snapshot the event-driven engine has at the same time."""
    return SignalSnapshot(float(row["close"]), float(row["atr"]), bool(row["long_entry"] > 0),
                          bool(row["short_entry"] > 0))


class UniverseSignalEngine:
    """Incremental per-pair ATR / momentum and the cross-sectional currency-strength score."""

    def __init__(self, p: StrategyParams, pairs: tuple[str, ...]):
        self.p, self.pairs = p, tuple(pairs)
        self._closes = {k: deque(maxlen=p.mom_n + 1) for k in self.pairs}
        self._tr = {k: deque(maxlen=p.atr_n) for k in self.pairs}
        self._atr = {k: math.nan for k in self.pairs}
        self._z_prev = {k: math.nan for k in self.pairs}
        self._ccys = sorted({x for k in self.pairs for x in currencies(k)})

    def update(self, pair: str, o: float, h: float, l: float, c: float) -> None:
        """Fold in one completed bar of one pair."""
        cl = self._closes[pair]
        pc = cl[-1] if cl else None
        tr = h - l if pc is None else max(h - l, abs(h - pc), abs(l - pc))
        trs = self._tr[pair]
        trs.append(tr)
        self._atr[pair] = math.fsum(trs) / self.p.atr_n if len(trs) == self.p.atr_n else math.nan
        cl.append(c)

    def _momentum(self, pair: str) -> float:
        cl, atr, n = self._closes[pair], self._atr[pair], self.p.mom_n
        if len(cl) < n + 1 or not atr > 0:
            return math.nan
        c = cl[-1]
        return math.log(c / cl[0]) / (atr / c * math.sqrt(n))

    def snapshots(self) -> dict[str, SignalSnapshot]:
        """Call once per hourly close after every pair's bar is in; returns each pair's snapshot."""
        m = {k: self._momentum(k) for k in self.pairs}
        strength = {}
        for ccy in self._ccys:
            vals = [m[k] if currencies(k)[0] == ccy else -m[k] for k in self.pairs if ccy in currencies(k)]
            strength[ccy] = math.nan if any(math.isnan(v) for v in vals) else sum(vals) / len(vals)
        out = {}
        for k in self.pairs:
            base, quote = currencies(k)
            z, zp = strength[base] - strength[quote], self._z_prev[k]
            out[k] = SignalSnapshot(self._closes[k][-1] if self._closes[k] else math.nan, self._atr[k],
                                    z > 0 and zp <= 0, z < 0 and zp >= 0)
            self._z_prev[k] = z
        return out


class PositionLogic:
    """Per-pair state machine: breakout entries, chandelier trailing stop, stop-and-reverse."""

    def __init__(self, p: StrategyParams):
        self.p = p
        self.direction = 0
        self.stop = math.nan
        self.exit_reason = ""

    def on_bar_close(self, snap: SignalSnapshot) -> int:
        """Fold in a completed bar; return the desired direction for the next open."""
        d = self.direction
        self.exit_reason = ""
        if d == 0:
            return 1 if snap.long_entry else (-1 if snap.short_entry else 0)
        if d * (snap.close - self.stop) < 0:
            self.exit_reason = "stop"
        else:
            s2 = snap.close - d * self.p.stop_mult * snap.atr
            if d * (s2 - self.stop) > 0:
                self.stop = s2
        opposite = snap.short_entry if d == 1 else snap.long_entry
        return -d if opposite else (0 if self.exit_reason else d)

    def on_entry(self, direction: int, snap: SignalSnapshot) -> None:
        self.direction = direction
        self.stop = snap.close - direction * self.p.stop_mult * snap.atr

    def on_exit(self) -> None:
        self.direction, self.stop = 0, math.nan


class RiskManager:
    """Volatility-targeted sizing, leverage cap, daily-loss halt and max-drawdown kill switch."""

    def __init__(self, limits: RiskLimits, nav0: float):
        self.limits = limits
        self.last_equity = self.peak = self.day_start = nav0
        self.day: int | None = None
        self.halt_until = -1
        self.killed = False
        self.n_halts = 0
        self._halt_ns = int(limits.halt_hours * 3_600_000_000_000)

    @staticmethod
    def position_size(nav: float, atr: float, price: float, quote_to_usd: float | None,
                      p: StrategyParams, limits: RiskLimits) -> int:
        """Units = NAV * risk% / (ATR * multiplier * quote->USD), whole lots, leverage-capped; 0 if degenerate."""
        vals = (nav, atr, price, quote_to_usd)
        if any(v is None or not math.isfinite(v) or v <= 0 for v in vals):
            return 0
        denom = p.stop_mult * atr * quote_to_usd
        if not denom > 0:
            return 0
        units = min(nav * p.risk_pct / denom, limits.max_leverage * nav / (price * quote_to_usd))
        return int(units // limits.lot) * limits.lot

    def on_close(self, ts_close: int, equity: float) -> None:
        """Called once per hourly close, before any decision taken at that close."""
        day = trading_day(ts_close - 3_600_000_000_000)  # day of the bar that just closed
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


@dataclass(frozen=True)
class Segment:
    start_ns: int            # first decision time (a bar close) traded with `params`
    end_ns: int              # decisions strictly before this time; equity recorded up to it
    params: StrategyParams


def warmup_start(plan: list[Segment]) -> int:
    bars = max(warmup_bars(s.params) for s in plan)
    return plan[0].start_ns - int((bars * 1.5 + 72) * 3_600_000_000_000)


# ============================================================================= NautilusTrader strategy
class PortfolioTrendStrategy(Strategy):
    """Wires Nautilus market data -> SignalEngine -> PositionLogic -> RiskManager -> Nautilus orders.

    One instance trades every pair in `pairs` on the SIM venue. `plan` is a list of contiguous segments,
    each with its own parameters (the stitched walk-forward switches parameters at segment boundaries and
    liquidates the previous segment's positions there).
    """

    def __init__(self, plan: list[Segment], pairs: tuple[str, ...], limits: RiskLimits = RiskLimits()):
        super().__init__(StrategyConfig(strategy_id="PortfolioTrend-001"))
        for s in plan:
            s.params.validate()
        self.plan, self.pairs, self.limits = plan, tuple(pairs), limits
        self.plan_start, self.plan_end = plan[0].start_ns, plan[-1].end_ns
        self.warmup_start_ns = warmup_start(plan)
        self._pair_of = {B.bar_type(p): p for p in self.pairs}
        uniq = list(dict.fromkeys(s.params for s in plan))
        self._signals = {q: UniverseSignalEngine(q, self.pairs) for q in uniq}
        self._seen: set[str] = set()
        self._logic = {pair: PositionLogic(plan[0].params) for pair in self.pairs}
        self._units = {pair: 0 for pair in self.pairs}
        self._risk = RiskManager(limits, B.STARTING_NAV)
        self._ts = -1
        self._seg = -1
        self._equity: list[tuple[int, float]] = []
        self._orders: list[dict] = []
        self._fills: list[dict] = []
        self._role: dict = {}
        self.desyncs = 0

    # ---------------------------------------------------------------- lifecycle
    def on_start(self) -> None:
        for p in self.pairs:
            self.subscribe_bars(B.bar_type(p))

    def on_bar(self, bar) -> None:
        ts = bar.ts_event
        pair = self._pair_of[bar.bar_type]
        if ts != self._ts:
            if self._seen:  # the previous close never completed (a pair had no bar): decide with what came in
                self._decide_all(self._ts)
            self._on_close_time(ts)
        o, h, l, c = bar.open.as_double(), bar.high.as_double(), bar.low.as_double(), bar.close.as_double()
        for eng in self._signals.values():
            eng.update(pair, o, h, l, c)
        self._seen.add(pair)
        if len(self._seen) == len(self.pairs):
            self._decide_all(ts)

    def _decide_all(self, ts: int) -> None:
        seen, self._seen = self._seen, set()
        snaps = {q: eng.snapshots() for q, eng in self._signals.items()}
        if self._seg < 0:
            return
        params = self.plan[self._seg].params
        for pair in self.pairs:
            if pair in seen:
                self._decide(pair, snaps[params][pair], params, ts)

    def _decide(self, pair: str, snap: SignalSnapshot, params: StrategyParams, ts: int) -> None:
        logic = self._logic[pair]
        target = logic.on_bar_close(snap)
        if not self._risk.allows_trading(ts):
            target = 0
        d = logic.direction
        if target == d:
            return
        if d != 0:
            self._exit(pair, ts, "reverse" if target == -d else logic.exit_reason)
        if target != 0:
            size = RiskManager.position_size(self._risk.last_equity, snap.atr, snap.close,
                                             self._quote_to_usd(pair), params, self.limits)
            if size > 0:
                self._submit(pair, target * size, ts, "ENTRY", "signal")
                logic.on_entry(target, snap)
                self._units[pair] = target * size

    # ---------------------------------------------------------------- per-timestamp work
    def _on_close_time(self, ts: int) -> None:
        self._ts = ts
        self._sync_positions()
        equity = self._equity_usd()
        if self.plan_start < ts <= self.plan_end:
            self._equity.append((ts, equity))
        self._risk.on_close(ts, equity)
        seg = next((k for k, s in enumerate(self.plan) if s.start_ns <= ts < s.end_ns), -1)
        if seg != self._seg:
            if self._seg >= 0 and seg >= 0:  # switching segments: liquidate what the old parameters opened
                for pair in self.pairs:
                    if self._units[pair] != 0:
                        self._exit(pair, ts, "segment_end")
            if seg >= 0:
                self._logic = {pair: PositionLogic(self.plan[seg].params) for pair in self.pairs}
            self._seg = seg
        if seg >= 0 and not self._risk.allows_trading(ts):
            reason = "kill" if self._risk.killed else "halt"
            for pair in self.pairs:
                if self._units[pair] != 0:
                    self._exit(pair, ts, reason)

    def _exit(self, pair: str, ts: int, reason: str) -> None:
        units = self._units[pair]
        if units != 0:
            self._submit(pair, -units, ts, "EXIT", reason)
        self._units[pair] = 0
        self._logic[pair].on_exit()

    def _submit(self, pair: str, signed_units: int, ts: int, role: str, reason: str) -> None:
        side = OrderSide.BUY if signed_units > 0 else OrderSide.SELL
        order = self.order_factory.market(B.instrument_id(pair), side, Quantity.from_int(abs(signed_units)),
                                          reduce_only=(role == "EXIT"))
        self._role[order.client_order_id] = (pair, role, reason)
        self._orders.append({"ts": ts, "pair": pair, "units": signed_units, "role": role, "reason": reason})
        self.submit_order(order)

    def _sync_positions(self) -> None:
        """The venue's net position is the truth; any mismatch (e.g. a rejected order) is counted and fixed."""
        for pair in self.pairs:
            net = int(self.portfolio.net_position(B.instrument_id(pair)))
            if net != self._units[pair]:
                self.desyncs += 1
                self._units[pair] = net
                if net == 0:
                    self._logic[pair].on_exit()

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
    """Pair each ENTRY fill with the EXIT fill that closes it (one position per pair at a time)."""
    rows, open_ = [], {}
    for f in fills.itertuples(index=False):
        if f.role == "ENTRY":
            open_[f.pair] = f
        elif f.role == "EXIT" and f.pair in open_:
            e = open_.pop(f.pair)
            d = 1 if e.units > 0 else -1
            rows.append({"pair": f.pair, "direction": d, "units": abs(e.units), "entry_ts": e.ts,
                         "entry_px": e.px, "exit_ts": f.ts, "exit_px": f.px, "exit_reason": f.reason,
                         "pnl_pips": d * (f.px - e.px) / B.pip_size(f.pair),
                         "hours_held": (f.ts - e.ts) / 3.6e12})
    for pair, e in open_.items():
        rows.append({"pair": pair, "direction": 1 if e.units > 0 else -1, "units": abs(e.units),
                     "entry_ts": e.ts, "entry_px": e.px, "exit_ts": np.nan, "exit_px": np.nan,
                     "exit_reason": "open", "pnl_pips": np.nan, "hours_held": np.nan})
    cols = ["pair", "direction", "units", "entry_ts", "entry_px", "exit_ts", "exit_px", "exit_reason",
            "pnl_pips", "hours_held"]
    return pd.DataFrame(rows, columns=cols)


# ============================================================================= vectorised reference (parity)
def reference_backtest(h1: pd.DataFrame, plan: list[Segment], limits: RiskLimits = RiskLimits(),
                       nav0: float = B.STARTING_NAV) -> dict:
    """Independent re-implementation for ONE USD-quoted pair: pandas signals, fills at the next open quote.

    Used only to cross-check the Nautilus plumbing (orders, fills, marks, timing) bar by bar.
    """
    sigs = {q: signal_frame(h1, q) for q in dict.fromkeys(s.params for s in plan)}
    closes = h1.index.as_unit("ns").asi8 + 3_600_000_000_000
    ob, oa, cb, ca = (h1[c].to_numpy() for c in ("open_bid", "open_ask", "close_bid", "close_ask"))
    risk = RiskManager(limits, nav0)
    logic, units, cash = None, 0, nav0
    seg_now, orders, equity = -1, [], []

    def order(ts, du, role, reason, i):
        nonlocal cash
        px = oa[i] if du > 0 else ob[i]
        cash -= du * px
        orders.append({"ts": ts, "units": du, "role": role, "reason": reason, "px": px})

    for i in range(1, len(h1)):
        t_dec = int(closes[i - 1])                         # decision at the close of bar i-1
        eq = cash + units * (cb[i - 1] if units > 0 else ca[i - 1])
        if plan[0].start_ns < t_dec <= plan[-1].end_ns:
            equity.append((t_dec, eq))
        risk.on_close(t_dec, eq)
        seg = next((k for k, s in enumerate(plan) if s.start_ns <= t_dec < s.end_ns), -1)
        if seg != seg_now:
            if seg_now >= 0 and seg >= 0 and units != 0:
                order(t_dec, -units, "EXIT", "segment_end", i)
                units = 0
            logic = PositionLogic(plan[seg].params) if seg >= 0 else None
            seg_now = seg
        if seg >= 0 and not risk.allows_trading(t_dec) and units != 0:
            order(t_dec, -units, "EXIT", "kill" if risk.killed else "halt", i)
            units = 0
            logic.on_exit()
        if seg < 0:
            continue
        q = plan[seg].params
        snap = snapshot_from_row(sigs[q].iloc[i])
        target = logic.on_bar_close(snap)
        if not risk.allows_trading(t_dec):
            target = 0
        d = logic.direction
        if target == d:
            continue
        if d != 0:
            order(t_dec, -units, "EXIT", "reverse" if target == -d else logic.exit_reason, i)
            units = 0
            logic.on_exit()
        if target != 0:
            size = RiskManager.position_size(risk.last_equity, snap.atr, snap.close, 1.0, q, limits)
            if size > 0:
                order(t_dec, target * size, "ENTRY", "signal", i)
                units = target * size
                logic.on_entry(target, snap)
    last = len(h1) - 1
    if plan[0].start_ns < int(closes[last]) <= plan[-1].end_ns:
        equity.append((int(closes[last]), cash + units * (cb[last] if units > 0 else ca[last])))
    eq = pd.Series([e for _, e in equity], index=pd.to_datetime([t for t, _ in equity], utc=True), dtype=float)
    return {"orders": pd.DataFrame(orders, columns=["ts", "units", "role", "reason", "px"]), "equity": eq}
