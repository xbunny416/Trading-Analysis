"""Strategy registry.

Cycle 4 (PREREGISTRATION.md, archived in research/cycle4): published FX factor strategies A1..A6 and A3F. Their
modules stay as a library for later trials.
Cycle 5 (README): an adaptive search; every trial has an immutable ID (C5-001, ...) and its own module.

Each module defines its signal (vectorised ``signal_frame`` and event-driven ``Signals``) on top of the shared
sizing / execution / risk engine in ``academic_base.py``. The cycle-3 hypotheses live in research/cycle3/.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

REGISTRY = {
    "A1": ("a1_tsmom_mop", "Moskowitz-Ooi-Pedersen 2012", "time-series momentum, sign of the L-month return"),
    "A2": ("a2_tsmom_blend", "Hurst-Ooi-Pedersen 2017", "time-series momentum, equal blend of 1/3/12 months"),
    "A3": ("a3_ewma_crossover", "Baz et al. 2015 (Man AHL)", "multi-speed EWMA crossover with response function"),
    "A4": ("a4_carry", "Koijen-Moskowitz-Pedersen-Vrugt 2018", "carry: sign of the known rate differential"),
    "A5": ("a5_xs_momentum", "Menkhoff-Sarno-Schmeling-Schrimpf 2012", "cross-sectional currency momentum"),
    "A6": ("a6_multifactor", "Asness-Moskowitz-Pedersen 2013; KMPV 2018", "equal blend of A2, A4 and A5"),
    # Addendum 2 (post-hoc, trial 8): A3 unchanged, re-evaluated with a fast rolling walk-forward
    "A3F": ("a3f_ewma_fast_wfa", "Baz et al. 2015 (Man AHL)", "A3 with <= 152-day rolling OOS windows, WFE >= 2/3"),
}
ORDER = tuple(REGISTRY)            # cycle 4, archived

# Cycle 5: ID -> (module, family, parent trial or "")
CYCLE5_REGISTRY: dict[str, tuple[str, str, str]] = {
    "C5-001": ("c5_001_ts_meanrev", "time-series mean reversion (hourly z-score)", ""),
    "C5-002": ("c5_002_xs_reversal", "cross-sectional short-term reversal", "A5"),
    "C5-003": ("c5_003_a3_threshold", "trend: A3 with a signal-strength threshold", "A3"),
    "C5-004": ("c5_004_trend_carry_filter", "trend: A2 that refuses to pay carry", "A2"),
    "C5-005": ("c5_005_trend_efficiency_regime", "trend: A2 only in efficient (trending) regimes", "A2"),
    "C5-006": ("c5_006_dollar_trend", "trend: dollar consensus of A3 over the USD pairs", "A3"),
    "C5-007": ("c5_007_eurgbp_relative_value", "relative value: EURUSD-GBPUSD spread mean reversion", "C5-001"),
    "C5-008": ("c5_008_shock_follow", "large-shock follow-through, bar-count hold", ""),
    "C5-009": ("c5_009_shock_fade", "large-shock reversal, bar-count hold", "C5-008"),
    "C5-010": ("c5_010_volmanaged_dollar_trend", "trend: volatility-managed dollar consensus", "C5-006"),
    "C5-011": ("c5_011_derisked_dollar_trend", "trend: dollar consensus, de-risk-only volatility scaling", "C5-006"),
    "C5-012": ("c5_012_blend_trend_volmanaged", "trend: vol-managed blend of dollar consensus and per-pair A3", "C5-010"),
    "C5-013": ("c5_013_volmanaged_a3", "trend: vol-managed per-pair A3", "C5-010"),
    "C5-014": ("c5_014_volmanaged_dollar_tsmom", "trend: vol-managed dollar consensus of A2 TSMOM", "C5-010"),
    "C5-015": ("c5_015_fast_dollar_trend", "trend: fast vol-managed dollar consensus (A3 spans x 1/8..1/2)", "C5-010"),
    "C5-016": ("c5_016_donchian_breakout", "trend: Donchian / Turtle breakout, per pair", ""),
    "C5-017": ("c5_017_volmanaged_dollar_carry", "carry: vol-managed dollar carry (LRV 2014)", "A4"),
    "C5-018": ("c5_018_rate_momentum", "rate momentum: change in the known rate differential, vol-managed", ""),
    "C5-019": ("c5_019_slow_dollar_trend", "trend: slow vol-managed dollar consensus (A3 spans x 1..2)", "C5-015"),
    "C5-020": ("c5_020_xs_value", "value: 3-year cross-sectional reversal (nominal proxy), vol-managed", "C5-002"),
    "C5-021": ("c5_021_value_trend_combo", "ensemble: 50/50 dollar trend + currency value, vol-managed", "C5-010"),
    "C5-022": ("c5_022_smooth_value", "value: 3-year value with smooth z-score weights, vol-managed", "C5-020"),
    "C5-023": ("c5_023_smooth_value_trend_combo", "ensemble: 50/50 dollar trend + smooth value, vol-managed", "C5-021"),
    "C5-024": ("c5_024_smooth_xs_momentum", "XS momentum with smooth z-score weights, vol-managed", "A5"),
    "C5-025": ("c5_025_trend_value_momentum", "ensemble: dollar trend + smooth value + smooth XS momentum", "C5-023"),
}
CYCLE5 = tuple(CYCLE5_REGISTRY)
REGISTRY.update({k: v for k, v in CYCLE5_REGISTRY.items()})
BASE_FILE = Path(__file__).resolve().parent / "academic_base.py"


def path(hid: str) -> Path:
    return Path(__file__).resolve().parent / f"{REGISTRY[hid][0]}.py"


def load(hid: str) -> ModuleType:
    return importlib.import_module(f"hypotheses.{REGISTRY[hid][0]}")


def depends(hid: str) -> list[Path]:
    """Other hypothesis files a module imports (part of its code fingerprint and complexity audit)."""
    return [Path(__file__).resolve().parent / f for f in getattr(load(hid), "DEPENDS", ())]
