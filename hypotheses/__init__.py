"""Pre-registered cycle-3 hypotheses (see PREREGISTRATION.md).

Each module is the cycle-2 ``strategy.py`` of the listed commit, copied byte for byte; H1-H4 additionally end
with an appended compatibility shim so their per-pair signal functions accept the {pair: frame} universe.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

REGISTRY = {
    "H1": ("h1_shock_chandelier", "0350c08", "range-shock entry + chandelier stop"),
    "H2": ("h2_high_conviction_shock", "7d66a4b", "H1 with large shocks only"),
    "H3": ("h3_donchian_breakout", "3951388", "multi-day Donchian breakout, stop-and-reverse"),
    "H4": ("h4_shock_wide_stops", "c7f2519", "H1 with wider stops"),
    "H5": ("h5_currency_strength", "e90f080", "cross-sectional currency-strength trend, zero-cross"),
    "H6": ("h6_currency_strength_band", "cded45d", "currency strength with a hysteresis band"),
    "H7": ("h7_parsimonious_shock", "992e3e7", "shock momentum with the stop tied to the threshold"),
}
ORDER = tuple(REGISTRY)


def path(hid: str) -> Path:
    return Path(__file__).resolve().parent / f"{REGISTRY[hid][0]}.py"


def load(hid: str) -> ModuleType:
    return importlib.import_module(f"hypotheses.{REGISTRY[hid][0]}")
