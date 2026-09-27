"""Pre-registered cycle-4 hypotheses (see PREREGISTRATION.md): published FX factor strategies.

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
}
ORDER = tuple(REGISTRY)
BASE_FILE = Path(__file__).resolve().parent / "academic_base.py"


def path(hid: str) -> Path:
    return Path(__file__).resolve().parent / f"{REGISTRY[hid][0]}.py"


def load(hid: str) -> ModuleType:
    return importlib.import_module(f"hypotheses.{REGISTRY[hid][0]}")
