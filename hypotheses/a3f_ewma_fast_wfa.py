"""
A3F - A3 (Baz et al. 2015 multi-speed EWMA crossover) re-evaluated with a fast rolling walk-forward.

Addendum 2 of PREREGISTRATION.md, trial 8, requested after A3's Stage-1 result was seen (post-hoc). The strategy is
A3 unchanged: signal, constants, sizing, execution and the band grid are imported from a3_ewma_crossover.py (part
of this file's code fingerprint). Only the walk-forward changes:

    OOS windows of equal length, at most WFA_MAX_OOS_DAYS = 152 days (< half a year), tiling the same stitched OOS
    span as the standard geometry (2011-10-02 -> 2022-12-30); each IS window is the 70 : 30 stretch immediately
    before its OOS window (rolling, not anchored); the band is re-chosen on every IS window.
    Gate (d): walk-forward efficiency (OOS Sharpe / mean IS Sharpe) >= WFA_MIN_WFE = 2/3 (instead of 0.60).
    Every other gate and integrity test is unchanged.
"""
from __future__ import annotations

from hypotheses.a3_ewma_crossover import *  # noqa: F401,F403
from hypotheses.a3_ewma_crossover import __doc__ as _A3_DOC  # noqa: F401

HYPOTHESIS = ("A3F = A3 (multi-speed EWMA crossover, Baz et al. 2015) with a fast rolling walk-forward: OOS windows "
              "<= 152 days, IS the preceding 70:30 stretch, WFE gate >= 2/3 (Addendum 2, post-hoc).")
DEPENDS = ("a3_ewma_crossover.py",)
WFA_MAX_OOS_DAYS = 152
WFA_MIN_WFE = 2 / 3
