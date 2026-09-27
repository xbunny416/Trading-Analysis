"""
strategy.py - entry point for the strategy logic.

Cycle 3 evaluates a fixed, pre-registered list of hypotheses (PREREGISTRATION.md). Each lives in
``hypotheses/`` as the unchanged cycle-2 strategy module it came from; load one with ``load("H1")``.
"""
from hypotheses import ORDER, REGISTRY, load  # noqa: F401
