"""
strategy.py - entry point for the strategy logic.

Cycle 4 evaluates a fixed, pre-registered list of published FX factor strategies (PREREGISTRATION.md). Each lives
in ``hypotheses/`` (A1..A6) on top of the shared engine ``hypotheses/academic_base.py``, which holds both the
event-driven NautilusTrader implementation (MarketData, Signals, RiskManager, execution) and the vectorised pandas
reference. Load one with ``load("A1")``.
"""
from hypotheses import ORDER, REGISTRY, load  # noqa: F401
