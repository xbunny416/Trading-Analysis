"""
strategy.py - entry point for the strategy logic.

Each strategy lives in ``hypotheses/`` (cycle-5 trials ``c5_*.py``; the cycle-4 library ``a1..a6_*.py``) on top of the
shared engine ``hypotheses/academic_base.py``, which holds both the event-driven NautilusTrader implementation
(MarketData, Signals, RiskManager, execution) and the vectorised pandas reference. Load one with ``load("C5-001")``.
"""
from hypotheses import CYCLE5, ORDER, REGISTRY, load  # noqa: F401
