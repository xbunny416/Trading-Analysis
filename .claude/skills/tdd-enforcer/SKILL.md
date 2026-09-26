---
name: tdd-enforcer
description: Requires writing automated tests and verification harnesses before implementing core strategy logic.
---

# Test-Driven Development (TDD) Protocol

1. Never write strategy code in `strategy.py` without first creating or verifying the test harness in `harness.py`.
2. Ensure the test harness verifies edge cases: missing data bars, extreme volatility spikes, and zero-division errors in position sizing.
3. Every implementation step must be followed by running the test suite via the bash tool.
