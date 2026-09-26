---
name: quant-wfa-validator
description: Enforces institutional robustness, walk-forward analysis (WFA), and anti-overfitting rules when developing trading strategies.
---

# Quantitative Strategy Protocol

Whenever writing or modifying trading strategy logic in this repository, follow these rules:

### 1. Zero Lookahead Leakage
- All indicators and alpha factors MUST use `.shift(1)` or equivalent prior-bar access.
- Signals evaluated at bar `t-1` execute on the open of bar `t`.
- Never access high, low, or close of bar `t` to generate signals for bar `t`.

### 2. Complexity & Parameter Limits
- Total tunable parameters per strategy must not exceed 4.
- Reject arbitrary calendar filters (no "day != Friday" or "hour != 14").
- Do not stack technical indicators without distinct economic drivers.

### 3. Verification Commands
- Always execute: `python scripts/eval_wfa.py`
- Do NOT declare the task complete until:
  1. Walk-Forward Efficiency (OOS Sharpe / IS Sharpe) >= 0.60
  2. Perturbation leakage check exits with status code 0
  3. Max Drawdown < 12% across out-of-sample segments
