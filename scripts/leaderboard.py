#!/usr/bin/env python3
"""Print the cycle-5 leaderboard (markdown) from results/cycle5/*.json, in trial order."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hypotheses as H  # noqa: E402

rows = []
for hid, (_, family, _) in H.CYCLE5_REGISTRY.items():
    f = ROOT / "results" / "cycle5" / f"{hid}.json"
    if not f.exists():
        rows.append(f"| {hid} | {family} | registered, not run yet | | | | | |")
        continue
    d = json.loads(f.read_text())
    m = d["metrics"]
    ds = m.get("oos_deflated_sharpe") or {}
    gates = [x[1] for x in d["failures"] if x.startswith("(")]
    if any(not x.startswith("(") for x in d["failures"]):
        gates.append("integrity")
    wfe = "n/a" if m["wfe"] is None else f"{m['wfe']:.2f}"
    rows.append(f"| {hid} | {family} | {m['oos_sharpe']:.2f} | {wfe} | {m['oos_max_drawdown']:.1%} | "
                f"{m['oos_trades_per_year']:.0f} | {ds.get('dsr')} ({ds.get('n_trials')}) | "
                f"{', '.join(gates) if gates else 'none: PASS'} |")
print("| Trial | Family | OOS Sharpe | WFE | max DD | fills/yr | DSR (N) | Failed gates |")
print("|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
