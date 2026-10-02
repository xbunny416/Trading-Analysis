"""
Run a Stage-1 FAIL once on the locked 2023 holdout as a DIAGNOSTIC (cycle 5, at the user's request). Never a pass.

The harness unlocks the holdout only after a Stage-1 PASS. After 63 trials without one, the user asked to see whether
the best book survives new data. This script runs the harness's own Stage-2 machinery (same splits, gates, tests and
reports) with three differences:
  - it requires a logged Stage-1 trial of the same code fingerprint (any status) instead of a logged PASS;
  - its report always carries the failure "diagnostic: ...", so its status is FAIL whatever the holdout shows;
  - it is logged in holdout.log like any holdout look, so the holdout stays used for that code fingerprint.
harness.py is not changed, so the code fingerprint is the one the Stage-1 trial logged.

    python scripts/holdout_diagnostic.py C5-052
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as B  # noqa: E402
import harness as Hn  # noqa: E402
import hypotheses as H  # noqa: E402


def diagnostic_guard(hid: str, fp: str, trials: list[dict], holdouts: list[dict], cycle5=None) -> str | None:
    """None if the diagnostic may run, else the refusal reason."""
    if hid not in (H.CYCLE5 if cycle5 is None else cycle5):
        return f"{hid} is not a cycle-5 trial"
    same = [t for t in trials if t["hyp"] == hid and t["code"] == fp]
    if not same:
        return f"{hid} (code {fp}) has no logged Stage-1 trial with this code"
    if any(t["status"] == "PASS" for t in same):
        return f"{hid} passed Stage 1; use harness.py --holdout"
    if any(h["code"] == fp for h in holdouts):
        return f"the holdout was already used for code {fp}; it is evaluated exactly once"
    return None


def guard_tests() -> dict:
    t = [{"hyp": "C5-X", "code": "aaa", "status": "FAIL", "n": 1}, {"hyp": "C5-P", "code": "ppp", "status": "PASS", "n": 2}]
    cyc = ("C5-X", "C5-P", "C5-Y")
    cases = {
        "allowed": diagnostic_guard("C5-X", "aaa", t, [], cyc) is None,
        "other_code_refused": diagnostic_guard("C5-X", "bbb", t, [], cyc) is not None,
        "never_run_refused": diagnostic_guard("C5-Y", "yyy", t, [], cyc) is not None,
        "stage1_pass_refused": diagnostic_guard("C5-P", "ppp", t, [], cyc) is not None,
        "second_look_refused": diagnostic_guard("C5-X", "aaa", t, [{"code": "aaa"}], cyc) is not None,
        "archived_refused": diagnostic_guard("A3", "aaa", t, [], cyc) is not None,
    }
    return {"passed": all(cases.values()), **cases}


def main(hid: str) -> int:
    g = guard_tests()
    if not g["passed"]:
        print(f"REFUSED: guard self-test failed {g}", file=sys.stderr)
        return 1
    t_start = time.time()
    Hn.use_hypothesis(hid)
    fp = Hn.code_fingerprint(hid)
    data_dir = B.DATA_DIR.resolve()
    trials = Hn.logged_trials()
    reason = diagnostic_guard(hid, fp, trials, Hn.logged_holdouts())
    if reason:
        print(f"REFUSED: {reason}", file=sys.stderr)
        return 1
    stage1 = [t for t in trials if t["hyp"] == hid and t["code"] == fp][-1]

    failures = [f"diagnostic: Stage 1 (trial {stage1['n']}) did not pass; this holdout look was requested as a "
                f"diagnostic and can never count as a pass"]
    m15, rates = Hn.load_market(B.PAIRS, data_dir, None, True)
    provenance = Hn.data_provenance(data_dir=data_dir, m15=m15)
    data_errors, data_info = Hn.validate_data(m15)
    failures += [f"data: {e}" for e in data_errors]
    market = B.Market(m15, rates=rates)
    tests = Hn.selftest()
    tests["diagnostic_guard"] = g
    splits = Hn.holdout_wfa_splits(market.index)
    with Hn._pool(market.pairs, data_dir, None, True) as pool:
        wfa = Hn.run_wfa(market, pool, splits)
    tests["no_order_desyncs"] = {"passed": wfa["oos_desyncs"] == 0 and wfa["is_desyncs"] == 0,
                                 "oos": wfa["oos_desyncs"], "is": wfa["is_desyncs"]}
    failures += Hn._gate_failures(wfa)
    failures += [f"integrity: {n} failed" for n in Hn._failed_names(tests)]
    trial = {"code_fingerprint": fp, "stage1_trial": stage1["n"], "stage1_status": stage1["status"],
             "diagnostic": True, "counts_as_trial": True, "data_dir": str(data_dir)}
    period = (f"holdout {Hn.pd.Timestamp(splits[0]['oos'][0], tz='UTC'):%Y-%m-%d} -> {market.index[-1]:%Y-%m-%d} "
              f"(DIAGNOSTIC)")
    report = Hn._report(2, "FAIL", failures, trial, provenance, data_info, period, wfa, tests, True, t_start)
    Hn.append_entry(Hn.HOLDOUT_LOG, "HOLDOUT", len(Hn.logged_holdouts()) + 1, fp, report)
    return Hn._finish(report, f"{hid}_holdout", wfa, market)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
