# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Personal workspace for writing and backtesting trading strategies. It holds a
NautilusTrader walk-forward harness for FX trend-following research (see
`README.md`) plus Claude Code configuration (`.claude/skills/`).

## Commands

| Purpose | Command or file |
|---|---|
| Install | `pip install -r requirements.txt` (NautilusTrader 1.221, Python 3.11+) |
| Unit / integrity tests (synthetic data, not a trial) | `python harness.py --selftest` |
| Walk-forward evaluation (logs a trial) | `python harness.py` or `python scripts/eval_wfa.py` |
| Evaluate other data (never logged) | `python harness.py --data-dir <folder of <PAIR>.csv>` |
| Nautilus data / engine plumbing | `backtest.py` |
| Strategy logic | `strategy.py` |

Real-data runs need the MT5 exports in `data/mt5/<PAIR>.csv` (git-ignored). Every
completed real-data run with changed code counts against the 8-trial budget in
`trials.log`; iterate with `--selftest` first.

When you add a dependency file or test runner, record the exact install and test
commands here, so that "run the tests" works on the first try.

## Skills

Project skills live in `.claude/skills/<name>/SKILL.md`.

| Directory | Skill | Use when |
|---|---|---|
| `quant-wfa/` | `quant-wfa-validator` | Writing or changing strategy logic |
| `tdd-enforcer/` | `tdd-enforcer` | Before touching `strategy.py` |
| `tdd/` | `test-driven-development` | Implementing any feature or bugfix |
| `debugging/` | `systematic-debugging` | Any bug, test failure or unexpected backtest result, before proposing a fix |

- `quant-wfa` and `tdd-enforcer` are specific to this repository. Where they are
  stricter than the general-purpose `tdd` and `debugging` skills, they win.
- `tdd/` and `debugging/` are vendored from obra/superpowers without changes
  (see each folder's `UPSTREAM.md`). Their examples use TypeScript and `npm`;
  apply them with the Python equivalents. A reference to
  `superpowers:test-driven-development` means `tdd/`.
  `superpowers:verification-before-completion` is not installed; follow
  [Verification](#verification) instead.
- `anthropics-tools/` holds reference copies of Anthropic's `webapp-testing`
  skill and the blank skill template. They are one level too deep to load
  automatically; see `anthropics-tools/README.md` to activate one.

<important if="you are writing or modifying trading strategy, signal, indicator or backtest code">

## Strategy code

- Follow `quant-wfa-validator` for lookahead, parameter-count and walk-forward
  rules. Its thresholds are the definition of done for strategy changes.
- Follow `tdd-enforcer`: the harness and its edge cases come before strategy code.

</important>

## Workflow

- Start multi-file or new-strategy work in plan mode. Split the plan into
  phases, each gated by tests that must pass before the next phase starts.
- Send broad codebase searches to a subagent so the main context keeps only
  the conclusions.
- Compact manually with a focus hint (for example `/compact focus on the WFA
  refactor`) before the context gets crowded, rather than waiting for
  auto-compact. Start a new session for an unrelated task.
- If you run the same workflow more than once a day, turn it into a skill in
  `.claude/skills/`.

## Verification

- Do not call a task done until you have run the relevant checks and watched
  them pass. Run the whole test suite, not only the test you touched.
- Report every failure by name with its output, including ones you did not cause.
- If no automated check exists for a change, say so and describe how you
  verified it.

## Maintaining this configuration

- Keep this file under 200 lines. Move instructions that apply only to some
  paths into `.claude/rules/*.md` with a `paths:` frontmatter glob, so they load
  only when matching files are touched.
- Put behavior the harness can enforce (permissions, hooks, model, commit
  attribution) in `.claude/settings.json`, not in prose here.
- Write a skill's `description` as a trigger ("Use when …"), not a summary.
- Do not edit vendored skill files. Update them by re-copying from upstream at
  a newer commit and updating `UPSTREAM.md`.
