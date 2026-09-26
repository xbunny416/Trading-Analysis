# Upstream source

Vendored from [obra/superpowers](https://github.com/obra/superpowers), directory
`skills/systematic-debugging/`, commit
[`8ca22db`](https://github.com/obra/superpowers/tree/8ca22dba9a94f28898bbce59f2537ff4d87c747d/skills/systematic-debugging).

- License: MIT, copyright (c) 2025 Jesse Vincent. See `LICENSE` (copied from the upstream repository root).
- Files other than `LICENSE` and this note are byte-identical to upstream.
  Keep local changes out of them so an update is a clean re-copy.

## Local notes

- Installed in `debugging/`; the upstream frontmatter `name` is `systematic-debugging`.
- `SKILL.md` refers to `superpowers:test-driven-development`; in this
  repository that skill lives in `.claude/skills/tdd/`. It also refers to
  `superpowers:verification-before-completion`, which is not installed; follow
  the Verification section of the root `CLAUDE.md` instead.
- `find-polluter.sh` runs `npm test <file>`. Swap in the Python test command
  before using it here.
- `test-academic.md`, `test-pressure-*.md` and `CREATION-LOG.md` are the
  upstream author's scenarios for testing the skill itself. Claude does not
  load them unless it opens them.
