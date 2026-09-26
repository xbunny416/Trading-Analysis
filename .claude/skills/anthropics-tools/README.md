# Anthropic skill references

Copied from [anthropics/skills](https://github.com/anthropics/skills) at commit
[`3337550`](https://github.com/anthropics/skills/tree/33375500bcea98d610eb30ce10ac4e59b89c390d).
Files are byte-identical to upstream.

| Folder | Upstream path | License |
|---|---|---|
| `webapp-testing/` | `skills/webapp-testing/` | Apache 2.0, see `webapp-testing/LICENSE.txt` |
| `template-skill/` | `template/` | No license file upstream (two-line skeleton) |

## These skills are not loaded automatically

Claude Code discovers project skills only at `.claude/skills/<name>/SKILL.md`.
Folders inside `anthropics-tools/` are one level too deep, so they act as
reference copies. To activate one, copy its folder up a level:

```bash
cp -r .claude/skills/anthropics-tools/webapp-testing .claude/skills/webapp-testing
```

To start a new skill, copy `template-skill/` to `.claude/skills/<new-name>/`
and fill in `name`, `description` and the instructions.

## Not copied

- `xlsx`, `pdf`, `docx`, `pptx`: their `LICENSE.txt` forbids reproducing or
  redistributing the files, so they cannot be committed here. The same skills
  are available through the `document-skills` plugin
  (`/plugin install document-skills@anthropic-agent-skills`).
- `skill-creator` (evals and benchmark analysis for skills) is available through
  the `example-skills` plugin (`/plugin install example-skills@anthropic-agent-skills`).
