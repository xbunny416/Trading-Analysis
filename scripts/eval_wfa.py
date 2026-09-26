#!/usr/bin/env python3
"""Entry point named by .claude/skills/quant-wfa: runs the walk-forward harness (same flags, same exit code)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
