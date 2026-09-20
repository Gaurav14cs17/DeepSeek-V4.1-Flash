#!/usr/bin/env python3
"""Fast verification subset for development loops."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent / "tests"

FAST = [
    "test_config.py",
    "test_tokenizer.py",
    "test_rope.py",
    "test_attention_causality.py",
    "test_shapes.py",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *[str(TESTS / t) for t in FAST]]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
