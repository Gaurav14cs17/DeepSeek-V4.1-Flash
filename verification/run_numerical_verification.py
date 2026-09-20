#!/usr/bin/env python3
"""Numerical / mathematical verification suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent / "tests"

NUM = [
    "test_rope.py",
    "test_attention_causality.py",
    "test_numerics.py",
    "test_router.py",
    "test_quantization.py",
    "test_moe.py",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *[str(TESTS / t) for t in NUM]]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
