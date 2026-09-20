#!/usr/bin/env python3
"""Thin wrapper: python -m DeepSeekFlashV4-Mini.benchmarks is awkward; run via path."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.regression import main

if __name__ == "__main__":
    main()
