#!/usr/bin/env python3
"""MoE-focused training entry (same loop, logs expert load)."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from training.pretrain import main

if __name__ == "__main__":
    main()
