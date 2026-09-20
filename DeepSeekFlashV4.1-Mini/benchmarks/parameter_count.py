#!/usr/bin/env python3
"""Parameter count (V4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV41Mini, preset


def main() -> None:
    for name in ("demo", "pc", "better"):
        cfg = preset(name)
        cfg.use_vision = False
        m = DeepSeekFlashV41Mini(cfg)
        print(f"{name:8s} params={m.num_parameters():,}")


if __name__ == "__main__":
    main()
