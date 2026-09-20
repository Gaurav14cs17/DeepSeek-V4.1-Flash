#!/usr/bin/env python3
"""Parameter count benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV4Mini, preset


def main() -> None:
    for name in ("demo", "pc", "better"):
        m = DeepSeekFlashV4Mini(preset(name))
        print(f"{name:8s} params={m.num_parameters():,}")


if __name__ == "__main__":
    main()
