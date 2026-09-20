#!/usr/bin/env python3
"""Expert utilization histogram."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV4Mini, preset


def main() -> None:
    m = DeepSeekFlashV4Mini(preset("pc"))
    m.train()
    x = torch.randint(0, m.cfg.vocab_size, (4, 32))
    log: list[str] = []
    m(x, log=log, trace=True)
    for line in log:
        if "MoE" in line:
            print(line)


if __name__ == "__main__":
    main()
