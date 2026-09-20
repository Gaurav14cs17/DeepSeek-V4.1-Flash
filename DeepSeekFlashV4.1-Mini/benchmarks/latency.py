#!/usr/bin/env python3
"""Latency benchmark (CPU)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV41Mini, preset


def main() -> None:
    cfg = preset("pc"); cfg.use_vision=False
m = DeepSeekFlashV41Mini(cfg)
    m.eval()
    x = torch.randint(0, m.cfg.vocab_size, (4, 64))
    # warmup
    for _ in range(3):
        m(x, trace=False)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        m(x, trace=False)
    ms = (time.perf_counter() - t0) * 1000 / n
    print(f"forward latency ~{ms:.2f} ms / batch")


if __name__ == "__main__":
    main()
