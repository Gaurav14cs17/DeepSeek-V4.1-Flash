#!/usr/bin/env python3
"""Throughput benchmark (tokens/s)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV4Mini, preset


def main() -> None:
    m = DeepSeekFlashV4Mini(preset("pc"))
    m.eval()
    B, T = 8, 64
    x = torch.randint(0, m.cfg.vocab_size, (B, T))
    for _ in range(3):
        m(x, trace=False)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        m(x, trace=False)
    toks = B * T * n
    tps = toks / (time.perf_counter() - t0)
    print(f"throughput ~{tps:,.0f} tokens/s")


if __name__ == "__main__":
    main()
