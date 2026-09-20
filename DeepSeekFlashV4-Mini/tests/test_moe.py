"""Tests for MoE."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.moe import MoE


def test_moe_forward():
    m = MoE(64, n_routed=8, n_shared=1, n_activated=2, hidden=128, vocab_size=32)
    x = torch.randn(2, 10, 64)
    tokens = torch.randint(0, 32, (2, 10))
    y = m(x, tokens=tokens)
    assert y.shape == x.shape


def test_moe_hash():
    m = MoE(
        64, n_routed=8, n_shared=1, n_activated=2, hidden=128, vocab_size=32, use_hash=True
    )
    x = torch.randn(2, 10, 64)
    tokens = torch.randint(0, 32, (2, 10))
    assert m(x, tokens=tokens).shape == x.shape


if __name__ == "__main__":
    test_moe_forward()
    test_moe_hash()
    print("ok")
