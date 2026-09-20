"""Tests for MoE (V4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.moe import MoE


def test_moe():
    m = MoE(64, 8, 1, 2, 128)
    x = torch.randn(2, 10, 64)
    assert m(x).shape == x.shape


if __name__ == "__main__":
    test_moe()
    print("ok")
