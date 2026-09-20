"""Tests for attention (V4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.attention import SlidingWindowAttention


def test_swa():
    m = SlidingWindowAttention(64, 4, 16, 8)
    x = torch.randn(2, 12, 64)
    assert m(x).shape == x.shape


if __name__ == "__main__":
    test_swa()
    print("ok")
