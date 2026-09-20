"""Tests for RoPE (V4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.rope import apply_rope


def test_rope_shape():
    x = torch.randn(2, 4, 8, 16)
    assert apply_rope(x).shape == x.shape


if __name__ == "__main__":
    test_rope_shape()
    print("ok")
