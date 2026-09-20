"""Tests for attention."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.attention import CausalSelfAttention, SlidingWindowAttention


def test_swa_shape():
    m = SlidingWindowAttention(64, 4, 16, window=8)
    x = torch.randn(2, 12, 64)
    y = m(x)
    assert y.shape == x.shape


def test_causal_shape():
    m = CausalSelfAttention(64, 4, 16)
    x = torch.randn(2, 10, 64)
    assert m(x).shape == x.shape


if __name__ == "__main__":
    test_swa_shape()
    test_causal_shape()
    print("ok")
