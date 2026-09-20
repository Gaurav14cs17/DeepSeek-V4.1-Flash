"""Tests for CSA2 / MLA."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.mla import CSA2, MLA, SharedGlobalState


def test_csa2():
    m = CSA2(64, 4, 16, topk=4, compress=2, mode="full")
    x = torch.randn(2, 16, 64)
    y, shared = m(x, SharedGlobalState())
    assert y.shape == x.shape


def test_mla():
    m = MLA(64, 4, 16)
    x = torch.randn(2, 12, 64)
    assert m(x).shape == x.shape


if __name__ == "__main__":
    test_csa2()
    test_mla()
    print("ok")
