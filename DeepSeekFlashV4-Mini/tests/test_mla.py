"""Tests for CSA / HCA / MLA."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.mla import CSA, HCA, MLA


def test_csa():
    m = CSA(64, 4, 16, compress=4, topk=4, swa_window=8)
    x = torch.randn(2, 16, 64)
    assert m(x).shape == x.shape


def test_hca():
    m = HCA(64, 4, 16, compress=8, swa_window=8)
    x = torch.randn(2, 16, 64)
    assert m(x).shape == x.shape


def test_mla():
    m = MLA(64, 4, 16, kv_lora_rank=16)
    x = torch.randn(2, 12, 64)
    assert m(x).shape == x.shape


if __name__ == "__main__":
    test_csa()
    test_hca()
    test_mla()
    print("ok")
