"""Tests for router."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.router import Router


def test_router_topk():
    r = Router(64, n_routed=8, n_activated=2)
    flat = torch.randn(20, 64)
    topv, topi = r(flat)
    assert topv.shape == (20, 2)
    assert topi.shape == (20, 2)
    assert topi.max() < 8


if __name__ == "__main__":
    test_router_topk()
    print("ok")
