"""Router numerical verification against reference."""

from __future__ import annotations

import pytest
import torch

from pkg_import import import_from_pkg
from reference import max_mean_abs, reference_router_topk


def test_router_matches_reference_for_fixed_gate():
    rmod = import_from_pkg("DeepSeekFlashV4-Mini", "model.router")
    torch.manual_seed(0)
    r = rmod.Router(16, n_routed=4, n_activated=2, routed_scaling=1.5)
    flat = torch.randn(6, 16)
    with torch.no_grad():
        logits = r.gate(flat)
        ref_v, ref_i = reference_router_topk(logits, 2, routed_scaling=1.5)
        topv, topi = r(flat)
    assert torch.equal(topi, ref_i)
    mx, _ = max_mean_abs(topv, ref_v)
    assert mx < 1e-5


def test_router_topk1():
    rmod = import_from_pkg("DeepSeekFlashV4-Mini", "model.router")
    r = rmod.Router(8, n_routed=3, n_activated=1, routed_scaling=1.0)
    flat = torch.randn(4, 8)
    topv, topi = r(flat)
    assert topv.shape == (4, 1) and topi.shape == (4, 1)
    assert torch.allclose(topv, torch.ones_like(topv), atol=1e-5)


def test_router_equal_logits_normalized():
    rmod = import_from_pkg("DeepSeekFlashV4-Mini", "model.router")
    r = rmod.Router(4, n_routed=3, n_activated=2, routed_scaling=1.0)
    with torch.no_grad():
        r.gate.weight.zero_()
    flat = torch.zeros(2, 4)
    topv, topi = r(flat)
    assert torch.allclose(topv[:, 0], topv[:, 1], atol=1e-5)
    assert torch.allclose(topv.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_topk_gt_experts_raises_in_reference():
    with pytest.raises(ValueError):
        reference_router_topk(torch.randn(2, 3), top_k=5)


def test_hash_router_deterministic():
    rmod = import_from_pkg("DeepSeekFlashV4-Mini", "model.router")
    r = rmod.Router(8, n_routed=4, n_activated=2, vocab_size=10, use_hash=True)
    flat = torch.randn(3, 8)
    tokens = torch.tensor([[1, 2, 3]])
    v1, i1 = r(flat, tokens=tokens)
    v2, i2 = r(flat, tokens=tokens)
    assert torch.equal(i1, i2) and torch.equal(v1, v2)
