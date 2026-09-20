"""MoE / shared-expert verification."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg


def test_shared_experts_always_active():
    se_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.shared_experts")
    se = se_mod.SharedExperts(32, n_shared=2, hidden=64)
    x = torch.randn(5, 32)
    y = se(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, torch.zeros_like(y))


def test_moe_shape_and_finite():
    moe_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.moe")
    m = moe_mod.MoE(32, n_routed=4, n_shared=1, n_activated=2, hidden=64, vocab_size=16)
    x = torch.randn(2, 8, 32)
    tokens = torch.randint(0, 16, (2, 8))
    y = m(x, tokens=tokens)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_moe_active_params_less_than_total():
    moe_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.moe")
    m = moe_mod.MoE(32, n_routed=8, n_shared=1, n_activated=2, hidden=64)
    total = sum(p.numel() for p in m.parameters())
    shared = sum(p.numel() for p in m.shared.parameters())
    one_expert = sum(p.numel() for p in m.routed[0].parameters())
    active = shared + m.n_activated * one_expert
    assert active < total
    assert active / total < 0.5


def test_expert_gradient_flows():
    exp = import_from_pkg("DeepSeekFlashV4-Mini", "model.experts")
    e = exp.Expert(16, 32)
    x = torch.randn(4, 16, requires_grad=True)
    e(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert e.w1.weight.grad is not None
