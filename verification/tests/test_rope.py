"""RoPE numerical verification vs reference."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg
from reference import cosine_sim, max_mean_abs, reference_rope


def test_rope_vs_reference_max_error():
    rope = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 17, 16)
    y = rope.apply_rope(x)
    y_ref = reference_rope(x)
    mx, mn = max_mean_abs(y, y_ref)
    assert mx < 1e-5, f"max_abs_error={mx}"
    assert mn < 1e-6
    assert cosine_sim(y, y_ref) > 0.999999


def test_rope_position_zero_identity():
    rope = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    x = torch.randn(1, 2, 1, 8)
    assert torch.allclose(rope.apply_rope(x), x, atol=1e-5)


def test_rope_long_positions():
    rope = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    x = torch.randn(1, 1, 256, 16)
    y = rope.apply_rope(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(x.float().norm(dim=-1), y.float().norm(dim=-1), atol=1e-4)


def test_rope_batch_broadcast():
    rope = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    x = torch.randn(4, 8, 9, 32)
    freqs = rope.rope_freqs(32, 9, x.device)
    y = rope.apply_rope(x, freqs=freqs)
    assert y.shape == x.shape


def test_rope_gradient_finite():
    rope = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    x = torch.randn(2, 2, 5, 8, requires_grad=True)
    rope.apply_rope(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
