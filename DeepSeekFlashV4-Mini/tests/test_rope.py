"""RoPE unit tests — Stage 1 gate.

Checks:
1. output shape
2. position-0 behavior (identity rotation at t=0 for the complex multiply form)
3. deterministic behavior
4. gradient propagation
5. comparison with rotate-half reference
6. even-dim requirement / dtype preservation
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.rope import apply_rope, rope_freqs, rope_reference_rotate_half


def test_output_shape():
    x = torch.randn(2, 4, 8, 16)
    y = apply_rope(x)
    assert y.shape == x.shape


def test_position_zero_is_identity():
    # At t=0, angles are 0 → cis(0)=1 → y == x (up to float noise)
    x = torch.randn(1, 2, 1, 8)
    y = apply_rope(x)
    assert torch.allclose(y, x, atol=1e-5)


def test_deterministic():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 4, 8)
    y1 = apply_rope(x)
    y2 = apply_rope(x)
    assert torch.equal(y1, y2)


def test_gradient_propagation():
    x = torch.randn(2, 2, 4, 8, requires_grad=True)
    y = apply_rope(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_matches_rotate_half_reference():
    B, H, T, D = 1, 2, 5, 8
    x = torch.randn(B, H, T, D)
    freqs = rope_freqs(D, T, x.device)
    cos = freqs.real
    sin = freqs.imag
    y_ref = rope_reference_rotate_half(x, cos, sin)
    y = apply_rope(x, freqs=freqs)
    assert torch.allclose(y, y_ref, atol=1e-5)


def test_dtype_preserved():
    x = torch.randn(1, 1, 4, 8, dtype=torch.float32)
    assert apply_rope(x).dtype == torch.float32


def test_even_dim_required():
    x = torch.randn(1, 1, 2, 7)
    try:
        apply_rope(x)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_norm_approximately_preserved():
    # RoPE is orthonormal per pair → L2 norm per token/head preserved
    x = torch.randn(2, 3, 6, 16)
    y = apply_rope(x)
    xn = x.float().norm(dim=-1)
    yn = y.float().norm(dim=-1)
    assert torch.allclose(xn, yn, atol=1e-4)


if __name__ == "__main__":
    test_output_shape()
    test_position_zero_is_identity()
    test_deterministic()
    test_gradient_propagation()
    test_matches_rotate_half_reference()
    test_dtype_preserved()
    test_even_dim_required()
    test_norm_approximately_preserved()
    print("test_rope: ALL PASSED")
