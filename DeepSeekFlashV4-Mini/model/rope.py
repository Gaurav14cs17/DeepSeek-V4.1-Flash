"""Rotary Position Embedding (RoPE).

[FAITHFUL] Standard RoPE (Su et al.) as used in DeepSeek attention paths.
Paper note (V4.1 report): RoPE preserves vector norm; KV quantization is applied
*after* RoPE (quantize-before-RoPE yields only marginal gains).

Math (per head dimension pairs)
-------------------------------
For dimension pair (2i, 2i+1) at position t with base θ:

    θ_i = base^{−2i/d}
    R_t x = rotate pairs of x by angle (t · θ_i)

Complex form used here:

    x_c = view_as_complex(x)          # [B, H, T, d/2]
    freqs_t = exp(i · t · θ)          # [T, d/2]
    y_c = x_c * freqs_t
    y = view_as_real(y_c)

Shapes
------
x:     [B, n_heads, T, d_head] with d_head even
freqs: [T, d_head/2] complex
out:   same as x

Memory: freqs are O(T · d_head/2); no KV growth from RoPE itself.
"""

from __future__ import annotations

import torch


def rope_freqs(
    d_head: int,
    seq_len: int,
    device: torch.device,
    base: float = 10000.0,
) -> torch.Tensor:
    """Build complex RoPE frequencies for positions 0..seq_len-1."""
    assert d_head % 2 == 0, "RoPE requires even head dimension"
    half = d_head // 2
    # θ_i = base^{-2i/d}  →  inv_freq[i] = base^{-i/half}
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    # outer: [T, half] angles
    angles = torch.outer(t, inv_freq)
    # cis(θ) = cos θ + i sin θ
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor | None = None,
    position_offset: int = 0,
) -> torch.Tensor:
    """
    Apply RoPE to q or k.

    x: [B, H, T, D]
    freqs: optional precomputed [S, D/2] complex covering positions used
    position_offset: start position index when decoding with cache (stage later)
    """
    B, H, T, D = x.shape
    if freqs is None:
        freqs = rope_freqs(D, position_offset + T, x.device)
        freqs = freqs[position_offset : position_offset + T]
    else:
        freqs = freqs[:T]

    # Pair last dim into complex numbers: [B,H,T,D/2]
    x_f = x.float().reshape(B, H, T, D // 2, 2)
    x_c = torch.view_as_complex(x_f)
    # Broadcast freqs [T, D/2] → [1,1,T,D/2]
    y_c = x_c * freqs.unsqueeze(0).unsqueeze(0)
    y = torch.view_as_real(y_c).reshape(B, H, T, D)
    return y.type_as(x)


def rope_reference_rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Reference rotate-half formulation (for tests).

    x: [B,H,T,D]; cos/sin: [T, D/2] broadcastable to pairs.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # cos/sin [T, D/2] → [1,1,T,D/2]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    out = torch.stack((y1, y2), dim=-1).flatten(-2)
    return out
