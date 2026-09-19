"""Rotary Position Embeddings (RoPE). Paper: RoPE on LLM; 2D-RoPE on DeepSeek-ViT."""

from __future__ import annotations

import torch


def rope_freqs(
    d_head: int, seq_len: int, device: torch.device, base: float = 10000.0
) -> torch.Tensor:
    """Complex cis frequencies [T, d_head/2]."""
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    half = d_head // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    return torch.polar(torch.ones_like(torch.outer(t, inv)), torch.outer(t, inv))


def apply_rope(x: torch.Tensor, freqs: torch.Tensor | None = None) -> torch.Tensor:
    """
    x: [B, H, T, D] with D even.
    freqs: [T, D/2] complex, or built from T if None.
    """
    B, H, T, D = x.shape
    if freqs is None:
        freqs = rope_freqs(D, T, x.device)
    x_ = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    out = x_ * freqs[:T].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(out).reshape(B, H, T, D).type_as(x)


def apply_rope_2d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Educational 2D-RoPE: token grid [B,H,h*w,D] — height freqs on first half dims,
    width freqs on second half, then 1D-style multiply.
    """
    B, nH, N, D = x.shape
    assert N == height * width and D % 4 == 0
    half = D // 2
    xh = x[..., :half]
    xw = x[..., half:]
    fh = rope_freqs(half, height, x.device)
    fw = rope_freqs(half, width, x.device)
    pos = torch.arange(N, device=x.device)
    row = pos // width
    col = pos % width
    return torch.cat(
        [apply_rope(xh, fh[row]), apply_rope(xw, fw[col])], dim=-1
    )
