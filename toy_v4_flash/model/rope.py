"""RoPE (educational)."""

from __future__ import annotations

import torch


def rope_freqs(
    d_head: int, seq_len: int, device: torch.device, base: float = 10000.0
) -> torch.Tensor:
    assert d_head % 2 == 0
    half = d_head // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor | None = None) -> torch.Tensor:
    B, H, T, D = x.shape
    if freqs is None:
        freqs = rope_freqs(D, T, x.device)
    x_ = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    out = x_ * freqs[:T].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(out).reshape(B, H, T, D).type_as(x)
