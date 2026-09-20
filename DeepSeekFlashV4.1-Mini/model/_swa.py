"""Sliding-Window Attention — local branch (+ RoPE)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        window: int,
        use_rope: bool = True,
        use_fp8_kv: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window = window
        self.use_rope = use_rope and (d_head % 2 == 0)
        self.use_fp8_kv = use_fp8_kv
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def _fp8_sim(self, t: torch.Tensor) -> torch.Tensor:
        """Paper keeps SWA KV in FP8 — toy E4M3-ish round."""
        if not self.use_fp8_kv:
            return t
        scale = t.detach().abs().amax().clamp_min(1e-6) / 448.0
        q = torch.round(t / scale).clamp(-448, 448)
        return q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)
        k = self._fp8_sim(k)
        v = self._fp8_sim(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        idx = torch.arange(T, device=x.device)
        allow = (idx[None, :] <= idx[:, None]) & (
            idx[:, None] - idx[None, :] < self.window
        )
        scores = scores.masked_fill(~allow[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)
