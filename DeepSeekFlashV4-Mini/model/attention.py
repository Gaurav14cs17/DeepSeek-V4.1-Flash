"""Sliding-window + causal multi-head attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class SlidingWindowAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_head: int, window: int, use_rope: bool = True
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window = window
        self.use_rope = use_rope and d_head % 2 == 0
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        idx = torch.arange(T, device=x.device)
        allow = (idx[None, :] <= idx[:, None]) & (
            idx[:, None] - idx[None, :] < self.window
        )
        scores = scores.masked_fill(~allow[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)


class CausalSelfAttention(nn.Module):
    """Full causal MHA (baseline for experiments)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, use_rope: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.use_rope = use_rope and d_head % 2 == 0
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)
