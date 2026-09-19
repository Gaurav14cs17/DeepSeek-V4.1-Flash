"""Heavily Compressed Attention (HCA) — V4-Flash §2.3.2 educational toy.

Paper: aggressive non-overlapping compress (m'=128), dense attention over
compressed entries (no indexer) ∪ local SWA.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope
from .swa import SlidingWindowAttention


def compress_nonoverlap(x: torch.Tensor, m: int) -> torch.Tensor:
    B, T, D = x.shape
    if m <= 1:
        return x
    pad = (m - T % m) % m
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    T2 = x.shape[1]
    return x.view(B, T2 // m, m, D).mean(dim=2)


class HCA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        compress: int = 16,
        swa_window: int = 8,
        use_rope: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.compress = compress
        self.use_rope = use_rope and d_head % 2 == 0
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.swa = SlidingWindowAttention(
            d_model, n_heads, d_head, swa_window, use_rope=use_rope
        )

    def forward(self, x: torch.Tensor, log: Optional[List[str]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k_c = compress_nonoverlap(k, self.compress)
        v_c = compress_nonoverlap(v, self.compress)
        Nc = k_c.shape[1]

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k_e = k_c.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        v_e = v_c.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        if self.use_rope:
            q = apply_rope(q)
        # Dense over compressed positions with causal: p < (q+1)//m
        scores = (q @ k_e.transpose(-2, -1)) / math.sqrt(self.d_head)
        q_pos = torch.arange(T, device=x.device)
        allow_p = (q_pos + 1) // max(self.compress, 1)  # [T]
        p_idx = torch.arange(Nc, device=x.device)
        causal = p_idx[None, :] < allow_p[:, None].clamp_min(1)
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        # rows with all -inf → zeros
        finite = torch.isfinite(scores).any(dim=-1, keepdim=True)
        scores = torch.where(finite, scores, torch.zeros_like(scores))
        att = F.softmax(scores, dim=-1)
        sparse = (att @ v_e).transpose(1, 2).contiguous().view(B, T, -1)
        sparse = self.out(sparse)
        local = self.swa(x)
        if log is not None:
            log.append(f"HCA m'={self.compress} Nc={Nc} dense-comp + SWA")
        return sparse + local
