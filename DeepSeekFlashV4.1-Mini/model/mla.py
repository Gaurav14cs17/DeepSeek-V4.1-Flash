"""CSA2 (V4.1 compressed sparse attention) + optional classic MLA."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._csa2 import CSA2, SharedGlobalState, compress_tokens
from .rope import apply_rope


class MLA(nn.Module):
    """Tiny Multi-head Latent Attention for comparison labs."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        kv_lora_rank: int = 16,
        use_rope: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.use_rope = use_rope and d_head % 2 == 0
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.kv_down = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.k_up = nn.Linear(kv_lora_rank, n_heads * d_head, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, log: Optional[List[str]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        c_kv = self.kv_down(x)
        k = self.k_up(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        att = F.softmax(scores, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        if log is not None:
            log.append(f"MLA latent_rank={c_kv.shape[-1]}")
        return self.out(out)


__all__ = ["CSA2", "SharedGlobalState", "compress_tokens", "MLA"]
