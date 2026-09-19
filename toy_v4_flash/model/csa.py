"""Compressed Sparse Attention (CSA) — V4-Flash §2.3.1 educational toy.

Paper: compress m=4 with overlapping windows + Lightning Indexer Top-K,
then attend selected compressed KV ∪ local SWA.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope
from .swa import SlidingWindowAttention


def compress_overlap(x: torch.Tensor, m: int) -> torch.Tensor:
    """
    Overlapping CSA-style compress: each entry pools ~2m tokens with stride m.
    Toy: mean of windows [i*m : i*m+2m].
    """
    B, T, D = x.shape
    if m <= 1:
        return x
    width = 2 * m
    pad = (m - T % m) % m
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    T2 = x.shape[1]
    outs = []
    for start in range(0, T2 - width + 1, m):
        outs.append(x[:, start : start + width].mean(dim=1))
    if not outs:
        return x.mean(dim=1, keepdim=True)
    return torch.stack(outs, dim=1)


class CSA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        compress: int = 4,
        topk: int = 4,
        swa_window: int = 8,
        use_rope: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.compress = compress
        self.topk = topk
        self.use_rope = use_rope and d_head % 2 == 0
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_head, bias=False)
        self.indexer_q = nn.Linear(d_model, d_head, bias=False)
        self.indexer_k = nn.Linear(d_head, d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.swa = SlidingWindowAttention(
            d_model, n_heads, d_head, swa_window, use_rope=use_rope
        )

    def forward(self, x: torch.Tensor, log: Optional[List[str]] = None) -> torch.Tensor:
        B, T, _ = x.shape
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k_c = compress_overlap(k, self.compress)
        v_c = compress_overlap(v, self.compress)
        # Lightning Indexer (toy): ReLU-ish scores → Top-K
        iq = self.indexer_q(x)
        ik = self.indexer_k(k_c)
        scores = F.relu(torch.einsum("btd,bnd->btn", iq, ik))
        kk = min(self.topk, scores.shape[-1])
        _, topk_idx = scores.topk(kk, dim=-1)

        k_e = k_c.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        v_e = v_c.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q)
        idx = topk_idx.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        k_sel = torch.gather(
            k_e.unsqueeze(2).expand(-1, -1, T, -1, -1),
            3,
            idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head),
        )
        v_sel = torch.gather(
            v_e.unsqueeze(2).expand(-1, -1, T, -1, -1),
            3,
            idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head),
        )
        att = (q.unsqueeze(-2) @ k_sel.transpose(-2, -1)).squeeze(-2) / math.sqrt(
            self.d_head
        )
        att = F.softmax(att, dim=-1)
        sparse = (att.unsqueeze(-2) @ v_sel).squeeze(-2)
        sparse = sparse.transpose(1, 2).contiguous().view(B, T, -1)
        sparse = self.out(sparse)
        local = self.swa(x)
        if log is not None:
            log.append(
                f"CSA m={self.compress} Nc={k_c.shape[1]} Top-{kk} + SWA"
            )
        return sparse + local
