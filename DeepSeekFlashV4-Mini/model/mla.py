"""Compressed / latent attention family used by V4-Flash (CSA + HCA).

Also includes a tiny Multi-head Latent Attention (MLA) block for labs that
compare classic DeepSeek-V2 MLA with V4's CSA–HCA hybrid.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SlidingWindowAttention
from .rope import apply_rope


def compress_overlap(x: torch.Tensor, m: int) -> torch.Tensor:
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


def compress_nonoverlap(x: torch.Tensor, m: int) -> torch.Tensor:
    B, T, D = x.shape
    if m <= 1:
        return x
    pad = (m - T % m) % m
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    T2 = x.shape[1]
    return x.view(B, T2 // m, m, D).mean(dim=2)


class CSA(nn.Module):
    """Compressed Sparse Attention (V4-Flash §2.3.1)."""

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
            log.append(f"CSA m={self.compress} Nc={k_c.shape[1]} Top-{kk} + SWA")
        return sparse + local


class HCA(nn.Module):
    """Heavily Compressed Attention (V4-Flash §2.3.2)."""

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
        scores = (q @ k_e.transpose(-2, -1)) / math.sqrt(self.d_head)
        q_pos = torch.arange(T, device=x.device)
        allow_p = (q_pos + 1) // max(self.compress, 1)
        p_idx = torch.arange(Nc, device=x.device)
        causal = p_idx[None, :] < allow_p[:, None].clamp_min(1)
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        finite = torch.isfinite(scores).any(dim=-1, keepdim=True)
        scores = torch.where(finite, scores, torch.zeros_like(scores))
        att = F.softmax(scores, dim=-1)
        sparse = (att @ v_e).transpose(1, 2).contiguous().view(B, T, -1)
        sparse = self.out(sparse)
        local = self.swa(x)
        if log is not None:
            log.append(f"HCA m'={self.compress} Nc={Nc} dense-comp + SWA")
        return sparse + local


class MLA(nn.Module):
    """Tiny Multi-head Latent Attention (DeepSeek-V2 style, educational)."""

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
