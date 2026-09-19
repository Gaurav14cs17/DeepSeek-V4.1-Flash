"""Compressed Sparse Attention 2 (CSA2) — Full / Reindex / Reuse + hierarchical pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CSA2Mode
from .rope import apply_rope


@dataclass
class SharedGlobalState:
    """Cross-layer reusable global KV + indexer K + Top-K + hierarchical pool."""

    main_kv: Optional[torch.Tensor] = None
    indexer_k: Optional[torch.Tensor] = None
    topk_idx: Optional[torch.Tensor] = None
    candidate_pool: Optional[torch.Tensor] = None


def compress_tokens(x: torch.Tensor, m: int) -> torch.Tensor:
    B, T, D = x.shape
    if m <= 1:
        return x
    pad = (m - T % m) % m
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    T2 = x.shape[1]
    return x.view(B, T2 // m, m, D).mean(dim=2)


class CSA2(nn.Module):
    """
    Modes (paper §2.3.1):
      Full    — main KV + indexer + Top-K (+ build hierarchical candidate pool)
      Reindex — reuse KV/indexer K; fresh Top-K, optionally restricted to pool
      Reuse   — reuse main KV + Top-K
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        topk: int,
        compress: int,
        mode: CSA2Mode,
        candidate_pool: int = 0,
        use_fp4: bool = True,
        swa_window: int = 8,
        use_rope: bool = True,
        use_fp8_swa: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.topk = topk
        self.compress = compress
        self.mode = mode
        self.candidate_pool = candidate_pool
        self.use_fp4 = use_fp4
        self.use_rope = use_rope and (d_head % 2 == 0)

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_head, bias=False)
        self.indexer_q = nn.Linear(d_model, d_head, bias=False)
        self.indexer_k_from_main = nn.Linear(d_head, d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

        from .swa import SlidingWindowAttention

        self.swa = SlidingWindowAttention(
            d_model, n_heads, d_head, window=swa_window, use_rope=use_rope, use_fp8_kv=use_fp8_swa
        )

    def _quantize_fp4_sim(self, t: torch.Tensor) -> torch.Tensor:
        """MXFP4-ish: E2M1 levels with per-16-channel scale (paper §2.4.4)."""
        if not self.use_fp4:
            return t
        # group along last dim into blocks of 16
        *lead, D = t.shape
        g = 16
        pad = (g - D % g) % g
        x = F.pad(t, (0, pad)) if pad else t
        x = x.view(*lead, -1, g)
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 6.0
        q = torch.round(x / scale).clamp(-6, 6) * scale
        q = q.view(*lead, -1)
        return q[..., :D] if pad else q

    def _build_main_kv(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv = self.kv_proj(h)
        k, v = kv.chunk(2, dim=-1)
        k = compress_tokens(k, self.compress)
        v = compress_tokens(v, self.compress)
        k = self._quantize_fp4_sim(k)
        v = self._quantize_fp4_sim(v)
        main_kv = torch.stack([k, v], dim=2)
        indexer_k = self.indexer_k_from_main(k)
        return main_kv, indexer_k

    def _score_and_select(
        self,
        x: torch.Tensor,
        indexer_k: torch.Tensor,
        candidate_pool: Optional[torch.Tensor],
        build_pool: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Hierarchical Sparse Indexer (§2.3.2):
          - Full: score all compressed positions; emit candidate pool (top-P)
          - Later Reindex: only score indices inside that pool
        """
        B, T, _ = x.shape
        iq = self.indexer_q(x)
        scores = torch.einsum("btd,bnd->btn", iq, indexer_k)
        new_pool = None
        Nc = indexer_k.shape[1]

        if candidate_pool is not None:
            mask = torch.full((B, T, Nc), float("-inf"), device=x.device, dtype=scores.dtype)
            P = candidate_pool.shape[-1]
            for p in range(P):
                idx = candidate_pool[..., p].clamp(0, Nc - 1)
                gathered = scores.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                mask.scatter_(-1, idx.unsqueeze(-1), gathered.unsqueeze(-1))
            scores = mask
        elif build_pool and self.candidate_pool > 0:
            # hierarchical: block-max then top-P as the shared candidate pool
            block = max(1, Nc // max(self.candidate_pool, 1))
            # coarse block maxima → expand to pool size via topk on full scores
            pool_k = min(self.candidate_pool, Nc)
            _, new_pool = scores.topk(pool_k, dim=-1)

        k = min(self.topk, scores.shape[-1])
        # avoid all -inf rows
        finite = torch.isfinite(scores).any(dim=-1, keepdim=True)
        scores = torch.where(finite, scores, torch.zeros_like(scores))
        _, topk_idx = scores.topk(k, dim=-1)
        return topk_idx, new_pool

    def forward(
        self,
        x: torch.Tensor,
        shared: SharedGlobalState,
        log: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, SharedGlobalState]:
        B, T, _ = x.shape
        notes = log if log is not None else []

        if self.mode == "full":
            main_kv, indexer_k = self._build_main_kv(x)
            topk_idx, pool = self._score_and_select(
                x, indexer_k, None, build_pool=True
            )
            shared = SharedGlobalState(
                main_kv=main_kv,
                indexer_k=indexer_k,
                topk_idx=topk_idx,
                candidate_pool=pool if pool is not None else shared.candidate_pool,
            )
            notes.append(
                f"CSA2[full] main_kv={tuple(main_kv.shape)}, Top-{topk_idx.shape[-1]}"
                + (
                    f", hier_pool={pool.shape[-1]}"
                    if pool is not None
                    else ""
                )
            )
        elif self.mode == "reindex":
            assert shared.main_kv is not None and shared.indexer_k is not None
            main_kv, indexer_k = shared.main_kv, shared.indexer_k
            topk_idx, _ = self._score_and_select(
                x, indexer_k, shared.candidate_pool, build_pool=False
            )
            shared = SharedGlobalState(
                main_kv=main_kv,
                indexer_k=indexer_k,
                topk_idx=topk_idx,
                candidate_pool=shared.candidate_pool,
            )
            notes.append(
                f"CSA2[reindex] rescored Top-K "
                f"(hier_pool={'yes' if shared.candidate_pool is not None else 'full'})"
            )
        elif self.mode == "reuse":
            assert shared.main_kv is not None and shared.topk_idx is not None
            main_kv = shared.main_kv
            topk_idx = shared.topk_idx
            notes.append("CSA2[reuse] reused KV + Top-K (no indexer)")
        else:
            raise ValueError(self.mode)

        k = main_kv[:, :, 0]
        v = main_kv[:, :, 1]
        k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q)
            # rotate K along compressed axis approximately via expand positions
            # educational: apply rope on gathered keys after gather using query positions

        idx = topk_idx.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        k_sel = torch.gather(
            k.unsqueeze(2).expand(-1, -1, T, -1, -1),
            3,
            idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head),
        )
        v_sel = torch.gather(
            v.unsqueeze(2).expand(-1, -1, T, -1, -1),
            3,
            idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_head),
        )

        scores = (q.unsqueeze(-2) @ k_sel.transpose(-2, -1)).squeeze(-2) / math.sqrt(
            self.d_head
        )
        attn = F.softmax(scores, dim=-1)
        sparse_out = (attn.unsqueeze(-2) @ v_sel).squeeze(-2)
        sparse_out = sparse_out.transpose(1, 2).contiguous().view(B, T, -1)
        sparse_out = self.out(sparse_out)

        local = self.swa(x)
        return sparse_out + local, shared
