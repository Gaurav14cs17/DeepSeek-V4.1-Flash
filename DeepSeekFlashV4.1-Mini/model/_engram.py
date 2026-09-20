"""Toy Engram (§2.4.2): multi-head N-gram hash memory + Sinkhorn-ish table balance."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyEngram(nn.Module):
    """
    Paper: orders {2,3,4}, 8 hash heads, context-aware gate, placed at layers 1 & 14.
    Toy: small tables, orders configurable, optional Sinkhorn row/col balance.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_entries: int,
        emb_dim: int,
        n_heads: int = 4,
        ngram_orders: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.n_entries = n_entries
        self.ngram_orders = ngram_orders
        self.n_heads = n_heads
        self.tables = nn.ParameterDict()
        # distinct prime-ish sizes per head (paper); toy: offset strides
        self._sizes = {}
        for order in ngram_orders:
            for h in range(n_heads):
                key = f"o{order}_h{h}"
                size = n_entries + h * 7 + order  # distinct
                self._sizes[key] = size
                self.tables[key] = nn.Parameter(torch.randn(size, emb_dim) * 0.02)
        self.proj = nn.Linear(
            emb_dim * len(ngram_orders) * n_heads, d_model, bias=False
        )
        self.gate = nn.Linear(
            d_model + emb_dim * len(ngram_orders) * n_heads, d_model
        )

    def _hash(self, tokens: torch.Tensor, order: int, head: int) -> torch.Tensor:
        B, T = tokens.shape
        pad = torch.zeros(B, order - 1, dtype=tokens.dtype, device=tokens.device)
        x = torch.cat([pad, tokens], dim=1)
        h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        base = 31 + head * 17
        for i in range(order):
            h = h * base + x[:, i : i + T].long()
        size = self._sizes[f"o{order}_h{head}"]
        return h.abs() % size

    def sinkhorn_balance_(self, steps: int = 3) -> None:
        """Momentum-free toy Sinkhorn: normalize table rows then cols lightly."""
        with torch.no_grad():
            for p in self.tables.values():
                t = p.data
                for _ in range(steps):
                    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    t = t / t.norm(dim=0, keepdim=True).clamp_min(1e-6)
                p.data.copy_(t * 0.02 * (t.shape[-1] ** 0.5))

    def forward(
        self, tokens: torch.Tensor, hidden: torch.Tensor, log: Optional[List[str]] = None
    ) -> torch.Tensor:
        pieces = []
        addrs = []
        for order in self.ngram_orders:
            for head in range(self.n_heads):
                idx = self._hash(tokens, order, head)
                addrs.append(idx)
                emb = self.tables[f"o{order}_h{head}"][idx]
                pieces.append(emb)
        mem = torch.cat(pieces, dim=-1)
        gated = torch.sigmoid(self.gate(torch.cat([hidden, mem], dim=-1)))
        out = hidden + gated * self.proj(mem)
        if log is not None:
            unique = sum(a.unique().numel() for a in addrs)
            log.append(
                f"Engram orders={self.ngram_orders} heads={self.n_heads} "
                f"unique_rows≈{unique}"
            )
        return out


def pack_engram_to_disk(module: ToyEngram, path: str) -> None:
    torch.save({k: v.detach().cpu() for k, v in module.tables.items()}, path)


def load_engram_from_disk(module: ToyEngram, path: str) -> None:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for k, v in blob.items():
            if k in module.tables:
                module.tables[k].copy_(v)
