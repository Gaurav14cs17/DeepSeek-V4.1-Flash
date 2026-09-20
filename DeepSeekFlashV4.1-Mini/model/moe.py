"""DeepSeekMoE for V4.1 (shared + routed, modality load balance)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .experts import Expert
from .router import Router
from .shared_experts import SharedExperts


class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_routed: int,
        n_shared: int,
        n_activated: int,
        hidden: int,
        balance_lr: float = 0.01,
    ):
        super().__init__()
        self.n_activated = n_activated
        self.n_routed = n_routed
        self.shared = SharedExperts(d_model, n_shared, hidden)
        self.routed = nn.ModuleList([Expert(d_model, hidden) for _ in range(n_routed)])
        self.router = Router(d_model, n_routed, n_activated, balance_lr=balance_lr)

    def forward(
        self,
        x: torch.Tensor,
        log: Optional[List[str]] = None,
        modality: str = "text",
    ) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = self.shared(flat)
        topv, topi = self.router(flat, modality=modality)
        for k in range(self.n_activated):
            idx = topi[:, k]
            w = topv[:, k].unsqueeze(-1)
            for e_id in idx.unique().tolist():
                mask = idx == int(e_id)
                out[mask] = out[mask] + w[mask] * self.routed[int(e_id)](flat[mask])
        if log is not None:
            hist = torch.bincount(topi.reshape(-1), minlength=self.n_routed)
            log.append(f"MoE[{modality}] hist={hist.tolist()}")
        return out.view(B, T, D)


TinyMoE = MoE
