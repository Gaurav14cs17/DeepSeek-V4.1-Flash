"""DeepSeekMoE: shared + routed experts."""

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
        vocab_size: int = 64,
        use_hash: bool = False,
        swiglu_limit: float = 10.0,
        routed_scaling: float = 1.5,
        balance_lr: float = 0.01,
    ):
        super().__init__()
        self.n_activated = n_activated
        self.n_routed = n_routed
        self.shared = SharedExperts(d_model, n_shared, hidden, swiglu_limit)
        self.routed = nn.ModuleList(
            [Expert(d_model, hidden, swiglu_limit) for _ in range(n_routed)]
        )
        self.router = Router(
            d_model,
            n_routed,
            n_activated,
            vocab_size=vocab_size,
            use_hash=use_hash,
            routed_scaling=routed_scaling,
            balance_lr=balance_lr,
        )

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        log: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = self.shared(flat)
        topv, topi = self.router(flat, tokens=tokens)
        if log is not None:
            if self.router.use_hash:
                log.append(f"MoE[hash] activated={self.n_activated}")
            else:
                hist = torch.bincount(topi.reshape(-1), minlength=self.n_routed)
                log.append(f"MoE[learned] hist={hist.tolist()}")

        for k in range(self.n_activated):
            idx = topi[:, k]
            w = topv[:, k].unsqueeze(-1)
            for e_id in idx.unique().tolist():
                mask = idx == int(e_id)
                out[mask] = out[mask] + w[mask] * self.routed[int(e_id)](flat[mask])
        return out.view(B, T, D)


# Alias
TinyMoE = MoE
