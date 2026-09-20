"""MoE router: learned sqrt-softplus gate + optional hash bootstrap."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashGate(nn.Module):
    """Frozen token-id → expert hash routing (first layers of V4-Flash)."""

    def __init__(self, vocab_size: int, n_routed: int, n_activated: int):
        super().__init__()
        self.n_activated = n_activated
        table = torch.arange(vocab_size).unsqueeze(1)
        experts = []
        for h in range(n_activated):
            experts.append((table * (17 + 13 * h) + h) % n_routed)
        self.register_buffer("tid2eid", torch.cat(experts, dim=1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1)
        return self.tid2eid[flat.clamp(0, self.tid2eid.shape[0] - 1)]


class Router(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_routed: int,
        n_activated: int,
        vocab_size: int = 64,
        use_hash: bool = False,
        routed_scaling: float = 1.5,
        balance_lr: float = 0.01,
    ):
        super().__init__()
        self.n_activated = n_activated
        self.n_routed = n_routed
        self.use_hash = use_hash
        self.routed_scaling = routed_scaling
        self.balance_lr = balance_lr
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.hash_gate = HashGate(vocab_size, n_routed, n_activated) if use_hash else None
        self.register_buffer("bias", torch.zeros(n_routed))

    def scores(self, flat: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.softplus(self.gate(flat)) + 1e-6)

    def forward(
        self,
        flat: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (top_values [BT,K], top_indices [BT,K])."""
        if self.use_hash and tokens is not None and self.hash_gate is not None:
            topi = self.hash_gate(tokens)
            topv = torch.ones_like(topi, dtype=flat.dtype) / self.n_activated
            return topv, topi

        scores = self.scores(flat)
        select = scores + self.bias.unsqueeze(0)
        _, topi = select.topk(self.n_activated, dim=-1)
        topv = scores.gather(-1, topi)
        topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        topv = topv * self.routed_scaling
        if self.training:
            with torch.no_grad():
                load = torch.bincount(topi.reshape(-1), minlength=self.n_routed).float()
                load = load / load.sum().clamp_min(1.0)
                self.bias.add_(self.balance_lr * (1.0 / self.n_routed - load))
        return topv, topi
