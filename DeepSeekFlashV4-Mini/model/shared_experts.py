"""Always-on shared experts."""

from __future__ import annotations

import torch
import torch.nn as nn

from .experts import Expert


class SharedExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_shared: int,
        hidden: int,
        swiglu_limit: float = 10.0,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden, swiglu_limit) for _ in range(n_shared)]
        )

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(flat)
        for e in self.experts:
            out = out + e(flat)
        return out
