"""SwiGLU expert FFNs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden: int, swiglu_limit: float = 10.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.w1(x).clamp(-self.limit, self.limit)
        u = self.w3(x).clamp(-self.limit, self.limit)
        return self.w2(F.silu(g) * u)
