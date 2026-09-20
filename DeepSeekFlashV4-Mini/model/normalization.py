"""RMSNorm.

[FAITHFUL] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
Cited in DeepSeek-V4.1-Flash report for ViT / backbone normalization practice;
DeepSeek stacks use RMSNorm rather than LayerNorm in the LLM path.

Math
----
Given x ∈ R^{d}:

    RMS(x) = sqrt( mean(x²) + ε )
    y      = (x / RMS(x)) ⊙ γ

where γ is a learned gain (no bias).

Shapes: [..., d] → [..., d]
Memory: O(d) parameters; activation traffic similar to x.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float for stable RMS, then return in input dtype.
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x_f / rms
        return (self.weight * y).type_as(x)
