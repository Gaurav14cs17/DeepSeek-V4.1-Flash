"""Language-model head."""

from __future__ import annotations

import torch.nn as nn


class LMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, h):
        return self.proj(h)

    @property
    def weight(self):
        return self.proj.weight
