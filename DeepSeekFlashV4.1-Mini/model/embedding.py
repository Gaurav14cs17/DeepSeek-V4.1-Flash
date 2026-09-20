"""Token embedding."""

from __future__ import annotations

import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        return self.emb(tokens)

    @property
    def weight(self):
        return self.emb.weight
