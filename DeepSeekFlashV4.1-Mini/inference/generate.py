"""Autoregressive generation."""

from __future__ import annotations

import torch

from .sampling import sample_logits


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    max_new: int = 32,
    temperature: float = 1.0,
    top_k: int = 20,
) -> torch.Tensor:
    model.eval()
    tokens = prompt
    max_ctx = getattr(model.cfg, "max_seq_len", 128)
    for _ in range(max_new):
        ctx = tokens[:, -max_ctx:]
        logits, _ = model(ctx, trace=False)
        nxt = sample_logits(logits[:, -1, :], temperature=temperature, top_k=top_k)
        tokens = torch.cat([tokens, nxt], dim=1)
    return tokens
