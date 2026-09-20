"""Sampling helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.clone()
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
