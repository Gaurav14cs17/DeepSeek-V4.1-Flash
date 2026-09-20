"""LM + optional MTP / load-balance losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_aux: torch.Tensor | None = None,
    mtp_weight: float = 0.1,
) -> torch.Tensor:
    loss = lm_loss(logits, targets)
    if mtp_aux is not None:
        loss = loss + mtp_weight * mtp_aux
    return loss
