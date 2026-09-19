"""Multi-Token Prediction (MTP) — V4-Flash keeps MTP; V4.1 replaces with DSpark."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .norms import RMSNorm


class ToyMTP(nn.Module):
    """
    Extra heads that predict tokens at offset +1..+depth from current hidden.
    Jointly trained with main LM head (paper V4); omitted in V4.1 backbone.
    """

    def __init__(self, d_model: int, vocab_size: int, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(depth)])
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(depth)]
        )

    def forward(
        self, h: torch.Tensor, log: Optional[List[str]] = None
    ) -> List[torch.Tensor]:
        outs = []
        for i in range(self.depth):
            outs.append(self.heads[i](self.norms[i](h)))
        if log is not None:
            log.append(f"MTP depth={self.depth} extra next-token heads")
        return outs

    def loss(
        self, h: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        """
        targets: [B,T] next-token labels for main LM (already shifted).
        MTP head i predicts targets shifted further by i+1 within the window.
        """
        import torch.nn.functional as F

        logits_list = self.forward(h)
        total = h.new_zeros(())
        n = 0
        for i, logits in enumerate(logits_list):
            # predict token at position t+(i+1) using hidden at t
            # labels: targets[:, i+1:] vs logits[:, :-(i+1)]
            shift = i + 1
            if targets.shape[1] <= shift:
                continue
            lab = targets[:, shift:]
            lg = logits[:, :-shift] if shift > 0 else logits
            T = min(lab.shape[1], lg.shape[1])
            total = total + F.cross_entropy(
                lg[:, :T].reshape(-1, lg.size(-1)),
                lab[:, :T].reshape(-1),
            )
            n += 1
        return total / max(n, 1)
