"""Original multi-pass mHC (V4) — not Single-Pass (that is V4.1)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm


class MultiPassMHC(nn.Module):
    """
    Paper V4 mHC (Eq. 2 style): n residual streams.
    Multi-pass: predict (A,B,C) from X_l then mix with current A_l
    (V4.1 Single-Pass shifts to A_{l-1} for fusion — we keep the older order).
    """

    def __init__(self, d_model: int, n_streams: int = 2):
        super().__init__()
        self.n = n_streams
        self.d = d_model
        out = n_streams + n_streams * n_streams + n_streams
        self.norm = RMSNorm(d_model)
        self.coeff = nn.Linear(d_model, out, bias=False)
        self.register_buffer(
            "init_A", torch.ones(1, n_streams) / n_streams, persistent=False
        )

    def expand_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(2).expand(-1, -1, self.n, -1).contiguous()

    def collapse(self, streams: torch.Tensor) -> torch.Tensor:
        return streams.mean(dim=2)

    def predict(
        self, streams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, n, _ = streams.shape
        h = self.norm(streams.mean(dim=2))
        raw = self.coeff(h)
        A = F.softmax(raw[..., :n], dim=-1)
        Bmat = F.softmax(raw[..., n : n + n * n].view(B, T, n, n), dim=-1)
        C = torch.sigmoid(raw[..., n + n * n :].unsqueeze(-1))
        return A, Bmat, C

    def mix_input(self, streams: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 3:
            A = A.unsqueeze(2)
        return (A @ streams).squeeze(2)

    def residual_update(
        self,
        streams: torch.Tensor,
        block_out: torch.Tensor,
        Bmat: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        y = block_out.unsqueeze(2)
        return (Bmat @ streams) + C * y
