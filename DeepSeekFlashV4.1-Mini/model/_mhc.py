"""Single-Pass mHC (paper §2.4.1 Eq. 6) — multi-stream residual mixing (toy)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm


class SinglePassMHC(nn.Module):
    """
    Maintains n residual streams. Single-Pass shift: block l uses A_{l-1}
    from the previous block so residual update + mixing fuse in one pass.

        X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)
        (A_l, B_l, C_l) = H(X_l)

    Toy: n streams × d_model, coefficient predictor H is a small MLP + RMS.
    """

    def __init__(self, d_model: int, n_streams: int = 2):
        super().__init__()
        self.n = n_streams
        self.d = d_model
        # predict A (1×n), flatten B (n×n), C (n×1) → n + n*n + n
        out = n_streams + n_streams * n_streams + n_streams
        self.norm = RMSNorm(d_model)
        self.coeff = nn.Linear(d_model, out, bias=False)
        # learnable initial A_{-1} mixing (uniform)
        self.register_buffer(
            "init_A", torch.ones(1, n_streams) / n_streams, persistent=False
        )

    def mix_input(self, streams: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """A: [B,T,1,n] or [B,T,n] — mix n streams → block input [B,T,D]."""
        # streams: [B,T,n,D]
        if A.dim() == 3:
            A = A.unsqueeze(2)  # [B,T,1,n]
        return (A @ streams).squeeze(2)  # [B,T,D]

    def predict(
        self, streams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """H(X): A [B,T,n], B [B,T,n,n], C [B,T,n,1] from mean-pooled streams."""
        B, T, n, D = streams.shape
        h = self.norm(streams.mean(dim=2))  # [B,T,D]
        raw = self.coeff(h)
        A = F.softmax(raw[..., :n], dim=-1)  # [B,T,n]
        Bmat = raw[..., n : n + n * n].view(B, T, n, n)
        Bmat = torch.softmax(Bmat, dim=-1)
        C = torch.sigmoid(raw[..., n + n * n :].unsqueeze(-1))  # [B,T,n,1]
        return A, Bmat, C

    def residual_update(
        self,
        streams: torch.Tensor,
        block_out: torch.Tensor,
        Bmat: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """X' = B X + C ⊗ Y  (Y broadcast to streams)."""
        # streams [B,T,n,D], block_out [B,T,D]
        y = block_out.unsqueeze(2)  # [B,T,1,D]
        mixed = (Bmat @ streams) + C * y
        return mixed

    def expand_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Seed n identical streams from a single hidden state [B,T,D]."""
        return x.unsqueeze(2).expand(-1, -1, self.n, -1).contiguous()

    def collapse(self, streams: torch.Tensor) -> torch.Tensor:
        return streams.mean(dim=2)
