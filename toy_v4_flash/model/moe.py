"""DeepSeekMoE for V4-Flash: shared + routed, hash bootstrap, sqrt-softplus, SwiGLU clamp."""

from __future__ import annotations

from typing import List, Optional

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


class HashGate(nn.Module):
    """Frozen token-id → expert hash routing (first layers of V4-Flash)."""

    def __init__(self, vocab_size: int, n_routed: int, n_activated: int):
        super().__init__()
        self.n_activated = n_activated
        table = torch.arange(vocab_size).unsqueeze(1)
        # deterministic multi-hash
        experts = []
        for h in range(n_activated):
            experts.append((table * (17 + 13 * h) + h) % n_routed)
        self.register_buffer("tid2eid", torch.cat(experts, dim=1))  # [V, K]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens [B,T] → [B*T, K]
        flat = tokens.reshape(-1)
        return self.tid2eid[flat.clamp(0, self.tid2eid.shape[0] - 1)]


class TinyMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_routed: int,
        n_shared: int,
        n_activated: int,
        hidden: int,
        vocab_size: int = 64,
        use_hash: bool = False,
        swiglu_limit: float = 10.0,
        routed_scaling: float = 1.5,
        balance_lr: float = 0.01,
    ):
        super().__init__()
        self.n_activated = n_activated
        self.n_routed = n_routed
        self.use_hash = use_hash
        self.routed_scaling = routed_scaling
        self.balance_lr = balance_lr
        self.shared = nn.ModuleList(
            [Expert(d_model, hidden, swiglu_limit) for _ in range(n_shared)]
        )
        self.routed = nn.ModuleList(
            [Expert(d_model, hidden, swiglu_limit) for _ in range(n_routed)]
        )
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.hash_gate = HashGate(vocab_size, n_routed, n_activated) if use_hash else None
        self.register_buffer("bias", torch.zeros(n_routed))

    def _scores(self, flat: torch.Tensor) -> torch.Tensor:
        # Sqrt(Softplus) affinity (V4 change from sigmoid)
        return torch.sqrt(F.softplus(self.gate(flat)) + 1e-6)

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        log: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = torch.zeros_like(flat)
        for e in self.shared:
            out = out + e(flat)

        if self.use_hash and tokens is not None and self.hash_gate is not None:
            topi = self.hash_gate(tokens)
            topv = torch.ones_like(topi, dtype=x.dtype) / self.n_activated
            if log is not None:
                log.append(f"MoE[hash] activated={self.n_activated}")
        else:
            scores = self._scores(flat)
            select = scores + self.bias.unsqueeze(0)
            topv_sel, topi = select.topk(self.n_activated, dim=-1)
            # weights from original scores, renormalize, scale 1.5
            topv = scores.gather(-1, topi)
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            topv = topv * self.routed_scaling
            if self.training:
                with torch.no_grad():
                    load = torch.bincount(topi.reshape(-1), minlength=self.n_routed).float()
                    load = load / load.sum().clamp_min(1.0)
                    self.bias.add_(self.balance_lr * (1.0 / self.n_routed - load))
            if log is not None:
                hist = torch.bincount(topi.reshape(-1), minlength=self.n_routed)
                log.append(f"MoE[learned] hist={hist.tolist()}")

        for k in range(self.n_activated):
            idx = topi[:, k]
            w = topv[:, k].unsqueeze(-1)
            for e_id in idx.unique().tolist():
                mask = idx == int(e_id)
                out[mask] = out[mask] + w[mask] * self.routed[int(e_id)](flat[mask])

        return out.view(B, T, D)
