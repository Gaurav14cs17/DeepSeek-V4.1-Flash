"""Tiny DeepSeekMoE + auxiliary-loss-free / modality load balancing (toy)."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TinyMoE(nn.Module):
    """
    Shared + routed experts. Aux-loss-free balancing: expert-wise bias updated
    from load (Wang et al.). Modality biases: separate text / image corrections.
    """

    def __init__(
        self,
        d_model: int,
        n_routed: int,
        n_shared: int,
        n_activated: int,
        hidden: int,
        balance_lr: float = 0.01,
    ):
        super().__init__()
        self.n_activated = n_activated
        self.n_routed = n_routed
        self.balance_lr = balance_lr
        self.shared = nn.ModuleList([Expert(d_model, hidden) for _ in range(n_shared)])
        self.routed = nn.ModuleList([Expert(d_model, hidden) for _ in range(n_routed)])
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        # aux-loss-free biases (not trained by autograd — updated after step)
        self.register_buffer("bias_text", torch.zeros(n_routed))
        self.register_buffer("bias_image", torch.zeros(n_routed))

    def forward(
        self,
        x: torch.Tensor,
        log: Optional[List[str]] = None,
        modality: str = "text",
    ) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        logits = self.gate(flat)
        bias = self.bias_text if modality == "text" else self.bias_image
        # bias for selection only; original scores for weighting (paper §2.1.1)
        select_logits = logits + bias.unsqueeze(0)
        select_scores = F.softmax(select_logits, dim=-1)
        weight_scores = F.softmax(logits, dim=-1)
        topv_sel, topi = select_scores.topk(self.n_activated, dim=-1)
        # gather weights from original scores at selected indices
        topv = weight_scores.gather(-1, topi)

        out = torch.zeros_like(flat)
        for e in self.shared:
            out = out + e(flat)

        for k in range(self.n_activated):
            idx = topi[:, k]
            w = topv[:, k].unsqueeze(-1)
            for e_id in idx.unique().tolist():
                mask = idx == e_id
                out[mask] = out[mask] + w[mask] * self.routed[e_id](flat[mask])

        # update biases toward uniform load (aux-loss-free)
        if self.training:
            with torch.no_grad():
                load = torch.bincount(topi.reshape(-1), minlength=self.n_routed).float()
                load = load / load.sum().clamp_min(1.0)
                target = 1.0 / self.n_routed
                delta = self.balance_lr * (target - load)
                if modality == "text":
                    self.bias_text.add_(delta)
                else:
                    self.bias_image.add_(delta)

        if log is not None:
            hist = torch.bincount(topi.reshape(-1), minlength=len(self.routed))
            log.append(f"MoE[{modality}] hist={hist.tolist()}")

        return out.view(B, T, D)
