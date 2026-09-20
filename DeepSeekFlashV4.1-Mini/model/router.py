"""V4.1 router with modality-specific aux-loss-free biases."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, d_model: int, n_routed: int, n_activated: int, balance_lr: float = 0.01):
        super().__init__()
        self.n_activated = n_activated
        self.n_routed = n_routed
        self.balance_lr = balance_lr
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        self.register_buffer("bias_text", torch.zeros(n_routed))
        self.register_buffer("bias_image", torch.zeros(n_routed))

    def forward(
        self, flat: torch.Tensor, modality: str = "text"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(flat)
        bias = self.bias_text if modality == "text" else self.bias_image
        select_scores = F.softmax(logits + bias.unsqueeze(0), dim=-1)
        weight_scores = F.softmax(logits, dim=-1)
        _, topi = select_scores.topk(self.n_activated, dim=-1)
        topv = weight_scores.gather(-1, topi)
        if self.training:
            with torch.no_grad():
                load = torch.bincount(topi.reshape(-1), minlength=self.n_routed).float()
                load = load / load.sum().clamp_min(1.0)
                delta = self.balance_lr * (1.0 / self.n_routed - load)
                if modality == "text":
                    self.bias_text.add_(delta)
                else:
                    self.bias_image.add_(delta)
        return topv, topi
