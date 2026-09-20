"""Memory reporting helpers."""

from __future__ import annotations

import torch


def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def activation_estimate(batch: int, seq: int, d_model: int, n_layers: int, bytes_per: int = 4) -> int:
    # rough: residual stream per layer
    return batch * seq * d_model * n_layers * bytes_per
