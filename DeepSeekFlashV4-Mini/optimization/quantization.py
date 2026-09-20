"""Toy quantization helpers (educational, not production QAT)."""

from __future__ import annotations

import torch


def fake_quantize(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    if n_bits >= 16:
        return x
    qmax = (1 << (n_bits - 1)) - 1
    scale = x.detach().abs().amax().clamp_min(1e-8) / qmax
    q = torch.clamp((x / scale).round(), -qmax - 1, qmax)
    return q * scale


def mxfp4_sim(x: torch.Tensor) -> torch.Tensor:
    """Very rough FP4-ish simulation for labs."""
    return fake_quantize(x, n_bits=4)
