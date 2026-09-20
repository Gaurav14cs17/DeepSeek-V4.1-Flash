"""Slow, correctness-first reference kernels for verification."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def reference_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Hand-computable RoPE via explicit cos/sin pair rotation.

    x: [B, H, T, D], D even.
    """
    B, H, T, D = x.shape
    assert D % 2 == 0
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)  # [T, half]
    cos = angles.cos()
    sin = angles.sin()
    x_f = x.float()
    x1 = x_f[..., ::2]
    x2 = x_f[..., 1::2]
    c = cos.view(1, 1, T, half)
    s = sin.view(1, 1, T, half)
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y.type_as(x)


def reference_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    window: int | None = None,
) -> torch.Tensor:
    """
    Reference attention: softmax(QK^T / sqrt(d) + mask) V

    q,k,v: [B, H, T, D]
    returns: [B, H, T, D]
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    allow = torch.ones(T, T, device=q.device, dtype=torch.bool)
    if causal:
        allow = torch.tril(allow)
    if window is not None:
        idx = torch.arange(T, device=q.device)
        allow = allow & ((idx[:, None] - idx[None, :]) < window)
    scores = scores.masked_fill(~allow[None, None], float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.type_as(q)


def reference_router_topk(
    logits: torch.Tensor,
    top_k: int,
    *,
    routed_scaling: float = 1.0,
    use_sqrt_softplus: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Manual router: softplus→sqrt scores → top-k → renormalize → scale.

    logits: [N, E]
    returns (weights [N,K], indices [N,K])
    """
    if top_k > logits.shape[-1]:
        raise ValueError(f"top_k={top_k} > n_experts={logits.shape[-1]}")
    if use_sqrt_softplus:
        scores = torch.sqrt(F.softplus(logits) + 1e-6)
    else:
        scores = F.softmax(logits, dim=-1)
    topv, topi = scores.topk(top_k, dim=-1)
    topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    topv = topv * routed_scaling
    return topv, topi


def reference_fake_quantize(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    if n_bits >= 16:
        return x
    qmax = (1 << (n_bits - 1)) - 1
    scale = x.abs().amax().clamp_min(1e-8) / qmax
    q = torch.clamp((x / scale).round(), -qmax - 1, qmax)
    return q * scale


def max_mean_abs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    return float(diff.max()), float(diff.mean())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    if float(denom) == 0.0:
        return 1.0 if torch.equal(af, bf) else 0.0
    return float((af @ bf) / denom)
