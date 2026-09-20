"""Numerical correctness + quality trade-off helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch

from .delta import compare_metrics, format_table


QUALITY_KEYS = (
    "loss",
    "train_loss",
    "validation_loss",
    "perplexity",
)


def numerical_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """max/mean abs error, relative error, cosine similarity."""
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    if af.numel() != bf.numel():
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (af - bf).abs()
    denom = af.abs().mean().clamp_min(1e-8)
    cos = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "relative_error": float(diff.mean() / denom),
        "cosine_similarity": float(cos),
    }


def quality_tradeoff_table(
    baseline: Mapping[str, Any],
    optimized: Mapping[str, Any],
    *,
    extra_keys: Optional[List[str]] = None,
) -> str:
    """Loss / perplexity / memory / latency side-by-side for quality-affecting opts."""
    keys = list(QUALITY_KEYS)
    for k in (
        "peak_gpu_memory_mb",
        "decode_latency_ms",
        "inference_tokens_per_sec",
        "kv_bytes_per_token",
    ):
        if k in baseline and k in optimized:
            keys.append(k)
    if extra_keys:
        keys.extend(extra_keys)
    rows = compare_metrics(dict(baseline), dict(optimized), keys=keys)
    # Relabel slight quality regressions as CHECK (not auto-fail)
    for r in rows:
        if r["metric"] in QUALITY_KEYS and r["better"] == "NO":
            r["better"] = "CHECK"
    return format_table(rows, candidate_label="Optimized")
