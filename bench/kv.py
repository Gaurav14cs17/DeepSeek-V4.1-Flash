"""KV-cache first-class benchmark — context sweep tables.

Sequence lengths: 128, 256, 512, 1024, 2048, 4096 (subset if HW-limited).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CONTEXT_LENGTHS: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)


def kv_bytes_per_token(
    *,
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    bytes_per_elem: int = 2,  # fp16 default
    n_tensors: int = 2,  # K and V
) -> float:
    """Theoretical KV bytes per token (all layers)."""
    return float(n_layers * n_kv_heads * d_head * bytes_per_elem * n_tensors)


def total_kv_mb(bytes_per_token: float, seq: int, batch: int = 1) -> float:
    return batch * seq * bytes_per_token / (1024**2)


def measure_kv_point(
    *,
    context: int,
    bytes_per_token: float,
    batch: int = 1,
    prefill_ms: float = 0.0,
    decode_ms: float = 0.0,
    peak_gpu_mb: float = 0.0,
    tokens_per_sec: float = 0.0,
) -> Dict[str, float]:
    return {
        "context": float(context),
        "kv_bytes_per_token": bytes_per_token,
        "kv_cache_mb": total_kv_mb(bytes_per_token, context, batch),
        "prefill_latency_ms": prefill_ms,
        "decode_latency_ms": decode_ms,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "inference_tokens_per_sec": tokens_per_sec,
        "cache_growth_mb_per_1k": total_kv_mb(bytes_per_token, 1000, batch),
    }


def format_kv_sweep_table(
    baseline: Sequence[Mapping[str, Any]],
    optimized: Sequence[Mapping[str, Any]],
) -> str:
    """
    | Context | Baseline KV | Optimized KV | Reduction |
    """
    b_by = {int(r["context"]): r for r in baseline}
    o_by = {int(r["context"]): r for r in optimized}
    contexts = sorted(set(b_by) & set(o_by))
    lines = [
        "| Context | Baseline KV MB | Optimized KV MB | Reduction |",
        "| ------: | -------------: | --------------: | --------: |",
    ]
    for c in contexts:
        b = float(b_by[c]["kv_cache_mb"])
        o = float(o_by[c]["kv_cache_mb"])
        if b == 0:
            red = "—"
        else:
            red = f"{(b - o) / abs(b) * 100:+.1f}%"
        lines.append(f"| {c:7d} | {b:14.4f} | {o:15.4f} | {red:>9s} |")
    return "\n".join(lines)


def format_kv_ascii_chart(
    baseline: Sequence[Mapping[str, Any]],
    optimized: Sequence[Mapping[str, Any]],
    *,
    width: int = 40,
) -> str:
    """Simple text chart: context length → KV memory (baseline vs optimized)."""
    b_by = {int(r["context"]): float(r["kv_cache_mb"]) for r in baseline}
    o_by = {int(r["context"]): float(r["kv_cache_mb"]) for r in optimized}
    contexts = sorted(set(b_by) & set(o_by))
    if not contexts:
        return "(no overlapping contexts)"
    vmax = max(max(b_by[c], o_by[c]) for c in contexts) or 1.0
    lines = ["KV MEMORY vs CONTEXT", "-" * 50]
    for c in contexts:
        b, o = b_by[c], o_by[c]
        nb = int(round(b / vmax * width))
        no = int(round(o / vmax * width))
        lines.append(f"{c:5d} B |{'█' * nb}{' ' * (width - nb)}| {b:.3f} MB")
        lines.append(f"      O |{'░' * no}{' ' * (width - no)}| {o:.3f} MB")
    lines.append("B=baseline  O=optimized")
    return "\n".join(lines)


def compression_ratio(baseline_bytes: float, optimized_bytes: float) -> float:
    if optimized_bytes <= 0:
        return float("inf")
    return baseline_bytes / optimized_bytes
