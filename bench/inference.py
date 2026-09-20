"""Inference benchmark protocol — fixed prompt / gen length grids.

Every inference benchmark must use the same grid unless the experiment
explicitly studies that variable:

  prompt lengths:  128, 256, 512, 1024, 2048
  generated tokens: 32, 128, 512

Hardware may force a subset; record which subset ran in config.extra.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .latency import time_ms

# Standard grids from the mandatory protocol
PROMPT_LENGTHS: Tuple[int, ...] = (128, 256, 512, 1024, 2048)
GEN_LENGTHS: Tuple[int, ...] = (32, 128, 512)


def allowed_configs(
    max_seq: int,
    *,
    prompt_lengths: Sequence[int] = PROMPT_LENGTHS,
    gen_lengths: Sequence[int] = GEN_LENGTHS,
) -> List[Tuple[int, int]]:
    """Return (prompt, gen) pairs that fit under max_seq."""
    out: List[Tuple[int, int]] = []
    for p in prompt_lengths:
        for g in gen_lengths:
            if p + g <= max_seq:
                out.append((p, g))
    return out


def measure_inference_point(
    *,
    run_prefill: Callable[[], None],
    run_decode_step: Callable[[], None],
    prompt_len: int,
    gen_len: int,
    warmup: int = 2,
    measure_iters: int = 5,
    peak_gpu_mb: float = 0.0,
    peak_cpu_mb: float = 0.0,
    kv_cache_bytes: float = 0.0,
) -> Dict[str, float]:
    """
    Measure one (prompt, gen) configuration.

    run_prefill: callable that runs full prefill once
    run_decode_step: callable that runs ONE decode step (caller loops gen_len)
    """
    prefill = time_ms(run_prefill, warmup=warmup, iters=measure_iters)

    def decode_all() -> None:
        for _ in range(gen_len):
            run_decode_step()

    decode = time_ms(decode_all, warmup=max(1, warmup // 2), iters=measure_iters)
    total_ms = prefill["mean_ms"] + decode["mean_ms"]
    tok_s = gen_len / (decode["mean_ms"] / 1000.0) if decode["mean_ms"] > 0 else 0.0
    return {
        "prompt_len": float(prompt_len),
        "gen_len": float(gen_len),
        "ttft_ms": prefill["mean_ms"],  # first-token ≈ prefill for greedy single stream
        "prefill_latency_ms": prefill["mean_ms"],
        "decode_latency_ms": decode["mean_ms"],
        "avg_token_latency_ms": decode["mean_ms"] / max(gen_len, 1),
        "total_inference_ms": total_ms,
        "inference_tokens_per_sec": tok_s,
        "p50_latency_ms": decode["p50_latency_ms"],
        "p95_latency_ms": decode["p95_latency_ms"],
        "peak_gpu_memory_mb": peak_gpu_mb,
        "peak_cpu_memory_mb": peak_cpu_mb,
        "kv_cache_mb": kv_cache_bytes / (1024**2),
        "kv_bytes_per_token": kv_cache_bytes / max(prompt_len + gen_len, 1),
    }


def format_inference_grid(rows: Iterable[Dict[str, Any]]) -> str:
    header = (
        "| Prompt | Gen | TTFT ms | Prefill ms | Decode ms | Total s | tok/s | GPU MB |"
    )
    sep = "| -----: | --: | ------: | ---------: | --------: | ------: | ----: | -----: |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {int(r['prompt_len']):6d} | {int(r['gen_len']):3d} | "
            f"{r['ttft_ms']:7.2f} | {r['prefill_latency_ms']:10.2f} | "
            f"{r['decode_latency_ms']:9.2f} | {r['total_inference_ms']/1000:7.3f} | "
            f"{r['inference_tokens_per_sec']:5.1f} | {r.get('peak_gpu_memory_mb', 0):6.1f} |"
        )
    return "\n".join(lines)


def inference_delta_summary(baseline: Dict[str, float], optimized: Dict[str, float]) -> str:
    """Print TTFT / decode / total / throughput / memory improvements."""
    keys = [
        ("ttft_ms", "TTFT", True),
        ("decode_latency_ms", "Decode", True),
        ("total_inference_ms", "Total latency", True),
        ("inference_tokens_per_sec", "Throughput", False),
        ("peak_gpu_memory_mb", "GPU memory", True),
    ]
    lines = ["INFERENCE IMPROVEMENT", "-" * 40]
    for k, label, lower_better in keys:
        if k not in baseline or k not in optimized:
            continue
        b, o = float(baseline[k]), float(optimized[k])
        if b == 0:
            continue
        if lower_better:
            pct = (b - o) / abs(b) * 100.0
        else:
            pct = (o - b) / abs(b) * 100.0
        lines.append(f"{label}: {o - b:+.4g}  (Improve % {pct:+.2f})")
    return "\n".join(lines)
