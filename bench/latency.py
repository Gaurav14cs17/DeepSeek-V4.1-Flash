"""Latency helpers (warmup + timed iters + component breakdown)."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence


def time_ms(fn: Callable[[], None], *, warmup: int = 3, iters: int = 10) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples_sorted = sorted(samples)
    return {
        "mean_ms": statistics.mean(samples),
        "p50_latency_ms": samples_sorted[len(samples_sorted) // 2],
        "p95_latency_ms": samples_sorted[max(0, int(0.95 * (len(samples_sorted) - 1)))],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


# Standard inference component keys (fill what applies)
COMPONENT_KEYS = (
    "tokenizer",
    "embedding",
    "attention",
    "router",
    "experts",
    "ffn",
    "lm_head",
    "sampling",
    "prefill_total",
    "decode_total",
)


def time_components(
    components: Mapping[str, Callable[[], None]],
    *,
    warmup: int = 2,
    iters: int = 10,
) -> Dict[str, float]:
    """Time each named component; returns {name_ms: mean_ms, ...} plus total_ms."""
    out: Dict[str, float] = {}
    total = 0.0
    for name, fn in components.items():
        m = time_ms(fn, warmup=warmup, iters=iters)["mean_ms"]
        out[f"{name}_ms"] = m
        total += m
    out["total_ms"] = total
    return out


def format_component_table(
    baseline: Mapping[str, float],
    optimized: Mapping[str, float],
    *,
    keys: Optional[Sequence[str]] = None,
) -> str:
    """
    | Component | Baseline | Optimized | Δ |
    """
    if keys is None:
        keys = sorted(
            {k[: -len("_ms")] for k in baseline if k.endswith("_ms")}
            | {k[: -len("_ms")] for k in optimized if k.endswith("_ms")}
        )
    lines = [
        "| Component | Baseline | Optimized |      Δ |",
        "| --------- | -------: | --------: | -----: |",
    ]
    for name in keys:
        bk, ok = f"{name}_ms", f"{name}_ms"
        if bk not in baseline or ok not in optimized:
            continue
        b, o = float(baseline[bk]), float(optimized[ok])
        pct = (b - o) / abs(b) * 100.0 if b else 0.0
        lines.append(f"| {name:9s} | {b:8.2f} | {o:9.2f} | {pct:+6.1f}% |")
    return "\n".join(lines)
