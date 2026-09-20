"""Optimization ladder — measure after EVERY level, never jump to final.

LEVEL 0  Reference
LEVEL 1  Memory optimization
LEVEL 2  Algorithmic optimization
LEVEL 3  Cache optimization
LEVEL 4  Quantization
LEVEL 5  Kernel/runtime optimization
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

LEVEL_NAMES = {
    0: "Reference",
    1: "Memory",
    2: "Algorithmic",
    3: "Cache",
    4: "Quantization",
    5: "Kernel/runtime",
}


def format_ladder_table(
    versions: Sequence[Mapping[str, Any]],
    *,
    memory_key: str = "peak_gpu_memory_mb",
    latency_key: str = "decode_latency_ms",
    throughput_key: str = "inference_tokens_per_sec",
    baseline_index: int = 0,
) -> str:
    """
    | Version | Memory | Latency | Throughput | Cum. mem Δ | Cum. lat Δ | Cum. thr Δ |
    cumulative % relative to versions[baseline_index].
    """
    if not versions:
        return "(empty ladder)"
    base = versions[baseline_index]
    bm = float(base.get(memory_key, 0) or 0)
    bl = float(base.get(latency_key, 0) or 0)
    bt = float(base.get(throughput_key, 0) or 0)

    lines = [
        "| Version | Memory | Latency | Throughput | Cum. mem vs A0 | Cum. lat vs A0 | Cum. thr vs A0 |",
        "| ------- | -----: | ------: | ---------: | -------------: | -------------: | -------------: |",
    ]
    for i, v in enumerate(versions):
        name = str(v.get("version", f"A{i}"))
        m = float(v.get(memory_key, 0) or 0)
        lat = float(v.get(latency_key, 0) or 0)
        thr = float(v.get(throughput_key, 0) or 0)
        cm = f"{(bm - m) / abs(bm) * 100:+.1f}%" if bm else "—"
        cl = f"{(bl - lat) / abs(bl) * 100:+.1f}%" if bl else "—"
        ct = f"{(thr - bt) / abs(bt) * 100:+.1f}%" if bt else "—"
        if i == baseline_index:
            cm = cl = ct = "—"
        lines.append(
            f"| {name:7s} | {m:6.3g} | {lat:7.3g} | {thr:10.3g} | {cm:>14s} | {cl:>14s} | {ct:>14s} |"
        )
    return "\n".join(lines)


def ladder_level_label(level: int) -> str:
    return f"L{level} {LEVEL_NAMES.get(level, 'Custom')}"
