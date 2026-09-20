"""Write optimization_report.md from BEFORE/AFTER bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .delta import compare_metrics, format_table, verdict
from .memory import format_memory_report, memory_delta


def _pct(b: float, o: float, *, lower_better: bool) -> str:
    if b == 0:
        return "n/a"
    if lower_better:
        return f"{(b - o) / abs(b) * 100:+.2f}%"
    return f"{(o - b) / abs(b) * 100:+.2f}%"


def _pair(b_m: Dict[str, Any], o_m: Dict[str, Any], key: str) -> tuple:
    return b_m.get(key), o_m.get(key)


def write_optimization_report(
    out_path: Path,
    *,
    name: str,
    baseline_dir: Path,
    optimized_dir: Path,
    analysis: str = "",
    tradeoffs: str = "",
    conclusion: str = "",
) -> Path:
    b_cfg = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    o_cfg = json.loads((optimized_dir / "config.json").read_text(encoding="utf-8"))
    b_m = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    o_m = json.loads((optimized_dir / "metrics.json").read_text(encoding="utf-8"))
    b_mem = json.loads((baseline_dir / "memory.json").read_text(encoding="utf-8"))
    o_mem = json.loads((optimized_dir / "memory.json").read_text(encoding="utf-8"))

    rows = compare_metrics(b_m, o_m)
    table = format_table(rows)
    vline = verdict(rows)

    hw = b_cfg.get("hardware", {})
    sw = b_cfg.get("software", {})

    def section_metric(label: str, key: str, lower_better: bool = True) -> str:
        b, o = _pair(b_m, o_m, key)
        if b is None or o is None:
            # try memory bundle
            b = b_mem.get(key, b)
            o = o_mem.get(key, o)
        if b is None or o is None:
            return f"### {label}\n\nNot measured in this experiment.\n"
        bf, of = float(b), float(o)
        return (
            f"### {label}\n\n"
            f"Before: `{bf}`\n\n"
            f"After: `{of}`\n\n"
            f"Absolute change: `{of - bf:+.6g}`\n\n"
            f"Percentage change (Improve %): `{_pct(bf, of, lower_better=lower_better)}`\n"
        )

    md = f"""# Optimization: {name}

## Baseline

Implementation: `{b_cfg.get('version')}`
Component: `{b_cfg.get('component')}`
Configuration: `{b_cfg.get('config_name')}`
Commit: `{b_cfg.get('git_commit')}`

## Optimized Version

Implementation: `{o_cfg.get('version')}`
Component: `{o_cfg.get('component')}`
Configuration: `{o_cfg.get('config_name')}`
Commit: `{o_cfg.get('git_commit')}`

## Hardware

CPU: {hw.get('cpu')}
GPU: {hw.get('gpu')}
VRAM: {hw.get('gpu_vram_total_mb')} MB
Platform: {hw.get('platform')}

## Software

Python: {sw.get('python')}
PyTorch: {sw.get('pytorch')}
CUDA: {sw.get('cuda_version')}

## Memory

### Before
```
{format_memory_report(b_mem, "MEMORY REPORT (BASELINE)")}
```

### After
```
{format_memory_report(o_mem, "MEMORY REPORT (OPTIMIZED)")}
```

```
{memory_delta(b_mem, o_mem)}
```

{section_metric("Peak GPU", "peak_gpu_memory_mb", True)}
{section_metric("CPU RAM", "peak_cpu_memory_mb", True)}
{section_metric("KV cache", "kv_cache_mb", True)}

## Latency

{section_metric("Prefill", "prefill_latency_ms", True)}
{section_metric("Decode", "decode_latency_ms", True)}
{section_metric("TTFT", "ttft_ms", True)}
{section_metric("Forward (microbench)", "forward_ms", True)}
{section_metric("Step", "step_ms" if "step_ms" in b_m or "step_ms" in o_m else "training_step_ms", True)}
{section_metric("Backward", "backward_ms", True)}
{section_metric("Optimizer", "optimizer_ms", True)}

## Throughput

{section_metric("Inference tokens/sec", "inference_tokens_per_sec", False)}
{section_metric("Training tokens/sec", "training_tokens_per_sec", False)}
{section_metric("Generic tokens/sec", "tokens_per_second", False)}

## Training

{section_metric("Train loss", "loss" if "loss" in b_m or "loss" in o_m else "train_loss", True)}
{section_metric("Validation loss", "validation_loss", True)}
{section_metric("Perplexity", "perplexity" if "perplexity" in b_m or "perplexity" in o_m else "perplexity_val", True)}
{section_metric("Training step", "training_step_ms", True)}
{section_metric("Training wall time", "training_time_s", True)}

## Accuracy

{section_metric("Max abs error", "max_abs_error", True)}
{section_metric("Mean abs error", "mean_abs_error", True)}
{section_metric("Cosine similarity", "cosine_similarity", False)}

## KV Cache

{section_metric("KV bytes/token", "kv_bytes_per_token", True)}

## Standard metric table

{table}

## Verdict

{vline}

## Numerical difference

- max_abs_error: {o_m.get('max_abs_error', 'n/a')}
- mean_abs_error: {o_m.get('mean_abs_error', 'n/a')}
- relative_error: {o_m.get('relative_error', 'n/a')}
- cosine_similarity: {o_m.get('cosine_similarity', 'n/a')}

## Analysis

{analysis or "_Fill in: why each metric changed._"}

## Trade-offs

{tradeoffs or "_List anything that became worse._"}

## Conclusion

{conclusion or "_State only measured results. No subjective labels (excellent / amazing / huge)._"}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path
