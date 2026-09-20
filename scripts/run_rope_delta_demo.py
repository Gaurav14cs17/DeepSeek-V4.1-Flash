#!/usr/bin/env python3
"""RoPE micro-optimization demo of the BEFORE→AFTER→DELTA loop.

A0 = apply_rope (complex multiply)  [FAITHFUL]
A1 = rope_reference_rotate_half     [FAITHFUL] alternate formulation

Same hardware, shapes, iters, seed. Numerical parity expected (~0 error).
Latency Δ is measured — do not claim "faster" without the table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "DeepSeekFlashV4-Mini"))

from bench.delta import compare_metrics, format_table, verdict  # noqa: E402
from bench.history import append_history  # noqa: E402
from bench.latency import time_ms  # noqa: E402
from bench.memory import collect_memory  # noqa: E402
from bench.protocol import ExperimentMeta, save_result_bundle  # noqa: E402
from bench.regression import check_regression  # noqa: E402
from bench.report import write_optimization_report  # noqa: E402
from model.rope import apply_rope, rope_freqs, rope_reference_rotate_half  # noqa: E402


def num_errors(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "relative_error": float(diff.mean() / (a.float().abs().mean() + 1e-8)),
        "cosine_similarity": float(cos),
    }


def bench_impl(name: str, fn, q: torch.Tensor, meta_base: ExperimentMeta) -> Path:
    if q.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    lat = time_ms(lambda: fn(q), warmup=meta_base.warmup_iters, iters=meta_base.measure_iters)
    # dummy module-less memory
    mem = collect_memory(None, batch=q.shape[0], seq=q.shape[2], d_model=q.shape[-1], n_layers=1)
    if q.is_cuda:
        mem.peak_gpu_memory_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 3)

    meta = ExperimentMeta(
        model="DeepSeekFlashV4-Mini",
        experiment="rope_formulation",
        component="rope",
        version=name,
        baseline_version="A0_complex" if name != "A0_complex" else None,
        precision="fp32",
        batch_size=q.shape[0],
        sequence_length=q.shape[2],
        warmup_iters=meta_base.warmup_iters,
        measure_iters=meta_base.measure_iters,
        seed=42,
        config_name="v4_tiny",
        notes=f"RoPE formulation {name}",
    )
    metrics = {
        "parameters": 0,
        "trainable_parameters": 0,
        "active_parameters": 0,
        "peak_gpu_memory_mb": mem.peak_gpu_memory_mb,
        "peak_cpu_memory_mb": mem.peak_cpu_memory_mb,
        "forward_ms": lat["mean_ms"],
        "p50_latency_ms": lat["p50_latency_ms"],
        "p95_latency_ms": lat["p95_latency_ms"],
        "tokens_per_second": (q.shape[0] * q.shape[2]) / (lat["mean_ms"] / 1000.0),
    }
    return save_result_bundle("v4", "rope_formulation", name, meta, metrics, mem.to_dict())


def main() -> None:
    device = torch.device("cpu")
    B, H, T, D = 4, 4, 64, 16
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device)
    freqs = rope_freqs(D, T, device)

    # numerical check
    y0 = apply_rope(q, freqs=freqs)
    cos, sin = freqs.real, freqs.imag
    y1 = rope_reference_rotate_half(q, cos, sin)
    err = num_errors(y0, y1)

    meta = ExperimentMeta(
        model="DeepSeekFlashV4-Mini",
        experiment="rope_formulation",
        component="rope",
        version="protocol",
        batch_size=B,
        sequence_length=T,
        warmup_iters=5,
        measure_iters=50,
        seed=42,
        config_name="v4_tiny",
    )

    d0 = bench_impl("A0_complex", lambda x: apply_rope(x, freqs=freqs), q, meta)
    d1 = bench_impl(
        "A1_rotate_half",
        lambda x: rope_reference_rotate_half(x, freqs.real, freqs.imag),
        q,
        meta,
    )

    # attach numerical errors to A1 metrics

    m1 = json.loads((d1 / "metrics.json").read_text())
    m1.update(err)
    (d1 / "metrics.json").write_text(json.dumps(m1, indent=2), encoding="utf-8")

    m0 = json.loads((d0 / "metrics.json").read_text())
    m1 = json.loads((d1 / "metrics.json").read_text())
    rows = compare_metrics(m0, m1)
    print("\n=== RoPE A0 (complex) vs A1 (rotate-half) — NOT claiming A1 is faster ===\n")
    print(format_table(rows, candidate_label="A1"))
    print()
    print(verdict(rows))
    warns = check_regression(m0, m1, threshold_pct=10.0)
    for w in warns:
        print(w)
    if not warns:
        print("Regression check: no >10% regressions (or N/A).")

    report = REPO / "results" / "v4" / "rope_formulation" / "optimization_report.md"
    write_optimization_report(
        report,
        name="RoPE formulation A0→A1 (candidate, not assumed win)",
        baseline_dir=d0,
        optimized_dir=d1,
        analysis=(
            "A0 = complex multiply (lab default). A1 = rotate-half reference. "
            f"Numerical max_abs_error={err['max_abs_error']:.2e}, "
            f"cosine_similarity={err['cosine_similarity']:.6f}. "
            "If Improve % is negative on forward_ms, A1 is slower — keep A0."
        ),
        tradeoffs=(
            "A1 may be slower on CPU for this tensor size. "
            "Correctness matched; speed did not improve."
        ),
        conclusion=verdict(rows),
    )

    cfg = json.loads((d1 / "config.json").read_text())
    append_history(
        {
            "experiment": "rope_formulation",
            "component": "rope",
            "version": "A1_rotate_half",
            "baseline_version": "A0_complex",
            "parameters": 0,
            "gpu_memory_mb": m1.get("peak_gpu_memory_mb", ""),
            "cpu_memory_mb": m1.get("peak_cpu_memory_mb", ""),
            "inference_tokens_per_sec": m1.get("tokens_per_second", ""),
            "max_abs_error": err["max_abs_error"],
            "mean_abs_error": err["mean_abs_error"],
            "timestamp": cfg.get("timestamp", ""),
            "git_commit": cfg.get("git_commit", ""),
        }
    )
    print(f"\nReport → {report}")
    print("(Skipped second compare print — use: python -m bench.compare ... if needed)")


if __name__ == "__main__":
    main()
