"""Training benchmark protocol — forward / backward / optimizer / step.

Canonical keys (write both aliases where noted):
  forward_ms, backward_ms, optimizer_ms
  step_ms  (== training_step_ms for history.csv)
  training_tokens_per_sec, training_time_s
  loss / train_loss, validation_loss, perplexity
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Optional

from .delta import compare_metrics, format_table
from .latency import time_ms

# Keys emitted by measure_training_step (+ aliases for history / older bundles)
TRAINING_METRIC_KEYS = (
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
    "step_ms",
    "training_step_ms",
    "training_tokens_per_sec",
    "training_time_s",
    "epoch_time_s",
    "loss",
    "train_loss",
    "validation_loss",
    "perplexity",
    "peak_gpu_memory_mb",
    "peak_cpu_memory_mb",
)


def normalize_training_metrics(m: Mapping[str, Any]) -> Dict[str, Any]:
    """Ensure step_ms ↔ training_step_ms and loss ↔ train_loss aliases exist."""
    out = dict(m)
    if "step_ms" in out and "training_step_ms" not in out:
        out["training_step_ms"] = out["step_ms"]
    if "training_step_ms" in out and "step_ms" not in out:
        out["step_ms"] = out["training_step_ms"]
    if "loss" in out and "train_loss" not in out:
        out["train_loss"] = out["loss"]
    if "train_loss" in out and "loss" not in out:
        out["loss"] = out["train_loss"]
    return out


def measure_training_step(
    *,
    forward_fn: Callable[[], Any],
    backward_fn: Callable[[Any], None],
    optimizer_fn: Callable[[], None],
    zero_grad_fn: Optional[Callable[[], None]] = None,
    warmup: int = 2,
    iters: int = 10,
    tokens_per_step: int = 0,
) -> Dict[str, float]:
    """
    Time forward / backward / optimizer separately, then a full step.

    forward_fn   → returns a loss tensor (graph attached)
    backward_fn  → runs loss.backward() (and any grad sync)
    optimizer_fn → runs optimizer.step()
    zero_grad_fn → optional; called before each timed forward that will backward

    Notes:
    - Forward-only timing uses a separate call and frees the graph (backward not run).
    - Backward timing = (forward+backward) − forward_mean.
    - Optimizer timing = full_step − (forward+backward)_mean.
    """

    def _zero() -> None:
        if zero_grad_fn is not None:
            zero_grad_fn()

    def forward_only() -> None:
        _zero()
        loss = forward_fn()
        # Drop graph so repeated fwd timing does not leak memory.
        if hasattr(loss, "detach"):
            loss.detach()
        del loss

    def forward_backward() -> None:
        _zero()
        loss = forward_fn()
        backward_fn(loss)
        del loss

    def full_step() -> None:
        _zero()
        loss = forward_fn()
        backward_fn(loss)
        optimizer_fn()
        del loss

    fwd = time_ms(forward_only, warmup=warmup, iters=iters)
    fb = time_ms(forward_backward, warmup=warmup, iters=iters)
    full = time_ms(full_step, warmup=warmup, iters=iters)

    forward_ms = fwd["mean_ms"]
    backward_ms = max(0.0, fb["mean_ms"] - forward_ms)
    optimizer_ms = max(0.0, full["mean_ms"] - fb["mean_ms"])
    step_ms = full["mean_ms"]
    tok_s = (
        tokens_per_step / (step_ms / 1000.0) if step_ms > 0 and tokens_per_step else 0.0
    )
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "step_ms": step_ms,
        "training_step_ms": step_ms,  # history.csv / Stage-1 alias
        "training_tokens_per_sec": tok_s,
        "p50_latency_ms": full["p50_latency_ms"],
        "p95_latency_ms": full["p95_latency_ms"],
    }


def format_training_table(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> str:
    """Standard BEFORE vs AFTER table for training metrics (Improve % + Better?)."""
    b = normalize_training_metrics(baseline)
    o = normalize_training_metrics(optimized)
    keys = [
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "step_ms",
        "training_tokens_per_sec",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "loss",
        "validation_loss",
        "perplexity",
    ]
    rows = compare_metrics(b, o, keys=keys)
    return format_table(rows, candidate_label="Optimized")


def training_delta_summary(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> str:
    b = normalize_training_metrics(baseline)
    o = normalize_training_metrics(optimized)
    lines = ["TRAINING IMPROVEMENT", "-" * 40]
    for k, label, lower in [
        ("peak_gpu_memory_mb", "Memory", True),
        ("step_ms", "Step time", True),
        ("forward_ms", "Forward", True),
        ("backward_ms", "Backward", True),
        ("optimizer_ms", "Optimizer", True),
        ("training_tokens_per_sec", "Training throughput", False),
        ("validation_loss", "Val loss", True),
        ("perplexity", "Perplexity", True),
    ]:
        if k not in b or k not in o:
            continue
        bv, ov = float(b[k]), float(o[k])
        if bv == 0:
            continue
        pct = ((bv - ov) / abs(bv) * 100.0) if lower else ((ov - bv) / abs(bv) * 100.0)
        lines.append(f"{label}: {pct:+.2f}%")
    return "\n".join(lines)


def wall_clock_s(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
