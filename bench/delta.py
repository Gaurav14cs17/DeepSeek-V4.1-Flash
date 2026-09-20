"""Delta calculation and BEFORE vs AFTER tables."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

Direction = Literal["lower", "higher", "neutral"]

# Metrics where lower is better
LOWER_IS_BETTER = {
    "peak_gpu_memory_mb",
    "allocated_gpu_memory_mb",
    "reserved_gpu_memory_mb",
    "peak_cpu_memory_mb",
    "param_memory_mb",
    "activation_memory_mb",
    "activation_memory_mb_est",
    "optimizer_memory_mb",
    "gradient_memory_mb",
    "kv_cache_mb",
    "kv_bytes_per_token",
    "checkpoint_size_mb",
    "temporary_mb",
    "prefill_latency_ms",
    "decode_latency_ms",
    "avg_token_latency_ms",
    "ttft_ms",
    "total_inference_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "init_time_ms",
    "forward_ms",
    "train_forward_ms",
    "backward_ms",
    "optimizer_ms",
    "step_ms",
    "training_step_ms",
    "epoch_time_s",
    "training_time_s",
    "loss",
    "validation_loss",
    "train_loss",
    "perplexity",
    "perplexity_val",
    "max_abs_error",
    "mean_abs_error",
    "relative_error",
    "routing_imbalance",
    "dropped_tokens",
}

# Metrics where higher is better
HIGHER_IS_BETTER = {
    "tokens_per_second",
    "training_tokens_per_sec",
    "inference_tokens_per_sec",
    "prefill_tokens_per_sec",
    "decode_tokens_per_sec",
    "samples_per_sec",
    "cosine_similarity",
    "throughput",
    "routing_entropy",
    "cache_compression_ratio",
}

NEUTRAL = {
    "parameters",
    "trainable_parameters",
    "active_parameters",
    "flops_est",
    "active_experts_per_token",
    "tokens_per_expert",
    "expert_utilization",
}


def metric_direction(name: str) -> Direction:
    if name in LOWER_IS_BETTER:
        return "lower"
    if name in HIGHER_IS_BETTER:
        return "higher"
    return "neutral"


def improvement_percent(baseline: float, optimized: float, direction: Direction) -> Optional[float]:
    if baseline == 0:
        return None
    if direction == "lower":
        return (baseline - optimized) / abs(baseline) * 100.0
    if direction == "higher":
        return (optimized - baseline) / abs(baseline) * 100.0
    return 0.0


def is_better(baseline: float, optimized: float, direction: Direction, tol: float = 1e-9) -> str:
    if direction == "neutral":
        return "—"
    if direction == "lower":
        if optimized < baseline - tol:
            return "YES"
        if optimized > baseline + tol:
            return "NO"
        return "SAME"
    # higher
    if optimized > baseline + tol:
        return "YES"
    if optimized < baseline - tol:
        return "NO"
    return "SAME"


def compare_metrics(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
    keys: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return list of row dicts for the standard benchmark table."""
    if keys is None:
        keys = sorted(set(baseline) | set(optimized))
    rows = []
    for k in keys:
        if k not in baseline or k not in optimized:
            continue
        b, o = baseline[k], optimized[k]
        if not isinstance(b, (int, float)) or not isinstance(o, (int, float)):
            continue
        b_f, o_f = float(b), float(o)
        direction = metric_direction(k)
        abs_d = o_f - b_f
        rel = improvement_percent(b_f, o_f, direction)
        rows.append(
            {
                "metric": k,
                "baseline": b_f,
                "optimized": o_f,
                "absolute_delta": abs_d,
                "relative_delta_pct": rel,
                "direction": direction,
                "better": is_better(b_f, o_f, direction),
            }
        )
    return rows


def format_table(rows: List[Dict[str, Any]], *, candidate_label: str = "Candidate") -> str:
    """
    Improvement %: positive = better for that metric's direction.
      lower-is-better: (baseline - candidate) / baseline * 100
      higher-is-better: (candidate - baseline) / baseline * 100
    Negative Improvement % = regression (worse).
    """
    header = (
        f"| {'Metric':28s} | {'Baseline':>12s} | {candidate_label:>12s} | "
        f"{'Absolute Δ':>12s} | {'Improve %':>10s} | {'Better?':>7s} |"
    )
    sep = (
        f"| {'-'*28} | {'-'*12}: | {'-'*12}: | {'-'*12}: | {'-'*10}: | {'-'*7} |"
    )
    lines = [
        header,
        sep,
        # footnote as markdown-ish comment line
    ]
    for r in rows:
        rel = r["relative_delta_pct"]
        rel_s = "—" if rel is None else f"{rel:+.2f}%"
        lines.append(
            f"| {r['metric']:28s} | {r['baseline']:12.4g} | {r['optimized']:12.4g} | "
            f"{r['absolute_delta']:+12.4g} | {rel_s:>10s} | {r['better']:>7s} |"
        )
    lines.append("")
    lines.append(
        "_Improve % > 0 means better; < 0 means worse. Absolute Δ = candidate − baseline._"
    )
    return "\n".join(lines)


def verdict(rows: List[Dict[str, Any]]) -> str:
    """One-line summary: keep baseline / mixed / adopt candidate."""
    better = sum(1 for r in rows if r["better"] == "YES")
    worse = sum(1 for r in rows if r["better"] == "NO")
    if worse and not better:
        return "VERDICT: REGRESSION — keep Baseline (A0). Candidate is not an optimization."
    if better and not worse:
        return "VERDICT: IMPROVEMENT — Candidate wins on measured metrics."
    if better and worse:
        return "VERDICT: TRADE-OFF — some metrics better, some worse; read the table."
    return "VERDICT: NEUTRAL — no material change."


def assert_same_protocol(a: Dict[str, Any], b: Dict[str, Any], fields: Tuple[str, ...]) -> None:
    """Refuse unfair comparisons."""
    bad = []
    for f in fields:
        if a.get(f) != b.get(f):
            bad.append(f"{f}: {a.get(f)!r} vs {b.get(f)!r}")
    if bad:
        raise ValueError(
            "Cannot compare — protocol mismatch (fix hardware/batch/seq/etc):\n  "
            + "\n  ".join(bad)
        )
