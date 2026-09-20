"""Memory measurement + breakdown reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import psutil


@dataclass
class MemoryReport:
    parameters: int = 0
    trainable_parameters: int = 0
    param_memory_mb: float = 0.0
    gradient_memory_mb: float = 0.0
    optimizer_memory_mb: float = 0.0
    activation_memory_mb_est: float = 0.0
    kv_cache_mb: float = 0.0
    temporary_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    allocated_gpu_memory_mb: float = 0.0
    reserved_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    checkpoint_size_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cpu_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def collect_memory(
    model=None,
    *,
    batch: int = 1,
    seq: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    kv_bytes_per_token: float = 0.0,
    optimizer_state_bytes_per_param: float = 8.0,  # AdamW ~2×fp32
    include_gradients: bool = True,
) -> MemoryReport:
    import torch

    rep = MemoryReport()
    rep.peak_cpu_memory_mb = round(_cpu_rss_mb(), 3)

    if model is not None:
        params = [p for p in model.parameters()]
        n = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        bytes_params = sum(p.numel() * p.element_size() for p in params)
        rep.parameters = int(n)
        rep.trainable_parameters = int(trainable)
        rep.param_memory_mb = round(bytes_params / (1024**2), 4)
        if include_gradients:
            rep.gradient_memory_mb = rep.param_memory_mb  # fp32 grads ≈ params
        rep.optimizer_memory_mb = round(
            trainable * optimizer_state_bytes_per_param / (1024**2), 4
        )

    # rough activation estimate (fp32 residual stream per layer)
    rep.activation_memory_mb_est = round(
        batch * seq * d_model * n_layers * 4 / (1024**2), 4
    )
    rep.kv_cache_mb = round(batch * seq * kv_bytes_per_token / (1024**2), 4)

    if torch.cuda.is_available():
        rep.allocated_gpu_memory_mb = round(torch.cuda.memory_allocated() / (1024**2), 3)
        rep.reserved_gpu_memory_mb = round(torch.cuda.memory_reserved() / (1024**2), 3)
        rep.peak_gpu_memory_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 3)
    else:
        rep.peak_gpu_memory_mb = 0.0

    return rep


def format_memory_report(rep: MemoryReport | Dict[str, Any], title: str = "MEMORY REPORT") -> str:
    d = rep.to_dict() if isinstance(rep, MemoryReport) else rep
    lines = [
        "=" * 50,
        title,
        "=" * 50,
        f"Model parameters:      {d.get('parameters', 0):,}",
        f"Trainable parameters:  {d.get('trainable_parameters', 0):,}",
        f"Parameter memory:      {d.get('param_memory_mb', 0):.3f} MB",
        f"Gradient memory:       {d.get('gradient_memory_mb', 0):.3f} MB",
        f"Optimizer memory:      {d.get('optimizer_memory_mb', 0):.3f} MB",
        f"Activation memory:     {d.get('activation_memory_mb_est', 0):.3f} MB",
        f"KV cache:              {d.get('kv_cache_mb', 0):.3f} MB",
        f"Temporary tensors:     {d.get('temporary_mb', 0):.3f} MB",
        f"Peak GPU memory:       {d.get('peak_gpu_memory_mb', 0):.3f} MB",
        f"Allocated GPU:         {d.get('allocated_gpu_memory_mb', 0):.3f} MB",
        f"Reserved GPU:          {d.get('reserved_gpu_memory_mb', 0):.3f} MB",
        f"Peak CPU RAM:          {d.get('peak_cpu_memory_mb', 0):.3f} MB",
        "=" * 50,
    ]
    return "\n".join(lines)


def memory_delta(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    keys = [
        "peak_gpu_memory_mb",
        "kv_cache_mb",
        "activation_memory_mb_est",
        "param_memory_mb",
        "peak_cpu_memory_mb",
    ]
    lines = ["MEMORY IMPROVEMENT", "-" * 40]
    for k in keys:
        if k not in before or k not in after:
            continue
        b, a = float(before[k]), float(after[k])
        abs_d = a - b
        pct = (b - a) / b * 100 if b else 0.0
        lines.append(f"{k}: {abs_d:+.3f}  ({pct:+.2f}% vs baseline; lower is better)")
    return "\n".join(lines)
