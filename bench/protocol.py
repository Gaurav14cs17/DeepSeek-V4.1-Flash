"""Experiment metadata + reproducible result bundles."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[1]


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["gpu_vram_total_mb"] = round(total / (1024**2), 2)
            info["gpu_vram_free_mb"] = round(free / (1024**2), 2)
        else:
            info["gpu"] = None
            info["gpu_vram_total_mb"] = 0.0
            info["gpu_vram_free_mb"] = 0.0
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    return info


def software_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "pytorch": None,
        "cuda_version": None,
    }
    try:
        import torch

        out["pytorch"] = torch.__version__
        out["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    return out


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


@dataclass
class ExperimentMeta:
    model: str  # DeepSeekFlashV4-Mini | DeepSeekFlashV4.1-Mini
    experiment: str
    component: str
    version: str  # baseline | optimized | A0 | A1 ...
    baseline_version: Optional[str] = None
    precision: str = "fp32"
    batch_size: int = 1
    sequence_length: int = 64
    generation_length: int = 0
    warmup_iters: int = 3
    measure_iters: int = 10
    seed: int = 42
    config_name: str = "v4_tiny"
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hardware"] = hardware_info()
        d["software"] = software_info()
        d["git_commit"] = git_commit()
        d["timestamp"] = datetime.now(timezone.utc).isoformat()
        return d


def result_dir(model_key: str, experiment: str, version: str) -> Path:
    """results/v4/<experiment>/<version>/"""
    root = REPO / "results" / model_key / experiment / version
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_result_bundle(
    model_key: str,
    experiment: str,
    version: str,
    meta: ExperimentMeta | Dict[str, Any],
    metrics: Dict[str, Any],
    memory: Dict[str, Any],
) -> Path:
    d = result_dir(model_key, experiment, version)
    cfg = meta.to_dict() if isinstance(meta, ExperimentMeta) else meta
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (d / "memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
    return d
