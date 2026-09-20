#!/usr/bin/env python3
"""Benchmark verification — hardware report + micro latency + existing B0 artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def hardware() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_vram_gb": None,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 3
        )
    return info


def micro_bench() -> dict:
    sys.path.insert(0, str(ROOT / "DeepSeekFlashV4-Mini"))
    from model.config import V4Config
    from model.model import DeepSeekFlashV4Mini

    cfg = V4Config(vocab_size=32, n_layers=4, max_seq_len=64)
    m = DeepSeekFlashV4Mini(cfg).eval()
    device = torch.device("cpu")
    m.to(device)
    results = {}
    for seq in (128, 256):
        if seq > cfg.max_seq_len:
            # still run but truncate context path uses min
            pass
        x = torch.randint(0, 32, (1, min(seq, cfg.max_seq_len)), device=device)
        with torch.no_grad():
            for _ in range(2):
                m(x, trace=False)
            t0 = time.perf_counter()
            n = 5
            for _ in range(n):
                m(x, trace=False)
            dt = (time.perf_counter() - t0) / n
        results[f"prefill_T{x.shape[1]}_ms"] = round(dt * 1000, 3)
    return results


def main() -> int:
    hw = hardware()
    print("hardware:", json.dumps(hw, indent=2))
    micro = micro_bench()
    print("micro:", json.dumps(micro, indent=2))
    payload = {"hardware": hw, "micro_latency": micro}
    baseline = ROOT / "results" / "v4" / "stage01" / "baseline" / "metrics.json"
    payload["stage01_baseline_exists"] = baseline.exists()
    if baseline.exists():
        payload["stage01_baseline"] = json.loads(baseline.read_text(encoding="utf-8"))
    rope_report = ROOT / "results" / "v4" / "rope_formulation" / "optimization_report.md"
    payload["rope_delta_report_exists"] = rope_report.exists()
    (OUT / "benchmark_verification.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    # also run pytest benchmark smoke
    rc = subprocess.call(
        [sys.executable, "-m", "pytest", "-q", str(Path(__file__).parent / "tests" / "test_benchmarks.py")],
        cwd=str(ROOT),
    )
    print(f"wrote {OUT / 'benchmark_verification.json'} pytest_rc={rc}")
    # No GPU: document inability to verify 2GB GPU path
    if not hw["cuda_available"]:
        print("NOTE: CUDA unavailable — 2 GB GPU verification cannot run on this host.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
