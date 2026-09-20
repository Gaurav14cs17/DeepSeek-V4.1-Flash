#!/usr/bin/env python3
"""Verify Stage-1 research gate for deepseek_flash_lab."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V4 = REPO / "DeepSeekFlashV4-Mini"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd} (exit {r.returncode})")


def main() -> None:
    print("=== deepseek_flash_lab Stage-1 verify ===")
    run([sys.executable, "tests/test_rope.py"], V4)
    run([sys.executable, "training/stage01_train.py"], V4)

    metrics_path = REPO / "results" / "stage01" / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("MISSING results/stage01/metrics.json")
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = [
        "parameters",
        "dataset_tokens",
        "train_loss",
        "validation_loss",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "training_time_s",
        "tokens_per_second",
    ]
    missing = [k for k in required if k not in m]
    if missing:
        raise SystemExit(f"metrics missing keys: {missing}")

    print("\n=== Stage-1 PASSED ===")
    print(json.dumps({k: m[k] for k in required}, indent=2))
    print("\nNext: unlock Stage 2 (vanilla attention) per docs/progress.md")
    print("Do NOT start V4.1 until V4-Mini Stages 1–3 are complete.")


if __name__ == "__main__":
    main()
