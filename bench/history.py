"""Append rows to comparison/optimization_history.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Mapping

REPO = Path(__file__).resolve().parents[1]
HISTORY_PATH = REPO / "comparison" / "optimization_history.csv"

COLUMNS = [
    "experiment",
    "component",
    "version",
    "baseline_version",
    "parameters",
    "active_parameters",
    "gpu_memory_mb",
    "cpu_memory_mb",
    "prefill_ms",
    "decode_ms",
    "total_inference_ms",
    "training_step_ms",
    "training_tokens_per_sec",
    "inference_tokens_per_sec",
    "kv_bytes_per_token",
    "loss",
    "validation_loss",
    "perplexity",
    "max_abs_error",
    "mean_abs_error",
    "timestamp",
    "git_commit",
]


def append_history(row: Mapping[str, Any], path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        out = {c: row.get(c, "") for c in COLUMNS}
        w.writerow(out)
