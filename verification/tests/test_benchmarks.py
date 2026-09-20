"""Benchmark harness smoke — ensure protocol modules import and produce numbers."""

from __future__ import annotations

import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

from pkg_import import import_from_pkg  # noqa: E402


def test_bench_protocol_import():
    import sys

    sys.path.insert(0, str(REPO))
    from bench import delta, protocol  # noqa: F401


def test_micro_latency_measurable():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(
        cfg_mod.V4Config(vocab_size=32, n_layers=4, max_seq_len=32)
    ).eval()
    x = torch.randint(0, 32, (1, 16))
    with torch.no_grad():
        for _ in range(2):
            m(x, trace=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            m(x, trace=False)
    dt = (time.perf_counter() - t0) / 5
    assert dt > 0
    assert dt < 5.0


def test_existing_stage01_baseline_json():
    path = REPO / "results" / "v4" / "stage01" / "baseline" / "metrics.json"
    assert path.exists(), "Stage-1 B0 metrics missing — run scripts/run_stage01_baseline.py"
