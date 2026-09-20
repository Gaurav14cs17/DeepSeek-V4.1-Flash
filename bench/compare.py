"""Compare two result bundles with identical protocol and print Δ table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .delta import assert_same_protocol, compare_metrics, format_table, verdict
from .memory import format_memory_report, memory_delta
from .report import write_optimization_report


PROTOCOL_FIELDS = (
    "batch_size",
    "sequence_length",
    "generation_length",
    "precision",
    "seed",
    "warmup_iters",
    "measure_iters",
)


def _assert_same_hardware(b_cfg: dict, o_cfg: dict) -> None:
    """Refuse compares across different GPU / platform when recorded."""
    bh = b_cfg.get("hardware") or {}
    oh = o_cfg.get("hardware") or {}
    fields = []
    for key in ("gpu", "platform", "machine"):
        if key in bh and key in oh and bh[key] != oh[key]:
            fields.append(f"hardware.{key}: {bh[key]!r} vs {oh[key]!r}")
    if fields:
        raise ValueError(
            "Cannot compare — hardware mismatch:\n  " + "\n  ".join(fields)
        )


def compare_dirs(baseline: Path, optimized: Path, report: Path | None = None) -> None:
    b_cfg = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    o_cfg = json.loads((optimized / "config.json").read_text(encoding="utf-8"))
    assert_same_protocol(b_cfg, o_cfg, PROTOCOL_FIELDS)
    _assert_same_hardware(b_cfg, o_cfg)

    b_m = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    o_m = json.loads((optimized / "metrics.json").read_text(encoding="utf-8"))
    b_mem = json.loads((baseline / "memory.json").read_text(encoding="utf-8"))
    o_mem = json.loads((optimized / "memory.json").read_text(encoding="utf-8"))

    print(format_memory_report(b_mem, "MEMORY REPORT — BASELINE (A0)"))
    print()
    print(format_memory_report(o_mem, "MEMORY REPORT — CANDIDATE (A1)"))
    print()
    print(memory_delta(b_mem, o_mem))
    print()
    rows = compare_metrics(b_m, o_m)
    print(format_table(rows, candidate_label="Candidate"))
    print()
    print(verdict(rows))

    if report:
        write_optimization_report(
            report,
            name=o_cfg.get("experiment", "unnamed"),
            baseline_dir=baseline,
            optimized_dir=optimized,
        )
        print(f"\nWrote {report}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--optimized", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()
    compare_dirs(args.baseline, args.optimized, args.report)


if __name__ == "__main__":
    main()
