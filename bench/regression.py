"""Automatic regression checks vs a previous metrics.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .delta import LOWER_IS_BETTER, HIGHER_IS_BETTER


def check_regression(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    *,
    threshold_pct: float = 10.0,
) -> List[str]:
    """
    Return WARNING strings if current is >threshold worse than previous
    on memory/latency/throughput/quality metrics.
    """
    warnings: List[str] = []

    def pct_change(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return (new - old) / abs(old) * 100.0

    for k in LOWER_IS_BETTER:
        if k not in previous or k not in current:
            continue
        old, new = float(previous[k]), float(current[k])
        # increase is worse for lower-is-better
        ch = pct_change(old, new)
        if ch > threshold_pct:
            warnings.append(
                f"WARNING: {k} increased {ch:+.1f}% (prev={old:.4g}, now={new:.4g})"
            )

    for k in HIGHER_IS_BETTER:
        if k not in previous or k not in current:
            continue
        old, new = float(previous[k]), float(current[k])
        ch = pct_change(old, new)
        if ch < -threshold_pct:
            warnings.append(
                f"WARNING: {k} decreased {ch:+.1f}% (prev={old:.4g}, now={new:.4g})"
            )

    return warnings


def compare_result_dirs(prev_dir: Path, curr_dir: Path, threshold_pct: float = 10.0) -> List[str]:
    prev = json.loads((prev_dir / "metrics.json").read_text(encoding="utf-8"))
    curr = json.loads((curr_dir / "metrics.json").read_text(encoding="utf-8"))
    return check_regression(prev, curr, threshold_pct=threshold_pct)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Regression check between two result bundles")
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=10.0)
    args = p.parse_args()
    warns = compare_result_dirs(args.previous, args.current, args.threshold)
    if not warns:
        print("OK: no regressions above threshold")
    else:
        for w in warns:
            print(w)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
