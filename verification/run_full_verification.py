#!/usr/bin/env python3
"""Full verification entrypoint.

Runs:
  1) automatic code audit
  2) full pytest suite under verification/tests
  3) numerical suite
  4) benchmark verification
  5) writes a machine-readable summary JSON

Human-readable reports live alongside this script:
  IMPLEMENTATION_INVENTORY.md
  paper_to_code.md
  gradient_audit.md
  PAPER_CLAIM_AUDIT.md
  FINAL_VERIFICATION_REPORT.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VER = Path(__file__).resolve().parent
OUT = VER / "results"
OUT.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> dict:
    print("\n>>>", " ".join(cmd))
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    return {"cmd": cmd, "returncode": rc, "seconds": round(time.perf_counter() - t0, 3)}


def main() -> int:
    summary = {"steps": []}
    summary["steps"].append(run([sys.executable, str(VER / "audit_code.py")]))
    summary["steps"].append(
        run([sys.executable, "-m", "pytest", "-q", str(VER / "tests")])
    )
    summary["steps"].append(run([sys.executable, str(VER / "run_numerical_verification.py")]))
    summary["steps"].append(run([sys.executable, str(VER / "run_benchmark_verification.py")]))

    failed = [s for s in summary["steps"] if s["returncode"] != 0]
    summary["ok"] = len(failed) == 0
    summary["failed_steps"] = failed
    (OUT / "full_verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n=== FULL VERIFICATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
