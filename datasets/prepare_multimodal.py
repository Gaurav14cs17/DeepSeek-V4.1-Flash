#!/usr/bin/env python3
"""Level 4 multimodal prepare stub — implemented in V4.1 stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=10_000)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "shards" / "mm_tiny.jsonl")
    args = p.parse_args()
    print(
        json.dumps(
            {
                "status": "deferred",
                "reason": "V4.1 multimodal stage not started",
                "args": vars(args),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
