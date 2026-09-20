#!/usr/bin/env python3
"""Download tiny dataset shards with hard caps (never huge by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cap-aware dataset download for the Mini lab")
    p.add_argument(
        "--dataset",
        default="tinystories",
        choices=["tinystories", "fineweb", "fineweb-edu"],
        help="Logical dataset name",
    )
    p.add_argument("--num-samples", type=int, default=256, help="Max examples to keep")
    p.add_argument("--max-tokens", type=int, default=50_000, help="Approx char/token budget")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--streaming", action="store_true", help="HF streaming mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "shards" / "tinystories_tiny.txt",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not download",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plan = {
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "max_tokens": args.max_tokens,
        "max_seq_len": args.max_seq_len,
        "streaming": args.streaming,
        "seed": args.seed,
        "out": str(args.out),
        "note": "Defaults are intentionally tiny for ≤2GB labs.",
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Install `datasets` or use Level-0 synthetic data. "
            "Stage-1 does not require a download."
        ) from e

    name_map = {
        "tinystories": ("roneneldan/TinyStories", "train"),
        "fineweb": ("HuggingFaceFW/fineweb", "default"),
        "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "default"),
    }
    hf_name, split = name_map[args.dataset]
    ds = load_dataset(hf_name, split=split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    chars = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or row.get("story") or ""
            if not text:
                continue
            # truncate long docs
            text = text[: args.max_seq_len * 4]
            f.write(text.replace("\n", " ").strip() + "\n")
            kept += 1
            chars += len(text)
            if kept >= args.num_samples or chars >= args.max_tokens:
                break

    meta = {"kept_samples": kept, "approx_chars": chars, "path": str(args.out)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
