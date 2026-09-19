#!/usr/bin/env python3
"""Download open-source text corpora for toy training.

Supported:
  tinystories      — TinyStories (roneneldan/TinyStories on Hugging Face)
                     Paper: TinyStories (Eldan & Li). Training recipe example:
                     https://github.com/SauravP97/tiny-stories-hf
  tinyshakespeare  — public-domain Shakespeare (karpathy/char-rnn mirror)

Usage:
  python -m toy_dsv41.data.download_opensource tinystories
  python -m toy_dsv41.data.download_opensource tinystories --max-stories 20000
  python -m toy_dsv41.train --dataset tinystories --preset better --steps 2000
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OPENSOURCE_DIR = DATA_DIR / "opensource"

DATASETS = {
    "tinystories": {
        "kind": "hf",
        "hf_id": "roneneldan/TinyStories",
        "filename": "tinystories.txt",
        "default_max_stories": 20_000,
        "license": (
            "TinyStories synthetic stories (Eldan & Li). "
            "HF: roneneldan/TinyStories. "
            "Example training repo: https://github.com/SauravP97/tiny-stories-hf (MIT)"
        ),
        "about": (
            "Short children’s stories with simple vocabulary — designed for "
            "tiny LMs that still speak coherent English. Full set is large (~2GB); "
            "we export a CPU-friendly subset as plain .txt for this toy."
        ),
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
    "tinyshakespeare": {
        "kind": "url",
        "url": (
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt"
        ),
        "filename": "tinyshakespeare.txt",
        "license": "Public domain (Shakespeare) — via karpathy/char-rnn",
        "about": "Classic open char-LM corpus (~1.1M characters).",
    },
}


def _download_url(url: str, out: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as resp:
        out.write_bytes(resp.read())


def _download_tinystories(out: Path, max_stories: int) -> None:
    """Stream HF TinyStories and write a plain-text subset (one story per block)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Need `datasets` for TinyStories. In your ml env:\n"
            "  pip install datasets\n"
            f"Original error: {e}"
        ) from e

    print(f"  loading Hugging Face dataset roneneldan/TinyStories …")
    print(f"  keeping first {max_stories:,} train stories (CPU subset)")
    # streaming avoids downloading the full ~2GB when we only need a slice
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    n = 0
    chars = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            n += 1
            chars += len(text) + 2
            if n >= max_stories:
                break
            if n % 2000 == 0:
                print(f"  … {n:,} stories / {chars:,} chars")

    if n == 0:
        out.unlink(missing_ok=True)
        raise SystemExit("downloaded 0 stories — check network / HF access")
    print(f"  exported {n:,} stories, ~{chars:,} characters")


def download(
    name: str,
    force: bool = False,
    max_stories: int | None = None,
) -> Path:
    if name not in DATASETS:
        raise SystemExit(
            f"unknown dataset {name!r}. Choose from: {', '.join(DATASETS)}"
        )
    meta = DATASETS[name]
    OPENSOURCE_DIR.mkdir(parents=True, exist_ok=True)
    out = OPENSOURCE_DIR / meta["filename"]
    if out.exists() and not force:
        print(f"already exists: {out} ({out.stat().st_size:,} bytes)")
        print(f"license: {meta['license']}")
        return out

    print(f"downloading {name} …")
    print(f"  about: {meta['about']}")
    print(f"  ref  : {meta.get('url', '')}")
    print(f"  license: {meta['license']}")

    if meta["kind"] == "url":
        _download_url(meta["url"], out)
    elif meta["kind"] == "hf":
        n = max_stories or int(meta.get("default_max_stories", 20_000))
        _download_tinystories(out, max_stories=n)
    else:
        raise SystemExit(f"unknown kind {meta['kind']}")

    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Download open-source toy corpora")
    p.add_argument(
        "name",
        nargs="?",
        default="tinystories",
        choices=list(DATASETS.keys()),
        help="dataset id (default: tinystories)",
    )
    p.add_argument("--force", action="store_true", help="re-download")
    p.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="TinyStories only: how many train stories to export (default 20000)",
    )
    p.add_argument("--list", action="store_true", help="list datasets and exit")
    args = p.parse_args()
    if args.list:
        for k, v in DATASETS.items():
            print(f"{k}")
            print(f"  file: {v['filename']}")
            print(f"  {v['about']}")
            print(f"  {v['license']}")
            print()
        return
    path = download(args.name, force=args.force, max_stories=args.max_stories)
    print("\nTrain with:")
    print(f"  python -m toy_dsv41.train --dataset {args.name} --preset better --steps 2000")
    print(f"  # or: --corpus {path}")


if __name__ == "__main__":
    main()
