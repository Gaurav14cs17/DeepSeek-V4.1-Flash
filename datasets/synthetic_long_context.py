#!/usr/bin/env python3
"""Level 5: synthetic long-context sequences for KV / CSA2 labs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096, help="tokens per sample")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "shards" / "long_ctx.pt")
    args = p.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    seq = min(args.max_seq_len, args.max_tokens)
    data = torch.randint(0, args.vocab_size, (args.num_samples, seq), generator=g)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": data, "meta": vars(args)}, args.out)
    print(json.dumps({"path": str(args.out), "shape": list(data.shape)}, indent=2))


if __name__ == "__main__":
    main()
