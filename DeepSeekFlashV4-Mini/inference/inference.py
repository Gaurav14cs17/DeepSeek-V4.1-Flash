#!/usr/bin/env python3
"""CLI inference for DeepSeekFlashV4-Mini."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from inference.generate import generate  # noqa: E402
from model import DeepSeekFlashV4Mini, preset  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.dataset import build_corpus  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--preset", default="pc")
    args = p.parse_args()
    device = torch.device("cpu")

    if args.ckpt and args.ckpt.exists():
        model, tok, _ = load_checkpoint(args.ckpt, device)
    else:
        _, tok, _ = build_corpus(None)
        model = DeepSeekFlashV4Mini(preset(args.preset, tok.vocab_size)).to(device)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    out = generate(model, ids, max_new=args.max_new, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
