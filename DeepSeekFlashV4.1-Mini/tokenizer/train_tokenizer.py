#!/usr/bin/env python3
"""Fit a char vocab from a text file and optionally save it."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer import CharTokenizer
from vocab import save_vocab


def main() -> None:
    p = argparse.ArgumentParser(description="Train char tokenizer from a corpus file")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("vocab.json"))
    args = p.parse_args()
    text = args.corpus.read_text(encoding="utf-8", errors="ignore")
    tok = CharTokenizer.from_text(text)
    save_vocab(str(args.out), tok.stoi)
    print(f"vocab_size={tok.vocab_size} → {args.out}")


if __name__ == "__main__":
    main()
