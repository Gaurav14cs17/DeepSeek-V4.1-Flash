"""Char corpus dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from tokenizer.tokenizer import CharTokenizer

DEFAULT_CORPUS = """Once upon a time there was a little cat.
The cat liked milk and sun and naps.
One day the cat met a dog by the river.
They became friends and shared stories.
The end.
"""


def build_corpus(path: Path | None = None) -> Tuple[str, CharTokenizer, torch.Tensor]:
    if path is not None and path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        shard = Path(__file__).resolve().parents[2] / "datasets" / "shards" / "level0_char.txt"
        text = shard.read_text(encoding="utf-8") if shard.exists() else DEFAULT_CORPUS
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return text, tok, ids


def random_batch(
    ids: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    hi = ids.numel() - seq_len - 1
    if hi <= 1:
        raise ValueError("corpus too short for seq_len")
    starts = torch.randint(0, hi, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([ids[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y
