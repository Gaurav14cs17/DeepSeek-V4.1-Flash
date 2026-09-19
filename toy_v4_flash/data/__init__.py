"""Corpus loading + train/val char dataset for CPU training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = DATA_DIR / "corpus_toy.txt"
OPENSOURCE_DIR = DATA_DIR / "opensource"
TINY_SHAKESPEARE = OPENSOURCE_DIR / "tinyshakespeare.txt"
TINY_STORIES = OPENSOURCE_DIR / "tinystories.txt"


def load_text(path: Optional[str | Path] = None) -> str:
    p = Path(path) if path else DEFAULT_CORPUS
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"empty corpus: {p}")
    return text


load_corpus = load_text


def ensure_opensource(name: str = "tinyshakespeare") -> Path:
    """Download open-source corpus if missing; return path."""
    from .download_opensource import download

    return download(name, force=False)


def resolve_corpus(
    corpus: Optional[str | Path] = None,
    dataset: str = "tinystories",
) -> Tuple[Path, str]:
    """
    Priority:
      1) explicit --corpus path
      2) named --dataset (default: tinystories — open TinyStories HF subset)
      3) builtin corpus_toy.txt if dataset == toy
    """
    if corpus is not None:
        path = Path(corpus)
        if not path.exists():
            raise FileNotFoundError(f"corpus not found: {path}")
        return path, f"custom:{path}"

    dataset = (dataset or "tinystories").lower().strip()
    if dataset in ("toy", "builtin", "corpus_toy"):
        return DEFAULT_CORPUS, "builtin:corpus_toy.txt"

    if dataset in ("tinystories", "tiny-stories", "tinystory"):
        path = ensure_opensource("tinystories")
        return path, "opensource:tinystories"

    if dataset in ("tinyshakespeare", "shakespeare", "open", "opensource"):
        path = ensure_opensource("tinyshakespeare")
        return path, "opensource:tinyshakespeare"

    raise ValueError(
        f"unknown dataset {dataset!r}; "
        "use tinystories | tinyshakespeare | toy | or --corpus PATH"
    )


@dataclass
class CorpusBundle:
    text: str
    tokenizer: CharTokenizer
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    source: str = ""

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def build_corpus(
    path: Optional[str | Path] = None,
    val_ratio: float = 0.1,
    dataset: str = "tinystories",
) -> CorpusBundle:
    resolved, source = resolve_corpus(path, dataset=dataset)
    text = load_text(resolved)
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = ids.numel()
    n_val = max(64, int(n * val_ratio))
    n_train = n - n_val
    if n_train < 128:
        train_ids, val_ids = ids, ids[: min(128, n)]
    else:
        train_ids, val_ids = ids[:n_train], ids[n_train:]
    return CorpusBundle(
        text=text,
        tokenizer=tok,
        train_ids=train_ids,
        val_ids=val_ids,
        source=source,
    )


class CharLMDataset(Dataset):
    def __init__(self, ids: torch.Tensor, seq_len: int):
        if ids.numel() <= seq_len + 1:
            raise ValueError(
                f"ids length {ids.numel()} too short for seq_len={seq_len}"
            )
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.ids.numel() - self.seq_len - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[idx : idx + self.seq_len]
        y = self.ids[idx + 1 : idx + 1 + self.seq_len]
        return x, y


def random_batch(
    ids: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = ids.numel() - seq_len - 1
    if n < 1:
        raise ValueError("corpus too short for seq_len")
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([ids[i : i + seq_len] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + seq_len] for i in ix])
    return x.to(device), y.to(device)
