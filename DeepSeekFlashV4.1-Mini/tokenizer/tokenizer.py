"""Character-level tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .vocab import build_char_vocab


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        stoi, itos = build_char_vocab(text)
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        unk = self.stoi.get(" ", next(iter(self.stoi.values())))
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        itos = {int(k): v for k, v in state["itos"].items()}
        return cls(stoi=state["stoi"], itos=itos)


# Alias used across the lab
Tokenizer = CharTokenizer
