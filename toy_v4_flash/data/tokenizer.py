"""Character-level tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
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
