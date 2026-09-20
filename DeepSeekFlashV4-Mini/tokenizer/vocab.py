"""Vocabulary helpers.

[SIMPLIFIED] Production DeepSeek uses a large BPE vocabulary (report Table 1: 129280 for V4.1).
This lab uses a character vocabulary so Level-0/1 experiments run on CPU / 2 GB GPU.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def build_char_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars: List[str] = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def save_vocab_json(path: str, stoi: Dict[str, int]) -> None:
    import json
    from pathlib import Path

    Path(path).write_text(json.dumps(stoi, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab_json(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    import json
    from pathlib import Path

    stoi = json.loads(Path(path).read_text(encoding="utf-8"))
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
