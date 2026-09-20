"""Vocabulary helpers for the char-level lab tokenizer."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")


def build_char_vocab(text: str, *, add_special: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    if add_special:
        ordered: List[str] = list(SPECIAL_TOKENS) + [c for c in chars if c not in SPECIAL_TOKENS]
    else:
        ordered = chars
    stoi = {ch: i for i, ch in enumerate(ordered)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def merge_vocabs(*texts: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    return build_char_vocab("".join(texts))


def vocab_size(stoi: Dict[str, int]) -> int:
    return len(stoi)


def save_vocab(path: str, stoi: Dict[str, int]) -> None:
    from pathlib import Path
    import json

    Path(path).write_text(json.dumps(stoi, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    from pathlib import Path
    import json

    stoi = json.loads(Path(path).read_text(encoding="utf-8"))
    itos = {int(v) if isinstance(v, str) and v.isdigit() else v: k for k, v in
            ({v: k for k, v in stoi.items()}).items()}
    # stoi is char -> id
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
