"""Checkpoint save/load for V4.1 Mini."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from model.config import ToyConfig
from model.model import DeepSeekFlashV41Mini
from tokenizer.tokenizer import CharTokenizer


def save_checkpoint(
    path: Path,
    model: DeepSeekFlashV41Mini,
    tok: CharTokenizer,
    step: int,
    train_loss: float,
    val_loss: float,
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(model.cfg, k) for k in model.cfg.__dataclass_fields__}
    torch.save(
        {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "config": cfg,
            "tokenizer": tok.state_dict(),
            "meta": meta or {},
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "params": model.num_parameters(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> Tuple[DeepSeekFlashV41Mini, CharTokenizer, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    tok = CharTokenizer.from_state_dict(blob["tokenizer"])
    fields = ToyConfig.__dataclass_fields__
    raw = {k: v for k, v in blob["config"].items() if k in fields}
    cfg = ToyConfig(**raw)
    model = DeepSeekFlashV41Mini(cfg).to(device)
    model.load_state_dict(blob["model"])
    return model, tok, blob
