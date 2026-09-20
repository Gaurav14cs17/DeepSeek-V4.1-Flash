"""Checkpoint save/load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import torch

from model.config import V4Config
from model.model import DeepSeekFlashV4Mini
from tokenizer.tokenizer import CharTokenizer


def save_checkpoint(
    path: Path,
    model: DeepSeekFlashV4Mini,
    tok: CharTokenizer,
    step: int,
    train_loss: float,
    val_loss: float,
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg.__dict__.copy()
    cfg["layer_types"] = list(model.cfg.layer_types)
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
) -> Tuple[DeepSeekFlashV4Mini, CharTokenizer, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    tok = CharTokenizer.from_state_dict(blob["tokenizer"])
    fields = V4Config.__dataclass_fields__
    raw = {k: v for k, v in blob["config"].items() if k in fields}
    cfg = V4Config(**raw)
    model = DeepSeekFlashV4Mini(cfg).to(device)
    model.load_state_dict(blob["model"])
    return model, tok, blob
