"""Tests for V4.1-specific behavior (CED, Engram, Bounded Replay)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV41Mini, preset
from optimization.kv_cache import swa_bounded_replay


def test_forward_ced():
    cfg = preset("demo")
    cfg.use_vision = False
    m = DeepSeekFlashV41Mini(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    logits, info = m(x, phase="prefill", use_bounded_replay=True, use_ced_prefill=True)
    assert logits.shape[0] == 2
    assert logits.shape[-1] == cfg.vocab_size
    assert isinstance(info.get("log"), list)


def test_bounded_replay_trim():
    h = torch.randn(2, 40, 64)
    out = swa_bounded_replay(h, 8)
    assert out.shape == (2, 8, 64)


if __name__ == "__main__":
    test_forward_ced()
    test_bounded_replay_trim()
    print("ok")
