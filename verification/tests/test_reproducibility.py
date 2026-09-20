"""Reproducibility under fixed seed."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg


def test_forward_reproducible_eval():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    cfg = cfg_mod.V4Config(vocab_size=24, n_layers=4, max_seq_len=16, dropout=0.0)

    def run(seed: int):
        torch.manual_seed(seed)
        m = model_mod.DeepSeekFlashV4Mini(cfg).eval()
        x = torch.randint(0, 24, (2, 10))
        with torch.no_grad():
            logits, _ = m(x, trace=False)
        return logits

    a = run(123)
    b = run(123)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    cfg = cfg_mod.V4Config(vocab_size=24, n_layers=4, max_seq_len=16, dropout=0.0)
    torch.manual_seed(1)
    m1 = model_mod.DeepSeekFlashV4Mini(cfg).eval()
    torch.manual_seed(2)
    m2 = model_mod.DeepSeekFlashV4Mini(cfg).eval()
    x = torch.randint(0, 24, (1, 6))
    with torch.no_grad():
        y1, _ = m1(x, trace=False)
        y2, _ = m2(x, trace=False)
    assert not torch.equal(y1, y2)
