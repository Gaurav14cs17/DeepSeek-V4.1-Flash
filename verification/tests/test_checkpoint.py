"""Checkpoint save/load output consistency."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg


def test_checkpoint_roundtrip_logits(tmp_path):
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    torch.manual_seed(0)
    cfg = cfg_mod.V4Config(vocab_size=24, n_layers=4, max_seq_len=16)
    m = model_mod.DeepSeekFlashV4Mini(cfg).eval()
    x = torch.randint(0, 24, (1, 8))
    with torch.no_grad():
        logits1, _ = m(x, trace=False)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg.__dict__}, path)
    m2 = model_mod.DeepSeekFlashV4Mini(cfg).eval()
    m2.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    with torch.no_grad():
        logits2, _ = m2(x, trace=False)
    assert torch.allclose(logits1, logits2, atol=0, rtol=0)
