"""Tiny overfit + gradient finiteness checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pkg_import import import_from_pkg


def test_v4_overfit_tiny_batch():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    torch.manual_seed(0)
    cfg = cfg_mod.V4Config(vocab_size=20, n_layers=4, max_seq_len=16, dropout=0.0)
    m = model_mod.DeepSeekFlashV4Mini(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    data = torch.randint(0, 20, (8, 12))
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    losses = []
    m.train()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        logits, _ = m(data[:, :-1], trace=False)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), data[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.85, f"loss did not drop: {losses[0]} → {losses[-1]}"
    changed = any(
        not torch.equal(before[n], p.detach()) for n, p in m.named_parameters() if n in before
    )
    assert changed


def test_stage01_modules_gradient():
    emb_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.embedding")
    norm_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.normalization")
    rope_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    emb = emb_mod.TokenEmbedding(16, 32)
    norm = norm_mod.RMSNorm(32)
    ids = torch.randint(0, 16, (2, 5))
    x = emb(ids)
    x = norm(x)
    q = x.view(2, 5, 2, 16).transpose(1, 2)
    y = rope_mod.apply_rope(q)
    y.sum().backward()
    assert emb.emb.weight.grad is not None
    assert norm.weight.grad is not None
