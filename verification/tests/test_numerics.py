"""Broader numerical checks (NaN/Inf, compression ratios, CSA vs dense gap)."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg
from reference import cosine_sim, max_mean_abs


def test_no_nan_in_v4_forward_backward():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(cfg_mod.V4Config(vocab_size=24, n_layers=4, max_seq_len=16))
    x = torch.randint(0, 24, (2, 10))
    logits, _ = m(x, trace=False)
    assert torch.isfinite(logits).all()
    logits.float().sum().backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_compression_ratio_csa_tensors():
    mla = import_from_pkg("DeepSeekFlashV4-Mini", "model.mla")
    x = torch.randn(2, 32, 64)
    y = mla.compress_overlap(x, m=4)
    assert y.shape[1] < x.shape[1]
    ratio = (x.numel() * 4) / (y.numel() * 4)
    assert ratio > 1.5


def test_csa_differs_from_dense_causal():
    """DSA/CSA is an approximation — quantify gap vs CausalSelfAttention."""
    attn = import_from_pkg("DeepSeekFlashV4-Mini", "model.attention")
    mla = import_from_pkg("DeepSeekFlashV4-Mini", "model.mla")
    torch.manual_seed(0)
    x = torch.randn(1, 16, 64)
    dense = attn.CausalSelfAttention(64, 4, 16, use_rope=True)
    sparse = mla.CSA(64, 4, 16, compress=4, topk=4, swa_window=8, use_rope=True)
    yd = dense(x)
    ys = sparse(x)
    assert torch.isfinite(yd).all() and torch.isfinite(ys).all()
    mx, mn = max_mean_abs(yd.detach(), ys.detach())
    assert mx > 0
    _ = cosine_sim(yd.detach(), ys.detach())


def test_v41_replay_zeros_prefix_logits():
    """Known behavior: bounded replay pads prefix logits with zeros."""
    cfg_mod = import_from_pkg("DeepSeekFlashV4.1-Mini", "model.config")
    mod = import_from_pkg("DeepSeekFlashV4.1-Mini", "model.model")
    cfg = cfg_mod.ToyConfig(use_vision=False, use_dspark=False, swa_window=8)
    m = mod.ToyDSV41(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 20))
    with torch.no_grad():
        logits, info = m(x, phase="prefill", use_bounded_replay=True, trace=True)
    assert info["replayed"] is True
    assert torch.count_nonzero(logits[:, :-cfg.swa_window]) == 0
