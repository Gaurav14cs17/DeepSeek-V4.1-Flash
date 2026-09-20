"""Tensor shape smoke tests across implemented modules."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg


def test_v4_full_model_shapes():
    mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    cfg = cfg_mod.V4Config(vocab_size=32, n_layers=4, max_seq_len=32)
    m = mod.DeepSeekFlashV4Mini(cfg)
    x = torch.randint(0, 32, (2, 12))
    logits, _ = m(x, trace=False)
    assert logits.shape == (2, 12, 32)


def test_v4_csa_hca_mla_shapes():
    mla = import_from_pkg("DeepSeekFlashV4-Mini", "model.mla")
    x = torch.randn(2, 16, 64)
    assert mla.CSA(64, 4, 16, compress=4, topk=4)(x).shape == x.shape
    assert mla.HCA(64, 4, 16, compress=8)(x).shape == x.shape
    assert mla.MLA(64, 4, 16, kv_lora_rank=8)(x).shape == x.shape


def test_v41_full_model_shapes():
    cfg_mod = import_from_pkg("DeepSeekFlashV4.1-Mini", "model.config")
    mod = import_from_pkg("DeepSeekFlashV4.1-Mini", "model.model")
    cfg = cfg_mod.ToyConfig(use_vision=False, use_dspark=False)
    m = mod.ToyDSV41(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 20))
    logits, info = m(x, phase="prefill", trace=True)
    assert logits.shape[0] == 1 and logits.shape[2] == cfg.vocab_size
    assert "replayed" in info


def test_deferred_public_modules_flagged():
    compression = import_from_pkg("DeepSeekFlashV4-Mini", "model.compression")
    assert getattr(compression, "DEFERRED", False) is True
