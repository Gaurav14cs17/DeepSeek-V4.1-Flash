"""Memory / parameter accounting checks."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg


def test_independent_param_count_matches():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(cfg_mod.V4Config(vocab_size=32, n_layers=4))
    indep = sum(p.numel() for p in m.parameters())
    assert m.num_parameters() == indep


def test_param_bytes_estimate():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(cfg_mod.V4Config(vocab_size=32, n_layers=4))
    n = m.num_parameters()
    bytes_fp32 = n * 4
    assert bytes_fp32 / (1024**2) < 50


def test_cpu_forward_peak_reasonable():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(cfg_mod.V4Config(vocab_size=32, n_layers=4, max_seq_len=32))
    x = torch.randint(0, 32, (2, 16))
    logits, _ = m(x, trace=False)
    assert logits.numel() * 4 / (1024**2) < 10
