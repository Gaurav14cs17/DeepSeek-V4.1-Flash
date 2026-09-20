"""KV cache correctness — current lab is accounting-only (document failure)."""

from __future__ import annotations

import inspect

import torch

from pkg_import import import_from_pkg


def test_v4_kv_estimate_runs():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    kv = import_from_pkg("DeepSeekFlashV4-Mini", "optimization.kv_cache")
    cfg = cfg_mod.V4Config()
    stats = kv.estimate_kv(cfg, seq_len=64)
    assert stats.tokens == 64
    assert kv.exact_swa_replay_span(cfg.n_layers, cfg.swa_window) == cfg.n_layers * cfg.swa_window


def test_no_incremental_kv_cache_api():
    """FAIL condition documented: generate() recomputes full context each step."""
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(
        cfg_mod.V4Config(vocab_size=32, n_layers=4, max_seq_len=32)
    )
    sig = inspect.signature(m.forward)
    assert "past_key_values" not in sig.parameters
    assert "cache" not in sig.parameters


def test_generate_greedy_recompute_only():
    torch.manual_seed(0)
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    m = model_mod.DeepSeekFlashV4Mini(
        cfg_mod.V4Config(vocab_size=32, n_layers=4, max_seq_len=32)
    )
    m.eval()
    prompt = torch.randint(0, 32, (1, 4))
    out = m.generate(prompt, max_new=4, temperature=1e-8, top_k=0)
    assert out.shape[1] == 8
