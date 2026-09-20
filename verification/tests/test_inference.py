"""Inference pipeline verification."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg


def test_prompt_to_decode_pipeline():
    tok_mod = import_from_pkg("DeepSeekFlashV4-Mini", "tokenizer")
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")

    text = "The cat sat"
    tok = tok_mod.CharTokenizer.from_text(text + " on the mat.")
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    cfg = cfg_mod.V4Config(vocab_size=tok.vocab_size, n_layers=4, max_seq_len=32)
    m = model_mod.DeepSeekFlashV4Mini(cfg)
    m.eval()
    with torch.no_grad():
        logits, _ = m(ids, trace=False)
    assert logits.shape == (1, ids.shape[1], tok.vocab_size)
    assert torch.isfinite(logits).all()
    torch.manual_seed(0)
    out = m.generate(ids, max_new=5, temperature=0.8, top_k=5)
    decoded = tok.decode(out[0].tolist())
    assert isinstance(decoded, str)
    assert len(out[0]) == ids.shape[1] + 5


def test_greedy_ish_temperature_near_zero():
    cfg_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.config")
    model_mod = import_from_pkg("DeepSeekFlashV4-Mini", "model.model")
    torch.manual_seed(1)
    cfg = cfg_mod.V4Config(vocab_size=32, n_layers=4, max_seq_len=24)
    m = model_mod.DeepSeekFlashV4Mini(cfg).eval()
    prompt = torch.randint(0, 32, (1, 3))
    a = m.generate(prompt, max_new=4, temperature=1e-6)
    torch.manual_seed(1)
    b = m.generate(prompt, max_new=4, temperature=1e-6)
    assert torch.equal(a, b)
