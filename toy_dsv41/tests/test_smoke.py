"""Smoke tests for extended toy (RoPE, mHC, Vision, DSpark, CED replay)."""

from __future__ import annotations

import torch

from toy_dsv41.data import build_corpus, random_batch
from toy_dsv41.model import ToyDSV41, preset


def test_forward_shapes():
    cfg = preset("demo", vocab_size=40)
    model = ToyDSV41(cfg)
    tokens = torch.randint(0, 40, (2, 16))
    logits, trace = model(
        tokens, phase="prefill", use_ced_prefill=False, use_bounded_replay=False, trace=True
    )
    assert logits.shape == (2, 16, 40)
    assert trace["kv"].tokens == 16


def test_ced_bounded_replay():
    cfg = preset("demo", vocab_size=40)
    model = ToyDSV41(cfg)
    tokens = torch.randint(0, 40, (1, 24))
    logits, trace = model(
        tokens, phase="prefill", use_ced_prefill=True, use_bounded_replay=True, trace=True
    )
    assert logits.shape[1] == 24
    assert trace["replayed"] is True
    assert any("Bounded Replay" in line for line in trace["log"])


def test_vision_path():
    cfg = preset("demo", vocab_size=40)
    model = ToyDSV41(cfg)
    tokens = torch.randint(0, 40, (1, 8))
    # 1×16×16 grayscale → patches
    images = torch.randn(1, 1, 16, 16)
    logits, trace = model(
        tokens,
        phase="prefill",
        images=images,
        use_ced_prefill=False,
        use_bounded_replay=False,
        trace=True,
    )
    assert logits.shape[0] == 1
    assert any("Vision" in line for line in trace["log"])


def test_dspark_generate():
    model = ToyDSV41(preset("demo", vocab_size=32))
    prompt = torch.randint(0, 32, (1, 4))
    out = model.generate(prompt, max_new=6, top_k=5, use_dspark=True)
    assert out.shape[1] == 10


def test_corpus_and_batch():
    bundle = build_corpus(dataset="toy")
    assert bundle.source.startswith("builtin:")
    assert bundle.vocab_size > 10
    assert bundle.train_ids.numel() > 100
    x, y = random_batch(bundle.train_ids, 4, 32, torch.device("cpu"))
    assert x.shape == (4, 32) and y.shape == (4, 32)


def test_presets():
    for name in ("demo", "pc", "better"):
        cfg = preset(name, vocab_size=32)
        m = ToyDSV41(cfg)
        assert m.num_parameters() > 1000
        assert cfg.use_mhc and cfg.use_rope and cfg.use_dspark and cfg.use_vision


def test_generate():
    model = ToyDSV41(preset("demo", vocab_size=32))
    prompt = torch.randint(0, 32, (1, 4))
    out = model.generate(prompt, max_new=3, top_k=5, use_dspark=False)
    assert out.shape[1] == 7


if __name__ == "__main__":
    test_forward_shapes()
    test_ced_bounded_replay()
    test_vision_path()
    test_dspark_generate()
    test_corpus_and_batch()
    test_presets()
    test_generate()
    print("all tests passed")
