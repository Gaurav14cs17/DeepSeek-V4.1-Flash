"""Unit tests for paper components (RoPE, mHC, CSA2, MoE, Engram, Vision, DSpark)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from toy_dsv41.model.config import ToyConfig, preset
from toy_dsv41.model.csa2 import CSA2, SharedGlobalState
from toy_dsv41.model.dspark import DSparkDrafter
from toy_dsv41.model.engram import ToyEngram, pack_engram_to_disk, load_engram_from_disk
from toy_dsv41.model.kv_cache import estimate_kv, swa_bounded_replay
from toy_dsv41.model.mhc import SinglePassMHC
from toy_dsv41.model.moe import TinyMoE
from toy_dsv41.model.rope import apply_rope, apply_rope_2d, rope_freqs
from toy_dsv41.model.swa import SlidingWindowAttention
from toy_dsv41.model.transformer import ToyDSV41
from toy_dsv41.model.vision import TinyDeepSeekViT


# ----- RoPE -----


def test_rope_preserves_shape():
    x = torch.randn(2, 4, 16, 16)  # B,H,T,D
    y = apply_rope(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_2d_grid():
    h, w = 4, 4
    x = torch.randn(1, 2, h * w, 16)
    y = apply_rope_2d(x, h, w)
    assert y.shape == x.shape


def test_rope_freqs_length():
    f = rope_freqs(16, 32, torch.device("cpu"))
    assert f.shape == (32, 8)
    assert f.is_complex()


# ----- mHC -----


def test_mhc_mix_and_update():
    mhc = SinglePassMHC(d_model=32, n_streams=2)
    x = torch.randn(2, 8, 32)
    streams = mhc.expand_from_hidden(x)
    assert streams.shape == (2, 8, 2, 32)
    A, B, C = mhc.predict(streams)
    mixed = mhc.mix_input(streams, A)
    assert mixed.shape == (2, 8, 32)
    out = mhc.residual_update(streams, mixed, B, C)
    assert out.shape == streams.shape
    assert mhc.collapse(out).shape == (2, 8, 32)


# ----- SWA -----


def test_swa_causal_window():
    swa = SlidingWindowAttention(32, 4, 8, window=4, use_rope=True)
    x = torch.randn(1, 12, 32)
    y = swa(x)
    assert y.shape == x.shape


# ----- CSA2 modes -----


def test_csa2_full_reindex_reuse():
    attn = CSA2(
        d_model=32,
        n_heads=4,
        d_head=8,
        topk=4,
        compress=2,
        mode="full",
        candidate_pool=8,
        use_fp4=True,
        swa_window=4,
    )
    x = torch.randn(1, 16, 32)
    shared = SharedGlobalState()
    y, shared = attn(x, shared)
    assert y.shape == x.shape
    assert shared.main_kv is not None
    assert shared.candidate_pool is not None

    attn.mode = "reindex"
    y2, shared2 = attn(x, shared)
    assert y2.shape == x.shape
    assert shared2.topk_idx is not None

    attn.mode = "reuse"
    y3, shared3 = attn(x, shared2)
    assert y3.shape == x.shape


# ----- MoE -----


def test_moe_aux_free_bias_updates():
    moe = TinyMoE(32, n_routed=4, n_shared=1, n_activated=2, hidden=64)
    moe.train()
    x = torch.randn(2, 8, 32)
    y = moe(x, modality="text")
    assert y.shape == x.shape
    assert torch.isfinite(moe.bias_text).all()
    y_img = moe(x, modality="image")
    assert y_img.shape == x.shape
    assert torch.isfinite(moe.bias_image).all()


# ----- Engram -----


def test_engram_forward_and_pack():
    from pathlib import Path

    eng = ToyEngram(40, 32, n_entries=64, emb_dim=16, n_heads=2, ngram_orders=(2, 3, 4))
    tokens = torch.randint(0, 40, (2, 10))
    h = torch.randn(2, 10, 32)
    out = eng(tokens, h)
    assert out.shape == h.shape

    path = Path(__file__).resolve().parent.parent / "_artifacts" / "engram_test.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_engram_to_disk(eng, str(path))
    eng2 = ToyEngram(40, 32, n_entries=64, emb_dim=16, n_heads=2, ngram_orders=(2, 3, 4))
    load_engram_from_disk(eng2, str(path))
    for k in eng.tables:
        assert torch.allclose(eng.tables[k], eng2.tables[k])

    eng.sinkhorn_balance_(steps=2)
    assert torch.isfinite(eng.tables["o2_h0"]).all()


# ----- Vision -----


def test_vision_unshuffle():
    vit = TinyDeepSeekViT(d_model=32, patch=4, n_heads=2, d_head=16, depth=1)
    images = torch.randn(1, 1, 16, 16)
    vis = vit(images)
    assert vis.dim() == 3 and vis.shape[0] == 1 and vis.shape[-1] == 32
    assert vis.shape[1] >= 1


# ----- DSpark -----


def test_dspark_draft_and_accept():
    ds = DSparkDrafter(d_model=32, vocab_size=50, n_heads=4, d_head=8, draft_len=5)
    h = torch.randn(1, 8, 32)
    draft, conf, base = ds.draft(h)
    assert draft.shape == (1, 5)
    assert conf.shape == (1, 5)
    assert base.shape == (1, 5, 50)
    n = ds.verify_and_accept(draft, conf, draft, conf_threshold=0.0)
    assert n == 5


# ----- KV / Bounded Replay -----


def test_kv_estimate_and_replay():
    cfg = preset("demo", vocab_size=32)
    stats = estimate_kv(cfg, seq_len=64)
    assert stats.tokens == 64
    assert stats.layers_with_own_kv >= 1
    h = torch.randn(1, 20, 32)
    r = swa_bounded_replay(h, n_win=8)
    assert r.shape == (1, 8, 32)


# ----- Train step -----


def test_one_train_step():
    cfg = preset("demo", vocab_size=40)
    model = ToyDSV41(cfg)
    model.train()
    x = torch.randint(0, 40, (2, 16))
    y = torch.randint(0, 40, (2, 16))
    logits, _ = model(
        x, phase="prefill", use_ced_prefill=False, use_bounded_replay=False, trace=False
    )
    loss = F.cross_entropy(logits.reshape(-1, 40), y.reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert loss.isfinite()
    assert len(grads) > 0


def test_comparison_table():
    cfg = ToyConfig(vocab_size=32)
    table = cfg.comparison_table()
    assert "DSpark" in table and "Vision" in table and "mHC" in table


def test_feature_flags_off():
    from dataclasses import replace

    cfg = replace(
        preset("demo", vocab_size=32),
        use_mhc=False,
        use_vision=False,
        use_dspark=False,
        use_rope=False,
    )
    model = ToyDSV41(cfg)
    assert model.vision is None and model.dspark is None and model.mhc_seed is None
    tokens = torch.randint(0, 32, (1, 12))
    logits, _ = model(
        tokens, phase="prefill", use_ced_prefill=False, use_bounded_replay=False, trace=False
    )
    assert logits.shape == (1, 12, 32)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("component tests passed")
