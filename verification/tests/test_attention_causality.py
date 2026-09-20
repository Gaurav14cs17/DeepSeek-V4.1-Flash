"""Attention numerical + causal leakage verification."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pkg_import import import_from_pkg
from reference import cosine_sim, max_mean_abs, reference_causal_attention


def test_attention_shapes():
    attn = import_from_pkg("DeepSeekFlashV4-Mini", "model.attention")
    m = attn.CausalSelfAttention(64, 4, 16, use_rope=True)
    x = torch.randn(2, 11, 64)
    assert m(x).shape == x.shape
    swa = attn.SlidingWindowAttention(64, 4, 16, window=8)
    assert swa(x).shape == x.shape


def test_core_scores_match_reference():
    rope = import_from_pkg("DeepSeekFlashV4-Mini", "model.rope")
    torch.manual_seed(1)
    B, H, T, D = 2, 2, 7, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    q, k = rope.apply_rope(q), rope.apply_rope(k)
    ref = reference_causal_attention(q, k, v, causal=True)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal[None, None], float("-inf"))
    out = F.softmax(scores, dim=-1) @ v
    mx, mn = max_mean_abs(out, ref)
    assert mx < 1e-5
    assert cosine_sim(out, ref) > 0.999999


def test_causal_leakage_prefix_invariant():
    attn = import_from_pkg("DeepSeekFlashV4-Mini", "model.attention")
    torch.manual_seed(2)
    m = attn.CausalSelfAttention(32, 2, 16, use_rope=True)
    m.eval()
    a = torch.randn(1, 3, 32)
    b = torch.cat([a, torch.randn(1, 1, 32)], dim=1)
    with torch.no_grad():
        ya = m(a)
        yb = m(b)
    mx, _ = max_mean_abs(ya, yb[:, :3])
    assert mx < 1e-5, f"causal leak max_abs={mx}"


def test_swa_window_blocks_distant_past():
    attn = import_from_pkg("DeepSeekFlashV4-Mini", "model.attention")
    torch.manual_seed(3)
    m = attn.SlidingWindowAttention(16, 1, 16, window=1, use_rope=False)
    x = torch.randn(1, 3, 16)
    x2 = x.clone()
    x2[:, 0] = torch.randn(1, 16)
    m.eval()
    with torch.no_grad():
        y = m(x)
        y2 = m(x2)
    mx, _ = max_mean_abs(y[:, 2], y2[:, 2])
    assert mx < 1e-5, f"SWA window leak max_abs={mx}"
