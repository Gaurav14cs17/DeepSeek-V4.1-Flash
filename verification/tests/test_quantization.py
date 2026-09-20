"""Quantization / FP4 simulation verification."""

from __future__ import annotations

import torch

from pkg_import import import_from_pkg
from reference import cosine_sim, max_mean_abs, reference_fake_quantize


def test_fake_quantize_matches_reference():
    q = import_from_pkg("DeepSeekFlashV4-Mini", "optimization.quantization")
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    y = q.fake_quantize(x, n_bits=8)
    y_ref = reference_fake_quantize(x, n_bits=8)
    mx, _ = max_mean_abs(y, y_ref)
    assert mx < 1e-6


def test_mxfp4_sim_error_bounded():
    q = import_from_pkg("DeepSeekFlashV4-Mini", "optimization.quantization")
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    y = q.mxfp4_sim(x)
    mx, mn = max_mean_abs(x, y)
    assert torch.isfinite(y).all()
    assert cosine_sim(x, y) > 0.9
    assert mx > 0


def test_fp4_not_packed_storage():
    """Public FP4 KV module is deferred; CSA2 sim keeps float tensors."""
    fp4 = import_from_pkg("DeepSeekFlashV4.1-Mini", "optimization.fp4_kv_cache")
    assert getattr(fp4, "DEFERRED", False) is True
    csa2 = import_from_pkg("DeepSeekFlashV4.1-Mini", "model._csa2")
    m = csa2.CSA2(64, 4, 16, topk=4, compress=2, mode="full", use_fp4=True)
    x = torch.randn(1, 16, 64)
    y, _ = m(x, csa2.SharedGlobalState())
    assert y.dtype == torch.float32
