"""Smoke + component tests for toy DeepSeek-V4-Flash."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from toy_v4_flash.data import build_corpus, random_batch
from toy_v4_flash.model import ToyV4Flash, preset
from toy_v4_flash.model.csa import CSA
from toy_v4_flash.model.hca import HCA
from toy_v4_flash.model.mtp import ToyMTP


def test_forward_shapes():
    cfg = preset("demo", vocab_size=40)
    model = ToyV4Flash(cfg)
    tokens = torch.randint(0, 40, (2, 16))
    logits, trace = model(tokens, trace=True)
    assert logits.shape == (2, 16, 40)
    assert "csa" in cfg.layer_types and "hca" in cfg.layer_types
    assert any("CSA" in line or "HCA" in line for line in trace["log"])


def test_csa_and_hca_units():
    x = torch.randn(1, 24, 32)
    y1 = CSA(32, 4, 8, compress=4, topk=4, swa_window=4)(x)
    y2 = HCA(32, 4, 8, compress=8, swa_window=4)(x)
    assert y1.shape == x.shape and y2.shape == x.shape


def test_mtp_loss():
    mtp = ToyMTP(32, 40, depth=1)
    h = torch.randn(2, 16, 32)
    y = torch.randint(0, 40, (2, 16))
    loss = mtp.loss(h, y)
    assert loss.isfinite()


def test_train_step():
    model = ToyV4Flash(preset("demo", vocab_size=40))
    model.train()
    x = torch.randint(0, 40, (2, 16))
    y = torch.randint(0, 40, (2, 16))
    logits, info = model(x, trace=False, return_hidden=True)
    loss = F.cross_entropy(logits.reshape(-1, 40), y.reshape(-1))
    loss = loss + 0.3 * model.mtp.loss(info["hidden"], y)
    loss.backward()
    assert loss.isfinite()


def test_corpus():
    bundle = build_corpus(dataset="toy")
    x, y = random_batch(bundle.train_ids, 4, 32, torch.device("cpu"))
    assert x.shape == (4, 32)


def test_generate():
    model = ToyV4Flash(preset("demo", vocab_size=32))
    out = model.generate(torch.randint(0, 32, (1, 4)), max_new=3, top_k=5)
    assert out.shape[1] == 7


def test_presets():
    for name in ("demo", "pc", "better"):
        m = ToyV4Flash(preset(name, vocab_size=32))
        assert m.num_parameters() > 1000


def test_no_v41_modules():
    """V4-Flash toy must not include CED/CSA2/Engram/DSpark/Vision."""
    import toy_v4_flash.model as M

    assert not hasattr(M, "ToyDSV41")
    cfg = preset("demo", vocab_size=32)
    assert cfg.use_mtp and cfg.use_mhc
    # layer schedule is CSA/HCA not CSA2 modes
    assert set(cfg.layer_types) <= {"swa", "csa", "hca"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("all tests passed")
