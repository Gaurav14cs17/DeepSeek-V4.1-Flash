"""Configuration mathematical consistency checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

from pkg_import import import_from_pkg  # noqa: E402


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))


def test_v4_config_head_math():
    V4Config = import_from_pkg("DeepSeekFlashV4-Mini", "model.config").V4Config
    cfg = V4Config()
    assert cfg.d_model % cfg.n_heads == 0
    assert cfg.d_model == cfg.n_heads * cfg.d_head
    assert cfg.n_activated_experts <= cfg.n_routed_experts
    assert cfg.n_activated_experts >= 1
    assert len(cfg.layer_types) == cfg.n_layers


def test_v4_tiny_yaml_consistent():
    raw = _load_yaml("v4_tiny.yaml")
    assert raw["d_model"] == raw["n_heads"] * raw["d_head"]
    assert raw["memory_budget_mb"] <= 2048


def test_v4_invalid_heads_fail():
    V4Config = import_from_pkg("DeepSeekFlashV4-Mini", "model.config").V4Config
    with pytest.raises(AssertionError):
        V4Config(d_model=65, n_heads=4, d_head=16)


def test_v41_config_head_math():
    V41Config = import_from_pkg("DeepSeekFlashV4.1-Mini", "model.config").ToyConfig
    cfg = V41Config()
    assert cfg.d_model == cfg.n_heads * cfg.d_head
    assert len(cfg.encoder_csa_modes) == cfg.n_encoder_layers
    assert len(cfg.decoder_csa_modes) == cfg.n_decoder_layers
    assert cfg.n_activated_experts <= cfg.n_routed_experts


def test_v41_layer_override_without_modes_fails():
    """Documented failure mode: changing layer counts without mode lists."""
    V41Config = import_from_pkg("DeepSeekFlashV4.1-Mini", "model.config").ToyConfig
    with pytest.raises(AssertionError):
        V41Config(n_decoder_layers=2)  # default modes still length 4
