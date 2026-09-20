"""Hyperparameters for DeepSeekFlashV4-Mini (loads configs/v4_tiny.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import List, Literal, Optional

AttnKind = Literal["swa", "csa", "hca"]


@dataclass
class V4Config:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16
    n_layers: int = 8
    swa_window: int = 8
    max_seq_len: int = 128
    dropout: float = 0.0
    layer_types: List[AttnKind] = field(default_factory=list)
    csa_compress: int = 4
    hca_compress: int = 16
    csa_topk: int = 4
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    expert_hidden: int = 128
    n_hash_moe_layers: int = 2
    swiglu_limit: float = 10.0
    routed_scaling: float = 1.5
    use_mhc: bool = True
    mhc_streams: int = 2
    use_mtp: bool = True
    mtp_depth: int = 1
    use_rope: bool = True
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.d_head
        if not self.layer_types:
            types: List[AttnKind] = []
            for i in range(self.n_layers):
                if i < 2:
                    types.append("swa")
                elif (i - 2) % 2 == 0:
                    types.append("csa")
                else:
                    types.append("hca")
            self.layer_types = types
        assert len(self.layer_types) == self.n_layers

    @classmethod
    def from_yaml(cls, path: str | Path) -> "V4Config":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    @classmethod
    def tiny(cls, repo_root: Optional[Path] = None) -> "V4Config":
        root = repo_root or Path(__file__).resolve().parents[2]
        path = root / "configs" / "v4_tiny.yaml"
        if path.exists():
            return cls.from_yaml(path)
        return cls(max_seq_len=96, dropout=0.05)


def preset(name: str = "pc", vocab_size: int = 64) -> V4Config:
    name = name.lower().strip()
    base = V4Config(vocab_size=vocab_size)
    if name == "demo":
        return base
    if name == "pc":
        return replace(base, dropout=0.05, max_seq_len=96)
    if name in ("better", "better_pc", "large", "tiny"):
        if name == "tiny":
            return replace(base, dropout=0.05, max_seq_len=96)
        return replace(
            base,
            d_model=96,
            n_heads=4,
            d_head=24,
            expert_hidden=192,
            dropout=0.08,
            max_seq_len=128,
            hca_compress=16,
        )
    raise ValueError(f"unknown preset {name!r}; use demo|pc|better|tiny")


# Back-compat alias
ToyConfig = V4Config
