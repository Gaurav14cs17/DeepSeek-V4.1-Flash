"""Hyperparameters + V4-Flash vs toy vs V4.1-Flash comparison."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Literal, Sequence, Tuple

AttnKind = Literal["swa", "csa", "hca"]

ORIGINAL_VS_TOY: Tuple[Tuple[str, str, str], ...] = (
    ("Architecture", "MoE + CSA–HCA hybrid", "Toy MoE + CSA–HCA (+ mHC, MTP)"),
    ("Backbone params", "284B", "~1–5M (preset)"),
    ("Activated / token", "13B", "full depth (no CED)"),
    ("Total layers", "43", "8 (toy)"),
    ("First layers", "2× SWA", "2× SWA"),
    ("Global attention", "CSA (m=4) ⊕ HCA (m=128)", "CSA ⊕ HCA interleaved"),
    ("CSA2 / CED / Engram / DSpark", "no (those are V4.1)", "no"),
    ("mHC", "original multi-pass (n=4)", "MultiPassMHC (n=2 toy)"),
    ("MTP", "yes (joint pretrain)", "ToyMTP heads"),
    ("SWA replay", "exact L×n_win", "exact-style L*n_win hint"),
    ("Main KV precision", "FP8-era (not FP4 QAT)", "FP16/sim"),
    ("MoE routed / activated", "256 / 6", "8 / 2"),
    ("Hash-MoE bootstrap", "first 3 layers", "first 2 layers (toy)"),
    ("Hidden size d_model", "~4096-class", "64 / 96"),
    ("Vocab / context", "BPE / 1M", "char / 96–128"),
)


def format_table(rows: Sequence[Tuple[str, str, str]] = ORIGINAL_VS_TOY) -> str:
    col0 = max(len("Setting"), max(len(r[0]) for r in rows))
    col1 = max(len("Original V4-Flash"), max(len(r[1]) for r in rows))
    col2 = max(len("Our toy"), max(len(r[2]) for r in rows))
    sep = f"+-{'-' * col0}-+-{'-' * col1}-+-{'-' * col2}-+"
    header = (
        f"| {'Setting'.ljust(col0)} | {'Original V4-Flash'.ljust(col1)} | "
        f"{'Our toy'.ljust(col2)} |"
    )
    lines = [sep, header, sep]
    for a, b, c in rows:
        lines.append(f"| {a.ljust(col0)} | {b.ljust(col1)} | {c.ljust(col2)} |")
    lines.append(sep)
    return "\n".join(lines)


V41_DIFF: Tuple[Tuple[str, str, str], ...] = (
    ("Layout", "Full-depth 43-layer", "CED 20+20 enc/dec"),
    ("Attention", "CSA–HCA hybrid", "Pure CSA2 Full/Reindex/Reuse"),
    ("Prefill activate", "13B (full)", "8B (encoder half)"),
    ("mHC", "Original multi-pass", "Single-Pass mHC"),
    ("Memory / decode", "MTP", "Engram + DSpark"),
    ("KV precision", "No FP4 main QAT", "FP4 main KV + Bounded Replay"),
    ("Vision native", "— (Table 1 multimodal blank)", "DeepSeek-ViT from start"),
)


@dataclass
class ToyConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16
    n_layers: int = 8
    swa_window: int = 8
    max_seq_len: int = 128
    dropout: float = 0.0

    # Layer schedule: 0,1=SWA; then CSA,HCA,CSA,HCA,...
    layer_types: List[AttnKind] = field(default_factory=list)

    csa_compress: int = 4  # paper m=4
    hca_compress: int = 16  # paper 128; toy smaller for short seq
    csa_topk: int = 4  # paper 512

    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    expert_hidden: int = 128
    n_hash_moe_layers: int = 2  # paper 3
    swiglu_limit: float = 10.0
    routed_scaling: float = 1.5

    use_mhc: bool = True
    mhc_streams: int = 2  # paper 4
    use_mtp: bool = True
    mtp_depth: int = 1  # extra next-token heads (toy 1 ⇒ predict +1)

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

    def comparison_table(self) -> str:
        live = list(ORIGINAL_VS_TOY)
        rep = {
            "Hidden size d_model": str(self.d_model),
            "Vocab / context": f"{self.vocab_size} / {self.max_seq_len}",
            "Total layers": str(self.n_layers),
            "MoE routed / activated": (
                f"{self.n_routed_experts} / {self.n_activated_experts}"
            ),
        }
        out = [(a, b, rep.get(a, c)) for a, b, c in live]
        return format_table(out)

    def vs_v41_table(self) -> str:
        return format_table(
            [("Aspect", "V4-Flash (this toy)", "V4.1-Flash (toy_dsv41)")]
            + list(V41_DIFF)
        )


def preset(name: str = "pc", vocab_size: int = 64) -> ToyConfig:
    name = name.lower().strip()
    base = ToyConfig(vocab_size=vocab_size)
    if name == "demo":
        return base
    if name == "pc":
        return replace(base, dropout=0.05, max_seq_len=96)
    if name in ("better", "better_pc", "large"):
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
    raise ValueError(f"unknown preset {name!r}; use demo|pc|better")
