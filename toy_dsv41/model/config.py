"""Hyperparameters + original-vs-toy table + named presets for this PC."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Literal, Sequence, Tuple


CSA2Mode = Literal["full", "reindex", "reuse"]

ORIGINAL_VS_TOY: Tuple[Tuple[str, str, str], ...] = (
    ("Architecture", "Multimodal MoE + CED", "MoE + CED + toy Vision/DSpark"),
    ("Backbone params", "552B", "~1–5M (preset)"),
    ("Engram params", "196B", "tiny hash tables ×2"),
    ("Activated / token (prefill)", "8B", "~half depth (CED)"),
    ("Activated / token (decode)", "16B", "~full depth (CED)"),
    ("Total Transformer layers", "40 (20 enc + 20 dec)", "8 (4 enc + 4 dec)"),
    ("First encoder layers", "2× SWA-only", "2× SWA-only"),
    ("Hidden size d_model", "5120", "64 / 96 (preset)"),
    ("Query heads", "64", "4"),
    ("Head / KV latent dim", "512", "16 / 24"),
    ("Vocab size", "129280", "char vocab from corpus"),
    ("Max context", "1,048,576 (1M)", "96–128"),
    ("SWA window n_win", "128", "8"),
    ("CSA2 enc compression m", "2", "2"),
    ("CSA2 dec compression m", "1", "1"),
    ("Attention Top-K", "512", "4"),
    ("Hier. candidate pool", "16384", "16"),
    ("MoE routed / activated", "384 / 6", "8 / 2"),
    ("Expert intermediate", "2304", "128–192"),
    ("Main KV precision", "FP4", "MXFP4-ish sim"),
    ("SWA KV precision", "FP8", "FP8 sim"),
    ("Vision", "DeepSeek-ViT + unshuffle", "TinyDeepSeekViT (toy)"),
    ("DSpark", "3-block drafter, 5 drafts", "DSparkDrafter (toy)"),
    ("Single-Pass mHC", "yes (Eq. 6)", "SinglePassMHC (toy)"),
    ("RoPE", "yes + 2D-RoPE ViT", "yes (toy)"),
    ("Pretrain tokens", "45T multimodal", "TinyStories / toy corpus"),
)


def format_original_vs_toy_table(
    rows: Sequence[Tuple[str, str, str]] = ORIGINAL_VS_TOY,
) -> str:
    col0 = max(len("Setting"), max(len(r[0]) for r in rows))
    col1 = max(len("Original V4.1-Flash"), max(len(r[1]) for r in rows))
    col2 = max(len("Our toy"), max(len(r[2]) for r in rows))
    sep = f"+-{'-' * col0}-+-{'-' * col1}-+-{'-' * col2}-+"
    header = (
        f"| {'Setting'.ljust(col0)} | {'Original V4.1-Flash'.ljust(col1)} | "
        f"{'Our toy'.ljust(col2)} |"
    )
    lines = [sep, header, sep]
    for setting, original, toy in rows:
        lines.append(
            f"| {setting.ljust(col0)} | {original.ljust(col1)} | {toy.ljust(col2)} |"
        )
    lines.append(sep)
    return "\n".join(lines)


@dataclass
class ToyConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    swa_window: int = 8
    max_seq_len: int = 128
    dropout: float = 0.0

    csa_topk: int = 4
    csa_compress: int = 2
    encoder_csa_modes: List[CSA2Mode] = field(
        default_factory=lambda: ["full", "reuse", "full", "reuse"]
    )
    decoder_csa_modes: List[CSA2Mode] = field(
        default_factory=lambda: ["full", "reuse", "reindex", "reuse"]
    )

    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    expert_hidden: int = 128

    engram_entries: int = 256
    engram_dim: int = 32
    engram_heads: int = 4
    engram_orders: Tuple[int, ...] = (2, 3, 4)

    use_fp4_main_kv: bool = True
    use_fp8_swa_kv: bool = True
    use_rope: bool = True
    use_mhc: bool = True
    mhc_streams: int = 2
    use_vision: bool = True
    vision_patch: int = 4
    vision_depth: int = 2
    use_dspark: bool = True
    dspark_draft_len: int = 5

    candidate_pool: int = 16
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model={self.d_model} != n_heads*d_head={self.n_heads * self.d_head}"
        )
        assert len(self.encoder_csa_modes) == self.n_encoder_layers
        assert len(self.decoder_csa_modes) == self.n_decoder_layers

    @property
    def n_layers(self) -> int:
        return self.n_encoder_layers + self.n_decoder_layers

    @property
    def bytes_per_main_kv_entry(self) -> float:
        main = self.d_head * (0.5 if self.use_fp4_main_kv else 2.0)
        indexer_k = self.d_head * 0.5
        return main + indexer_k

    def comparison_table(self) -> str:
        live = list(ORIGINAL_VS_TOY)
        replacements = {
            "Hidden size d_model": str(self.d_model),
            "Query heads": str(self.n_heads),
            "Head / KV latent dim": str(self.d_head),
            "Vocab size": str(self.vocab_size),
            "Max context": str(self.max_seq_len),
            "SWA window n_win": str(self.swa_window),
            "CSA2 enc compression m": str(self.csa_compress),
            "Attention Top-K": str(self.csa_topk),
            "Hier. candidate pool": str(self.candidate_pool),
            "MoE routed / activated": (
                f"{self.n_routed_experts} / {self.n_activated_experts}"
            ),
            "Expert intermediate": str(self.expert_hidden),
            "Total Transformer layers": (
                f"{self.n_layers} ({self.n_encoder_layers} enc + "
                f"{self.n_decoder_layers} dec)"
            ),
        }
        out: List[Tuple[str, str, str]] = []
        for setting, original, toy in live:
            out.append((setting, original, replacements.get(setting, toy)))
        return format_original_vs_toy_table(out)


def preset(name: str = "pc", vocab_size: int = 64) -> ToyConfig:
    """
    Named configs for this laptop:
      demo  — smallest, for architecture walkthrough
      pc    — default CPU training (balanced)
      better — slightly larger, still CPU-friendly
    """
    name = name.lower().strip()
    base = ToyConfig(vocab_size=vocab_size)
    if name == "demo":
        return base
    if name == "pc":
        return replace(
            base,
            d_model=64,
            d_head=16,
            expert_hidden=128,
            dropout=0.05,
            max_seq_len=96,
            tie_embeddings=True,
        )
    if name in ("better", "better_pc", "large"):
        return replace(
            base,
            d_model=96,
            n_heads=4,
            d_head=24,
            expert_hidden=192,
            engram_entries=512,
            engram_dim=48,
            dropout=0.08,
            max_seq_len=128,
            tie_embeddings=True,
        )
    raise ValueError(f"unknown preset {name!r}; use demo|pc|better")
