"""KV cache accounting + SWA Bounded Replay (educational).

Paper:
  - Global KV in HBM ≈ 890 B/token (CSA2 reuse + FP4)
  - Persistent KV ≈ 1/8 of V4-Flash via SWA Bounded Replay (don't persist SWA)

MiaAI-Lab DGX Sparks notes the same idea in production: only a few kv_source
layers store global KV → ~1670 B/token/rank on their TP setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .config import ToyConfig


@dataclass
class KVStats:
    tokens: int = 0
    global_entries: int = 0  # compressed main-KV positions stored
    layers_with_own_kv: int = 0
    layers_reusing_kv: int = 0
    bytes_hbm_global: float = 0.0
    bytes_persistent: float = 0.0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"tokens={self.tokens}, global_entries={self.global_entries}, "
            f"own_kv_layers={self.layers_with_own_kv}, "
            f"reuse_layers={self.layers_reusing_kv}, "
            f"HBM_global≈{self.bytes_hbm_global:.1f}B, "
            f"persistent≈{self.bytes_persistent:.1f}B"
        )


def estimate_kv(config: ToyConfig, seq_len: int) -> KVStats:
    """Count how CSA2 modes + FP4 shrink the global KV footprint."""
    stats = KVStats(tokens=seq_len)
    # Encoder: after 2 SWA-only layers, CSA2 with compress=m
    enc_comp = max(1, (seq_len + config.csa_compress - 1) // config.csa_compress)
    # Decoder: compress=1 (paper m=1)
    dec_comp = seq_len

    bpe = config.bytes_per_main_kv_entry

    # Only Full mode layers *create* new global KV in the toy (Reuse/Reindex share)
    for i, mode in enumerate(config.encoder_csa_modes):
        if i < 2:
            stats.notes.append(f"enc[{i}]=SWA-only (no global KV; mode list unused)")
            continue
        if mode == "full":
            stats.layers_with_own_kv += 1
            stats.global_entries += enc_comp
            stats.bytes_hbm_global += enc_comp * bpe
            stats.notes.append(f"enc[{i}]=full → +{enc_comp} entries")
        else:
            stats.layers_reusing_kv += 1
            stats.notes.append(f"enc[{i}]={mode} → share previous Full KV")

    for i, mode in enumerate(config.decoder_csa_modes):
        if mode == "full":
            stats.layers_with_own_kv += 1
            stats.global_entries += dec_comp
            stats.bytes_hbm_global += dec_comp * bpe
            stats.notes.append(f"dec[{i}]=full → +{dec_comp} entries")
        else:
            stats.layers_reusing_kv += 1
            stats.notes.append(f"dec[{i}]={mode} → share/reuse")

    # Persistent: global only (SWA Bounded Replay → do NOT persist SWA)
    stats.bytes_persistent = stats.bytes_hbm_global
    # Naive baseline without reuse: every CSA2 layer stores own KV
    naive_layers = (config.n_encoder_layers - 2) + config.n_decoder_layers
    naive = (
        (config.n_encoder_layers - 2) * enc_comp + config.n_decoder_layers * dec_comp
    ) * (config.d_head * 2.0)  # FP16-ish
    stats.notes.append(
        f"naive_no_reuse_fp16≈{naive:.0f}B vs toy_global≈{stats.bytes_hbm_global:.0f}B "
        f"(~{naive / max(stats.bytes_hbm_global, 1):.1f}x larger)"
    )
    stats.notes.append(
        f"SWA Bounded Replay: persist only global KV "
        f"(replay last n_win={config.swa_window} tokens to rebuild SWA)"
    )
    return stats


def swa_bounded_replay(
    prefix_hidden: torch.Tensor, n_win: int
) -> torch.Tensor:
    """
    Approximate SWA rebuild: keep only the last `n_win` tokens of a cached prefix.

    Exact recovery would need L * n_win tokens; paper uses just n_win.
    """
    if prefix_hidden.shape[1] <= n_win:
        return prefix_hidden
    return prefix_hidden[:, -n_win:, :]
