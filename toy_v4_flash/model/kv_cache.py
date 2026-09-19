"""KV accounting — V4-Flash exact SWA replay (L × n_win), not V4.1 Bounded Replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import ToyConfig


@dataclass
class KVStats:
    tokens: int = 0
    csa_layers: int = 0
    hca_layers: int = 0
    swa_layers: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"tokens={self.tokens}, CSA={self.csa_layers}, HCA={self.hca_layers}, "
            f"SWA={self.swa_layers}"
        )


def estimate_kv(cfg: ToyConfig, seq_len: int) -> KVStats:
    stats = KVStats(tokens=seq_len)
    for i, kind in enumerate(cfg.layer_types):
        if kind == "swa":
            stats.swa_layers += 1
            stats.notes.append(f"L{i}=SWA (local only)")
        elif kind == "csa":
            stats.csa_layers += 1
            nc = max(1, (seq_len + cfg.csa_compress - 1) // cfg.csa_compress)
            stats.notes.append(f"L{i}=CSA m={cfg.csa_compress} ~{nc} entries + indexer")
        else:
            stats.hca_layers += 1
            nc = max(1, (seq_len + cfg.hca_compress - 1) // cfg.hca_compress)
            stats.notes.append(f"L{i}=HCA m'={cfg.hca_compress} ~{nc} dense-comp")
    # V4 exact replay cost
    exact = cfg.n_layers * cfg.swa_window
    stats.notes.append(
        f"V4 exact SWA replay needs ~L*n_win={exact} tokens "
        f"(V4.1 Bounded Replay uses only n_win={cfg.swa_window})"
    )
    return stats


def exact_swa_replay_span(n_layers: int, n_win: int) -> int:
    return n_layers * n_win
