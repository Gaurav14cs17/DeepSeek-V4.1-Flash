"""Decoder / transformer blocks: SWA | CSA | HCA + MoE + multi-pass mHC."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .attention import SlidingWindowAttention
from .config import AttnKind, V4Config
from .mhc import MultiPassMHC
from .mla import CSA, HCA
from .moe import MoE
from .normalization import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, cfg: V4Config, layer_idx: int, kind: AttnKind):
        super().__init__()
        self.layer_idx = layer_idx
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        if kind == "swa":
            self.attn = SlidingWindowAttention(
                cfg.d_model, cfg.n_heads, cfg.d_head, cfg.swa_window, cfg.use_rope
            )
        elif kind == "csa":
            self.attn = CSA(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_head,
                compress=cfg.csa_compress,
                topk=cfg.csa_topk,
                swa_window=cfg.swa_window,
                use_rope=cfg.use_rope,
            )
        else:
            self.attn = HCA(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_head,
                compress=cfg.hca_compress,
                swa_window=cfg.swa_window,
                use_rope=cfg.use_rope,
            )
        self.moe = MoE(
            cfg.d_model,
            cfg.n_routed_experts,
            cfg.n_shared_experts,
            cfg.n_activated_experts,
            cfg.expert_hidden,
            vocab_size=cfg.vocab_size,
            use_hash=(layer_idx < cfg.n_hash_moe_layers),
            swiglu_limit=cfg.swiglu_limit,
            routed_scaling=cfg.routed_scaling,
        )
        self.mhc = MultiPassMHC(cfg.d_model, cfg.mhc_streams) if cfg.use_mhc else None

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        log: Optional[List[str]] = None,
        mhc_streams: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mhc is not None and mhc_streams is not None:
            A, Bmat, C = self.mhc.predict(mhc_streams)
            h_in = self.mhc.mix_input(mhc_streams, A)
        else:
            h_in = x
            Bmat = C = None

        h = self.norm1(h_in)
        if self.kind == "swa":
            y = self.drop(self.attn(h))
            if log is not None:
                log.append(f"Layer[{self.layer_idx}] SWA-only")
        else:
            y = self.drop(self.attn(h, log=log))

        y2 = self.drop(self.moe(self.norm2(h_in + y), tokens=tokens, log=log))
        block_out = y + y2

        if self.mhc is not None and mhc_streams is not None and Bmat is not None:
            mhc_streams = self.mhc.residual_update(mhc_streams, block_out, Bmat, C)
            x_out = self.mhc.collapse(mhc_streams)
            if log is not None and self.layer_idx == 0:
                log.append(f"mHC multi-pass streams={self.mhc.n}")
        else:
            x_out = h_in + block_out
        return x_out, mhc_streams


TransformerBlock = DecoderBlock
