"""Encoder / CED-decoder transformer blocks (+ Single-Pass mHC)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import CSA2Mode, ToyConfig
from .csa2 import CSA2, SharedGlobalState
from .mhc import SinglePassMHC
from .moe import TinyMoE
from .norms import RMSNorm
from .swa import SlidingWindowAttention


class EncoderBlock(nn.Module):
    """Layers 0–1: SWA only. Later: CSA2(mode) + MoE. Optional mHC streams."""

    def __init__(self, cfg: ToyConfig, layer_idx: int, mode: CSA2Mode, swa_only: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.swa_only = swa_only
        self.use_mhc = cfg.use_mhc
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        if swa_only:
            self.attn = SlidingWindowAttention(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_head,
                cfg.swa_window,
                use_rope=cfg.use_rope,
                use_fp8_kv=cfg.use_fp8_swa_kv,
            )
            self.csa2 = None
        else:
            self.attn = None
            self.csa2 = CSA2(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_head,
                cfg.csa_topk,
                cfg.csa_compress,
                mode,
                candidate_pool=cfg.candidate_pool if mode == "full" else 0,
                use_fp4=cfg.use_fp4_main_kv,
                swa_window=cfg.swa_window,
                use_rope=cfg.use_rope,
                use_fp8_swa=cfg.use_fp8_swa_kv,
            )
        self.moe = TinyMoE(
            cfg.d_model,
            cfg.n_routed_experts,
            cfg.n_shared_experts,
            cfg.n_activated_experts,
            cfg.expert_hidden,
        )
        self.mhc = SinglePassMHC(cfg.d_model, cfg.mhc_streams) if cfg.use_mhc else None
        self._prev_A: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        shared: SharedGlobalState,
        log: Optional[List[str]] = None,
        modality: str = "text",
        mhc_streams: Optional[torch.Tensor] = None,
        mhc_A_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SharedGlobalState, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.mhc is not None and mhc_streams is not None:
            A_use = mhc_A_prev if mhc_A_prev is not None else self.mhc.init_A.expand(
                x.shape[0], x.shape[1], -1
            )
            h_in = self.mhc.mix_input(mhc_streams, A_use)
        else:
            h_in = x
            mhc_streams = None

        h = self.norm1(h_in)
        if self.swa_only:
            y = self.drop(self.attn(h))
            if log is not None:
                log.append(f"Encoder[{self.layer_idx}] SWA-only")
        else:
            y, shared = self.csa2(h, shared, log=log)
            y = self.drop(y)

        h2 = self.norm2(h_in + y)
        y2 = self.drop(self.moe(h2, log=log, modality=modality))
        block_out = y + y2

        A_new = None
        if self.mhc is not None and mhc_streams is not None:
            A_new, Bmat, C = self.mhc.predict(mhc_streams)
            mhc_streams = self.mhc.residual_update(mhc_streams, block_out, Bmat, C)
            x_out = self.mhc.collapse(mhc_streams)
            if log is not None and self.layer_idx == 0:
                log.append(f"mHC Single-Pass streams={self.mhc.n}")
        else:
            x_out = h_in + block_out

        return x_out, shared, mhc_streams, A_new


class DecoderBlock(nn.Module):
    """CED (Eq. 1): Full-mode global KV from encoder final hidden states."""

    def __init__(self, cfg: ToyConfig, layer_idx: int, mode: CSA2Mode):
        super().__init__()
        self.layer_idx = layer_idx
        self.mode = mode
        self.use_mhc = cfg.use_mhc
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.kv_from_encoder = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.csa2 = CSA2(
            cfg.d_model,
            cfg.n_heads,
            cfg.d_head,
            cfg.csa_topk,
            compress=1,
            mode=mode,
            candidate_pool=cfg.candidate_pool if mode == "full" else 0,
            use_fp4=cfg.use_fp4_main_kv,
            swa_window=cfg.swa_window,
            use_rope=cfg.use_rope,
            use_fp8_swa=cfg.use_fp8_swa_kv,
        )
        self.moe = TinyMoE(
            cfg.d_model,
            cfg.n_routed_experts,
            cfg.n_shared_experts,
            cfg.n_activated_experts,
            cfg.expert_hidden,
        )
        self.mhc = SinglePassMHC(cfg.d_model, cfg.mhc_streams) if cfg.use_mhc else None

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        shared: SharedGlobalState,
        log: Optional[List[str]] = None,
        modality: str = "text",
        mhc_streams: Optional[torch.Tensor] = None,
        mhc_A_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SharedGlobalState, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.mhc is not None and mhc_streams is not None:
            A_use = mhc_A_prev if mhc_A_prev is not None else self.mhc.init_A.expand(
                x.shape[0], x.shape[1], -1
            )
            h_in = self.mhc.mix_input(mhc_streams, A_use)
        else:
            h_in = x

        h = self.norm1(h_in)
        if self.mode == "full":
            # Align CED source length with current decoder tokens (bounded replay)
            src = encoder_out
            if src.shape[1] != h.shape[1]:
                src = src[:, -h.shape[1] :]
            ced_src = self.kv_from_encoder(src)
            if log is not None:
                log.append(
                    f"Decoder[{self.layer_idx}] CED Full KV from encoder "
                    f"{tuple(ced_src.shape)}"
                )
            main_kv, indexer_k = self.csa2._build_main_kv(ced_src)
            topk_idx, pool = self.csa2._score_and_select(
                h, indexer_k, None, build_pool=True
            )
            shared = SharedGlobalState(
                main_kv=main_kv,
                indexer_k=indexer_k,
                topk_idx=topk_idx,
                candidate_pool=pool,
            )
            saved = self.csa2.mode
            self.csa2.mode = "reuse"
            y, shared = self.csa2(h, shared, log=log)
            self.csa2.mode = saved
        else:
            if log is not None:
                log.append(f"Decoder[{self.layer_idx}] CSA2[{self.mode}]")
            y, shared = self.csa2(h, shared, log=log)

        y = self.drop(y)
        y2 = self.drop(self.moe(self.norm2(h_in + y), log=log, modality=modality))
        block_out = y + y2

        A_new = None
        if self.mhc is not None and mhc_streams is not None:
            A_new, Bmat, C = self.mhc.predict(mhc_streams)
            mhc_streams = self.mhc.residual_update(mhc_streams, block_out, Bmat, C)
            x_out = self.mhc.collapse(mhc_streams)
        else:
            x_out = h_in + block_out

        return x_out, shared, mhc_streams, A_new
