"""ToyV4Flash — DeepSeek-V4-Flash educational model (CSA–HCA, mHC, MTP)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import TransformerBlock
from .config import ToyConfig
from .kv_cache import estimate_kv
from .mhc import MultiPassMHC
from .mtp import ToyMTP
from .norms import RMSNorm


class ToyV4Flash(nn.Module):
    """
    V4-Flash layout (tiny):

        tokens → Embedding
              → Layers: SWA, SWA, CSA, HCA, CSA, HCA, ...
              → each: attn + MoE (+ multi-pass mHC)
              → RMSNorm → LM head
              → MTP extra heads (training)
    No CED / CSA2 / Engram / DSpark / Vision (those are V4.1).
    """

    def __init__(self, cfg: ToyConfig | None = None):
        super().__init__()
        self.cfg = cfg or ToyConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(c, i, c.layer_types[i])
                for i in range(c.n_layers)
            ]
        )
        self.mhc_seed = MultiPassMHC(c.d_model, c.mhc_streams) if c.use_mhc else None
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.mtp = ToyMTP(c.d_model, c.vocab_size, c.mtp_depth) if c.use_mtp else None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        log: Optional[List[str]] = None,
        trace: bool = True,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        notes: Optional[List[str]] = log if log is not None else ([] if trace else None)
        c = self.cfg
        x = self.embed(tokens)
        if notes is not None:
            notes.append(
                f"V4-Flash toy T={tokens.shape[1]} layers={c.layer_types}"
            )

        mhc_streams = None
        if self.mhc_seed is not None:
            mhc_streams = self.mhc_seed.expand_from_hidden(x)

        for i, layer in enumerate(self.layers):
            if notes is not None:
                notes.append(f"— layer {i} ({c.layer_types[i]}) —")
            x, mhc_streams = layer(x, tokens, log=notes, mhc_streams=mhc_streams)

        h = self.norm(x)
        logits = self.lm_head(h)
        info = {
            "log": notes or [],
            "kv": estimate_kv(c, tokens.shape[1]) if trace else None,
            "hidden": h if return_hidden else None,
        }
        if self.mtp is not None and notes is not None:
            _ = self.mtp(h, log=notes)
        return logits, info

    def mtp_loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mtp is None:
            return tokens.new_zeros(())
        _, info = self.forward(tokens, trace=False, return_hidden=True)
        return self.mtp.loss(info["hidden"], targets)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        tokens = prompt
        for _ in range(max_new):
            ctx = tokens[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(ctx, trace=False)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens
