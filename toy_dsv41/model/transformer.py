"""ToyDSV41 — Causal Encoder-Decoder + Vision + DSpark + mHC + Engram."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DecoderBlock, EncoderBlock
from .config import ToyConfig
from .csa2 import SharedGlobalState
from .dspark import DSparkDrafter
from .engram import ToyEngram
from .kv_cache import estimate_kv, swa_bounded_replay
from .mhc import SinglePassMHC
from .norms import RMSNorm
from .vision import TinyDeepSeekViT


class ToyDSV41(nn.Module):
    """
    Structure (paper Fig. 3 at tiny scale):

        [images → ViT → unshuffle → proj] ⊕ text embed
              → Engram @ emb / layer-1
              → Encoder[0..1] SWA + MoE (+ mHC)
              → Encoder[2..]  CSA2 + MoE
              → encoder_out  ──CED──→ Decoder (Full KV from encoder_out)
              → Engram mid  → RMSNorm → LM head
              → DSpark (generate)
    """

    def __init__(self, cfg: ToyConfig | None = None):
        super().__init__()
        self.cfg = cfg or ToyConfig()
        c = self.cfg

        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        # Engram at embedding + mid (stand-in for layers 1 & 14)
        self.engram = ToyEngram(
            c.vocab_size,
            c.d_model,
            c.engram_entries,
            c.engram_dim,
            n_heads=c.engram_heads,
            ngram_orders=tuple(c.engram_orders),
        )
        self.engram_mid = ToyEngram(
            c.vocab_size,
            c.d_model,
            max(64, c.engram_entries // 2),
            c.engram_dim,
            n_heads=max(2, c.engram_heads // 2),
            ngram_orders=tuple(c.engram_orders),
        )
        self.vision = (
            TinyDeepSeekViT(
                c.d_model,
                patch=c.vision_patch,
                n_heads=max(2, c.n_heads // 2),
                d_head=c.d_head if c.d_head % 4 == 0 else 16,
                depth=c.vision_depth,
            )
            if c.use_vision
            else None
        )
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(c, i, c.encoder_csa_modes[i], swa_only=(i < 2))
                for i in range(c.n_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(c, i, c.decoder_csa_modes[i])
                for i in range(c.n_decoder_layers)
            ]
        )
        self.mhc_seed = SinglePassMHC(c.d_model, c.mhc_streams) if c.use_mhc else None
        self.norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.dspark = (
            DSparkDrafter(
                c.d_model,
                c.vocab_size,
                n_heads=c.n_heads,
                d_head=c.d_head,
                swa_window=c.swa_window,
                draft_len=c.dspark_draft_len,
            )
            if c.use_dspark
            else None
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.embed.weight.numel()
        return n

    def _fuse_vision(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        images: Optional[torch.Tensor],
        log: Optional[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Prepend visual tokens; modality=image if any vision tokens present."""
        modality = "text"
        if images is not None and self.vision is not None:
            vis = self.vision(images, log=log)  # [B, Nv, D]
            B = x.shape[0]
            # pad tokens for engram mid with zeros for visual prefix
            pad = torch.zeros(
                B, vis.shape[1], dtype=tokens.dtype, device=tokens.device
            )
            tokens = torch.cat([pad, tokens], dim=1)
            x = torch.cat([vis, x], dim=1)
            modality = "image"
        return tokens, x, modality

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        phase: str = "prefill",
        log: Optional[List[str]] = None,
        use_bounded_replay: bool = True,
        use_ced_prefill: bool = True,
        trace: bool = True,
        images: Optional[torch.Tensor] = None,
        modality: str = "text",
    ) -> Tuple[torch.Tensor, dict]:
        notes: Optional[List[str]] = log if log is not None else ([] if trace else None)
        c = self.cfg

        x = self.embed(tokens)
        tokens, x, mod = self._fuse_vision(tokens, x, images, notes)
        if images is not None:
            modality = mod

        x = self.engram(tokens, x, log=notes)
        if notes is not None:
            notes.append(f"phase={phase} T={tokens.shape[1]} d={c.d_model} mod={modality}")

        mhc_streams = None
        A_prev = None
        if self.mhc_seed is not None:
            mhc_streams = self.mhc_seed.expand_from_hidden(x)

        shared = SharedGlobalState()
        for i, layer in enumerate(self.encoder):
            if notes is not None:
                notes.append(f"— encoder {i} —")
            # Engram mid stand-in for paper layer-1 placement
            if i == 1:
                x = self.engram_mid(tokens, x, log=notes)
                if mhc_streams is not None:
                    mhc_streams = self.mhc_seed.expand_from_hidden(x)
            x, shared, mhc_streams, A_prev = layer(
                x,
                shared,
                log=notes,
                modality=modality,
                mhc_streams=mhc_streams,
                mhc_A_prev=A_prev,
            )

        encoder_out = x
        if notes is not None:
            notes.append(f"CED source encoder_out={tuple(encoder_out.shape)}")

        # CED prefill + SWA Bounded Replay: decoder only on last n_win tokens
        x_dec = x
        tok_dec = tokens
        full_len = x.shape[1]
        replayed = False
        if (
            phase == "prefill"
            and use_ced_prefill
            and use_bounded_replay
            and full_len > c.swa_window
        ):
            x_dec = swa_bounded_replay(x, c.swa_window)
            tok_dec = tokens[:, -c.swa_window :]
            replayed = True
            if notes is not None:
                notes.append(
                    f"CED prefill + SWA Bounded Replay: decoder on last "
                    f"n_win={c.swa_window}/{full_len} tokens"
                )

        if mhc_streams is not None:
            mhc_streams = self.mhc_seed.expand_from_hidden(x_dec)
            A_prev = None

        shared_dec = SharedGlobalState()
        for i, layer in enumerate(self.decoder):
            if notes is not None:
                notes.append(f"— decoder {i} —")
            x_dec, shared_dec, mhc_streams, A_prev = layer(
                x_dec,
                encoder_out,
                shared_dec,
                log=notes,
                modality=modality,
                mhc_streams=mhc_streams,
                mhc_A_prev=A_prev,
            )

        logits_tail = self.lm_head(self.norm(x_dec))
        if replayed:
            # pad logits to full length (prefix = copy first tail logit / zeros)
            B, Td, V = logits_tail.shape
            logits = logits_tail.new_zeros(B, full_len, V)
            logits[:, -Td:] = logits_tail
            # for LM training, prefer phase without replay; demo uses replay
        else:
            logits = logits_tail

        if trace:
            return logits, {
                "log": notes or [],
                "kv": estimate_kv(c, tokens.shape[1]),
                "encoder_shared": shared,
                "decoder_shared": shared_dec,
                "replayed": replayed,
            }
        return logits, {"log": [], "kv": None, "replayed": replayed}

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 16,
        temperature: float = 1.0,
        top_k: int = 0,
        use_dspark: bool = True,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        tokens = prompt
        log: List[str] = []
        while tokens.shape[1] - prompt.shape[1] < max_new:
            ctx = tokens[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(
                ctx,
                phase="decode",
                use_bounded_replay=False,
                use_ced_prefill=False,
                trace=False,
                images=images,
            )
            # hidden proxy: use embed of ctx for DSpark (lightweight)
            if use_dspark and self.dspark is not None and max_new - (
                tokens.shape[1] - prompt.shape[1]
            ) >= self.dspark.draft_len:
                h = self.embed(ctx)
                draft, conf, _ = self.dspark.draft(h, log=log)
                # verify: one backbone step per draft position (toy accept)
                accepted = []
                cur = tokens
                for i in range(draft.shape[1]):
                    ctx_i = cur[:, -self.cfg.max_seq_len :]
                    lg, _ = self.forward(
                        ctx_i,
                        phase="decode",
                        use_bounded_replay=False,
                        use_ced_prefill=False,
                        trace=False,
                    )
                    nxt_logits = lg[:, -1, :] / max(temperature, 1e-5)
                    if top_k > 0:
                        v, _ = torch.topk(nxt_logits, min(top_k, nxt_logits.size(-1)))
                        nxt_logits[nxt_logits < v[:, [-1]]] = float("-inf")
                    target = nxt_logits.argmax(dim=-1)
                    if bool((draft[:, i] == target).all()) and bool(
                        (conf[:, i] > 0.25).all()
                    ):
                        accepted.append(draft[:, i : i + 1])
                        cur = torch.cat([cur, draft[:, i : i + 1]], dim=1)
                    else:
                        # reject rest; sample one from backbone
                        probs = F.softmax(nxt_logits, dim=-1)
                        samp = torch.multinomial(probs, 1)
                        cur = torch.cat([cur, samp], dim=1)
                        break
                else:
                    pass
                tokens = cur
                continue

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, nxt], dim=1)
        # trim to requested length
        return tokens[:, : prompt.shape[1] + max_new]
