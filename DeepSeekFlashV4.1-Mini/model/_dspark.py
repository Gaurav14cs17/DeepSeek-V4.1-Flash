"""DSpark speculative decoding (paper §2.4.3) — educational toy drafter."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm
from ._swa import SlidingWindowAttention


class DSparkDrafter(nn.Module):
    """
    Paper: 3 Transformer blocks with SWA window 128; one pass → base logits for
    5 draft positions; lightweight Markov head; confidence head for acceptance.

    Toy: 3 tiny SWA blocks, draft_len=5, Markov + confidence heads.
    Backbone stays frozen w.r.t. DSpark loss in the real system; here we only
    use DSpark at generate-time for speculative accept/reject.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_heads: int = 4,
        d_head: int = 16,
        swa_window: int = 8,
        draft_len: int = 5,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.draft_len = draft_len
        self.vocab_size = vocab_size
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "n1": RMSNorm(d_model),
                        "attn": SlidingWindowAttention(
                            d_model, n_heads, d_head, swa_window
                        ),
                        "n2": RMSNorm(d_model),
                        "mlp": nn.Sequential(
                            nn.Linear(d_model, d_model * 2, bias=False),
                            nn.SiLU(),
                            nn.Linear(d_model * 2, d_model, bias=False),
                        ),
                    }
                )
            )
        self.norm = RMSNorm(d_model)
        # base logits for draft_len positions from last hidden
        self.draft_head = nn.Linear(d_model, draft_len * vocab_size, bias=False)
        # Markov: P(t_i | t_{i-1}) over vocab (toy bigram)
        self.markov = nn.Linear(vocab_size, vocab_size, bias=False)
        # confidence: per draft position acceptance logit
        self.confidence = nn.Linear(d_model, draft_len, bias=False)

    def forward_hidden(self, h: torch.Tensor) -> torch.Tensor:
        x = h
        for blk in self.blocks:
            x = x + blk["attn"](blk["n1"](x))
            x = x + blk["mlp"](blk["n2"](x))
        return self.norm(x)

    def draft(
        self, hidden: torch.Tensor, log: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden: [B,T,D] backbone states (usually last positions).
        Returns:
          draft_tokens [B, draft_len]
          conf [B, draft_len] in (0,1)
          base_logits [B, draft_len, V]
        """
        x = self.forward_hidden(hidden)
        last = x[:, -1]  # [B,D]
        B = last.shape[0]
        base = self.draft_head(last).view(B, self.draft_len, self.vocab_size)
        # greedy draft then Markov refine
        tokens = []
        prev_logits = base[:, 0]
        for i in range(self.draft_len):
            if i > 0:
                # mix base with Markov from previous one-hot soft
                soft = F.softmax(prev_logits, dim=-1)
                prev_logits = base[:, i] + self.markov(soft)
            tok = prev_logits.argmax(dim=-1)
            tokens.append(tok)
            prev_logits = base[:, i]  # keep for next markov input via soft of chosen
            # feed chosen token as soft target
            oh = F.one_hot(tok, self.vocab_size).float()
            prev_logits = base[:, min(i + 1, self.draft_len - 1)] + (
                self.markov(oh) if i + 1 < self.draft_len else prev_logits
            )
        draft_tokens = torch.stack(tokens, dim=1)
        conf = torch.sigmoid(self.confidence(last))
        if log is not None:
            log.append(
                f"DSpark drafted {self.draft_len} tokens, "
                f"mean_conf={conf.mean().item():.2f}"
            )
        return draft_tokens, conf, base

    @torch.no_grad()
    def verify_and_accept(
        self,
        draft_tokens: torch.Tensor,
        conf: torch.Tensor,
        target_tokens: torch.Tensor,
        conf_threshold: float = 0.3,
    ) -> int:
        """
        Accept longest prefix where draft matches target and conf > threshold.
        target_tokens: [B, draft_len] from backbone greedy/sample.
        Returns accepted length (min over batch for toy).
        """
        match = draft_tokens == target_tokens
        ok = match & (conf > conf_threshold)
        # prefix survival: first False stops
        B, L = ok.shape
        accepted = L
        for b in range(B):
            n = 0
            for i in range(L):
                if not bool(ok[b, i]):
                    break
                n += 1
            accepted = min(accepted, n)
        return int(accepted)
