"""Tiny DeepSeek-ViT + MLP projector (paper §2.1.1) — educational multimodal path."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm
from .rope import apply_rope_2d


class TinyDeepSeekViT(nn.Module):
    """
    Minimal ViT: linear patch embed (Muon-friendly), RMSNorm, SwiGLU MLP,
    2D-RoPE attention. Output spatial grid → 3×3 pixel-unshuffle → MLP projector.
    """

    def __init__(
        self,
        d_model: int,
        patch: int = 4,
        in_ch: int = 1,
        n_heads: int = 2,
        d_head: int = 16,
        depth: int = 2,
        max_grid: int = 8,
    ):
        super().__init__()
        self.patch = patch
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.max_grid = max_grid
        self.patch_proj = nn.Linear(in_ch * patch * patch, d_model, bias=False)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "n1": RMSNorm(d_model),
                        "qkv": nn.Linear(d_model, 3 * n_heads * d_head, bias=False),
                        "out": nn.Linear(n_heads * d_head, d_model, bias=False),
                        "n2": RMSNorm(d_model),
                        "w1": nn.Linear(d_model, d_model * 2, bias=False),
                        "w2": nn.Linear(d_model * 2, d_model, bias=False),
                        "w3": nn.Linear(d_model, d_model * 2, bias=False),
                    }
                )
            )
        # after 3×3 unshuffle: channels ×9, spatial /9
        self.projector = nn.Sequential(
            nn.Linear(d_model * 9, d_model, bias=False),
            RMSNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
        )

    def _patches(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """images [B,C,H,W] → tokens [B,Gh*Gw,D], Gh, Gw."""
        B, C, H, W = images.shape
        p = self.patch
        # crop to multiple of patch
        H2, W2 = (H // p) * p, (W // p) * p
        images = images[:, :, :H2, :W2]
        Gh, Gw = H2 // p, W2 // p
        # unfold
        x = images.unfold(2, p, p).unfold(3, p, p)  # [B,C,Gh,Gw,p,p]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, Gh * Gw, C * p * p)
        return self.patch_proj(x), Gh, Gw

    def _attn(self, layer: nn.ModuleDict, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        B, N, D = x.shape
        qkv = layer["qkv"](layer["n1"](x)).view(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope_2d(q, gh, gw)
        k = apply_rope_2d(k, gh, gw)
        scores = (q @ k.transpose(-2, -1)) * (self.d_head**-0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x + layer["out"](out)

    def _mlp(self, layer: nn.ModuleDict, x: torch.Tensor) -> torch.Tensor:
        h = layer["n2"](x)
        return x + layer["w2"](F.silu(layer["w1"](h)) * layer["w3"](h))

    def pixel_unshuffle_3x3(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """Rearrange 3×3 neighborhoods along channel → /9 spatial tokens."""
        B, N, D = x.shape
        # pad grid to multiple of 3
        pad_h = (3 - gh % 3) % 3
        pad_w = (3 - gw % 3) % 3
        grid = x.view(B, gh, gw, D)
        if pad_h or pad_w:
            grid = F.pad(grid, (0, 0, 0, pad_w, 0, pad_h))
        gh2, gw2 = grid.shape[1], grid.shape[2]
        nh, nw = gh2 // 3, gw2 // 3
        grid = grid.view(B, nh, 3, nw, 3, D).permute(0, 1, 3, 2, 4, 5)
        return grid.reshape(B, nh * nw, 9 * D)

    def forward(
        self, images: torch.Tensor, log: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        images: [B, C, H, W] float.
        returns visual embeddings [B, Nv, d_model] for concat with text.
        """
        x, gh, gw = self._patches(images)
        for layer in self.layers:
            x = self._attn(layer, x, gh, gw)
            x = self._mlp(layer, x)
        x = self.pixel_unshuffle_3x3(x, gh, gw)
        out = self.projector(x)
        if log is not None:
            log.append(
                f"Vision ViT grid={gh}x{gw} → unshuffle → vis_tokens={out.shape[1]}"
            )
        return out
