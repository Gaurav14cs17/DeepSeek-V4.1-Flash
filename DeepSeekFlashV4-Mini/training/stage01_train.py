#!/usr/bin/env python3
"""Stage-1 tiny training experiment.

Uses ONLY implemented stage-1 modules:
  tokenizer, TokenEmbedding, RMSNorm, apply_rope

Plus a *temporary* inline causal attention + LM head marked [EXPERIMENTAL]
so we can measure a real train/val loss before Stage-2 attention lands.

Writes: ../../results/stage01/metrics.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[0]
sys.path.insert(0, str(PKG))

from model.embedding import TokenEmbedding  # noqa: E402
from model.normalization import RMSNorm  # noqa: E402
from model.rope import apply_rope  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

LEVEL0_TEXT = """Once upon a time there was a little cat.
The cat liked milk and sun and naps on the mat.
One day the cat met a dog by the river.
They became friends and shared stories about the moon.
The end. Once upon a time the dog ran fast.
The cat sat still and watched the clouds.
Friends play near the water every day.
"""


class TemporaryCausalBlock(nn.Module):
    """[EXPERIMENTAL] Placeholder attention until Stage-2 `attention.py`."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        assert d_model == n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head
        self.norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ff_norm = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = [t.transpose(1, 2) for t in qkv.unbind(2)]
        # Stage-1 RoPE on q,k
        q, k = apply_rope(q), apply_rope(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.out(y)
        x = x + self.ff(self.ff_norm(x))
        return x


class Stage01Baseline(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_head: int, n_layers: int):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TemporaryCausalBlock(d_model, n_heads, d_head) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.emb.weight  # tie

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.final_norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def estimate_mem_mb(params: int, batch: int, seq: int, d: int, layers: int) -> dict:
    # rough float32 estimates
    param_mb = params * 4 / (1024**2)
    # AdamW ~2 states
    optim_mb = params * 8 / (1024**2)
    act_mb = batch * seq * d * layers * 4 / (1024**2)
    return {
        "param_mb": round(param_mb, 3),
        "optimizer_mb_est": round(optim_mb, 3),
        "activation_mb_est": round(act_mb, 3),
        "total_est_mb": round(param_mb + optim_mb + act_mb, 3),
    }


def random_batch(ids: torch.Tensor, batch: int, seq: int, device: torch.device):
    hi = max(1, ids.numel() - seq - 1)
    starts = torch.randint(0, hi, (batch,))
    x = torch.stack([ids[s : s + seq] for s in starts]).to(device)
    y = torch.stack([ids[s + 1 : s + 1 + seq] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def eval_loss(model, ids, batch, seq, device, n=20) -> float:
    model.eval()
    losses = []
    for _ in range(n):
        x, y = random_batch(ids, batch, seq, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def pick_device(budget_mb: int) -> torch.device:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024**2)
        print(f"CUDA free≈{free_mb:.0f} MB / total≈{total/(1024**2):.0f} MB")
        if free_mb >= 256:  # need some headroom; tiny model is small
            return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    cfg_path = REPO / "configs" / "v4_tiny.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    device = pick_device(int(cfg.get("memory_budget_mb", 2048)))

    # Level-0 corpus (offline): bundled synthetic text, or datasets/shards if present
    shard = REPO / "datasets" / "shards" / "level0_char.txt"
    text = shard.read_text(encoding="utf-8") if shard.exists() else LEVEL0_TEXT
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    d_model = int(cfg["d_model"])
    n_heads = int(cfg["n_heads"])
    d_head = int(cfg["d_head"])
    n_layers = int(cfg["n_layers"])
    seq = min(int(cfg["max_seq_len"]) - 1, 48)
    batch = int(cfg["batch_size"])
    steps = int(cfg["steps"])
    lr = float(cfg["lr"])

    model = Stage01Baseline(tok.vocab_size, d_model, n_heads, d_head, n_layers).to(device)
    n_params = model.num_parameters()
    mem = estimate_mem_mb(n_params, batch, seq, d_model, n_layers)

    print("=== Stage-1 memory print ===")
    print(f"device={device}")
    print(f"parameters={n_params:,}")
    print(json.dumps(mem, indent=2))
    print(f"seq={seq} batch={batch} steps={steps}")
    if mem["total_est_mb"] > float(cfg.get("memory_budget_mb", 2048)):
        raise SystemExit("Abort: estimated memory exceeds budget")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    train_loss = 0.0
    process = psutil.Process()
    peak_cpu_mb = process.memory_info().rss / (1024**2)
    peak_gpu_mb = 0.0

    model.train()
    tokens_seen = 0
    for step in range(1, steps + 1):
        x, y = random_batch(ids, batch, seq, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        tokens_seen += batch * seq
        peak_cpu_mb = max(peak_cpu_mb, process.memory_info().rss / (1024**2))
        if device.type == "cuda":
            peak_gpu_mb = max(peak_gpu_mb, torch.cuda.max_memory_allocated() / (1024**2))

    elapsed = time.perf_counter() - t0
    val_loss = eval_loss(model, ids, batch, seq, device)
    tps = tokens_seen / max(elapsed, 1e-9)

    out_dir = REPO / "results" / "stage01"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "DeepSeekFlashV4-Mini",
        "experiment": "stage01_rope_baseline",
        "fidelity": {
            "tokenizer": "[SIMPLIFIED]",
            "embedding": "[FAITHFUL]",
            "rmsnorm": "[FAITHFUL]",
            "rope": "[FAITHFUL]",
            "attention_in_train_loop": "[EXPERIMENTAL] temporary inline",
        },
        "parameters": n_params,
        "dataset": "level0_char_corpus",
        "dataset_chars": len(text),
        "dataset_tokens": int(ids.numel()),
        "vocab_size": tok.vocab_size,
        "sequence_length": seq,
        "batch_size": batch,
        "steps": steps,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "perplexity_val": math.exp(min(val_loss, 20)),
        "device": str(device),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 3),
        "peak_cpu_memory_mb": round(peak_cpu_mb, 3),
        "training_time_s": round(elapsed, 3),
        "tokens_per_second": round(tps, 2),
        "memory_estimates_mb": mem,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("=== Stage-1 results ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
