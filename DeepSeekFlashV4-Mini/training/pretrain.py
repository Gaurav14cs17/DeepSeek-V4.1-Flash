#!/usr/bin/env python3
"""Pretrain DeepSeekFlashV4-Mini on a char corpus."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Package root (hyphenated folder) on path
PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(PKG))

from model import DeepSeekFlashV4Mini, preset  # noqa: E402
from training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from training.dataset import build_corpus, random_batch  # noqa: E402
from training.losses import combined_loss, lm_loss  # noqa: E402
from training.scheduler import cosine_with_warmup  # noqa: E402


@torch.no_grad()
def evaluate(model, ids, batch_size, seq_len, device) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        x, y = random_batch(ids, batch_size, seq_len, device)
        logits, _ = model(x, trace=False)
        losses.append(lm_loss(logits, y).item())
    model.train()
    return sum(losses) / len(losses)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    _, tok, ids = build_corpus(Path(args.corpus) if args.corpus else None)
    cfg = preset(args.preset, vocab_size=tok.vocab_size)
    model = DeepSeekFlashV4Mini(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"params={model.num_parameters():,} vocab={tok.vocab_size} preset={args.preset}")
    t0 = time.time()
    train_loss = val_loss = 0.0
    for step in range(1, args.steps + 1):
        lr = cosine_with_warmup(step, args.steps, args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = random_batch(ids, args.batch_size, min(cfg.max_seq_len - 1, args.seq_len), device)
        logits, _ = model(x, trace=False)
        mtp_aux = model.mtp_loss(x, y) if cfg.use_mtp else None
        loss = combined_loss(logits, y, mtp_aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_loss = loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, ids, args.batch_size, x.shape[1], device)
            print(
                f"step {step}/{args.steps} train={train_loss:.4f} val={val_loss:.4f} "
                f"lr={lr:.2e} ({time.time() - t0:.1f}s)"
            )
            save_checkpoint(out / "ckpt.pt", model, tok, step, train_loss, val_loss)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="pc")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--corpus", default="")
    p.add_argument("--out", default=str(PKG / "_artifacts" / "pc_run"))
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
