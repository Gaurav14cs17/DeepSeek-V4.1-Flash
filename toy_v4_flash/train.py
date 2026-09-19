#!/usr/bin/env python3
"""
Train Toy DeepSeek-V4-Flash on CPU.

  python -m toy_v4_flash.train --preset better --steps 500
  python -m toy_v4_flash.train --generate-only --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from toy_v4_flash.data import build_corpus, random_batch
from toy_v4_flash.data.tokenizer import CharTokenizer
from toy_v4_flash.model import ToyV4Flash, preset
from toy_v4_flash.utils import append_history, plot_from_file, save_history

DEFAULT_OUT = Path(__file__).resolve().parent / "_artifacts" / "pc_run"


@torch.no_grad()
def evaluate(
    model: ToyV4Flash, ids: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        x, y = random_batch(ids, batch_size, seq_len, device)
        logits, _ = model(x, trace=False)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1)
            ).item()
        )
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(
    model: ToyV4Flash,
    tok: CharTokenizer,
    prompt: str,
    max_new: int,
    temperature: float,
    device: torch.device,
    top_k: int = 20,
) -> str:
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    max_ctx = model.cfg.max_seq_len - 1
    if ids.shape[1] > max_ctx:
        ids = ids[:, -max_ctx:]
    out = model.generate(ids, max_new=max_new, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def save_checkpoint(
    path: Path,
    model: ToyV4Flash,
    tok: CharTokenizer,
    step: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg.__dict__.copy()
    cfg["layer_types"] = list(model.cfg.layer_types)
    torch.save(
        {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "config": cfg,
            "tokenizer": tok.state_dict(),
            "args": vars(args),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "checkpoint": str(path),
                "preset": args.preset,
                "params": model.num_parameters(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> Tuple[ToyV4Flash, CharTokenizer, dict]:
    from toy_v4_flash.model.config import ToyConfig

    blob = torch.load(path, map_location=device, weights_only=False)
    tok = CharTokenizer.from_state_dict(blob["tokenizer"])
    fields = ToyConfig.__dataclass_fields__
    raw = {k: v for k, v in blob["config"].items() if k in fields}
    cfg = ToyConfig(**raw)
    model = ToyV4Flash(cfg).to(device)
    model.load_state_dict(blob["model"])
    return model, tok, blob


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "toy_pc.pt"

    bundle = build_corpus(
        args.corpus, val_ratio=args.val_ratio, dataset=args.dataset
    )
    tok = bundle.tokenizer
    cfg = preset(args.preset, vocab_size=tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len + 8)
    model = ToyV4Flash(cfg).to(device)

    if args.prompt == "Once upon a time" and "shakespeare" in bundle.source:
        args.prompt = "First Citizen:\n"

    print("=" * 64)
    print("Toy DeepSeek-V4-Flash — CSA–HCA hybrid (older than V4.1)")
    print("=" * 64)
    print(f"  preset     : {args.preset}")
    print(f"  dataset    : {bundle.source}")
    print(f"  corpus     : {len(bundle.text):,} chars  vocab={tok.vocab_size}")
    print(f"  params     : {model.num_parameters():,}")
    print(f"  layers     : {cfg.layer_types}")
    print(f"  MTP weight : {args.mtp_weight}")
    print(cfg.comparison_table())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    last_train = last_val = 0.0
    history: list = []
    history_path = out_dir / "history.json"
    plot_path = out_dir / "loss_curves.png"

    for step in range(1, args.steps + 1):
        x, y = random_batch(bundle.train_ids, args.batch_size, args.seq_len, device)
        logits, info = model(x, trace=False, return_hidden=True)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        if model.mtp is not None and args.mtp_weight > 0:
            loss = loss + args.mtp_weight * model.mtp.loss(info["hidden"], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_train = loss.item()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            last_val = evaluate(
                model, bundle.val_ids, args.batch_size, args.seq_len, device
            )
            append_history(history, step, last_train, last_val)
            elapsed = time.time() - t0
            toks = step * args.batch_size * args.seq_len
            print(
                f"  step {step:5d}/{args.steps}  "
                f"train={last_train:.4f}  val={last_val:.4f}  "
                f"tok/s≈{toks / max(elapsed, 1e-6):.0f}  t={elapsed:.1f}s"
            )

        if step % args.sample_every == 0 or step == args.steps:
            sample = sample_text(
                model, tok, args.prompt, args.max_new, args.temperature, device
            )
            print(f"  sample@{step}: {sample!r}")
            model.train()

    save_history(
        history_path,
        history,
        meta={
            "preset": args.preset,
            "steps": args.steps,
            "params": model.num_parameters(),
            "model": "toy_v4_flash",
        },
    )
    print(f"\n  history  → {history_path}")
    if args.plot:
        try:
            plot_from_file(history_path, plot_path)
            print(f"  plot     → {plot_path}")
        except SystemExit as e:
            print(f"  plot skipped: {e}")

    save_checkpoint(ckpt_path, model, tok, args.steps, last_train, last_val, args)
    print(f"  saved    → {ckpt_path}")
    print("  sample: python -m toy_v4_flash.train --generate-only")
    print("=" * 64)


def generate_only(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    ckpt = Path(args.out_dir) / "toy_pc.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}; train first")
    model, tok, blob = load_checkpoint(ckpt, device)
    print(f"loaded step={blob['step']} train={blob.get('train_loss', 0):.4f}")
    text = sample_text(
        model, tok, args.prompt, args.max_new, args.temperature, device
    )
    print(f"prompt : {args.prompt!r}")
    print(f"output : {text!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="CPU train toy DeepSeek-V4-Flash")
    p.add_argument("--preset", choices=["demo", "pc", "better"], default="pc")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--mtp-weight", type=float, default=0.3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--dataset", type=str, default="tinystories")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--generate-only", action="store_true")
    p.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    args = p.parse_args()
    if args.generate_only:
        generate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
