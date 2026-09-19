#!/usr/bin/env python3
"""
Train Toy DeepSeek-V4.1-Flash on this PC (CPU).

Usage:
  python -m toy_dsv41.train
  python -m toy_dsv41.train --preset better --steps 800
  python -m toy_dsv41.train --generate-only --prompt "the causal encoder"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from toy_dsv41.data import build_corpus, random_batch
from toy_dsv41.data.tokenizer import CharTokenizer
from toy_dsv41.model import ToyDSV41, preset
from toy_dsv41.utils import append_history, plot_from_file, save_history

DEFAULT_OUT = Path(__file__).resolve().parent / "_artifacts" / "pc_run"


@torch.no_grad()
def evaluate(
    model: ToyDSV41, ids: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        x, y = random_batch(ids, batch_size, seq_len, device)
        logits, _ = model(
            x,
            phase="prefill",
            use_bounded_replay=False,
            use_ced_prefill=False,
            trace=False,
        )
        loss = F.cross_entropy(
            logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1)
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(
    model: ToyDSV41,
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
    model: ToyDSV41,
    tok: CharTokenizer,
    step: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg.__dict__.copy()
    cfg["encoder_csa_modes"] = list(model.cfg.encoder_csa_modes)
    cfg["decoder_csa_modes"] = list(model.cfg.decoder_csa_modes)
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
) -> Tuple[ToyDSV41, CharTokenizer, dict]:
    from toy_dsv41.model.config import ToyConfig

    blob = torch.load(path, map_location=device, weights_only=False)
    tok = CharTokenizer.from_state_dict(blob["tokenizer"])
    fields = ToyConfig.__dataclass_fields__
    cfg = ToyConfig(**{k: v for k, v in blob["config"].items() if k in fields})
    model = ToyDSV41(cfg).to(device)
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
    model = ToyDSV41(cfg).to(device)

    # Sensible sample prompt for each corpus
    if args.prompt == "the causal encoder":
        if "tinystories" in bundle.source:
            args.prompt = "Once upon a time"
        elif "tinyshakespeare" in bundle.source:
            args.prompt = "First Citizen:\n"

    print("=" * 64)
    print("Toy DeepSeek-V4.1-Flash — structured PC training")
    print("=" * 64)
    print(
        "  flow: docs/ → model/ → data/ → train.py → _artifacts/pc_run/ → generate"
    )
    print(f"  preset     : {args.preset}")
    print(f"  device     : cpu  threads={torch.get_num_threads()}")
    print(f"  dataset    : {bundle.source}")
    print(f"  corpus     : {len(bundle.text):,} chars  vocab={tok.vocab_size}")
    print(f"  out dir    : {out_dir}")
    print(f"  split      : train={bundle.train_ids.numel():,}  val={bundle.val_ids.numel():,}")
    print(f"  params     : {model.num_parameters():,}")
    print(f"  batch/seq  : {args.batch_size} / {args.seq_len}")
    print(f"  steps      : {args.steps}  lr={args.lr}")
    print(cfg.comparison_table())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    t0 = time.time()
    last_train = 0.0
    last_val = 0.0
    history: list = []
    history_path = out_dir / "history.json"
    plot_path = out_dir / "loss_curves.png"

    for step in range(1, args.steps + 1):
        x, y = random_batch(
            bundle.train_ids, args.batch_size, args.seq_len, device
        )
        # Full-length logits for CE (CED/SWA replay is demo-only in forward)
        logits, _ = model(
            x,
            phase="prefill",
            use_bounded_replay=False,
            use_ced_prefill=False,
            trace=False,
        )
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
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

    meta = {
        "preset": args.preset,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "params": model.num_parameters(),
        "corpus_chars": len(bundle.text),
        "vocab": tok.vocab_size,
    }
    save_history(history_path, history, meta=meta)
    print(f"\n  history  → {history_path}")

    if args.plot:
        try:
            plot_from_file(history_path, plot_path)
            print(f"  plot     → {plot_path}")
        except SystemExit as e:
            print(f"  plot skipped: {e}")

    save_checkpoint(
        ckpt_path, model, tok, args.steps, last_train, last_val, args
    )
    print(f"  saved    → {ckpt_path}")
    print("  sample: python -m toy_dsv41.train --generate-only")
    print("  replot: python -m toy_dsv41.utils.plot_history")
    print("=" * 64)


def generate_only(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    ckpt = Path(args.out_dir) / "toy_pc.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}; train first")
    model, tok, blob = load_checkpoint(ckpt, device)
    print(
        f"loaded step={blob['step']} "
        f"train={blob.get('train_loss', blob.get('loss', 0)):.4f} "
        f"val={blob.get('val_loss', float('nan')):.4f}"
    )
    text = sample_text(
        model, tok, args.prompt, args.max_new, args.temperature, device
    )
    print(f"prompt : {args.prompt!r}")
    print(f"output : {text!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="CPU toy training (structured)")
    p.add_argument("--preset", choices=["demo", "pc", "better"], default="pc")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--prompt", type=str, default="the causal encoder")
    p.add_argument("--max-new", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        help=(
            "tinystories (default, HF TinyStories subset) | "
            "tinyshakespeare | toy (builtin educational text)"
        ),
    )
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="optional path to a .txt file (overrides --dataset)",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--generate-only", action="store_true")
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save loss_curves.png (train + val). Use --no-plot to skip.",
    )
    args = p.parse_args()
    if args.generate_only:
        generate_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
