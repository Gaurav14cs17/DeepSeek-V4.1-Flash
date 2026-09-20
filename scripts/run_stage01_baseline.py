#!/usr/bin/env python3
"""Record Stage-1 BASELINE (B0) with full protocol metadata.

This is the BEFORE measurement for all future Stage-1+ optimizations.
Never optimize without this (or a newer recorded baseline) first.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "DeepSeekFlashV4-Mini"))

from bench.history import append_history  # noqa: E402
from bench.latency import time_ms  # noqa: E402
from bench.memory import collect_memory, format_memory_report  # noqa: E402
from bench.protocol import ExperimentMeta, save_result_bundle  # noqa: E402
from bench.training_metrics import measure_training_step  # noqa: E402
from model.rope import apply_rope  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from training.stage01_train import (  # noqa: E402
    LEVEL0_TEXT,
    Stage01Baseline,
    eval_loss,
    pick_device,
    random_batch,
)


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "v4_tiny.yaml").read_text())
    device = pick_device(int(cfg.get("memory_budget_mb", 2048)))
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
    warmup, measure = 3, 10

    model = Stage01Baseline(tok.vocab_size, d_model, n_heads, d_head, n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- training timed (wall + per-phase breakdown) ---
    model.train()
    tokens_seen = 0
    train_loss = 0.0
    t0 = time.perf_counter()
    step_times = []
    for step in range(1, steps + 1):
        st = time.perf_counter()
        x, y = random_batch(ids, batch, seq, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        tokens_seen += batch * seq
        step_times.append((time.perf_counter() - st) * 1000.0)
    train_s = time.perf_counter() - t0
    val_loss = eval_loss(model, ids, batch, seq, device)

    # --- training step breakdown (fwd / bwd / opt) — mandatory protocol ---
    x_b, y_b = random_batch(ids, batch, seq, device)

    def forward_fn():
        logits = model(x_b)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_b.reshape(-1))

    def backward_fn(loss):
        loss.backward()

    def optimizer_fn():
        opt.step()

    def zero_grad_fn():
        opt.zero_grad(set_to_none=True)

    train_parts = measure_training_step(
        forward_fn=forward_fn,
        backward_fn=backward_fn,
        optimizer_fn=optimizer_fn,
        zero_grad_fn=zero_grad_fn,
        warmup=warmup,
        iters=measure,
        tokens_per_step=batch * seq,
    )
    print("\nTRAINING STEP BREAKDOWN (B0)")
    print(f"  forward_ms:   {train_parts['forward_ms']:.4f}")
    print(f"  backward_ms:  {train_parts['backward_ms']:.4f}")
    print(f"  optimizer_ms: {train_parts['optimizer_ms']:.4f}")
    print(f"  step_ms:      {train_parts['step_ms']:.4f}")
    print(f"  tokens/sec:   {train_parts['training_tokens_per_sec']:.1f}")

    # --- forward latency (eval / inference-style) ---
    model.eval()
    x, _ = random_batch(ids, batch, seq, device)

    def fwd():
        with torch.no_grad():
            model(x)

    lat = time_ms(fwd, warmup=warmup, iters=measure)

    # --- RoPE micro-bench (component) ---
    q = torch.randn(batch, n_heads, seq, d_head, device=device)

    def rope_fn():
        apply_rope(q)

    rope_lat = time_ms(rope_fn, warmup=warmup, iters=measure)

    mem = collect_memory(
        model,
        batch=batch,
        seq=seq,
        d_model=d_model,
        n_layers=n_layers,
        kv_bytes_per_token=0.0,
    )
    print(format_memory_report(mem, "MEMORY REPORT — STAGE01 BASELINE"))

    meta = ExperimentMeta(
        model="DeepSeekFlashV4-Mini",
        experiment="stage01",
        component="tokenizer+embedding+rmsnorm+rope+temp_attn",
        version="baseline",
        baseline_version=None,
        precision="fp32",
        batch_size=batch,
        sequence_length=seq,
        generation_length=0,
        warmup_iters=warmup,
        measure_iters=measure,
        seed=42,
        config_name="v4_tiny",
        notes="Stage-1 B0. Temporary inline attention is [EXPERIMENTAL].",
    )

    metrics = {
        "parameters": mem.parameters,
        "trainable_parameters": mem.trainable_parameters,
        "active_parameters": mem.trainable_parameters,
        "peak_gpu_memory_mb": mem.peak_gpu_memory_mb,
        "peak_cpu_memory_mb": mem.peak_cpu_memory_mb,
        "param_memory_mb": mem.param_memory_mb,
        "activation_memory_mb": mem.activation_memory_mb_est,
        "optimizer_memory_mb": mem.optimizer_memory_mb,
        "kv_bytes_per_token": 0.0,
        "forward_ms": lat["mean_ms"],
        "p50_latency_ms": lat["p50_latency_ms"],
        "p95_latency_ms": lat["p95_latency_ms"],
        "rope_forward_ms": rope_lat["mean_ms"],
        # training breakdown (mandatory)
        "train_forward_ms": train_parts["forward_ms"],
        "backward_ms": train_parts["backward_ms"],
        "optimizer_ms": train_parts["optimizer_ms"],
        "step_ms": train_parts["step_ms"],
        "training_step_ms": sum(step_times) / len(step_times),  # full train-loop mean
        "training_time_s": train_s,
        "training_tokens_per_sec": tokens_seen / max(train_s, 1e-9),
        "tokens_per_second": tokens_seen / max(train_s, 1e-9),
        "bench_training_tokens_per_sec": train_parts["training_tokens_per_sec"],
        "loss": train_loss,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "perplexity": math.exp(min(val_loss, 20)),
        "dataset_tokens": int(ids.numel()),
        "steps": steps,
    }

    out = save_result_bundle("v4", "stage01", "baseline", meta, metrics, mem.to_dict())
    append_history(
        {
            "experiment": "stage01",
            "component": meta.component,
            "version": "baseline",
            "baseline_version": "",
            "parameters": metrics["parameters"],
            "active_parameters": metrics["active_parameters"],
            "gpu_memory_mb": metrics["peak_gpu_memory_mb"],
            "cpu_memory_mb": metrics["peak_cpu_memory_mb"],
            "training_step_ms": metrics["training_step_ms"],
            "training_tokens_per_sec": metrics["training_tokens_per_sec"],
            "inference_tokens_per_sec": "",
            "kv_bytes_per_token": 0,
            "loss": metrics["loss"],
            "validation_loss": metrics["validation_loss"],
            "perplexity": metrics["perplexity"],
            "timestamp": meta.to_dict()["timestamp"],
            "git_commit": meta.to_dict()["git_commit"] or "",
        }
    )
    print(f"\nSaved baseline bundle → {out}")
    print("NEXT: implement an optimization only AFTER this B0 exists, then run AFTER + delta.")


if __name__ == "__main__":
    main()
