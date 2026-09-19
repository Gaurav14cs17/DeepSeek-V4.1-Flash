#!/usr/bin/env python3
"""Architecture walkthrough for toy DeepSeek-V4-Flash (older CSA–HCA model)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from toy_v4_flash.model import ToyV4Flash, preset
from toy_v4_flash.model.config import V41_DIFF, format_table
from toy_v4_flash.model.kv_cache import estimate_kv


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["demo", "pc", "better"], default="demo")
    args = p.parse_args()

    banner("Project flow")
    print(
        """
  DeepSeek-V4.1-Flash/
  ├── docs/                      paper (mentions V4-Flash as predecessor)
  ├── toy_dsv41/                 V4.1-Flash toy (CED + CSA2 + Engram + DSpark)
  └── toy_v4_flash/              ← you are here (older V4-Flash)
      ├── model/                 CSA–HCA + mHC + MTP + MoE
      ├── data/ / utils/
      ├── demo.py / train.py
      └── _artifacts/pc_run/
"""
    )

    cfg = preset(args.preset, vocab_size=48)
    model = ToyV4Flash(cfg)
    model.eval()
    Path(__file__).resolve().parent.joinpath("_artifacts").mkdir(exist_ok=True)

    banner("Original V4-Flash vs this toy")
    print(cfg.comparison_table())
    print(f"params={model.num_parameters():,}")

    banner("How this differs from toy_dsv41 (V4.1)")
    rows = [("Aspect", "V4-Flash (this)", "V4.1-Flash (toy_dsv41)")] + list(V41_DIFF)
    # custom header print
    print(format_table([(a, b, c) for a, b, c in rows[1:]]))
    for a, b, c in V41_DIFF:
        print(f"  {a}:  V4={b}  |  V4.1={c}")

    banner("KV / replay note")
    for T in (32, 64):
        print(f"  {estimate_kv(cfg, T).summary()}")
        for n in estimate_kv(cfg, T).notes[-1:]:
            print(f"    {n}")

    banner("Forward trace")
    tokens = torch.randint(0, cfg.vocab_size, (1, 20))
    logits, trace = model(tokens, trace=True)
    print(f"  logits={tuple(logits.shape)}")
    for line in (trace["log"] or [])[:28]:
        print(f"  {line}")
    if len(trace["log"]) > 28:
        print(f"  ... ({len(trace['log']) - 28} more)")

    banner("Next")
    print("  python -m toy_v4_flash.tests")
    print("  python -m toy_v4_flash.train --preset better --steps 500")
    print("  python -m toy_v4_flash.train --generate-only")


if __name__ == "__main__":
    main()
