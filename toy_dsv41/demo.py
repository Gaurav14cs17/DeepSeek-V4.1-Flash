#!/usr/bin/env python3
"""Architecture walkthrough for the restructured toy model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from toy_dsv41.model import ToyDSV41, preset
from toy_dsv41.model.engram import load_engram_from_disk, pack_engram_to_disk
from toy_dsv41.model.kv_cache import estimate_kv


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def print_project_flow() -> None:
    banner("Project flow (always follow this structure)")
    print(
        """
  DeepSeek-V4.1-Flash/
  ├── docs/                 ① paper (PDF)
  └── toy_dsv41/
      ├── model/            ② CED / CSA2 / MoE / Engram
      ├── data/             ③ small corpus
      ├── utils/            ④ plotting.py, plot_history.py
      ├── demo.py           ← you are here
      ├── train.py          ⑤ train + history + plot (uses utils/)
      └── _artifacts/pc_run/ ⑥ toy_pc.pt, history.json, loss_curves.png

  Flow:  docs/ → demo → train → _artifacts/ → --generate-only
"""
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["demo", "pc", "better"], default="demo")
    args = p.parse_args()

    print_project_flow()

    cfg = preset(args.preset, vocab_size=48)
    model = ToyDSV41(cfg)
    model.eval()
    out = Path(__file__).resolve().parent / "_artifacts"
    out.mkdir(exist_ok=True)

    banner("Structure detail (model/ + data/)")
    print(
        """
  toy_dsv41/model/   transformer, blocks, csa2, swa, moe, engram,
                     rope, mhc, vision, dspark, kv_cache, config
  toy_dsv41/data/    corpus_toy.txt, opensource/, tokenizer, build_corpus()
"""
    )

    banner("Original vs toy")
    print(cfg.comparison_table())
    print(f"params={model.num_parameters():,}")

    banner("Pack Engram (MiaAI pack stand-in)")
    path = out / "engram_demo.pt"
    pack_engram_to_disk(model.engram, str(path))
    load_engram_from_disk(model.engram, str(path))
    print(f"  wrote/reloaded {path}")

    banner("KV math")
    for T in (32, 64):
        print(f"  {estimate_kv(cfg, T).summary()}")

    banner("Prefill trace")
    tokens = torch.randint(0, cfg.vocab_size, (1, 20))
    logits, trace = model(tokens, phase="prefill", trace=True)
    print(f"  logits={tuple(logits.shape)}")
    for line in (trace["log"] or [])[:24]:
        print(f"  {line}")
    if len(trace["log"]) > 24:
        print(f"  ... ({len(trace['log']) - 24} more lines)")

    banner("Next in the flow")
    print("  python -m toy_dsv41.train --preset pc")
    print("  python -m toy_dsv41.train --preset better --steps 1000")
    print("  python -m toy_dsv41.utils.plot_history")
    print("  xdg-open toy_dsv41/_artifacts/pc_run/loss_curves.png")
    print("  python -m toy_dsv41.train --generate-only")


if __name__ == "__main__":
    main()
