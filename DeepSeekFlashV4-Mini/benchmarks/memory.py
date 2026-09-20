#!/usr/bin/env python3
"""Memory estimate benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import DeepSeekFlashV4Mini, preset
from optimization.memory import activation_estimate, param_bytes


def main() -> None:
    cfg = preset("pc")
    m = DeepSeekFlashV4Mini(cfg)
    print(f"param_bytes={param_bytes(m):,}")
    print(
        f"act_est={activation_estimate(8, 64, cfg.d_model, cfg.n_layers):,}"
    )


if __name__ == "__main__":
    main()
