#!/usr/bin/env python3
"""Compare V4 Mini vs V4.1 Mini parameter counts and layout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "DeepSeekFlashV4-Mini"))
sys.path.insert(0, str(REPO / "DeepSeekFlashV4.1-Mini"))

from model import DeepSeekFlashV4Mini, preset as preset_v4  # type: ignore  # noqa: E402

# reload v4.1 under alias — clear conflicting 'model'
import importlib

sys.path.insert(0, str(REPO / "DeepSeekFlashV4.1-Mini"))
# Import v41 by path
import importlib.util


def load_pkg_model(pkg_dir: Path, attr: str):
    sys.path.insert(0, str(pkg_dir))
    # force fresh
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    mod = importlib.import_module("model")
    return getattr(mod, attr), getattr(mod, "preset")


def main() -> None:
    V4, preset4 = load_pkg_model(REPO / "DeepSeekFlashV4-Mini", "DeepSeekFlashV4Mini")
    m4 = V4(preset4("pc"))
    V41, preset41 = load_pkg_model(REPO / "DeepSeekFlashV4.1-Mini", "DeepSeekFlashV41Mini")
    cfg = preset41("pc")
    cfg.use_vision = False
    m41 = V41(cfg)
    print(f"V4-Mini     params={m4.num_parameters():,} layers={m4.cfg.n_layers}")
    print(
        f"V4.1-Mini   params={m41.num_parameters():,} "
        f"enc={m41.cfg.n_encoder_layers} dec={m41.cfg.n_decoder_layers}"
    )


if __name__ == "__main__":
    main()
