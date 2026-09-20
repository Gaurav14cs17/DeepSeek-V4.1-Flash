"""DeepSeekFlashV4.1-Mini — CED + CSA2 + Engram + Single-Pass mHC + DSpark."""

from __future__ import annotations

from ._transformer import ToyDSV41
from .config import ToyConfig, V41Config, preset

DeepSeekFlashV41Mini = ToyDSV41
FlashV41Mini = ToyDSV41


class DeepSeekFlashV41MiniWrapped(ToyDSV41):
    """Alias class with lab naming; identical to ToyDSV41."""


# Prefer explicit name for labs
def build_model(cfg: ToyConfig | None = None) -> ToyDSV41:
    return ToyDSV41(cfg or ToyConfig())


__all__ = [
    "DeepSeekFlashV41Mini",
    "FlashV41Mini",
    "ToyDSV41",
    "ToyConfig",
    "V41Config",
    "preset",
    "build_model",
]
