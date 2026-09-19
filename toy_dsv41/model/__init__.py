"""Toy model: CED + CSA2 + MoE + Engram + mHC + RoPE + Vision + DSpark."""

from .config import ORIGINAL_VS_TOY, ToyConfig, format_original_vs_toy_table, preset
from .transformer import ToyDSV41

__all__ = [
    "ToyConfig",
    "ToyDSV41",
    "ORIGINAL_VS_TOY",
    "format_original_vs_toy_table",
    "preset",
]
