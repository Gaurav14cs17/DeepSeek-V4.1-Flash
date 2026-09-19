"""Toy DeepSeek-V4-Flash model package."""

from .config import ORIGINAL_VS_TOY, V41_DIFF, ToyConfig, format_table, preset
from .transformer import ToyV4Flash

__all__ = [
    "ToyConfig",
    "ToyV4Flash",
    "ORIGINAL_VS_TOY",
    "V41_DIFF",
    "format_table",
    "preset",
]
