"""Load package root onto sys.path (folder name has a dot and hyphen)."""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent


def ensure_path() -> Path:
    root = str(PKG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PKG_ROOT
