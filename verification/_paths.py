"""Repo path helpers for verification runners/tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V4_PKG = REPO_ROOT / "DeepSeekFlashV4-Mini"
V41_PKG = REPO_ROOT / "DeepSeekFlashV4.1-Mini"
VERIFICATION = Path(__file__).resolve().parent
RESULTS = VERIFICATION / "results"


def ensure_pkg_paths() -> None:
    for p in (str(REPO_ROOT), str(V4_PKG), str(V41_PKG), str(VERIFICATION)):
        if p not in sys.path:
            sys.path.insert(0, p)
