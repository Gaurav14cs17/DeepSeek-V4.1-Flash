"""Import helpers that avoid V4 / V4.1 package-name collisions (`model`, `optimization`)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]


def _purge(prefixes: Iterable[str]) -> None:
    doomed = [
        k
        for k in list(sys.modules)
        if any(k == p or k.startswith(p + ".") for p in prefixes)
    ]
    for k in doomed:
        del sys.modules[k]


def import_from_pkg(pkg_dirname: str, module: str) -> ModuleType:
    """
    Import `module` (e.g. 'model.config') from DeepSeekFlashV4-Mini or
    DeepSeekFlashV4.1-Mini after clearing colliding modules from sys.modules.
    """
    root = REPO / pkg_dirname
    _purge(("model", "optimization", "tokenizer", "inference", "training", "benchmarks"))
    # Drop prior pkg roots from path
    sys.path = [
        p
        for p in sys.path
        if "DeepSeekFlashV4-Mini" not in p and "DeepSeekFlashV4.1-Mini" not in p
    ]
    sys.path.insert(0, str(root))
    return importlib.import_module(module)
