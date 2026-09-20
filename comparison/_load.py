
from __future__ import annotations
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def load(pkg: str, attr: str):
    root = REPO / pkg
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    mod = importlib.import_module("model")
    return getattr(mod, attr), getattr(mod, "preset")
