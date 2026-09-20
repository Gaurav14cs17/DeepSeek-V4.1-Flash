"""Shared pytest fixtures / path bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_VER = Path(__file__).resolve().parents[1]
_REPO = _VER.parents[0]
# Put verification/ first so `pkg_import` / `reference` resolve; do NOT put both
# Mini packages on sys.path at once (they share top-level names `model`, etc.).
for p in (_VER, _REPO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@pytest.fixture
def seed():
    torch.manual_seed(0)
    return 0
