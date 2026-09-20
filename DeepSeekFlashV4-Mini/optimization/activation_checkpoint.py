"""Activation checkpointing wrapper."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_sequential(module, x, *args, use_reentrant: bool = False, **kwargs):
    if not torch.is_grad_enabled():
        return module(x, *args, **kwargs)
    return checkpoint(module, x, *args, use_reentrant=use_reentrant, **kwargs)
