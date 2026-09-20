"""Simple batch iterator over the char corpus."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch

from .dataset import random_batch


class BatchLoader:
    def __init__(
        self,
        ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        steps: int | None = None,
    ):
        self.ids = ids
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.steps = steps

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.steps if self.steps is not None else 10**9
        for _ in range(n):
            yield random_batch(self.ids, self.batch_size, self.seq_len, self.device)
