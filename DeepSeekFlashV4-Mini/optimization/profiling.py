"""Simple timing profiler."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


class Timer:
    def __init__(self) -> None:
        self.times: Dict[str, float] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        yield
        self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - t0)

    def report(self) -> str:
        return "\n".join(f"{k}: {v * 1000:.2f} ms" for k, v in sorted(self.times.items()))
