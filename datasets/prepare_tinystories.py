#!/usr/bin/env python3
"""Prepare a capped TinyStories text shard (Level 1–2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from download import main as download_main


if __name__ == "__main__":
    # Reuse download.py CLI; defaults stay tiny.
    import sys

    sys.argv = [sys.argv[0], "--dataset", "tinystories", *sys.argv[1:]]
    download_main()
