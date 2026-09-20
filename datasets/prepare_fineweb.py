#!/usr/bin/env python3
"""Prepare a capped FineWeb subset (Level 3). Opt-in sizes only."""

from __future__ import annotations

import sys

from download import main as download_main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--dataset", "fineweb", *sys.argv[1:]]
    download_main()
