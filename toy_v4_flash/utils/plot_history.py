#!/usr/bin/env python3
"""Replot train/val curves from a saved history.json.

Usage:
  python -m toy_v4_flash.utils.plot_history
  python -m toy_v4_flash.utils.plot_history --history path/to/history.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from toy_v4_flash.utils.plotting import plot_from_file

DEFAULT_HISTORY = (
    Path(__file__).resolve().parents[1] / "_artifacts" / "pc_run" / "history.json"
)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot train/val loss from history.json")
    p.add_argument("--history", type=str, default=str(DEFAULT_HISTORY))
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="PNG path (default: same folder as history → loss_curves.png)",
    )
    args = p.parse_args()
    history = Path(args.history)
    if not history.exists():
        raise SystemExit(
            f"missing {history}\n"
            "Train first: python -m toy_v4_flash.train --preset better"
        )
    out = Path(args.out) if args.out else None
    png = plot_from_file(history, out)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
